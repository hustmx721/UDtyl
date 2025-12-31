"""
Embedding and linear-probe analysis for task/UID representations.

This script builds balanced embedding sets, applies the recommended
standardize → PCA(50) → UMAP pipeline, and reports quantitative
probes/cluster metrics for both clean and UD variants.

Key features:
- Balanced sampling per task and per user to avoid class imbalance artifacts.
- Penultimate-layer embedding extraction via a lightweight forward hook.
- Shared UMAP fit on clean embeddings with transform on UD to keep a
  consistent coordinate system (falls back to independent fits when
  transform is unavailable).
- Linear probes, silhouette/Davies–Bouldin scores, and inter/intra-class
  distance ratios for both task and UID labels.
- Four UMAP figures:
    * z_task colored by task (clean vs UD)
    * z_task colored by uid  (clean vs UD)
    * z_uid  colored by task (clean vs UD)
    * z_uid  colored by uid  (clean vs UD)
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, pairwise_distances, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from utils.dataset import set_seed
from utils.init_all import apply_thread_limits, init_args, load_all, set_args
from utils.preprocess import preprocessing


# --------------------
# Data loading helpers
# --------------------

OPEN_BMI = {"MI", "SSVEP", "ERP"}
M3CV = {"Rest", "Transient", "Steady", "P300", "Motor", "SSVEP_SA"}


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _process_in_chunks(data: np.ndarray, processor, max_workers: int = 4) -> np.ndarray:
    """Apply a CPU-bound processor in small chunks to save RAM."""
    from concurrent.futures import ThreadPoolExecutor

    processed = np.empty_like(data)
    chunk = max(1, min(data.shape[0], 1024))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for start in range(0, data.shape[0], chunk):
            futures.append((start, pool.submit(processor, data[start : start + chunk])))
        for start, fut in futures:
            processed[start : start + chunk] = fut.result()
    return processed


def load_full_dataset(
    dataset: str,
    data_root: Path,
    *,
    seed: int,
    max_workers: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the full (train + test) set with both task and UID labels.

    Returns:
        data: np.ndarray [N, C, T]
        task_labels: np.ndarray [N]
        uid_labels: np.ndarray [N]
        session_labels: np.ndarray [N] (0=train, 1=test) for optional stratification
    """
    if dataset in OPEN_BMI:
        task_root = data_root / "OpenBMI"
        train_task = _load_pickle(task_root / "Task" / dataset / "train.pkl")
        test_task = _load_pickle(task_root / "Task" / dataset / "test.pkl")

        train_x = train_task["data"].astype(np.float32)
        test_x = test_task["data"].astype(np.float32)
        train_x, test_x = [x.reshape((-1, x.shape[-2], x.shape[-1])) for x in [train_x, test_x]]

        task_train_y = train_task["label"].astype(np.int16).reshape(-1)
        task_test_y = test_task["label"].astype(np.int16).reshape(-1)

        subj_train = _load_pickle(task_root / "processed" / dataset / "train.pkl")
        subj_test = _load_pickle(task_root / "processed" / dataset / "test.pkl")
        uid_train_y = (subj_train["ori_train_s"] - 1).astype(np.int16).reshape(-1)
        uid_test_y = (subj_test["ori_test_s"] - 1).astype(np.int16).reshape(-1)

        fs = 250
    elif dataset in M3CV:
        task_root = data_root / "M3CV"
        train_task = _load_pickle(task_root / "Task" / f"Session1_{dataset}.pkl")
        test_task = _load_pickle(task_root / "Task" / f"Session2_{dataset}.pkl")

        train_x = train_task["data"][:, :-1, :].astype(np.float32)
        test_x = test_task["data"][:, :-1, :].astype(np.float32)

        task_train_y = train_task["label"].astype(np.int16).reshape(-1)
        task_test_y = test_task["label"].astype(np.int16).reshape(-1)

        subj_train = _load_pickle(task_root / "Train" / f"T_{dataset}.pkl")
        subj_test = _load_pickle(task_root / "Test" / f"{dataset}.pkl")
        uid_train_y = subj_train["label"].astype(np.int16).reshape(-1)
        uid_test_y = subj_test["label"].astype(np.int16).reshape(-1)

        fs = 250
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    raw_x = np.concatenate([train_x, test_x], axis=0)
    task_labels = np.concatenate([task_train_y, task_test_y], axis=0)
    uid_labels = np.concatenate([uid_train_y, uid_test_y], axis=0)
    session_labels = np.concatenate(
        [np.zeros_like(task_train_y, dtype=np.int16), np.ones_like(task_test_y, dtype=np.int16)]
    )

    processor = preprocessing(fs=fs).EEGpipline
    processed_x = _process_in_chunks(raw_x, processor, max_workers=max_workers)

    return processed_x, task_labels, uid_labels, session_labels


# -------------------------
# Sampling & dataset helper
# -------------------------


def sample_balanced(labels: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    """Sample up to ``per_class`` items per unique label without replacement."""
    selected: List[int] = []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        take = min(per_class, idx.size)
        chosen = rng.choice(idx, size=take, replace=False)
        selected.extend(chosen.tolist())
    return np.array(sorted(selected), dtype=np.int64)


class DualLabelDataset(Dataset):
    def __init__(self, data: np.ndarray, task_labels: np.ndarray, uid_labels: np.ndarray, indices: Iterable[int]):
        idx = np.array(list(indices), dtype=np.int64)
        self.data = torch.from_numpy(data[idx]).unsqueeze(1)  # [N, 1, C, T]
        self.task_labels = torch.from_numpy(task_labels[idx]).long()
        self.uid_labels = torch.from_numpy(uid_labels[idx]).long()

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, i: int):
        return self.data[i], self.task_labels[i], self.uid_labels[i]


# -----------------------
# Embedding extraction
# -----------------------


def build_model(model_name: str, dataset: str, *, is_task: bool, ckpt: Path, gpuid: int, threads: int) -> torch.nn.Module:
    args = init_args()
    args.dataset = dataset
    args.is_task = is_task
    args.model = model_name
    args.gpuid = gpuid
    args = set_args(args)
    apply_thread_limits(threads)
    model, _, device = load_all(args)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def _get_penultimate_hook(model: torch.nn.Module):
    target = None
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            target = module
            break
    if target is None:
        raise RuntimeError("No Linear layer found to hook penultimate features.")

    features: List[torch.Tensor] = []

    def hook(module, inputs, output):
        features.append(inputs[0].detach())

    handle = target.register_forward_hook(hook)
    return handle, features


def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    handle, features = _get_penultimate_hook(model)
    all_task: List[torch.Tensor] = []
    all_uid: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            x, task_y, uid_y = batch
            x = x.to(device, non_blocking=True)
            _ = model(x)
            all_task.append(task_y)
            all_uid.append(uid_y)

    handle.remove()
    feats = torch.cat(features, dim=0).cpu().numpy()
    task_labels = torch.cat(all_task, dim=0).numpy()
    uid_labels = torch.cat(all_uid, dim=0).numpy()
    return feats, task_labels, uid_labels


# -----------------------
# Dimensionality reduction
# -----------------------


def reduce_embeddings(
    clean_feats: np.ndarray,
    ud_feats: np.ndarray,
    *,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    pca_dims: int = 50,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Standardize -> PCA -> UMAP (fit on clean, transform UD when available).
    Falls back to TSNE when umap-learn is unavailable.
    """
    scaler = StandardScaler()
    clean_std = scaler.fit_transform(clean_feats)
    ud_std = scaler.transform(ud_feats)

    pca = PCA(n_components=min(pca_dims, clean_std.shape[1]))
    clean_pca = pca.fit_transform(clean_std)
    ud_pca = pca.transform(ud_std)

    projections: Dict[str, np.ndarray] = {"pca_clean": clean_pca, "pca_ud": ud_pca}

    try:
        import umap  # type: ignore

        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=seed,
        )
        clean_2d = umap_model.fit_transform(clean_pca)
        if hasattr(umap_model, "transform"):
            ud_2d = umap_model.transform(ud_pca)
        else:
            ud_2d = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=seed,
            ).fit_transform(ud_pca)
    except Exception as exc:  # pragma: no cover - best-effort fallback
        print(f"UMAP unavailable ({exc}); falling back to TSNE.")
        tsne = TSNE(
            n_components=2,
            random_state=seed,
            perplexity=max(5, min(30, clean_pca.shape[0] // 5)),
            metric=metric,
        )
        clean_2d = tsne.fit_transform(clean_pca)
        ud_2d = tsne.fit_transform(ud_pca)

    projections["embed_clean"] = clean_2d
    projections["embed_ud"] = ud_2d
    return clean_2d, ud_2d, projections


# -----------------------
# Metrics & probing
# -----------------------


def _safe_silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if unique.size < 2 or x.shape[0] < 2:
        return float("nan")
    try:
        return silhouette_score(x, labels)
    except Exception:
        return float("nan")


def _safe_db(x: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if unique.size < 2 or x.shape[0] < 2:
        return float("nan")
    try:
        return davies_bouldin_score(x, labels)
    except Exception:
        return float("nan")


def _distance_ratio(x: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if unique.size < 2 or x.shape[0] < 2:
        return float("nan")
    dists = pairwise_distances(x)
    intra, inter = [], []
    for c in unique:
        mask = labels == c
        if mask.sum() > 1:
            intra.append(dists[np.ix_(mask, mask)].mean())
        inter.append(dists[np.ix_(mask, ~mask)].mean())
    if not intra or not inter:
        return float("nan")
    return float(np.nanmean(inter) / max(np.nanmean(intra), 1e-8))


def linear_probe(x: np.ndarray, y: np.ndarray, seed: int, max_iter: int, n_jobs: int) -> float:
    if np.unique(y).size < 2 or x.shape[0] < 4:
        return float("nan")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, n_jobs=n_jobs, multi_class="auto"),
    )
    clf.fit(x_train, y_train)
    return clf.score(x_test, y_test)


# -----------------------
# Plotting
# -----------------------


def _scatter(ax, points: np.ndarray, labels: np.ndarray, title: str):
    scatter = ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab20", s=6, alpha=0.8)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return scatter


def save_umap_pair(
    clean_2d: np.ndarray,
    ud_2d: np.ndarray,
    clean_labels: np.ndarray,
    ud_labels: np.ndarray,
    *,
    title_clean: str,
    title_ud: str,
    outfile: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _scatter(axes[0], clean_2d, clean_labels, title_clean)
    _scatter(axes[1], ud_2d, ud_labels, title_ud)
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


# -----------------------
# Main pipeline
# -----------------------


def run(args: argparse.Namespace):
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")

    clean_x, clean_task, clean_uid, _ = load_full_dataset(
        args.dataset, Path(args.data_root), seed=args.seed, max_workers=args.data_workers
    )
    ud_root = Path(args.ud_data_root or args.data_root)
    ud_x, ud_task, ud_uid, _ = load_full_dataset(
        args.dataset if args.ud_dataset is None else args.ud_dataset,
        ud_root,
        seed=args.seed,
        max_workers=args.data_workers,
    )

    task_idx = sample_balanced(clean_task, args.samples_per_task, rng)
    uid_idx = sample_balanced(clean_uid, args.samples_per_user, rng)
    selected_idx = np.unique(np.concatenate([task_idx, uid_idx]))

    ud_task_idx = sample_balanced(ud_task, args.samples_per_task, rng)
    ud_uid_idx = sample_balanced(ud_uid, args.samples_per_user, rng)
    ud_selected_idx = np.unique(np.concatenate([ud_task_idx, ud_uid_idx]))

    clean_ds = DualLabelDataset(clean_x, clean_task, clean_uid, selected_idx)
    ud_ds = DualLabelDataset(ud_x, ud_task, ud_uid, ud_selected_idx)

    clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    ud_loader = DataLoader(ud_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    task_model_clean = build_model(
        args.task_model, args.dataset, is_task=True, ckpt=Path(args.task_ckpt_clean), gpuid=args.gpuid, threads=args.torch_threads
    )
    task_model_ud = build_model(
        args.task_model, args.dataset, is_task=True, ckpt=Path(args.task_ckpt_ud), gpuid=args.gpuid, threads=args.torch_threads
    )
    uid_model_clean = build_model(
        args.uid_model, args.dataset, is_task=False, ckpt=Path(args.uid_ckpt_clean), gpuid=args.gpuid, threads=args.torch_threads
    )
    uid_model_ud = build_model(
        args.uid_model, args.dataset, is_task=False, ckpt=Path(args.uid_ckpt_ud), gpuid=args.gpuid, threads=args.torch_threads
    )

    z_task_clean, task_labels_clean, uid_labels_clean = extract_embeddings(task_model_clean, clean_loader, device)
    z_task_ud, task_labels_ud, uid_labels_ud = extract_embeddings(task_model_ud, ud_loader, device)
    z_uid_clean, task_labels_uid_clean, uid_labels_uid_clean = extract_embeddings(uid_model_clean, clean_loader, device)
    z_uid_ud, task_labels_uid_ud, uid_labels_uid_ud = extract_embeddings(uid_model_ud, ud_loader, device)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Dict[str, float]] = {}

    def record_metrics(prefix: str, feats: np.ndarray, t_labels: np.ndarray, u_labels: np.ndarray):
        scaled = StandardScaler().fit_transform(feats)
        metrics[prefix] = {
            "silhouette_task": _safe_silhouette(scaled, t_labels),
            "silhouette_uid": _safe_silhouette(scaled, u_labels),
            "davies_bouldin_task": _safe_db(scaled, t_labels),
            "davies_bouldin_uid": _safe_db(scaled, u_labels),
            "distance_ratio_task": _distance_ratio(scaled, t_labels),
            "distance_ratio_uid": _distance_ratio(scaled, u_labels),
            "probe_task_acc": linear_probe(scaled, t_labels, seed=args.seed, max_iter=args.probe_max_iter, n_jobs=args.probe_jobs),
            "probe_uid_acc": linear_probe(scaled, u_labels, seed=args.seed, max_iter=args.probe_max_iter, n_jobs=args.probe_jobs),
        }

    record_metrics("z_task_clean", z_task_clean, task_labels_clean, uid_labels_clean)
    record_metrics("z_task_ud", z_task_ud, task_labels_ud, uid_labels_ud)
    record_metrics("z_uid_clean", z_uid_clean, task_labels_uid_clean, uid_labels_uid_clean)
    record_metrics("z_uid_ud", z_uid_ud, task_labels_uid_ud, uid_labels_uid_ud)

    z_task_clean_2d_task, z_task_ud_2d_task, proj_task = reduce_embeddings(
        z_task_clean,
        z_task_ud,
        seed=args.seed,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        pca_dims=args.pca_dims,
    )
    z_uid_clean_2d_uid, z_uid_ud_2d_uid, proj_uid = reduce_embeddings(
        z_uid_clean,
        z_uid_ud,
        seed=args.seed,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        pca_dims=args.pca_dims,
    )

    save_umap_pair(
        z_task_clean_2d_task,
        z_task_ud_2d_task,
        task_labels_clean,
        task_labels_ud,
        title_clean="z_task | task color (clean)",
        title_ud="z_task | task color (UD)",
        outfile=outdir / "z_task_taskcolor.png",
    )
    save_umap_pair(
        z_task_clean_2d_task,
        z_task_ud_2d_task,
        uid_labels_clean,
        uid_labels_ud,
        title_clean="z_task | uid color (clean)",
        title_ud="z_task | uid color (UD)",
        outfile=outdir / "z_task_uidcolor.png",
    )
    save_umap_pair(
        z_uid_clean_2d_uid,
        z_uid_ud_2d_uid,
        task_labels_uid_clean,
        task_labels_uid_ud,
        title_clean="z_uid | task color (clean)",
        title_ud="z_uid | task color (UD)",
        outfile=outdir / "z_uid_taskcolor.png",
    )
    save_umap_pair(
        z_uid_clean_2d_uid,
        z_uid_ud_2d_uid,
        uid_labels_uid_clean,
        uid_labels_uid_ud,
        title_clean="z_uid | uid color (clean)",
        title_ud="z_uid | uid color (UD)",
        outfile=outdir / "z_uid_uidcolor.png",
    )

    with open(outdir / "projections.npz", "wb") as f:
        np.savez_compressed(
            f,
            z_task_clean=z_task_clean,
            z_task_ud=z_task_ud,
            z_uid_clean=z_uid_clean,
            z_uid_ud=z_uid_ud,
            **{f"task_{k}": v for k, v in proj_task.items()},
            **{f"uid_{k}": v for k, v in proj_uid.items()},
        )

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Figures saved to {outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedding visualization and probe analysis")
    parser.add_argument("--dataset", required=True, help="Dataset name (MI/SSVEP/ERP/Rest/Transient/Steady/P300/Motor/SSVEP_SA)")
    parser.add_argument("--ud_dataset", default=None, help="Optional dataset name for UD variant (defaults to --dataset)")
    parser.add_argument("--data_root", default="/mnt/data1/tyl/data", help="Root directory for clean data")
    parser.add_argument("--ud_data_root", default=None, help="Root directory for UD data (defaults to --data_root)")
    parser.add_argument("--task_model", default="EEGNet")
    parser.add_argument("--uid_model", default="EEGNet")
    parser.add_argument("--task_ckpt_clean", required=True)
    parser.add_argument("--task_ckpt_ud", required=True)
    parser.add_argument("--uid_ckpt_clean", required=True)
    parser.add_argument("--uid_ckpt_ud", required=True)
    parser.add_argument("--samples_per_task", type=int, default=300)
    parser.add_argument("--samples_per_user", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--umap_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_metric", type=str, default="cosine")
    parser.add_argument("--pca_dims", type=int, default=50)
    parser.add_argument("--probe_max_iter", type=int, default=500)
    parser.add_argument("--probe_jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--data_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="embedding_probe_outputs")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

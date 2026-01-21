"""
Minimal embedding visualization for clean vs protected samples.

Workflow:
1) Load testloader (task or UID) from the existing dataset pipeline.
2) Load a trained delta perturber (STFTDeltaPerturber) and teacher model.
3) Extract penultimate-layer features for clean and protected samples.
4) Run t-SNE on the combined features and plot clean vs protected.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from Distill import STFTDeltaPerturber
from utils.dataset import set_seed
from utils.init_all import apply_thread_limits, init_args, load_all, load_data, set_args


def build_model(dataset: str, model_name: str, is_task: bool, ckpt: Path, gpuid: int, threads: int) -> nn.Module:
    args = init_args()
    args.dataset = dataset
    args.model = model_name
    args.is_task = is_task
    args.gpuid = gpuid
    args = set_args(args)
    apply_thread_limits(threads)
    model, _, device = load_all(args)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def load_delta(delta_path: Path, device: torch.device) -> STFTDeltaPerturber:
    payload = torch.load(delta_path, map_location=device)
    if isinstance(payload, dict) and "delta" in payload:
        perturber = payload["delta"]
    elif isinstance(payload, STFTDeltaPerturber):
        perturber = payload
    else:
        raise ValueError(
            "Unsupported delta checkpoint format. Expected a saved STFTDeltaPerturber "
            "or a dict with key 'delta'."
        )
    perturber.to(device)
    perturber.eval()
    return perturber


def _get_penultimate_hook(model: nn.Module):
    target = None
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            target = module
            break
    if target is None:
        raise RuntimeError("No Linear layer found to hook penultimate features.")

    features: List[torch.Tensor] = []

    def hook(_module, inputs, _output):
        features.append(inputs[0].detach())

    handle = target.register_forward_hook(hook)
    return handle, features


@torch.no_grad()
def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    perturber: STFTDeltaPerturber | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    handle, features = _get_penultimate_hook(model)
    labels: List[torch.Tensor] = []

    for batch in dataloader:
        x, y = batch[:2]
        x = x.to(device, non_blocking=True)
        if perturber is not None:
            x = perturber(x)
        _ = model(x)
        labels.append(y.detach())

    handle.remove()
    feats = torch.cat(features, dim=0).cpu().numpy()
    labels_np = torch.cat(labels, dim=0).cpu().numpy()
    return feats, labels_np


def plot_tsne(
    clean_feats: np.ndarray,
    prot_feats: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    seed: int,
):
    combined = np.concatenate([clean_feats, prot_feats], axis=0)
    tsne = TSNE(n_components=2, random_state=seed, init="pca")
    embedded = tsne.fit_transform(combined)
    clean_2d = embedded[: clean_feats.shape[0]]
    prot_2d = embedded[clean_feats.shape[0] :]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(clean_2d[:, 0], clean_2d[:, 1], c=labels, cmap="tab20", s=6, alpha=0.8)
    axes[0].set_title("clean (t-SNE)")
    axes[1].scatter(prot_2d[:, 0], prot_2d[:, 1], c=labels, cmap="tab20", s=6, alpha=0.8)
    axes[1].set_title("protected (t-SNE)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple clean vs protected t-SNE visualization")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="EEGNet")
    parser.add_argument("--ckpt", required=True, help="Teacher model checkpoint path")
    parser.add_argument("--delta_ckpt", required=True, help="Delta perturber checkpoint path")
    parser.add_argument("--is_task", action="store_true", help="Use task labels (default: UID labels)")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output", default="embedding_tsne.png")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    args_cfg = init_args()
    args_cfg.dataset = args.dataset
    args_cfg.model = args.model
    args_cfg.is_task = args.is_task
    args_cfg.gpuid = args.gpuid
    args_cfg.bs = args.batch_size
    args_cfg = set_args(args_cfg)

    _train, _val, testloader = load_data(args_cfg)
    device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")

    model = build_model(args.dataset, args.model, args.is_task, Path(args.ckpt), args.gpuid, args.torch_threads)
    perturber = load_delta(Path(args.delta_ckpt), device)

    clean_feats, labels = extract_features(model, testloader, device, perturber=None)
    prot_feats, _ = extract_features(model, testloader, device, perturber=perturber)

    plot_tsne(clean_feats, prot_feats, labels, Path(args.output), seed=args.seed)
    print(f"Saved t-SNE figure to {args.output}")


if __name__ == "__main__":
    main()

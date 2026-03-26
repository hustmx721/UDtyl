import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from Distill import STFTDeltaPerturber
from utils.dataset import set_seed
from utils.init_all import load_data, set_args
from utils.models import LoadModel


# M3CV 64导联参考顺序（用户指定）
M3CV_CHAN_LABELS = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8',
    'Fz','Cz','Pz','FC1','FC2','CP1','CP2','FC5','FC6','CP5','CP6','FT9','FT10','TP9','TP10',
    'F1','F2','C1','C2','P1','P2','AF3','AF4','FC3','FC4','CP3','CP4','PO3','PO4','F5','F6',
    'C5','C6','P5','P6','AF7','AF8','FT7','FT8','TP7','TP8','PO7','PO8','Fpz','CPz','POz','Oz','FCz'
]


def get_channel_order(labels):
    """
    返回：排序索引、排序后的标签、分割线位置、脑区中心点和名称
    """

    def get_info(l):
        l = l.upper()
        if l.startswith('FP'): return 10, 'Fp'
        if l.startswith('AF'): return 20, 'AF'
        if l.startswith('F') and 'C' not in l and 'T' not in l: return 30, 'Frontal'
        if l.startswith('FC'): return 40, 'FC'
        if l.startswith('C') and 'P' not in l: return 50, 'Central'
        if l.startswith('CP'): return 60, 'CP'
        if l.startswith('P') and 'O' not in l: return 70, 'Parietal'
        if l.startswith('PO'): return 80, 'PO'
        if l.startswith('O'): return 90, 'Occipital'
        if l.startswith('FT'): return 100, 'FT'
        if l.startswith('T') and 'P' not in l: return 110, 'Temporal'
        if l.startswith('TP'): return 120, 'TP'
        return 200, 'Other'

    indexed_info = []
    for i, l in enumerate(labels):
        score, region = get_info(l)
        indexed_info.append({'idx': i, 'label': l, 'score': score, 'region': region})

    sorted_info = sorted(indexed_info, key=lambda x: (x['score'], x['label']))

    sorted_indices = [x['idx'] for x in sorted_info]
    sorted_labels = [x['label'] for x in sorted_info]

    boundaries = []
    region_marks = []

    current_region = sorted_info[0]['region']
    start_pos = 0

    for i, info in enumerate(sorted_info):
        if info['region'] != current_region:
            boundaries.append(i)
            region_marks.append({'name': current_region, 'center': (start_pos + i - 1) / 2})
            start_pos = i
            current_region = info['region']

    region_marks.append({'name': current_region, 'center': (start_pos + len(sorted_info) - 1) / 2})
    return sorted_indices, sorted_labels, boundaries, region_marks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("SpectraDistill Topo visualization (ME/Motor + ShallowConvNet)")
    parser.add_argument("--dataset", type=str, default="Motor")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpuid", type=int, default=0)

    parser.add_argument("--task_model", type=str, default="ShallowConvNet")
    parser.add_argument("--uid_model", type=str, default="ShallowConvNet")
    parser.add_argument("--task_ckpt", type=str, default="")
    parser.add_argument("--uid_ckpt", type=str, default="")
    parser.add_argument("--delta_ckpt", type=str, required=True)

    parser.add_argument("--channel_names", type=str, default="", help="txt/json 文件，覆盖默认 M3CV 通道顺序")
    parser.add_argument("--out_dir", type=str, default="figures/spectradistill_topo")
    parser.add_argument("--representative_mode", type=str, default="task_correct_uid_flip", choices=["task_correct_uid_flip", "first"])
    return parser.parse_args()


def load_channel_names(path: str, channel_count: int) -> List[str]:
    if path:
        p = Path(path)
        if p.suffix.lower() == ".json":
            names = json.loads(p.read_text(encoding="utf-8"))
        else:
            names = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(names) != channel_count:
            raise ValueError(f"channel_names 数量({len(names)}) != 通道数({channel_count})")
        return names
    if channel_count == len(M3CV_CHAN_LABELS):
        return M3CV_CHAN_LABELS
    return [f"Ch{i+1}" for i in range(channel_count)]


def to_device(batch, device: torch.device):
    x = batch[0]
    y = batch[1]
    if torch.is_tensor(x):
        x = x.to(device)
    if torch.is_tensor(y):
        y = y.to(device)
    return x, y


def build_model(model_name: str, ckpt: str, nclass: int, chans: int, samples: int, device: torch.device):
    if not ckpt:
        return None
    model = LoadModel(model_name, chans, samples, nclass).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def stft_log_mag(x: torch.Tensor, perturber: STFTDeltaPerturber) -> torch.Tensor:
    b, _, c, t = x.shape
    x_flat = x.squeeze(1).reshape(b * c, t)
    s = torch.stft(
        x_flat,
        n_fft=perturber.n_fft,
        hop_length=perturber.hop_length,
        win_length=perturber.win_length,
        window=perturber.window,
        center=True,
        return_complex=True,
    )
    s = s.reshape(b, c, s.shape[-2], s.shape[-1])
    return torch.log(torch.abs(s) + 1e-6)


@torch.no_grad()
def select_representative_sample(testloader, perturber, task_model, uid_model, mode: str, device):
    fallback = None
    for batch in testloader:
        x, y_task = to_device(batch, device)
        x_prot = perturber(x)
        if fallback is None:
            fallback = (x[0:1].detach().cpu(), x_prot[0:1].detach().cpu(), int(y_task[0].item()))

        if mode == "first" or task_model is None or uid_model is None:
            return fallback

        task_clean_pred = task_model(x).argmax(dim=1)
        task_prot_pred = task_model(x_prot).argmax(dim=1)
        uid_clean_pred = uid_model(x).argmax(dim=1)
        uid_prot_pred = uid_model(x_prot).argmax(dim=1)

        mask = (task_clean_pred == y_task) & (task_prot_pred == y_task) & (uid_clean_pred != uid_prot_pred)
        idxs = torch.where(mask)[0]
        if idxs.numel() > 0:
            i = int(idxs[0].item())
            return x[i:i+1].detach().cpu(), x_prot[i:i+1].detach().cpu(), int(y_task[i].item())

    if fallback is None:
        raise RuntimeError("测试集为空")
    return fallback


@torch.no_grad()
def aggregate_stats(testloader, perturber: STFTDeltaPerturber, device: torch.device):
    band_defs: Dict[str, Tuple[float, float]] = {
        "delta(1-4)": (1.0, 4.0),
        "theta(4-8)": (4.0, 8.0),
        "alpha(8-13)": (8.0, 13.0),
        "beta(13-30)": (13.0, 30.0),
        "gamma(30-45)": (30.0, 45.0),
    }

    sfreq = 250.0
    freq_bins = perturber.freq_bins
    freqs = np.fft.rfftfreq(perturber.n_fft, d=1.0 / sfreq)
    if len(freqs) != freq_bins:
        freqs = np.linspace(0, sfreq / 2, freq_bins)

    band_energy = {k: 0.0 for k in band_defs}
    total_abs = 0.0
    band_power_clean = {k: [] for k in band_defs}
    band_power_prot = {k: [] for k in band_defs}
    channel_scores = []

    for batch in testloader:
        x, _ = to_device(batch, device)
        x_p = perturber(x)

        a = stft_log_mag(x, perturber)
        ap = stft_log_mag(x_p, perturber)
        da = ap - a
        da_abs = da.abs()

        total_abs += da_abs.sum().item()
        channel_scores.append(da_abs.mean(dim=(0, 2, 3)).cpu().numpy())

        mag_clean = torch.exp(a)
        mag_prot = torch.exp(ap)

        for band_name, (f_low, f_high) in band_defs.items():
            mask = (freqs >= f_low) & (freqs < f_high)
            if not mask.any():
                continue
            f_idx = torch.from_numpy(np.where(mask)[0]).to(device)
            band_energy[band_name] += da_abs[:, :, f_idx, :].sum().item()

            p_clean = (mag_clean[:, :, f_idx, :] ** 2).mean(dim=(1, 2, 3))
            p_prot = (mag_prot[:, :, f_idx, :] ** 2).mean(dim=(1, 2, 3))
            band_power_clean[band_name].extend(p_clean.cpu().tolist())
            band_power_prot[band_name].extend(p_prot.cpu().tolist())

    band_ratio = {k: v / (total_abs + 1e-12) for k, v in band_energy.items()}
    rel_power = {
        k: (np.mean(band_power_prot[k]) - np.mean(band_power_clean[k])) / (np.mean(band_power_clean[k]) + 1e-12)
        for k in band_defs
    }
    channel_score = np.mean(np.stack(channel_scores, axis=0), axis=0)

    return band_ratio, rel_power, channel_score, band_power_clean, band_power_prot


def plot_figure1(x: np.ndarray, x_p: np.ndarray, perturber: STFTDeltaPerturber, ch_names: Sequence[str], out_path: Path):
    dx = x_p - x

    x_t = torch.tensor(x[None, None], dtype=torch.float32)
    xp_t = torch.tensor(x_p[None, None], dtype=torch.float32)
    a = stft_log_mag(x_t, perturber)[0].mean(dim=0).cpu().numpy()
    ap = stft_log_mag(xp_t, perturber)[0].mean(dim=0).cpu().numpy()
    da = ap - a

    t = np.arange(x.shape[-1]) / 250.0
    chan_map = {n: i for i, n in enumerate(ch_names)}
    chosen = [n for n in ["C3", "Cz", "C4"] if n in chan_map]
    if len(chosen) < 3:
        chosen = [ch_names[0], ch_names[len(ch_names)//2], ch_names[-1]]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for n in chosen:
        axes[0, 0].plot(t, x[chan_map[n]], label=n, linewidth=1.0)
        axes[0, 1].plot(t, x_p[chan_map[n]], label=n, linewidth=1.0)
        axes[0, 2].plot(t, dx[chan_map[n]], label=n, linewidth=1.0)

    axes[0, 0].set_title("(a) Original EEG")
    axes[0, 1].set_title("(b) Protected EEG")
    axes[0, 2].set_title("(c) Δx")
    for j in range(3):
        axes[0, j].set_xlabel("Time (s)")
        axes[0, j].set_ylabel("Amplitude")
        axes[0, j].legend(fontsize=8)

    vmin = min(a.min(), ap.min())
    vmax = max(a.max(), ap.max())
    im1 = axes[1, 0].imshow(a, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    im2 = axes[1, 1].imshow(ap, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    dmax = np.abs(da).max() + 1e-9
    im3 = axes[1, 2].imshow(da, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-dmax, vmax=dmax)

    axes[1, 0].set_title("(d) Mean log-STFT clean")
    axes[1, 1].set_title("(e) Mean log-STFT protected")
    axes[1, 2].set_title("(f) ΔA")

    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return a, ap, da, dx


def plot_figure2(
    band_ratio: Dict[str, float],
    rel_power: Dict[str, float],
    channel_score: np.ndarray,
    channel_names: Sequence[str],
    out_path: Path,
):
    bands = list(band_ratio.keys())
    br = np.array([band_ratio[b] for b in bands])
    rp = np.array([rel_power[b] for b in bands])

    sorted_indices, sorted_labels, boundaries, region_marks = get_channel_order(channel_names)
    ch_sorted = channel_score[sorted_indices]

    fig = plt.figure(figsize=(18, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(bands, br, color="#4C72B0")
    ax1.set_title("(a) Perturbation ratio by band")
    ax1.tick_params(axis="x", rotation=20)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(bands, rp, color="#DD8452")
    ax2.axhline(0, color="k", linewidth=1)
    ax2.set_title("(b) Relative band-power change")
    ax2.tick_params(axis="x", rotation=20)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(np.arange(len(ch_sorted)), ch_sorted, color="#55A868", linewidth=1.8)
    for b in boundaries:
        ax3.axvline(b - 0.5, color="gray", linestyle="--", linewidth=0.8)
    for m in region_marks:
        ax3.text(m['center'], ch_sorted.max() * 1.03, m['name'], ha="center", va="bottom", fontsize=8, rotation=45)
    ax3.set_xlim(-0.5, len(ch_sorted) - 0.5)
    ax3.set_title("(c) Channel perturbation (region-ordered)")
    ax3.set_ylabel("Mean |ΔA|")
    ax3.set_xticks(np.arange(len(ch_sorted)))
    ax3.set_xticklabels(sorted_labels, rotation=90, fontsize=6)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return sorted_indices, sorted_labels, boundaries, region_marks


def save_features(
    out_dir: Path,
    x: np.ndarray,
    x_p: np.ndarray,
    dx: np.ndarray,
    a: np.ndarray,
    ap: np.ndarray,
    da: np.ndarray,
    band_ratio: Dict[str, float],
    rel_power: Dict[str, float],
    channel_score: np.ndarray,
    channel_names: Sequence[str],
    sorted_indices: Sequence[int],
    sorted_labels: Sequence[str],
    boundaries: Sequence[int],
    region_marks: Sequence[Dict[str, float]],
    band_power_clean: Dict[str, List[float]],
    band_power_prot: Dict[str, List[float]],
):
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "topo_features.npz",
        x_clean=x,
        x_protected=x_p,
        delta_x=dx,
        A_log_clean=a,
        A_log_protected=ap,
        delta_A=da,
        channel_score=channel_score,
        channel_names=np.array(channel_names, dtype=object),
        sorted_indices=np.array(sorted_indices),
        sorted_labels=np.array(sorted_labels, dtype=object),
        boundaries=np.array(boundaries),
    )

    pd.DataFrame({"band": list(band_ratio.keys()), "R_b": list(band_ratio.values()), "DeltaP_b": [rel_power[k] for k in band_ratio.keys()]}).to_csv(
        out_dir / "band_statistics.csv", index=False
    )

    pd.DataFrame(
        {
            "channel": list(channel_names),
            "mean_abs_deltaA": channel_score,
            "sorted_rank": pd.Series(range(len(sorted_indices)), index=np.array(sorted_indices)).reindex(range(len(channel_names))).values,
        }
    ).to_csv(out_dir / "channel_statistics.csv", index=False)

    pd.DataFrame(region_marks).to_csv(out_dir / "region_marks.csv", index=False)

    serializable = {
        "band_ratio": band_ratio,
        "rel_power": rel_power,
        "band_power_clean": band_power_clean,
        "band_power_protected": band_power_prot,
    }
    (out_dir / "summary_stats.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")

    class Obj:
        pass

    cfg = Obj()
    cfg.dataset = args.dataset
    cfg.seed = args.seed
    cfg.bs = args.batch_size
    cfg.gpuid = args.gpuid
    cfg.is_task = True
    cfg = set_args(cfg)

    _, _, testloader = load_data(cfg, include_index=False)
    sample = next(iter(testloader))[0]
    _, _, chans, time_steps = sample.shape

    state = torch.load(args.delta_ckpt, map_location=device)
    if isinstance(state, dict) and "delta" in state and isinstance(state["delta"], STFTDeltaPerturber):
        perturber = state["delta"].to(device)
    elif isinstance(state, STFTDeltaPerturber):
        perturber = state.to(device)
    else:
        raise ValueError("无法识别 delta checkpoint 格式")
    perturber.eval()

    task_model = build_model(args.task_model, args.task_ckpt, cfg.nclass, chans, time_steps, device)

    uid_cfg = Obj()
    uid_cfg.dataset = args.dataset
    uid_cfg.seed = args.seed
    uid_cfg.bs = args.batch_size
    uid_cfg.gpuid = args.gpuid
    uid_cfg.is_task = False
    uid_cfg = set_args(uid_cfg)
    uid_model = build_model(args.uid_model, args.uid_ckpt, uid_cfg.nclass, chans, time_steps, device)

    ch_names = load_channel_names(args.channel_names, chans)

    x, x_p, y = select_representative_sample(testloader, perturber, task_model, uid_model, args.representative_mode, device)
    x_np = x.squeeze(0).squeeze(0).numpy()
    xp_np = x_p.squeeze(0).squeeze(0).numpy()

    fig1_path = out_dir / "fig1_sample_timefreq.png"
    a, ap, da, dx = plot_figure1(x_np, xp_np, perturber.cpu(), ch_names, fig1_path)

    band_ratio, rel_power, channel_score, band_power_clean, band_power_prot = aggregate_stats(testloader, perturber.to(device), device)
    fig2_path = out_dir / "fig2_band_region_stats.png"
    sorted_indices, sorted_labels, boundaries, region_marks = plot_figure2(band_ratio, rel_power, channel_score, ch_names, fig2_path)

    save_features(
        out_dir=out_dir,
        x=x_np,
        x_p=xp_np,
        dx=dx,
        a=a,
        ap=ap,
        da=da,
        band_ratio=band_ratio,
        rel_power=rel_power,
        channel_score=channel_score,
        channel_names=ch_names,
        sorted_indices=sorted_indices,
        sorted_labels=sorted_labels,
        boundaries=boundaries,
        region_marks=region_marks,
        band_power_clean=band_power_clean,
        band_power_prot=band_power_prot,
    )

    print(f"Representative sample task label: {y}")
    print(f"Saved figure1: {fig1_path}")
    print(f"Saved figure2: {fig2_path}")
    print(f"Saved features to: {out_dir}")


if __name__ == "__main__":
    main()

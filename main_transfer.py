import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from Distill import EOTDistribution, IdentityTransform, STFTDeltaPerturber
from evaluate import calculate_metrics, evaluate
from utils.Logging import Logger
from utils.dataset import set_seed
from utils.init_all import apply_thread_limits, load_all, load_data, set_args


def format_lambda_tag(lambda_task: float, lambda_uid: float, lambda_reg: float) -> str:
    return f"lt{lambda_task}_lu{lambda_uid}_lr{lambda_reg}"


def build_eot_distribution(args: argparse.Namespace) -> Optional[EOTDistribution]:
    if args.disable_eot:
        return None

    enabled = any(
        [
            args.eot_shift_prob > 0 and args.eot_shift > 0,
            args.eot_scale_prob > 0 and args.eot_scale,
            args.eot_channel_dropout_prob > 0 and args.eot_channel_dropout > 0,
            args.eot_resample_prob > 0 and args.eot_resample > 0,
        ]
    )
    if not enabled:
        return None

    return EOTDistribution(
        max_shift=args.eot_shift,
        shift_prob=args.eot_shift_prob,
        scale_low=args.eot_scale_min,
        scale_high=args.eot_scale_max,
        scale_prob=args.eot_scale_prob,
        channel_dropout=args.eot_channel_dropout,
        channel_dropout_prob=args.eot_channel_dropout_prob,
        resample_max_rate_delta=args.eot_resample,
        resample_prob=args.eot_resample_prob,
    )


def describe_eot(args: argparse.Namespace) -> str:
    if args.disable_eot:
        return "eot_disabled"

    parts = []
    if args.eot_shift_prob > 0 and args.eot_shift > 0:
        parts.append(f"shift{args.eot_shift}_p{args.eot_shift_prob}")
    if args.eot_scale_prob > 0 and args.eot_scale:
        parts.append(
            f"scale{args.eot_scale_min}-{args.eot_scale_max}_p{args.eot_scale_prob}"
        )
    if args.eot_channel_dropout_prob > 0 and args.eot_channel_dropout > 0:
        parts.append(
            f"chdrop{args.eot_channel_dropout}_p{args.eot_channel_dropout_prob}"
        )
    if args.eot_resample_prob > 0 and args.eot_resample > 0:
        parts.append(f"resample{args.eot_resample}_p{args.eot_resample_prob}")

    if not parts:
        return "noeot"
    return "eot_" + "_".join(parts)


def _load_perturber(path: Path, device: torch.device) -> STFTDeltaPerturber:
    if not path.is_file():
        raise FileNotFoundError(f"Perturbation checkpoint not found: {path}")

    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "delta" in state:
        perturber = state["delta"]
    else:
        perturber = state

    if not isinstance(perturber, STFTDeltaPerturber):
        raise TypeError("Loaded object is not an STFTDeltaPerturber")

    perturber = perturber.to(device)
    perturber.eval()
    for p in perturber.parameters():
        p.requires_grad_(False)
    return perturber


def _sample_transform(
    eot_distribution: Optional[EOTDistribution], device: torch.device
) -> IdentityTransform:
    if eot_distribution is None:
        return IdentityTransform()
    return eot_distribution.sample(device=device)


@torch.no_grad()
def _apply_ud(
    x: torch.Tensor,
    perturber: STFTDeltaPerturber,
    transform: IdentityTransform,
) -> torch.Tensor:
    x_t_bar = transform.apply(x.squeeze(1))
    x_prime = perturber(x_t_bar.unsqueeze(1))
    x_ud = transform.apply(x_prime.squeeze(1)).unsqueeze(1)
    return x_ud


@torch.no_grad()
def evaluate_on_ud(
    model: nn.Module,
    dataloader,
    device: torch.device,
    perturber: STFTDeltaPerturber,
    eot_distribution: Optional[EOTDistribution],
) -> Tuple[float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_labels = []

    clf_loss_func = nn.CrossEntropyLoss().to(device)

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        transform = _sample_transform(eot_distribution, device)
        x_ud = _apply_ud(x, perturber, transform)

        logits = model(x_ud)
        loss = clf_loss_func(logits, y.long())

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
        all_logits.append(logits.detach())
        all_labels.append(y.detach())

    avg_loss = total_loss / max(1, total_samples)
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    acc, f1, bca, eer = calculate_metrics(labels_cat, logits_cat)
    return avg_loss, acc, f1, bca, eer


def _candidate_delta_paths(
    delta_root: Path,
    dataset: str,
    src_model: str,
    seed: int,
    tags: Sequence[str],
) -> Sequence[Path]:
    base_dir = delta_root / dataset / src_model
    names = []
    for tag in tags:
        # names.append(base_dir / f"{tag}_seed{seed}_TestWOEOT.pth")
        names.append(base_dir / f"{tag}_seed{seed}.pth")
    return names


def resolve_delta_path(args: argparse.Namespace, seed: int) -> Path:
    if args.perturbation_path is not None:
        return Path(args.perturbation_path)

    lambda_tag = format_lambda_tag(args.lambda_task, args.lambda_uid, args.lambda_reg)
    tag = f"{describe_eot(args)}_{lambda_tag}"
    tags = [tag]
    candidates = _candidate_delta_paths(args.delta_root, args.dataset, args.src_model, seed, tags)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    joined = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find perturbation checkpoint. Tried:\n" + joined
    )

def resolve_tgt_checkpoint(args: argparse.Namespace, seed: int) -> Path:
    if args.tgt_checkpoint is not None:
        return Path(args.tgt_checkpoint)

    model_path = args.model_root 
    return model_path / f"UID_Teacher_{args.tgt_model}_{args.dataset}_{seed}.pth"


def run_one_seed(
    args: argparse.Namespace,
    perturber: STFTDeltaPerturber,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    args.seed = seed
    args.is_task = False
    args.model = args.tgt_model
    args = set_args(args)
    set_seed(seed)

    eot_distribution = build_eot_distribution(args)
    _, _, testloader = load_data(args)

    model, _, device = load_all(args)
    model.to(device)

    checkpoint_path = resolve_tgt_checkpoint(args, seed)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Target checkpoint not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    clean_loss, clean_acc, clean_f1, clean_bca, clean_eer = evaluate(
        model=model, dataloader=testloader, args=args, device=device
    )
    ud_loss, ud_acc, ud_f1, ud_bca, ud_eer = evaluate_on_ud(
        model=model,
        dataloader=testloader,
        device=device,
        perturber=perturber,
        eot_distribution=eot_distribution,
    )

    clean_metrics = np.array([clean_acc, clean_f1, clean_bca, clean_eer])
    ud_metrics = np.array([ud_acc, ud_f1, ud_bca, ud_eer])

    print(
        "Clean test | loss={:.6f}, acc={:.4f}, f1={:.4f}, bca={:.4f}, eer={:.4f}".format(
            clean_loss, clean_acc, clean_f1, clean_bca, clean_eer
        )
    )
    print(
        "UD test    | loss={:.6f}, acc={:.4f}, f1={:.4f}, bca={:.4f}, eer={:.4f}".format(
            ud_loss, ud_acc, ud_f1, ud_bca, ud_eer
        )
    )

    return clean_metrics, ud_metrics


def build_argument_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent
    default_log_root = project_root / "logs"
    default_model_root = project_root / "ModelSave" / "Distiil_Pretrain"
    default_csv_root = project_root / "csv"
    default_delta_root = project_root / "ModelSave" / "Distill_Delta"

    parser = argparse.ArgumentParser(description="Transferability evaluation for distill-STFT delta")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--initlr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--channel", type=int, default=22)
    parser.add_argument("--timepoint", type=float, default=4.0)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--nclass", type=int, default=9)
    parser.add_argument("--src_model", type=str, required=True)
    parser.add_argument("--tgt_model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--disable_eot", action="store_true", help="Disable EOT when regenerating UD data")
    parser.add_argument("--eot_shift", type=int, default=16)
    parser.add_argument("--eot_scale", action="store_true", help="Enable amplitude scaling transform")
    parser.add_argument("--eot_scale_min", type=float, default=0.9)
    parser.add_argument("--eot_scale_max", type=float, default=1.1)
    parser.add_argument("--eot_channel_dropout", type=float, default=0.05)
    parser.add_argument("--eot_resample", type=float, default=0.02)
    parser.add_argument("--eot_shift_prob", type=float, default=1.0, help="Probability to apply time shift")
    parser.add_argument("--eot_scale_prob", type=float, default=1.0, help="Probability to apply scaling")
    parser.add_argument(
        "--eot_channel_dropout_prob", type=float, default=1.0, help="Probability to apply channel dropout"
    )
    parser.add_argument("--eot_resample_prob", type=float, default=1.0, help="Probability to apply resampling jitter")
    parser.add_argument("--lambda_task", type=float, default=1.0)
    parser.add_argument("--lambda_uid", type=float, default=5.0)
    parser.add_argument("--lambda_reg", type=float, default=1e-3)
    parser.add_argument("--perturbation_path", type=Path, default=None)
    parser.add_argument("--tgt_checkpoint", type=Path, default=None)
    parser.add_argument("--log_root", type=Path, default=default_log_root)
    parser.add_argument("--model_root", type=Path, default=default_model_root)
    parser.add_argument("--csv_root", type=Path, default=default_csv_root)
    parser.add_argument("--delta_root", type=Path, default=default_delta_root)
    parser.add_argument("--torch_threads", type=int, default=10, help="Max torch threads")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    apply_thread_limits(getattr(args, "torch_threads", 10))
    args.device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")

    seeds = list(range(args.seed, args.seed + args.repeats))
    clean_results = np.zeros((len(seeds), 4))
    ud_results = np.zeros((len(seeds), 4))

    eot_tag = describe_eot(args)
    lambda_tag = format_lambda_tag(args.lambda_task, args.lambda_uid, args.lambda_reg)
    run_tag = f"transfer_{args.src_model}_to_{args.tgt_model}_{eot_tag}_{lambda_tag}"

    log_dir = args.log_root / args.dataset / f"transfer_{args.src_model}_to_{args.tgt_model}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_tag}.log"
    sys.stdout = Logger(log_path)

    for idx, seed in enumerate(seeds):
        start_time = time.time()
        print("=" * 30)
        print(f"dataset    : {args.dataset}")
        print(f"src model  : {args.src_model}")
        print(f"tgt model  : {args.tgt_model}")
        print(f"seed       : {seed}")
        print(f"gpu        : {args.gpuid}")
        print(f"eot tag    : {eot_tag}")

        perturber_path = resolve_delta_path(args, seed)
        print(f"delta path : {perturber_path}")
        perturber = _load_perturber(perturber_path, args.device)

        clean_metrics, ud_metrics = run_one_seed(args, perturber, seed)
        clean_results[idx] = clean_metrics
        ud_results[idx] = ud_metrics
        print(
            f"Seed {seed} done in {time.time() - start_time:.2f}s | "
            f"UD Acc={ud_metrics[0]:.4f}, F1={ud_metrics[1]:.4f}, BCA={ud_metrics[2]:.4f}, EER={ud_metrics[3]:.4f}"
        )

    row_labels = [str(s) for s in seeds] + ["Avg", "Std"]
    col_labels = ["Acc", "F1", "BCA", "EER"]

    print("\nClean results summary")
    print(f"{'SEED':<10} {col_labels[0]:<10} {col_labels[1]:<10} {col_labels[2]:<10} {col_labels[3]:<10}")
    for i, seed in enumerate(seeds):
        row = clean_results[i]
        print(f"{row_labels[i]:<10} {row[0]:<10.4f} {row[1]:<10.4f} {row[2]:<10.4f} {row[3]:<10.4f}")
    print(
        f"{row_labels[-2]:<10} {np.mean(clean_results[:, 0]):<10.4f} {np.mean(clean_results[:, 1]):<10.4f} "
        f"{np.mean(clean_results[:, 2]):<10.4f} {np.mean(clean_results[:, 3]):<10.4f}"
    )
    print(
        f"{row_labels[-1]:<10} {np.std(clean_results[:, 0]):<10.4f} {np.std(clean_results[:, 1]):<10.4f} "
        f"{np.std(clean_results[:, 2]):<10.4f} {np.std(clean_results[:, 3]):<10.4f}"
    )

    print("\nUD results summary")
    print(f"{'SEED':<10} {col_labels[0]:<10} {col_labels[1]:<10} {col_labels[2]:<10} {col_labels[3]:<10}")
    for i, seed in enumerate(seeds):
        row = ud_results[i]
        print(f"{row_labels[i]:<10} {row[0]:<10.4f} {row[1]:<10.4f} {row[2]:<10.4f} {row[3]:<10.4f}")
    print(
        f"{row_labels[-2]:<10} {np.mean(ud_results[:, 0]):<10.4f} {np.mean(ud_results[:, 1]):<10.4f} "
        f"{np.mean(ud_results[:, 2]):<10.4f} {np.mean(ud_results[:, 3]):<10.4f}"
    )
    print(
        f"{row_labels[-1]:<10} {np.std(ud_results[:, 0]):<10.4f} {np.std(ud_results[:, 1]):<10.4f} "
        f"{np.std(ud_results[:, 2]):<10.4f} {np.std(ud_results[:, 3]):<10.4f}"
    )

    final_clean = np.vstack([clean_results, np.mean(clean_results, axis=0), np.std(clean_results, axis=0)])
    final_ud = np.vstack([ud_results, np.mean(ud_results, axis=0), np.std(ud_results, axis=0)])
    combined = np.hstack([final_clean, final_ud])

    csv_dir = args.csv_root / args.dataset / f"transfer_{args.src_model}_to_{args.tgt_model}"
    os.makedirs(csv_dir, exist_ok=True)
    csv_name = f"{run_tag}.csv"
    np.savetxt(csv_dir / csv_name, np.round(combined, 4), delimiter=",", fmt="%.4f")
    print(f"Saved summary CSV to {csv_dir / csv_name}")


if __name__ == "__main__":
    main()

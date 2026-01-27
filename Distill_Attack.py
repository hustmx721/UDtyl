import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from Distill import EOTDistribution, IdentityTransform, STFTDeltaPerturber
from evaluate import calculate_metrics
from utils.Logging import Logger
from utils.dataset import set_seed
from utils.init_all import apply_thread_limits, load_all, load_data, set_args


def format_lambda_tag(lambda_task: float, lambda_uid: float, lambda_reg: float) -> str:
    """Return a file-system friendly tag for the current lambda configuration."""
    return f"lt{lambda_task}_lu{lambda_uid}_lr{lambda_reg}"


def build_eot_distribution(args: argparse.Namespace) -> Optional[EOTDistribution]:
    """Construct the shared EOT distribution used to regenerate UD samples."""

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
    """Build a short tag that captures the enabled EOT transforms for file names."""

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


def _normalize_l2(delta: torch.Tensor, eps: float) -> torch.Tensor:
    flat = delta.view(delta.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    scale = torch.minimum(torch.ones_like(norms), eps / norms)
    scaled = (flat * scale).view_as(delta)
    return scaled


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
    eps: float,
    alpha: float,
    steps: int,
    norm: str = "linf",
    random_start: bool = True,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> torch.Tensor:
    if steps <= 0 or eps <= 0:
        return x.detach()

    x_adv = x.detach()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)

    if clip_min is not None or clip_max is not None:
        lower = clip_min if clip_min is not None else -float("inf")
        upper = clip_max if clip_max is not None else float("inf")
        x_adv = x_adv.clamp(min=lower, max=upper)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = loss_fn(logits, y.long())
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

        if norm == "l2":
            step = _normalize_l2(grad, eps=1.0)
        else:
            step = grad.sign()

        x_adv = x_adv.detach() + alpha * step
        delta = x_adv - x

        if norm == "l2":
            delta = _normalize_l2(delta, eps)
        else:
            delta = delta.clamp(-eps, eps)

        x_adv = x + delta
        if clip_min is not None or clip_max is not None:
            lower = clip_min if clip_min is not None else -float("inf")
            upper = clip_max if clip_max is not None else float("inf")
            x_adv = x_adv.clamp(min=lower, max=upper)

    return x_adv.detach()


def train_one_epoch_adversarial(
    model: nn.Module,
    dataloader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    clf_loss_func: nn.Module,
    perturber: STFTDeltaPerturber,
    eot_distribution: Optional[EOTDistribution],
    eps: float,
    alpha: float,
    steps: int,
    norm: str,
    random_start: bool,
    clip_min: Optional[float],
    clip_max: Optional[float],
) -> Tuple[float, float, float, float, float]:
    model.train()

    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_labels = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        transform = _sample_transform(eot_distribution, device)
        x_ud = _apply_ud(x, perturber, transform)

        x_adv = pgd_attack(
            model=model,
            x=x_ud,
            y=y,
            loss_fn=clf_loss_func,
            eps=eps,
            alpha=alpha,
            steps=steps,
            norm=norm,
            random_start=random_start,
            clip_min=clip_min,
            clip_max=clip_max,
        )

        output = model(x_adv)
        loss = clf_loss_func(output, y.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
        all_logits.append(output.detach())
        all_labels.append(y.detach())

    avg_loss = total_loss / max(1, total_samples)
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    acc, f1, bca, eer = calculate_metrics(labels_cat, logits_cat)
    return avg_loss, acc, f1, bca, eer


@torch.no_grad()
def evaluate_uid_on_ud(
    model: nn.Module,
    dataloader,
    device: torch.device,
    clf_loss_func: nn.Module,
    perturber: STFTDeltaPerturber,
    eot_distribution: Optional[EOTDistribution],
) -> Tuple[float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_labels = []

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


def run_one_seed(args: argparse.Namespace, perturber: STFTDeltaPerturber) -> Tuple[np.ndarray, np.ndarray]:
    args.is_task = False
    args = set_args(args)
    set_seed(args.seed)

    eot_distribution = build_eot_distribution(args)
    clf_loss_func = nn.CrossEntropyLoss().to(args.device)

    trainloader, valloader, testloader = load_data(args)
    model, optimizer, device = load_all(args)
    model.to(device)

    best_state = None
    best_val_acc = -float("inf")
    best_epoch = 0

    for epoch in range(args.epoch):
        train_loss, train_acc, train_f1, train_bca, train_eer = train_one_epoch_adversarial(
            model=model,
            dataloader=trainloader,
            device=device,
            optimizer=optimizer,
            clf_loss_func=clf_loss_func,
            perturber=perturber,
            eot_distribution=eot_distribution,
            eps=args.attack_eps,
            alpha=args.attack_alpha,
            steps=args.attack_steps,
            norm=args.attack_norm,
            random_start=args.attack_random_start,
            clip_min=args.attack_clip_min,
            clip_max=args.attack_clip_max,
        )

        val_loss, val_acc, val_f1, val_bca, val_eer = evaluate_uid_on_ud(
            model=model,
            dataloader=valloader,
            device=device,
            clf_loss_func=clf_loss_func,
            perturber=perturber,
            eot_distribution=eot_distribution,
        )

        print(
            f"[Epoch {epoch + 1}] Train loss={train_loss:.6f}, Acc={train_acc:.4f}, "
            f"F1={train_f1:.4f}, BCA={train_bca:.4f}, EER={train_eer:.4f} | "
            f"Val loss={val_loss:.6f}, Acc={val_acc:.4f}, F1={val_f1:.4f}, BCA={val_bca:.4f}, EER={val_eer:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            best_epoch = epoch
            if args.save_model:
                torch.save(
                    best_state,
                    args.save_model / f"UID_AT_{args.model}_eps{args.attack_eps}_k{args.attack_steps}_seed{args.seed}.pth",
                )
        elif (epoch - best_epoch) > args.earlystop:
            print(f"Early stopping at epoch {epoch + 1} (no val improvement for {args.earlystop} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_f1, test_bca, test_eer = evaluate_uid_on_ud(
        model=model,
        dataloader=testloader,
        device=device,
        clf_loss_func=clf_loss_func,
        perturber=perturber,
        eot_distribution=eot_distribution,
    )

    return (
        np.array([train_acc, train_f1, train_bca, train_eer]),
        np.array([test_acc, test_f1, test_bca, test_eer]),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent
    default_log_root = project_root / "logs"
    default_model_root = project_root / "ModelSave"
    default_csv_root = project_root / "csv"
    default_delta_root = project_root / "ModelSave" / "Distill_Delta"

    parser = argparse.ArgumentParser(description="UID adversarial training on UD data")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--initlr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--channel", type=int, default=22)
    parser.add_argument("--timepoint", type=float, default=4.0)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--nclass", type=int, default=9)
    parser.add_argument("--model", type=str, default="EEGNet")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--earlystop", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--disable_eot", action="store_true", default=True, help="Disable EOT when regenerating UD data")
    parser.add_argument("--n_fft", type=int, default=256)
    parser.add_argument("--hop_length", type=int, default=None)
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
    parser.add_argument("--perturbation_path", type=Path, default=None, help="Checkpoint containing the STFT delta")
    parser.add_argument("--lambda_task", type=float, default=1.0, help="Task weight used in distillation (for checkpoint naming)")
    parser.add_argument("--lambda_uid", type=float, default=5.0, help="UID weight used in distillation (for checkpoint naming)")
    parser.add_argument("--lambda_reg", type=float, default=1e-3, help="Reg weight used in distillation (for checkpoint naming)")
    parser.add_argument("--attack_norm", type=str, choices=["linf", "l2"], default="linf")
    parser.add_argument("--attack_eps", type=float, default=0.01, help="Recommended: 0.1–0.2 for normalized inputs")
    parser.add_argument("--attack_steps", type=int, default=10, help="PGD iterations (10 is a strong default)")
    parser.add_argument("--attack_alpha", type=float, default=None, help="Step size for PGD (defaults to 1.5*eps/steps)")
    parser.add_argument("--attack_random_start", action="store_true", default=True, help="Use random PGD initialization")
    parser.add_argument(
        "--no_attack_random_start",
        action="store_false",
        dest="attack_random_start",
        help="Disable random PGD initialization",
    )
    parser.add_argument("--attack_clip_min", type=float, default=None)
    parser.add_argument("--attack_clip_max", type=float, default=None)
    parser.add_argument("--log_root", type=Path, default=default_log_root)
    parser.add_argument("--model_root", type=Path, default=default_model_root)
    parser.add_argument("--csv_root", type=Path, default=default_csv_root)
    parser.add_argument(
        "--delta_root",
        type=Path,
        default=default_delta_root,
        help="Root directory where STFT perturbations from distillation are stored",
    )
    parser.add_argument("--save_model", type=Path, default=None, help="Directory to store best checkpoints")
    parser.add_argument("--torch_threads", type=int, default=4, help="Max torch threads")
    
    return parser

def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent

    apply_thread_limits(getattr(args, "torch_threads", 10))
    args.device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")

    if args.attack_alpha is None:
        args.attack_alpha = 1.5 * args.attack_eps / max(1, args.attack_steps)

    args.eot_tag = "eot_scale0.95-1.05_p1.0"
    lambda_tag = format_lambda_tag(args.lambda_task, args.lambda_uid, args.lambda_reg)
    combined_tag = f"{args.eot_tag}_{lambda_tag}"

    if args.perturbation_path is None:
        perturbation_dir = args.delta_root / args.dataset / args.model
        perturbation_name = f"{combined_tag}_seed{args.seed}.pth"
        perturbation_path = perturbation_dir / perturbation_name
    else:
        perturbation_path = Path(args.perturbation_path)
    args.perturbation_path = perturbation_path

    if args.save_model is not None:
        os.makedirs(args.save_model, exist_ok=True)
    log_dir = args.log_root / args.dataset / args.model
    log_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"UID_AT_{combined_tag}_{args.model}_norm{args.attack_norm}_eps{args.attack_eps}_k{args.attack_steps}"
    log_path = log_dir / f"{run_tag}.log"
    sys.stdout = Logger(log_path)

    print("=" * 30)
    print(f"dataset: {args.dataset}")
    print(f"model  : {args.model}")
    print(f"eps    : {args.attack_eps}")
    print(f"steps  : {args.attack_steps}")
    print(f"alpha  : {args.attack_alpha}")
    print(f"random : {args.attack_random_start}")
    print(f"norm   : {args.attack_norm}")
    print(f"eot    : disabled")
    print(f"lambda : task={args.lambda_task}, uid={args.lambda_uid}, reg={args.lambda_reg}")
    print(f"delta  : {args.perturbation_path}")

    perturber = _load_perturber(Path(args.perturbation_path), args.device)

    seeds = list(range(args.seed, args.seed + args.repeats))
    train_results = np.zeros((len(seeds), 4))
    test_results = np.zeros((len(seeds), 4))

    for idx, seed in enumerate(seeds):
        args.seed = seed
        start = time.time()
        print(f"\n----- Seed {seed} -----")
        train_metrics, test_metrics = run_one_seed(args, perturber)
        train_results[idx] = train_metrics
        test_results[idx] = test_metrics
        print(
            f"Seed {seed} done in {time.time() - start:.2f}s | Test Acc={test_metrics[0]:.4f}, "
            f"F1={test_metrics[1]:.4f}, BCA={test_metrics[2]:.4f}, EER={test_metrics[3]:.4f}"
        )

    row_labels = [str(s) for s in seeds] + ["Avg", "Std"]
    col_labels = ["Acc", "F1", "BCA", "EER"]
    print("\nTest results summary")
    print(f"{'SEED':<10} {col_labels[0]:<10} {col_labels[1]:<10} {col_labels[2]:<10} {col_labels[3]:<10}")
    for i, seed in enumerate(seeds):
        row = test_results[i]
        print(f"{row_labels[i]:<10} {row[0]:<10.4f} {row[1]:<10.4f} {row[2]:<10.4f} {row[3]:<10.4f}")
    print(
        f"{row_labels[-2]:<10} {np.mean(test_results[:, 0]):<10.4f} {np.mean(test_results[:, 1]):<10.4f} "
        f"{np.mean(test_results[:, 2]):<10.4f} {np.mean(test_results[:, 3]):<10.4f}"
    )
    print(
        f"{row_labels[-1]:<10} {np.std(test_results[:, 0]):<10.4f} {np.std(test_results[:, 1]):<10.4f} "
        f"{np.std(test_results[:, 2]):<10.4f} {np.std(test_results[:, 3]):<10.4f}"
    )

    final_results = np.vstack([test_results, np.mean(test_results, axis=0), np.std(test_results, axis=0)])
    df = np.round(final_results, 4)
    csv_dir = args.csv_root / args.dataset / args.model
    os.makedirs(csv_dir, exist_ok=True)
    csv_name = f"{run_tag}.csv"
    np.savetxt(csv_dir / csv_name, df, delimiter=",", fmt="%.4f")
    print(f"Saved summary CSV to {csv_dir / csv_name}")


if __name__ == "__main__":
    main()

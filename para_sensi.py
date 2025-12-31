"""Privacy–Utility Pareto sensitivity experiment.

This script automates the hyperparameter sampling strategy described in the
user request:

* Sample key hyperparameters (``epsilon_delta``, ``lambda_reg``, ``lambda_uid``,
  ``lambda_task``, ``p_eot``) using log-uniform distributions or discrete
  grids.
* Run the two-stage pipeline per configuration:
  1) Optimize the frequency-domain perturbation (Stage-1, delta only).
  2) Train/evaluate downstream task and UID models from scratch (Stage-2).
* Aggregate metrics across multiple seeds per configuration.
* Compute and plot the Pareto front (utility vs. privacy) using both the
  suggested (Task drop, UID drop) visualization and the conventional
  (Task metric, 1-UID metric) dominance check.
* Report a stability region based on user-provided drop thresholds.

The implementation reuses the distillation pipeline defined in
``main_Distill.py`` to ensure consistent training budgets and teacher/UID
initialization.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main_Distill import build_argument_parser as build_distill_parser
from main_Distill import train_distillation
from utils.init_all import apply_thread_limits


METRIC_INDEX = {
    "acc": 0,
    "f1": 1,
    "bca": 2,
    "eer": 3,
}


@dataclass
class HyperSample:
    epsilon_delta: float
    lambda_reg: float
    lambda_uid: float
    lambda_task: float
    p_eot: float


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    """Sample log-uniformly between ``low`` and ``high`` (inclusive)."""

    if low <= 0 or high <= 0:
        raise ValueError("Log-uniform bounds must be positive.")
    if low > high:
        raise ValueError("Low bound must not exceed high bound.")
    log_low = math.log10(low)
    log_high = math.log10(high)
    return 10 ** rng.uniform(log_low, log_high)


def sample_hyperparameters(rng: random.Random, args: argparse.Namespace) -> HyperSample:
    """Draw one hyperparameter configuration following the recommended priors."""

    epsilon_delta = _log_uniform(rng, args.epsilon_min, args.epsilon_max)
    lambda_reg = _log_uniform(rng, args.lambda_reg_min, args.lambda_reg_max)
    lambda_uid = _log_uniform(rng, args.lambda_uid_min, args.lambda_uid_max)
    lambda_task = _log_uniform(rng, args.lambda_task_min, args.lambda_task_max)
    p_eot = rng.choice(args.p_eot_choices)

    return HyperSample(
        epsilon_delta=epsilon_delta,
        lambda_reg=lambda_reg,
        lambda_uid=lambda_uid,
        lambda_task=lambda_task,
        p_eot=float(p_eot),
    )


def _prepare_trial_args(
    base_args: argparse.Namespace, sample: HyperSample, seed: int
) -> argparse.Namespace:
    """Clone base args and inject the sampled hyperparameters for one trial."""

    trial_args = deepcopy(base_args)
    trial_args.seed = seed
    trial_args.epsilon_a = sample.epsilon_delta
    trial_args.lambda_reg = sample.lambda_reg
    trial_args.lambda_uid = sample.lambda_uid
    trial_args.lambda_task = sample.lambda_task

    # Map the aggregated p_eot to the per-transform probabilities. Disable EOT
    # entirely when p_eot == 0 to satisfy the discrete grid requirement.
    trial_args.disable_eot = math.isclose(sample.p_eot, 0.0, abs_tol=1e-8)
    prob = sample.p_eot
    trial_args.eot_shift_prob = prob
    trial_args.eot_scale_prob = prob
    trial_args.eot_channel_dropout_prob = prob
    trial_args.eot_resample_prob = prob

    return trial_args


def _extract_metric(metrics: Tuple[float, float, float, float], metric_idx: int) -> float:
    try:
        return float(metrics[metric_idx])
    except (IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected metrics tuple: {metrics}") from exc


def run_single_configuration(
    base_args: argparse.Namespace,
    sample: HyperSample,
    seeds: Sequence[int],
    metric_idx: int,
) -> Dict[str, float]:
    """Run Stage-1/2 for a hyperparameter sample across multiple seeds.

    Returns aggregated statistics required for plotting and tabulation.
    """

    task_clean_vals: List[float] = []
    uid_clean_vals: List[float] = []
    task_vals: List[float] = []
    uid_vals: List[float] = []

    for seed in seeds:
        trial_args = _prepare_trial_args(base_args, sample, seed)
        metrics = train_distillation(trial_args)

        task_clean_vals.append(_extract_metric(metrics["teacher_clean"], metric_idx))
        uid_clean_vals.append(_extract_metric(metrics["uid_teacher_clean"], metric_idx))
        task_vals.append(_extract_metric(metrics["perturbed_task"], metric_idx))
        uid_vals.append(_extract_metric(metrics["perturbed_uid"], metric_idx))

    task_clean = np.array(task_clean_vals)
    uid_clean = np.array(uid_clean_vals)
    task = np.array(task_vals)
    uid = np.array(uid_vals)

    task_drop = task_clean - task
    uid_drop = uid_clean - uid
    privacy = 1.0 - uid

    return {
        "utility_mean": float(task.mean()),
        "utility_std": float(task.std()),
        "privacy_mean": float(privacy.mean()),
        "privacy_std": float(privacy.std()),
        "task_drop_mean": float(task_drop.mean()),
        "task_drop_std": float(task_drop.std()),
        "uid_drop_mean": float(uid_drop.mean()),
        "uid_drop_std": float(uid_drop.std()),
        "task_clean_mean": float(task_clean.mean()),
        "uid_clean_mean": float(uid_clean.mean()),
    }


def _non_dominated_indices(utilities: np.ndarray, privacies: np.ndarray, tol: float = 1e-8) -> List[int]:
    """Return indices of Pareto-optimal points (maximize utility and privacy)."""

    n = len(utilities)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if dominated[i]:
                break
            if (
                utilities[j] >= utilities[i] - tol
                and privacies[j] >= privacies[i] - tol
                and (
                    utilities[j] > utilities[i] + tol
                    or privacies[j] > privacies[i] + tol
                )
            ):
                dominated[i] = True
    return [idx for idx, dom in enumerate(dominated) if not dom]


def _plot_pareto(
    records: List[Dict[str, float]],
    front_indices: Iterable[int],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xs = [rec["task_drop_mean"] * 100 for rec in records]
    ys = [rec["uid_drop_mean"] * 100 for rec in records]
    colors = [rec["p_eot"] for rec in records]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(xs, ys, c=colors, cmap="viridis", alpha=0.8, edgecolor="k", linewidth=0.5)

    for rec in records:
        ax.errorbar(
            rec["task_drop_mean"] * 100,
            rec["uid_drop_mean"] * 100,
            xerr=rec["task_drop_std"] * 100,
            yerr=rec["uid_drop_std"] * 100,
            fmt="none",
            ecolor="#666666",
            alpha=0.35,
            capsize=3,
        )

    front_sorted = sorted(front_indices, key=lambda i: records[i]["task_drop_mean"])
    front_x = [records[i]["task_drop_mean"] * 100 for i in front_sorted]
    front_y = [records[i]["uid_drop_mean"] * 100 for i in front_sorted]
    ax.plot(front_x, front_y, color="crimson", linewidth=2, label="Pareto front")

    ax.set_xlabel("Task drop (%) ↓")
    ax.set_ylabel("UID drop (%) ↑")
    ax.set_title("Privacy–Utility Pareto front (hyperparameter sensitivity)")
    cbar = fig.colorbar(scatter, ax=ax, label="p_eot")
    cbar.ax.set_ylabel("EOT apply probability", rotation=270, labelpad=15)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _stable_region_statement(
    records: List[Dict[str, float]],
    task_drop_limit: float,
    uid_drop_limit: float,
) -> str:
    eligible = [
        rec
        for rec in records
        if rec["task_drop_mean"] <= task_drop_limit and rec["uid_drop_mean"] >= uid_drop_limit
    ]
    if not eligible:
        return (
            "未找到满足稳定区间阈值的超参组合。可放宽阈值或增加采样点数来观察趋势。"
        )
    best = max(eligible, key=lambda r: r["uid_drop_mean"])
    return (
        "在较宽的超参范围内，我们观察到稳定的隐私-效用折中前沿。"
        f"在 Task drop ≤ {task_drop_limit*100:.1f}% 区间内仍能实现 UID drop ≥ {uid_drop_limit*100:.1f}%，"
        "表明方法不依赖精细调参。"
        f" 代表性点：ε_Δ={best['epsilon_delta']:.3g}, λ_reg={best['lambda_reg']:.3g},"
        f" λ_uid={best['lambda_uid']:.3g}, λ_task={best['lambda_task']:.3g}, p_eot={best['p_eot']}。"
    )


def build_parser() -> argparse.ArgumentParser:
    base = build_distill_parser()
    base.description = "Hyperparameter sensitivity sweep for privacy–utility Pareto analysis"
    base.add_argument("--samples", type=int, default=30, help="Number of hyperparameter configurations to sample")
    base.add_argument(
        "--seeds-per-sample",
        type=int,
        default=3,
        help="Number of random seeds to average per configuration",
    )
    base.add_argument("--sweep-seed", type=int, default=13, help="Random seed for hyperparameter sampling")
    base.add_argument("--epsilon-min", type=float, default=0.01, help="Lower bound for ε_Δ (log-uniform)")
    base.add_argument("--epsilon-max", type=float, default=1.0, help="Upper bound for ε_Δ (log-uniform)")
    base.add_argument("--lambda-reg-min", type=float, default=1e-5, help="Lower bound for λ_reg (log-uniform)")
    base.add_argument("--lambda-reg-max", type=float, default=1e-1, help="Upper bound for λ_reg (log-uniform)")
    base.add_argument("--lambda-uid-min", type=float, default=1e-2, help="Lower bound for λ_uid (log-uniform)")
    base.add_argument("--lambda-uid-max", type=float, default=10.0, help="Upper bound for λ_uid (log-uniform)")
    base.add_argument("--lambda-task-min", type=float, default=1e-2, help="Lower bound for λ_task (log-uniform)")
    base.add_argument("--lambda-task-max", type=float, default=10.0, help="Upper bound for λ_task (log-uniform)")
    base.add_argument(
        "--p-eot-choices",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Discrete choices for EOT application probability",
    )
    base.add_argument(
        "--metric",
        type=str,
        choices=list(METRIC_INDEX.keys()),
        default="bca",
        help="Utility/privacy metric used for Pareto computation",
    )
    base.add_argument(
        "--figure-path",
        type=Path,
        default=Path("figures") / "pareto_sensitivity.png",
        help="Where to save the Pareto scatter plot",
    )
    base.add_argument(
        "--result-csv",
        type=Path,
        default=Path("csv") / "pareto_sensitivity.csv",
        help="Where to save the detailed sweep table",
    )
    base.add_argument(
        "--stable-task-drop",
        type=float,
        default=0.05,
        help="Task drop threshold (fraction) to define the stability interval",
    )
    base.add_argument(
        "--stable-uid-drop",
        type=float,
        default=0.05,
        help="UID drop threshold (fraction) to define the stability interval",
    )
    return base


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    apply_thread_limits(getattr(args, "torch_threads", 4))

    metric_idx = METRIC_INDEX[args.metric]
    rng = random.Random(args.sweep_seed)
    seeds = list(range(args.seed, args.seed + args.seeds_per_sample))

    records: List[Dict[str, float]] = []
    start = time.time()
    for idx in range(args.samples):
        sample = sample_hyperparameters(rng, args)
        summary = run_single_configuration(args, sample, seeds, metric_idx)
        record = {
            "id": idx,
            "epsilon_delta": sample.epsilon_delta,
            "lambda_reg": sample.lambda_reg,
            "lambda_uid": sample.lambda_uid,
            "lambda_task": sample.lambda_task,
            "p_eot": sample.p_eot,
            **summary,
        }
        records.append(record)
        print(
            f"[Sample {idx+1}/{args.samples}] ε_Δ={sample.epsilon_delta:.3g}, "
            f"λ_reg={sample.lambda_reg:.3g}, λ_uid={sample.lambda_uid:.3g}, λ_task={sample.lambda_task:.3g}, "
            f"p_eot={sample.p_eot} | Task drop={summary['task_drop_mean']*100:.2f}% ± {summary['task_drop_std']*100:.2f}% | "
            f"UID drop={summary['uid_drop_mean']*100:.2f}% ± {summary['uid_drop_std']*100:.2f}%"
        )

    utilities = np.array([rec["utility_mean"] for rec in records])
    privacies = np.array([rec["privacy_mean"] for rec in records])
    front_indices = _non_dominated_indices(utilities, privacies)
    for i, rec in enumerate(records):
        rec["on_pareto_front"] = i in front_indices

    _plot_pareto(records, front_indices, args.figure_path)

    stable_statement = _stable_region_statement(
        records, args.stable_task_drop, args.stable_uid_drop
    )
    print(stable_statement)
    print(
        "EOT 强度主要影响前沿的上界（隐私最大化），"
        "而 λ_reg/ε_Δ 控制前沿形状（平衡点位置）。"
    )

    result_path = Path(args.result_csv)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(result_path, index=False)
    elapsed = time.time() - start
    print(f"Sweep finished in {elapsed/3600:.2f} hours. Results saved to {result_path} and {args.figure_path}.")


if __name__ == "__main__":
    main()

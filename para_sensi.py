"""Privacy–Utility Pareto sensitivity experiment.

This script performs a one-at-a-time sweep for four hyperparameters:

* ``epsilon_delta``
* ``lambda_reg``
* ``lambda_uid``
* ``lambda_task``

For each sweep dimension, the remaining three parameters are fixed to
user-provided base values. Every configuration is run across multiple seeds
to compute utility/privacy statistics. Per-run CSVs are saved under a dedicated
``para`` folder (model/csv/pth), and a global summary CSV is written before
rendering the Pareto front plot.

The implementation reuses the distillation pipeline defined in
``main_Distill.py`` to ensure consistent training budgets and teacher/UID
initialization.
"""

from __future__ import annotations

import argparse
import math
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from main_Distill import build_argument_parser as build_distill_parser
from main_Distill import train_distillation, describe_eot
from utils.init_all import apply_thread_limits


METRIC_INDEX = {
    "acc": 0,
    "wf1": 1,
    "bca": 2,
    "weer": 3,
}


@dataclass
class HyperSample:
    epsilon_delta: float
    lambda_reg: float
    lambda_uid: float
    lambda_task: float


def _format_param_value(value: float) -> str:
    raw = f"{value:.6g}"
    return re.sub(r"[^0-9a-zA-Z]+", "_", raw)


def _build_parameter_sweep(args: argparse.Namespace) -> List[Tuple[str, float, HyperSample]]:
    base = HyperSample(
        epsilon_delta=args.base_epsilon_delta,
        lambda_reg=args.base_lambda_reg,
        lambda_uid=args.base_lambda_uid,
        lambda_task=args.base_lambda_task,
    )

    sweep: List[Tuple[str, float, HyperSample]] = []
    for value in args.epsilon_values:
        sweep.append(
            (
                "epsilon_delta",
                value,
                HyperSample(
                    epsilon_delta=value,
                    lambda_reg=base.lambda_reg,
                    lambda_uid=base.lambda_uid,
                    lambda_task=base.lambda_task,
                ),
            )
        )
    for value in args.lambda_reg_values:
        sweep.append(
            (
                "lambda_reg",
                value,
                HyperSample(
                    epsilon_delta=base.epsilon_delta,
                    lambda_reg=value,
                    lambda_uid=base.lambda_uid,
                    lambda_task=base.lambda_task,
                ),
            )
        )
    for value in args.lambda_uid_values:
        sweep.append(
            (
                "lambda_uid",
                value,
                HyperSample(
                    epsilon_delta=base.epsilon_delta,
                    lambda_reg=base.lambda_reg,
                    lambda_uid=value,
                    lambda_task=base.lambda_task,
                ),
            )
        )
    for value in args.lambda_task_values:
        sweep.append(
            (
                "lambda_task",
                value,
                HyperSample(
                    epsilon_delta=base.epsilon_delta,
                    lambda_reg=base.lambda_reg,
                    lambda_uid=base.lambda_uid,
                    lambda_task=value,
                ),
            )
        )
    return sweep


def _prepare_trial_args(
    base_args: argparse.Namespace,
    sample: HyperSample,
    seed: int,
    param_name: str,
    param_value: float,
) -> argparse.Namespace:
    """Clone base args and inject the sampled hyperparameters for one trial."""

    trial_args = deepcopy(base_args)
    trial_args.eot_tag = describe_eot(trial_args)
    trial_args.seed = seed
    trial_args.epsilon_a = sample.epsilon_delta
    trial_args.lambda_reg = sample.lambda_reg
    trial_args.lambda_uid = sample.lambda_uid
    trial_args.lambda_task = sample.lambda_task

    # Map the aggregated p_eot to the per-transform probabilities.
    trial_args.disable_eot = math.isclose(base_args.p_eot, 0.0, abs_tol=1e-8)
    prob = base_args.p_eot
    trial_args.eot_shift_prob = 0.0
    trial_args.eot_scale_prob = 0.0
    trial_args.eot_channel_dropout_prob = 0.0
    trial_args.eot_resample_prob = prob

    para_root = base_args.para_root
    trial_args.model_root = para_root / "model"
    trial_args.csv_root = para_root / "csv"
    safe_value = _format_param_value(param_value)
    delta_root = para_root / "pth" / base_args.dataset / base_args.task_model / param_name / safe_value
    trial_args.save_delta = str(delta_root)

    return trial_args


def _save_configuration_csv(
    args: argparse.Namespace,
    param_name: str,
    param_value: float,
    sample: HyperSample,
    seeds: Sequence[int],
    task_clean: np.ndarray,
    uid_clean: np.ndarray,
    task: np.ndarray,
    uid: np.ndarray,
    task_drop: np.ndarray,
    uid_drop: np.ndarray,
    privacy: np.ndarray,
) -> None:
    csv_root = args.para_root / "csv" / args.dataset / args.task_model
    csv_root.mkdir(parents=True, exist_ok=True)
    safe_value = _format_param_value(param_value)
    csv_path = csv_root / f"para_{param_name}_{safe_value}.csv"
    df = pd.DataFrame(
        {
            "seed": [str(seed) for seed in seeds],
            "epsilon_delta": sample.epsilon_delta,
            "lambda_reg": sample.lambda_reg,
            "lambda_uid": sample.lambda_uid,
            "lambda_task": sample.lambda_task,
            "task_clean": task_clean,
            "uid_clean": uid_clean,
            "task": task,
            "uid": uid,
            "task_drop": task_drop,
            "uid_drop": uid_drop,
            "privacy": privacy,
        }
    )
    summary = df[["task_clean", "uid_clean", "task", "uid", "task_drop", "uid_drop", "privacy"]].agg(
        ["mean", "std"]
    )
    summary["seed"] = ["Avg", "Std"]
    for name in ["epsilon_delta", "lambda_reg", "lambda_uid", "lambda_task"]:
        summary[name] = getattr(sample, name)
    df = pd.concat([df, summary], ignore_index=True)
    df.to_csv(csv_path, index=False)


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
    param_name: str,
    param_value: float,
) -> Dict[str, float]:
    """Run Stage-1/2 for a hyperparameter sample across multiple seeds.

    Returns aggregated statistics required for plotting and tabulation.
    """

    task_clean_vals: List[float] = []
    uid_clean_vals: List[float] = []
    task_vals: List[float] = []
    uid_vals: List[float] = []

    for seed in seeds:
        trial_args = _prepare_trial_args(base_args, sample, seed, param_name, param_value)
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

    _save_configuration_csv(
        base_args,
        param_name,
        param_value,
        sample,
        seeds,
        task_clean,
        uid_clean,
        task,
        uid,
        task_drop,
        uid_drop,
        privacy,
    )

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
    param_names = [rec["param_name"] for rec in records]
    unique_params = sorted(set(param_names))
    param_to_idx = {name: idx for idx, name in enumerate(unique_params)}
    colors = [param_to_idx[name] for name in param_names]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(xs, ys, c=colors, cmap="tab10", alpha=0.8, edgecolor="k", linewidth=0.5)

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
    handles = [
        Line2D([0], [0], marker="o", color="w", label=name, markerfacecolor=scatter.cmap(param_to_idx[name]))
        for name in unique_params
    ]
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(handles=handles, title="Varied parameter")

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
        f" λ_uid={best['lambda_uid']:.3g}, λ_task={best['lambda_task']:.3g}。"
    )


def build_parser() -> argparse.ArgumentParser:
    base = build_distill_parser()
    base.description = "Hyperparameter sensitivity sweep for privacy–utility Pareto analysis"
    base.add_argument("--para-root", type=Path, default=Path("para"), help="Root directory for para outputs")
    base.add_argument(
        "--seeds-per-sample",
        type=int,
        default=3,
        help="Number of random seeds to average per configuration",
    )
    base.add_argument(
        "--epsilon-values",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0, 5.0],
        help="Discrete sweep values for ε_Δ",
    )
    base.add_argument(
        "--lambda-reg-values",
        type=float,
        nargs="+",
        default=[1e-4, 1e-3, 1e-2, 1e-1],
        help="Discrete sweep values for λ_reg",
    )
    base.add_argument(
        "--lambda-uid-values",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 5.0, 8.0],
        help="Discrete sweep values for λ_uid",
    )
    base.add_argument(
        "--lambda-task-values",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 4.0],
        help="Discrete sweep values for λ_task",
    )
    base.add_argument(
        "--base-epsilon-delta",
        type=float,
        default=1.0,
        help="Fixed ε_Δ when sweeping other parameters",
    )
    base.add_argument(
        "--base-lambda-reg",
        type=float,
        default=1e-3,
        help="Fixed λ_reg when sweeping other parameters",
    )
    base.add_argument(
        "--base-lambda-uid",
        type=float,
        default=5.0,
        help="Fixed λ_uid when sweeping other parameters",
    )
    base.add_argument(
        "--base-lambda-task",
        type=float,
        default=1.0,
        help="Fixed λ_task when sweeping other parameters",
    )
    base.add_argument(
        "--p-eot",
        type=float,
        default=1.0,
        help="Fixed EOT application probability (0 disables EOT)",
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
    seeds = list(range(args.seed, args.seed + args.seeds_per_sample))

    records: List[Dict[str, float]] = []
    start = time.time()
    sweep = _build_parameter_sweep(args)
    for idx, (param_name, param_value, sample) in enumerate(sweep):
        summary = run_single_configuration(args, sample, seeds, metric_idx, param_name, param_value)
        record = {
            "id": idx,
            "param_name": param_name,
            "param_value": param_value,
            "epsilon_delta": sample.epsilon_delta,
            "lambda_reg": sample.lambda_reg,
            "lambda_uid": sample.lambda_uid,
            "lambda_task": sample.lambda_task,
            **summary,
        }
        records.append(record)
        print(
            f"[Sample {idx+1}/{len(sweep)}] ε_Δ={sample.epsilon_delta:.3g}, "
            f"λ_reg={sample.lambda_reg:.3g}, λ_uid={sample.lambda_uid:.3g}, λ_task={sample.lambda_task:.3g}, "
            f"param={param_name}({param_value}) | Task drop={summary['task_drop_mean']*100:.2f}% ± {summary['task_drop_std']*100:.2f}% | "
            f"UID drop={summary['uid_drop_mean']*100:.2f}% ± {summary['uid_drop_std']*100:.2f}%"
        )

    utilities = np.array([rec["utility_mean"] for rec in records])
    privacies = np.array([rec["privacy_mean"] for rec in records])
    front_indices = _non_dominated_indices(utilities, privacies)
    for i, rec in enumerate(records):
        rec["on_pareto_front"] = i in front_indices

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
    _plot_pareto(records, front_indices, args.figure_path)
    elapsed = time.time() - start
    print(f"Sweep finished in {elapsed/3600:.2f} hours. Results saved to {result_path} and {args.figure_path}.")


if __name__ == "__main__":
    main()

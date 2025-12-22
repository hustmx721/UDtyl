import argparse
import math
import os
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import Tensor
from tqdm import tqdm

from Distill import EOTDistribution, IdentityTransform, STFTDeltaPerturber, kl_pq, uniform_kl
from evaluate import calculate_metrics, evaluate
from train import train_one_epoch
from utils.Logging import Logger
from utils.dataset import set_seed
from utils.init_all import load_all, load_data, set_args



def build_eot_distribution(args: argparse.Namespace) -> Optional[EOTDistribution]:
    """Construct the shared EOT distribution used by both distillation and evaluation."""

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

def _resolve_checkpoint_path(
    args: argparse.Namespace, provided: str, prefix: str, model_name: str
) -> Path:
    if provided:
        return Path(provided)
    default_dir = Path(args.model_root) / "Distill_Pretrain_Task"
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir / f"{prefix}_{model_name}_seed{args.seed}.pth"


def _resolve_metrics_path(args: argparse.Namespace, prefix: str, model_name: str) -> Path:
    csv_dir = Path(args.csv_root) / "Distill_Pretrain_Task"
    csv_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir / f"{prefix}_Clean_{model_name}.csv"


def _save_teacher_metrics(
    metrics_path: Path, metrics: Tuple[float, float, float, float], seed: int
) -> None:
    df_new = pd.DataFrame(
        [metrics], columns=["Acc", "F1", "BCA", "EER"], index=[str(seed)]
    ).round(4)
    if metrics_path.exists():
        df = pd.read_csv(metrics_path, index_col=0)
        df.loc[str(seed)] = df_new.iloc[0]
    else:
        df = df_new
    df.to_csv(metrics_path)
    print(f"Teacher clean metrics saved to {metrics_path}")


def _train_teacher_from_scratch(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    checkpoint_path: Path,
    metrics_path: Path,
    prefix: str,
) -> Tuple[nn.Module, Tuple[float, float, float, float]]:
    print(f"No {prefix} checkpoint provided. Training model from scratch...")
    trainloader, valloader, testloader = load_data(args, include_index=False)

    clf_loss_func = nn.CrossEntropyLoss().to(device)
    best_state = deepcopy(model.state_dict())
    best_val_acc = -float("inf")
    epochs_since_improve = 0

    for epoch in range(args.teacher_epochs):
        train_loss, train_acc, train_f1, train_bca, train_eer = train_one_epoch(
            model=model,
            dataloader=trainloader,
            device=device,
            optimizer=optimizer,
            clf_loss_func=clf_loss_func,
        )
        val_loss, val_acc, val_f1, val_bca, val_eer = evaluate(
            model=model, dataloader=valloader, args=args, device=device
        )

        print(
            f"[Teacher][Epoch {epoch + 1}] Train loss={train_loss:.6f}, Acc={train_acc:.4f}, F1={train_f1:.4f}, BCA={train_bca:.4f}, EER={train_eer:.4f} | "
            f"Val loss={val_loss:.6f}, Acc={val_acc:.4f}, F1={val_f1:.4f}, BCA={val_bca:.4f}, EER={val_eer:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = deepcopy(model.state_dict())
            epochs_since_improve = 0
            torch.save(best_state, checkpoint_path)
            print(f"Updated best {prefix} checkpoint -> {checkpoint_path}")
        else:
            epochs_since_improve += 1

        if epochs_since_improve > args.teacher_earlystop:
            print(f"Teacher early stopping at epoch {epoch + 1} (no val improvement for {epochs_since_improve} epochs)")
            break

    model.load_state_dict(best_state)
    _, test_acc, test_f1, test_bca, test_eer = evaluate(
        model=model, dataloader=testloader, args=args, device=device
    )
    clean_metrics = (test_acc, test_f1, test_bca, test_eer)
    print(
        "[Teacher][Clean Test] Acc={:.4f}, F1={:.4f}, BCA={:.4f}, EER={:.4f}".format(
            test_acc, test_f1, test_bca, test_eer
        )
    )
    _save_teacher_metrics(metrics_path, clean_metrics, args.seed)
    return model, clean_metrics


def _prepare_supervised_model(
    args: argparse.Namespace,
    model_name: str,
    is_task: bool,
    checkpoint: str,
    prefix: str,
) -> Tuple[nn.Module, torch.device, Tuple[float, float, float, float]]:
    model_args = deepcopy(args)
    model_args.is_task = is_task
    model_args.model = model_name
    model_args = set_args(model_args)
    model, optimizer, device = load_all(model_args)

    checkpoint_path = _resolve_checkpoint_path(model_args, checkpoint, prefix, model_name)
    metrics_path = _resolve_metrics_path(model_args, prefix, model_name)

    if checkpoint and Path(checkpoint).is_file():
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded {prefix} checkpoint from {checkpoint}")
    elif checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded {prefix} checkpoint from {checkpoint_path}")
    else:
        model, clean_metrics = _train_teacher_from_scratch(
            model_args, model, optimizer, device, checkpoint_path, metrics_path, prefix
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model, device, clean_metrics

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    _, _, testloader = load_data(model_args, include_index=False)
    _, test_acc, test_f1, test_bca, test_eer = evaluate(
        model=model, dataloader=testloader, args=model_args, device=device
    )
    clean_metrics = (test_acc, test_f1, test_bca, test_eer)
    print(
        f"[{prefix}][Clean Test] Acc={test_acc:.4f}, F1={test_f1:.4f}, BCA={test_bca:.4f}, EER={test_eer:.4f}"
    )
    _save_teacher_metrics(metrics_path, clean_metrics, model_args.seed)
    return model, device, clean_metrics


def _prepare_teacher(args: argparse.Namespace) -> Tuple[nn.Module, torch.device, Tuple[float, float, float, float]]:
    return _prepare_supervised_model(
        args=args,
        model_name=args.task_model,
        is_task=True,
        checkpoint=args.task_checkpoint,
        prefix="Task_Teacher",
    )


def _prepare_uid_adv(args: argparse.Namespace, device: torch.device) -> Tuple[nn.Module, Tuple[float, float, float, float]]:
    model, _, metrics = _prepare_supervised_model(
        args=args,
        model_name=args.uid_model,
        is_task=False,
        checkpoint=args.uid_checkpoint,
        prefix="UID_Teacher",
    )
    # Ensure UID model on the provided device
    model = model.to(device)
    return model, metrics


def _build_uid_maps_aligned(args: argparse.Namespace) -> Tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """Build UID label maps aligned to the task splits (train/val/test).

    This mirrors the splitting logic used by ``load_data`` for task loaders so that
    indices from task DataLoaders align with the correct UID labels, avoiding
    missing-key errors when the UID split differs from the task split.
    """

    OpenBMI = ["MI", "SSVEP", "ERP"]
    M3CV = ["Rest", "Transient", "Steady", "P300", "Motor", "SSVEP_SA"]

    if args.dataset in OpenBMI:
        data_train = pickle.load(open(f"/mnt/data1/tyl/data/OpenBMI/Task/{args.dataset}/train.pkl", "rb"))
        data_test = pickle.load(open(f"/mnt/data1/tyl/data/OpenBMI/Task/{args.dataset}/test.pkl", "rb"))

        task_train_labels = data_train["label"].astype(np.int16).reshape(-1)
        uid_train_labels = (
            pickle.load(open(f"/mnt/data1/tyl/data/OpenBMI/processed/{args.dataset}/train.pkl", "rb"))["ori_train_s"]
            - 1
        ).astype(np.int16).reshape(-1)
        uid_test_labels = (
            pickle.load(open(f"/mnt/data1/tyl/data/OpenBMI/processed/{args.dataset}/test.pkl", "rb"))["ori_test_s"]
            - 1
        ).astype(np.int16).reshape(-1)
    elif args.dataset in M3CV:
        data_train = pickle.load(open(f"/mnt/data1/tyl/data/M3CV/Task/Session1_{args.dataset}.pkl", "rb"))
        data_test = pickle.load(open(f"/mnt/data1/tyl/data/M3CV/Task/Session2_{args.dataset}.pkl", "rb"))

        task_train_labels = data_train["label"].astype(np.int16)
        uid_train_labels = pickle.load(open(f"/mnt/data1/tyl/data/M3CV/Train/T_{args.dataset}.pkl", "rb"))["label"].astype(
            np.int16
        )
        uid_test_labels = pickle.load(open(f"/mnt/data1/tyl/data/M3CV/Test/{args.dataset}.pkl", "rb"))["label"].astype(
            np.int16
        )
    else:
        raise ValueError("Invalid dataset name")

    indices = np.arange(len(task_train_labels))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=args.seed, stratify=task_train_labels
    )

    train_uid_seq = uid_train_labels[train_idx]
    val_uid_seq = uid_train_labels[val_idx]

    train_map = {int(i): int(lbl) for i, lbl in enumerate(train_uid_seq)}
    val_map = {int(i): int(lbl) for i, lbl in enumerate(val_uid_seq)}
    test_map = {int(i): int(lbl) for i, lbl in enumerate(uid_test_labels)}

    return train_map, val_map, test_map


def _gather_uid_labels(indices: Tensor, mapping: dict[int, int], device: torch.device) -> Tensor:
    return torch.tensor([mapping[int(i)] for i in indices.tolist()], device=device)


def train_distillation(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args = set_args(args)
    eot_distribution = build_eot_distribution(args)

    teacher, device, teacher_clean_metrics = _prepare_teacher(args)

    uid_args = deepcopy(args)
    uid_args.is_task = False
    uid_args = set_args(uid_args)
    uid_adv, uid_clean_metrics = _prepare_uid_adv(uid_args, device)

    trainloader_task, valloader_task, testloader_task = load_data(args, include_index=True)
    train_uid_map, val_uid_map, test_uid_map = _build_uid_maps_aligned(args)

    sample_x = next(iter(trainloader_task))[0]
    channels = sample_x.shape[2]
    time_steps = sample_x.shape[3]

    perturber = STFTDeltaPerturber(
        channels=channels,
        time_steps=time_steps,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        epsilon_delta=args.epsilon_a,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam([perturber.delta], lr=args.lr)

    for epoch in range(args.epochs):
        perturber.train()
        running_loss = 0.0
        for batch in tqdm(trainloader_task, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            x, y_task, indices = batch
            x = x.to(device)
            y_task = y_task.to(device)
            if eot_distribution is None:
                transform = IdentityTransform()
            else:
                transform = eot_distribution.sample(device=device)

            x_t_bar = transform.apply(x.squeeze(1))
            x_t = x_t_bar.unsqueeze(1)

            with torch.no_grad():
                teacher_prob = F.softmax(teacher(x_t), dim=1)

            x_prime = perturber(x_t)
            x_prime_t = transform.apply(x_prime.squeeze(1)).unsqueeze(1)

            teacher_logits_prime = teacher(x_prime_t)
            teacher_prob_prime = F.softmax(teacher_logits_prime, dim=1)

            uid_logits_prime = uid_adv(x_prime_t)
            uid_prob_prime = F.softmax(uid_logits_prime, dim=1)

            kl_task = kl_pq(teacher_prob, teacher_prob_prime, reduction="batchmean")
            ce_task = F.cross_entropy(teacher_logits_prime, y_task.long())
            kl_uid = uniform_kl(uid_prob_prime, uid_args.nclass, reduction="batchmean")
            reg = perturber.l2_regularizer()

            loss = (
                args.lambda_task * kl_task
                + args.lambda_ce * ce_task
                + args.lambda_uid * kl_uid
                + args.lambda_reg * reg
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            perturber.clip_()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(trainloader_task))
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.6f}, Reg={reg.item():.6f}")

        if (epoch + 1) % args.val_interval == 0:
            perturber.eval()
            (
                val_task_acc,
                val_task_f1,
                val_task_bca,
                val_task_eer,
                val_uid_acc,
                val_uid_f1,
                val_uid_bca,
                val_uid_eer,
            ) = evaluate_metrics(
                perturber,
                teacher,
                uid_adv,
                valloader_task,
                val_uid_map,
                eot_distribution,
                device,
                apply_perturb=True,
            )
            print(
                f"Validation @ epoch {epoch + 1}: "
                f"task_acc={val_task_acc:.4f}, task_f1={val_task_f1:.4f}, task_bca={val_task_bca:.4f}, task_eer={val_task_eer:.4f} | "
                f"uid_acc={val_uid_acc:.4f}, uid_f1={val_uid_f1:.4f}, uid_bca={val_uid_bca:.4f}, uid_eer={val_uid_eer:.4f}"
            )

    if args.save_delta:
        save_path = Path(args.save_delta)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(perturber.state_dict(), save_path)
        print(f"Saved perturbation template to {save_path}")

    perturber.eval()
    # Report clean vs perturbed accuracies on the held-out test split
    clean_task_acc, clean_task_f1, clean_task_bca, clean_task_eer, clean_uid_acc, clean_uid_f1, clean_uid_bca, clean_uid_eer = evaluate_metrics(
        perturber,
        teacher,
        uid_adv,
        testloader_task,
        test_uid_map,
        eot_distribution,
        device,
        apply_perturb=False,
    )
    test_task_acc, test_task_f1, test_task_bca, test_task_eer, test_uid_acc, test_uid_f1, test_uid_bca, test_uid_eer = evaluate_metrics(
        perturber,
        teacher,
        uid_adv,
        testloader_task,
        test_uid_map,
        eot_distribution,
        device,
        apply_perturb=True,
    )
    print(
        "[Final Test] clean_task: acc={:.4f}, f1={:.4f}, bca={:.4f}, eer={:.4f} | "
        "clean_uid: acc={:.4f}, f1={:.4f}, bca={:.4f}, eer={:.4f} | "
        "perturbed_task: acc={:.4f}, f1={:.4f}, bca={:.4f}, eer={:.4f} | "
        "perturbed_uid: acc={:.4f}, f1={:.4f}, bca={:.4f}, eer={:.4f}".format(
            clean_task_acc,
            clean_task_f1,
            clean_task_bca,
            clean_task_eer,
            clean_uid_acc,
            clean_uid_f1,
            clean_uid_bca,
            clean_uid_eer,
            test_task_acc,
            test_task_f1,
            test_task_bca,
            test_task_eer,
            test_uid_acc,
            test_uid_f1,
            test_uid_bca,
            test_uid_eer,
        )
    )
    return {
        "teacher_clean": teacher_clean_metrics,
        "uid_teacher_clean": uid_clean_metrics,
        "clean_task": (clean_task_acc, clean_task_f1, clean_task_bca, clean_task_eer),
        "clean_uid": (clean_uid_acc, clean_uid_f1, clean_uid_bca, clean_uid_eer),
        "perturbed_task": (test_task_acc, test_task_f1, test_task_bca, test_task_eer),
        "perturbed_uid": (test_uid_acc, test_uid_f1, test_uid_bca, test_uid_eer),
    }


@torch.no_grad()
def evaluate_metrics(
    perturber: STFTDeltaPerturber,
    teacher: nn.Module,
    uid_adv: nn.Module,
    dataloader,
    uid_map: dict[int, int],
    eot_distribution: Optional[EOTDistribution],
    device: torch.device,
    apply_perturb: bool,
) -> Tuple[float, float, float, float, float, float, float, float]:
    total_task_logits = []
    total_task_labels = []
    total_uid_logits = []
    total_uid_labels = []
    for x, y_task, indices in dataloader:
        x = x.to(device)
        y_task = y_task.to(device)
        uid_labels = _gather_uid_labels(indices, uid_map, device)

        if apply_perturb:
            if eot_distribution is None:
                transform = IdentityTransform()
            else:
                transform = eot_distribution.sample(device=device)

            x_t_bar = transform.apply(x.squeeze(1))
            x_t = x_t_bar.unsqueeze(1)
            x_prime = perturber(x_t)
            x_eval = transform.apply(x_prime.squeeze(1)).unsqueeze(1)
        else:
            x_eval = x

        task_logits = teacher(x_eval)
        uid_logits = uid_adv(x_eval)

        total_task_logits.append(task_logits)
        total_task_labels.append(y_task)
        total_uid_logits.append(uid_logits)
        total_uid_labels.append(uid_labels)

    task_logits_cat = torch.cat(total_task_logits, dim=0)
    task_labels_cat = torch.cat(total_task_labels, dim=0)
    uid_logits_cat = torch.cat(total_uid_logits, dim=0)
    uid_labels_cat = torch.cat(total_uid_labels, dim=0)

    task_acc, task_f1, task_bca, task_eer = calculate_metrics(task_labels_cat, task_logits_cat)
    uid_acc, uid_f1, uid_bca, uid_eer = calculate_metrics(uid_labels_cat, uid_logits_cat)

    return task_acc, task_f1, task_bca, task_eer, uid_acc, uid_f1, uid_bca, uid_eer


def summarize_results(results: np.ndarray, seeds: List[int], prefix: str) -> None:
    row_labels = [str(seed) for seed in seeds] + ["Avg", "Std"]
    col_labels = ["Acc", "F1", "BCA", "EER"]
    print(f"{prefix}结果汇总")
    print(
        f"{'SEED':<10} {col_labels[0]:<10} {col_labels[1]:<10} {col_labels[2]:<10} {col_labels[3]:<10}"
    )
    for i, seed in enumerate(seeds):
        row = results[i]
        print(f"{row_labels[i]:<10} {row[0]:<10.4f} {row[1]:<10.4f} {row[2]:<10.4f} {row[3]:<10.4f}")
    print(
        f"{row_labels[-2]:<10} {np.mean(results[:, 0]):<10.4f} {np.mean(results[:, 1]):<10.4f} "
        f"{np.mean(results[:, 2]):<10.4f} {np.mean(results[:, 3]):<10.4f}"
    )
    print(
        f"{row_labels[-1]:<10} {np.std(results[:, 0]):<10.4f} {np.std(results[:, 1]):<10.4f} "
        f"{np.std(results[:, 2]):<10.4f} {np.std(results[:, 3]):<10.4f}"
    )


def save_results_csv(results: np.ndarray, args, prefix: str, seeds: List[int]) -> None:
    final_results = np.vstack([results, np.mean(results, axis=0), np.std(results, axis=0)])
    df = pd.DataFrame(
        final_results,
        columns=["Acc", "F1", "BCA", "EER"],
        index=[*(str(seed) for seed in seeds), "Avg", "Std"],
    ).round(4)
    csv_path = args.csv_root / f"{args.dataset}"
    os.makedirs(csv_path, exist_ok=True)
    df.to_csv(csv_path / f"Distill_{prefix}_{args.task_model}.csv")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frequency-domain privacy distillation training")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--initlr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--channel", type=int, default=22)
    parser.add_argument("--timepoint", type=float, default=4.0)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--nclass", type=int, default=9)
    parser.add_argument("--task_model", type=str, default="EEGNet")
    parser.add_argument("--uid_model", type=str, default="EEGNet")
    parser.add_argument("--task_checkpoint", type=str, default="", help="Pretrained task teacher checkpoint")
    parser.add_argument("--uid_checkpoint", type=str, default="", help="Pretrained UID adversary checkpoint")
    parser.add_argument("--model_root", type=Path, default=Path("ModelSave"), help="Where to store trained teacher checkpoints")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--teacher_epochs", type=int, default=300, help="Max epochs when training teacher from scratch")
    parser.add_argument("--teacher_earlystop", type=int, default=30, help="Early-stop patience for teacher training")
    parser.add_argument("--n_fft", type=int, default=256)
    parser.add_argument("--hop_length", type=int, default=None)
    parser.add_argument("--epsilon_a", type=float, default=0.05)
    parser.add_argument("--lambda_task", type=float, default=1.0)
    parser.add_argument("--lambda_ce", type=float, default=0.1)
    parser.add_argument("--lambda_uid", type=float, default=2.0)
    parser.add_argument("--lambda_reg", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--save_delta", type=str, default="", help="Path to save the perturbation template")
    parser.add_argument("--eot_shift", type=int, default=16)
    parser.add_argument("--eot_scale", action="store_true", help="Enable amplitude scaling transform")
    parser.add_argument("--eot_scale_min", type=float, default=0.9)
    parser.add_argument("--eot_scale_max", type=float, default=1.1)
    parser.add_argument("--eot_channel_dropout", type=float, default=0.05)
    parser.add_argument("--eot_resample", type=float, default=0.02)
    parser.add_argument("--eot_shift_prob", type=float, default=1.0, help="Probability to apply time shift")
    parser.add_argument("--eot_scale_prob", type=float, default=1.0, help="Probability to apply scaling")
    parser.add_argument("--eot_channel_dropout_prob", type=float, default=1.0, help="Probability to apply channel dropout")
    parser.add_argument("--eot_resample_prob", type=float, default=1.0, help="Probability to apply resampling jitter")
    parser.add_argument("--repeats", type=int, default=5, help="Number of seeds to run")
    parser.add_argument("--log_root", type=Path, default=Path("logs"))
    parser.add_argument("--is_task", type=bool, default=True)
    parser.add_argument("--csv_root", type=Path, default=Path("csv"), help="Directory to store CSV results")
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    seeds = list(range(args.seed, args.seed + args.repeats))
    task_results = np.zeros((len(seeds), 4))
    uid_results = np.zeros((len(seeds), 4))
    clean_task_results = np.zeros((len(seeds), 4))
    clean_uid_results = np.zeros((len(seeds), 4))

    for idx, seed in enumerate(seeds):
        args.seed = seed
        set_seed(seed)

        log_path = args.log_root / f"Distill_{args.dataset}_{args.task_model}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = Logger(log_path)

        start_time = time.time()
        print("=" * 30)
        print(f"dataset: {args.dataset}")
        print(f"task model  : {args.task_model}")
        print(f"uid model   : {args.uid_model}")
        print(f"seed        : {seed}")
        print(f"gpu         : {args.gpuid}")
        print(f"epsilon_a   : {args.epsilon_a}")
        print(f"n_fft/hop   : {args.n_fft}/{args.hop_length or args.n_fft // 4}")
        print(f"lambda task : {args.lambda_task}")
        print(f"lambda uid  : {args.lambda_uid}")
        print(f"lambda reg  : {args.lambda_reg}")

        metrics = train_distillation(args)
        pert_task = metrics["perturbed_task"]
        pert_uid = metrics["perturbed_uid"]
        clean_task = metrics["clean_task"]
        clean_uid = metrics["clean_uid"]

        task_results[idx] = pert_task
        uid_results[idx] = pert_uid
        clean_task_results[idx] = clean_task
        clean_uid_results[idx] = clean_uid

        print(f"Seed {seed} finished. Time used: {time.time() - start_time:.2f}s")
        sys.stdout = sys.__stdout__
        print(f"Seed {seed} log saved to {log_path}")

    summarize_results(task_results, seeds, "Distill Perturbed Task")
    summarize_results(uid_results, seeds, "Distill Perturbed UID")
    save_results_csv(task_results, args, "Distill_TaskPerturbed", seeds)
    save_results_csv(uid_results, args, "Distill_UIDPerturbed", seeds)
    save_results_csv(clean_task_results, args, "Distill_TaskClean", seeds)
    save_results_csv(clean_uid_results, args, "Distill_UIDClean", seeds)
    print("All seeds finished.")


if __name__ == "__main__":
    main()

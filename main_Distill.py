import argparse
import math
import os
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import Tensor
from tqdm import tqdm

from evaluate import calculate_metrics
from utils.Logging import Logger
from utils.dataset import set_seed
from utils.init_all import load_all, load_data, set_args
from utils.models import LoadModel


class FrequencyDomainPerturber(nn.Module):
    """Learnable frequency-domain magnitude perturbation template.

    Supports **class-wise templates** so each UID shares the same perturbation.
    """

    def __init__(
        self,
        channels: int,
        n_fft: int = 256,
        hop_length: Optional[int] = None,
        epsilon: float = 0.05,
        mask: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
        *,
        num_uid_classes: int,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.epsilon = epsilon
        self.device = device or torch.device("cpu")
        if num_uid_classes is None or num_uid_classes <= 0:
            raise ValueError("Class-wise perturbation requires a positive num_uid_classes.")
        self.num_uid_classes = num_uid_classes

        self.register_buffer("window", torch.hann_window(self.n_fft, device=self.device))
        n_freq = n_fft // 2 + 1
        delta_shape = (self.num_uid_classes, channels, n_freq, 1)
        delta = torch.zeros(delta_shape, device=self.device)
        self.delta = nn.Parameter(delta)

        mask_shape = (1, channels, n_freq, 1)
        if mask is not None:
            self.register_buffer("mask", mask)
        else:
            self.register_buffer("mask", torch.ones(mask_shape, device=self.device))

    def forward(
        self,
        x: Tensor,
        uid_labels: Optional[Tensor] = None,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> Tensor:
        """Apply the learnable perturbation in the STFT magnitude domain.

        Args:
            x: EEG batch of shape ``(B, 1, C, T)``.
            uid_labels: Required when ``classwise=True``; UID indices for each sample.
            transform: Optional callable for EOT; applied before STFT on shape ``(B, C, T)``.
        """

        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(f"Expected input shape (B,1,C,T), got {tuple(x.shape)}")
        if uid_labels is None:
            raise ValueError("uid_labels must be provided for class-wise perturbation.")

        batch, _, channels, time_steps = x.shape
        x_bar = x.squeeze(1)
        if transform is not None:
            x_bar = transform(x_bar)

        flat = x_bar.reshape(batch * channels, time_steps)
        stft = torch.stft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        stft = stft.reshape(batch, channels, stft.size(-2), stft.size(-1))

        amplitude = torch.abs(stft)
        phase = torch.angle(stft)

        delta_tanh = torch.tanh(self.delta) * self.epsilon
        delta_selected = delta_tanh[uid_labels.long()]  # (B, C, F, 1)
        masked_delta = self.mask * delta_selected
        updated_amplitude = torch.relu(amplitude + masked_delta)

        modified = torch.polar(updated_amplitude, phase)
        modified_flat = modified.reshape(batch * channels, modified.size(-2), modified.size(-1))
        x_rec = torch.istft(
            modified_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            length=time_steps,
        )
        x_rec = x_rec.reshape(batch, channels, time_steps)
        return x_rec.unsqueeze(1)

    @property
    def delta_regularizer(self) -> Tensor:
        delta_tanh = torch.tanh(self.delta) * self.epsilon
        return torch.sum(delta_tanh ** 2)


def _random_shift(x: Tensor, max_shift: int) -> Tensor:
    if max_shift <= 0:
        return x
    shift = torch.randint(-max_shift, max_shift + 1, (1,), device=x.device).item()
    if shift == 0:
        return x
    return torch.roll(x, shifts=shift, dims=-1)


def _random_scale(x: Tensor, low: float = 0.9, high: float = 1.1) -> Tensor:
    scale = torch.empty(1, device=x.device).uniform_(low, high)
    return x * scale


def _channel_dropout(x: Tensor, drop_prob: float = 0.1) -> Tensor:
    if drop_prob <= 0:
        return x
    if x.dim() < 2:
        return x
    # Create channel mask and broadcast across remaining dims (e.g., time)
    broadcast_shape = (x.size(0), x.size(1), *([1] * (x.dim() - 2)))
    mask = torch.ones(broadcast_shape, device=x.device)
    dropout_mask = torch.bernoulli((1 - drop_prob) * mask)
    return x * dropout_mask


def _resample_jitter(x: Tensor, max_rate_delta: float = 0.05) -> Tensor:
    if max_rate_delta <= 0:
        return x
    batch, channels, time_steps = x.shape
    rate = 1.0 + float(torch.empty(1, device=x.device).uniform_(-max_rate_delta, max_rate_delta))
    new_length = max(1, int(math.ceil(time_steps * rate)))
    flat = x.reshape(batch * channels, 1, time_steps)  # (B*C, 1, T)
    up = F.interpolate(flat, size=new_length, mode="linear", align_corners=False)
    back = F.interpolate(up, size=time_steps, mode="linear", align_corners=False)
    return back.reshape(batch, channels, time_steps)


def build_eot_transform(args: argparse.Namespace) -> Callable[[Tensor], Tensor]:
    """Build an EOT transform with per-augmentation toggles and probabilities."""

    steps: list[tuple[Callable[[Tensor], Tensor], float]] = []
    if args.enable_eot_shift and args.eot_shift > 0:
        steps.append((lambda t: _random_shift(t, args.eot_shift), args.eot_shift_prob))
    if args.enable_eot_scale and args.eot_scale:
        steps.append((lambda t: _random_scale(t, args.eot_scale_min, args.eot_scale_max), args.eot_scale_prob))
    if args.enable_eot_channel_dropout and args.eot_channel_dropout > 0:
        steps.append((lambda t: _channel_dropout(t, args.eot_channel_dropout), args.eot_channel_dropout_prob))
    if args.enable_eot_resample and args.eot_resample > 0:
        steps.append((lambda t: _resample_jitter(t, args.eot_resample), args.eot_resample_prob))

    def apply_all(signal: Tensor) -> Tensor:
        out = signal
        for fn, prob in steps:
            if prob <= 0:
                continue
            if prob >= 1.0 or torch.rand(1, device=signal.device).item() < prob:
                out = fn(out)
        return out

    return apply_all


def _prepare_teacher(args: argparse.Namespace) -> Tuple[nn.Module, torch.device]:
    teacher_args = deepcopy(args)
    teacher_args.is_task = True
    teacher_args.model = args.task_model
    teacher_args = set_args(teacher_args)
    model, _, device = load_all(teacher_args)
    if args.task_checkpoint and Path(args.task_checkpoint).is_file():
        state = torch.load(args.task_checkpoint, map_location=device)
        model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, device


def _prepare_uid_adv(args: argparse.Namespace, device: torch.device) -> nn.Module:
    uid_args = deepcopy(args)
    uid_args.is_task = False
    uid_args.model = args.uid_model
    uid_args = set_args(uid_args)
    model = LoadModel(
        model_name=uid_args.model,
        Chans=uid_args.channel,
        Samples=int(uid_args.fs * uid_args.timepoint),
        n_classes=uid_args.nclass,
    ).to(device)
    if args.uid_checkpoint and Path(args.uid_checkpoint).is_file():
        state = torch.load(args.uid_checkpoint, map_location=device)
        model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _extract_labels(batch: Iterable[Tensor]) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    if len(batch) == 3:
        x, y, u = batch
        return x, y, u
    x, y = batch
    return x, y, None


def _compute_uniform_kl(p: Tensor) -> Tensor:
    num_classes = p.size(-1)
    uniform = torch.full_like(p, 1.0 / num_classes)
    return F.kl_div(torch.log(p + 1e-8), uniform, reduction="batchmean")


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
    eot_transform = build_eot_transform(args)

    teacher, device = _prepare_teacher(args)

    uid_args = deepcopy(args)
    uid_args.is_task = False
    uid_args = set_args(uid_args)
    uid_adv = _prepare_uid_adv(uid_args, device)

    trainloader_task, valloader_task, testloader_task = load_data(args, include_index=True)
    train_uid_map, val_uid_map, test_uid_map = _build_uid_maps_aligned(args)

    sample_x = next(iter(trainloader_task))[0]
    channels = sample_x.shape[2]

    perturber = FrequencyDomainPerturber(
        channels=channels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        epsilon=args.epsilon_a,
        device=device,
        num_uid_classes=uid_args.nclass,
    ).to(device)

    optimizer = torch.optim.Adam([perturber.delta], lr=args.lr)

    for epoch in range(args.epochs):
        perturber.train()
        running_loss = 0.0
        for batch in tqdm(trainloader_task, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            x, y_task, indices = batch
            x = x.to(device)
            y_task = y_task.to(device)
            uid_labels = _gather_uid_labels(indices, train_uid_map, device)

            x_bar = x.squeeze(1)
            with torch.no_grad():
                x_eot = eot_transform(x_bar)
                teacher_logits = teacher(x_eot.unsqueeze(1))
                teacher_prob = F.softmax(teacher_logits, dim=1)

            x_prime = perturber(x, uid_labels=uid_labels, transform=lambda _: x_eot)
            teacher_logits_prime = teacher(x_prime)
            teacher_prob_prime = F.softmax(teacher_logits_prime, dim=1)

            uid_logits_prime = uid_adv(x_prime)
            uid_prob_prime = F.softmax(uid_logits_prime, dim=1)

            kl_task = F.kl_div(torch.log(teacher_prob_prime + 1e-8), teacher_prob, reduction="batchmean")
            ce_task = F.cross_entropy(teacher_logits_prime, y_task.long())
            kl_uid = _compute_uniform_kl(uid_prob_prime)
            reg = perturber.delta_regularizer

            loss = (
                args.lambda_task * kl_task
                + args.lambda_ce * ce_task
                + args.lambda_uid * kl_uid
                + args.lambda_reg * reg
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
                eot_transform,
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
        eot_transform,
        device,
        apply_perturb=False,
    )
    test_task_acc, test_task_f1, test_task_bca, test_task_eer, test_uid_acc, test_uid_f1, test_uid_bca, test_uid_eer = evaluate_metrics(
        perturber,
        teacher,
        uid_adv,
        testloader_task,
        test_uid_map,
        eot_transform,
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
        "clean_task": (clean_task_acc, clean_task_f1, clean_task_bca, clean_task_eer),
        "clean_uid": (clean_uid_acc, clean_uid_f1, clean_uid_bca, clean_uid_eer),
        "perturbed_task": (test_task_acc, test_task_f1, test_task_bca, test_task_eer),
        "perturbed_uid": (test_uid_acc, test_uid_f1, test_uid_bca, test_uid_eer),
    }


@torch.no_grad()
def evaluate_metrics(
    perturber: FrequencyDomainPerturber,
    teacher: nn.Module,
    uid_adv: nn.Module,
    dataloader,
    uid_map: dict[int, int],
    eot_transform: Callable[[Tensor], Tensor],
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
            x_bar = x.squeeze(1)
            x_eot = eot_transform(x_bar)
            x_eval = perturber(x, uid_labels=uid_labels, transform=lambda _: x_eot)
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
    parser.add_argument("--channel", type=int, default=22)
    parser.add_argument("--timepoint", type=float, default=4.0)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--nclass", type=int, default=9)
    parser.add_argument("--task_model", type=str, default="EEGNet")
    parser.add_argument("--uid_model", type=str, default="EEGNet")
    parser.add_argument("--task_checkpoint", type=str, default="", help="Pretrained task teacher checkpoint")
    parser.add_argument("--uid_checkpoint", type=str, default="", help="Pretrained UID adversary checkpoint")
    parser.add_argument("--seed", type=int, default=2024)
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
    parser.add_argument("--enable_eot_shift", action="store_true", default=True)
    parser.add_argument("--enable_eot_scale", action="store_true", default=True)
    parser.add_argument("--enable_eot_channel_dropout", action="store_true", default=True)
    parser.add_argument("--enable_eot_resample", action="store_true", default=True)
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
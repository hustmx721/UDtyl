import argparse
import math
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from utils.Logging import Logger
from utils.dataset import set_seed
from utils.init_all import load_all, load_data, set_args
from utils.models import LoadModel


class FrequencyDomainPerturber(nn.Module):
    """Learnable frequency-domain magnitude perturbation template.

    Owns the learnable template ``delta`` with shape ``(1, C, F, 1)`` and
    applies it to the STFT magnitude of the input EEG.
    """

    def __init__(
        self,
        channels: int,
        n_fft: int = 256,
        hop_length: Optional[int] = None,
        epsilon: float = 0.05,
        mask: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.epsilon = epsilon
        self.device = device or torch.device("cpu")

        self.register_buffer("window", torch.hann_window(self.n_fft, device=self.device))
        n_freq = n_fft // 2 + 1
        delta = torch.zeros(1, channels, n_freq, 1, device=self.device)
        self.delta = nn.Parameter(delta)

        if mask is not None:
            self.register_buffer("mask", mask)
        else:
            self.register_buffer("mask", torch.ones(1, channels, n_freq, 1, device=self.device))

    def forward(self, x: Tensor, transform: Optional[Callable[[Tensor], Tensor]] = None) -> Tensor:
        """Apply the learnable perturbation in the STFT magnitude domain.

        Args:
            x: EEG batch of shape ``(B, 1, C, T)``.
            transform: Optional callable for EOT; applied before STFT.
        """

        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(f"Expected input shape (B,1,C,T), got {tuple(x.shape)}")

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
        masked_delta = self.mask * delta_tanh
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
    mask = torch.ones(x.shape[:-1], device=x.device)
    dropout_mask = torch.bernoulli((1 - drop_prob) * mask)
    return x * dropout_mask


def _resample_jitter(x: Tensor, max_rate_delta: float = 0.05) -> Tensor:
    if max_rate_delta <= 0:
        return x
    batch, channels, time_steps = x.shape
    rate = 1.0 + float(torch.empty(1, device=x.device).uniform_(-max_rate_delta, max_rate_delta))
    new_length = max(1, int(math.ceil(time_steps * rate)))
    x_resampled = F.interpolate(x.unsqueeze(1), size=new_length, mode="linear", align_corners=False)
    return F.interpolate(x_resampled, size=time_steps, mode="linear", align_corners=False).squeeze(1)


def build_eot_transform(args: argparse.Namespace) -> Callable[[Tensor], Tensor]:
    transforms = []
    if args.eot_shift > 0:
        transforms.append(lambda t: _random_shift(t, args.eot_shift))
    if args.eot_scale:
        transforms.append(lambda t: _random_scale(t, args.eot_scale_min, args.eot_scale_max))
    if args.eot_channel_dropout > 0:
        transforms.append(lambda t: _channel_dropout(t, args.eot_channel_dropout))
    if args.eot_resample > 0:
        transforms.append(lambda t: _resample_jitter(t, args.eot_resample))

    def apply_all(signal: Tensor) -> Tensor:
        out = signal
        for fn in transforms:
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
    return F.kl_div(torch.log(uniform), p, reduction="batchmean")


def train_distillation(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args = set_args(args)
    eot_transform = build_eot_transform(args)

    teacher, device = _prepare_teacher(args)
    uid_adv = _prepare_uid_adv(args, device)

    trainloader, valloader, _ = load_data(args, include_index=True)
    sample_x = next(iter(trainloader))[0]
    channels = sample_x.shape[2]

    perturber = FrequencyDomainPerturber(
        channels=channels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        epsilon=args.epsilon_a,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam([perturber.delta], lr=args.lr)

    for epoch in range(args.epochs):
        perturber.train()
        running_loss = 0.0
        for batch in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            x, y, u = _extract_labels(batch)
            x = x.to(device)
            y = y.to(device)
            if u is not None:
                u = u.to(device)

            x_bar = x.squeeze(1)
            with torch.no_grad():
                x_eot = eot_transform(x_bar)
                teacher_logits = teacher(x_eot.unsqueeze(1))
                teacher_prob = F.softmax(teacher_logits, dim=1)

            x_prime = perturber(x, transform=lambda _: x_eot)
            teacher_logits_prime = teacher(x_prime)
            teacher_prob_prime = F.softmax(teacher_logits_prime, dim=1)

            uid_logits_prime = uid_adv(x_prime)
            uid_prob_prime = F.softmax(uid_logits_prime, dim=1)

            kl_task = F.kl_div(torch.log(teacher_prob_prime + 1e-8), teacher_prob, reduction="batchmean")
            ce_task = F.cross_entropy(teacher_logits_prime, y.long())
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

        avg_loss = running_loss / max(1, len(trainloader))
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.6f}, Reg={reg.item():.6f}")

        if (epoch + 1) % args.val_interval == 0:
            perturber.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in valloader:
                    x, y, u = _extract_labels(batch)
                    x = x.to(device)
                    y = y.to(device)
                    x_bar = x.squeeze(1)
                    x_eot = eot_transform(x_bar)
                    x_prime = perturber(x, transform=lambda _: x_eot)
                    logits = teacher(x_prime)
                    val_loss += F.cross_entropy(logits, y.long()).item()
            val_loss /= max(1, len(valloader))
            print(f"Validation loss at epoch {epoch + 1}: {val_loss:.6f}")

    if args.save_delta:
        save_path = Path(args.save_delta)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(perturber.state_dict(), save_path)
        print(f"Saved perturbation template to {save_path}")


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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--save_delta", type=str, default="", help="Path to save the perturbation template")
    parser.add_argument("--eot_shift", type=int, default=16)
    parser.add_argument("--eot_scale", action="store_true", help="Enable amplitude scaling transform")
    parser.add_argument("--eot_scale_min", type=float, default=0.9)
    parser.add_argument("--eot_scale_max", type=float, default=1.1)
    parser.add_argument("--eot_channel_dropout", type=float, default=0.05)
    parser.add_argument("--eot_resample", type=float, default=0.02)
    parser.add_argument("--repeats", type=int, default=1, help="Number of seeds to run")
    parser.add_argument("--log_root", type=Path, default=Path("logs"))
    parser.add_argument("--is_task", type=bool, default=True)
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    seeds = list(range(args.seed, args.seed + args.repeats))

    for idx, seed in enumerate(seeds):
        args.seed = seed
        set_seed(seed)

        log_path = args.log_root / f"{args.dataset}_Distill_seed{seed}.log"
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

        train_distillation(args)

        print(f"Seed {seed} finished. Time used: {time.time() - start_time:.2f}s")
        sys.stdout = sys.__stdout__
        print(f"Seed {seed} log saved to {log_path}")

    print("All seeds finished.")


if __name__ == "__main__":
    main()

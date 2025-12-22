"""Distill-STFT-UD (Simple Version A, M=1)

Implements the algorithm you provided:
- Only \Delta is learnable.
- STFT -> magnitude + \Delta -> ReLU -> iSTFT.
- Task distillation: KL(p_T || p_T') with frozen teacher T.
- UID uniformization: KL(p_D' || uniform) with frozen adversary D.
- L2 regularizer on \Delta.
- \Delta is clipped to [-epsilon_delta, +epsilon_delta] each iteration.

This is a **self-contained** reference implementation (PyTorch only) designed for:
- correctness (shape checks, correct KL direction),
- reproducibility (deterministic EOT transform objects),
- easy integration into an existing training pipeline.

Author: ChatGPT (GPT-5.2 Thinking)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# ---------------------------
# 0) Utility: KL definitions
# ---------------------------

def kl_pq(p: Tensor, q: Tensor, eps: float = 1e-8, reduction: str = "batchmean") -> Tensor:
    """Compute KL(p || q) where p and q are probabilities.

    Args:
        p: (..., K) probabilities (sum to 1).
        q: (..., K) probabilities (sum to 1).
        eps: numerical stability.
        reduction: 'batchmean' (recommended), 'mean', or 'sum'.

    Returns:
        Scalar tensor.
    """
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: p{tuple(p.shape)} vs q{tuple(q.shape)}")
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    kl = (p * (p.log() - q.log())).sum(dim=-1)  # (...,)
    if reduction == "sum":
        return kl.sum()
    if reduction == "mean":
        return kl.mean()
    if reduction == "batchmean":
        # convention: sum over batch / batch_size
        return kl.sum() / max(1, kl.numel() if kl.dim() == 0 else kl.shape[0])
    raise ValueError(f"Unknown reduction={reduction}")


def uniform_kl(p: Tensor, num_classes: int, eps: float = 1e-8, reduction: str = "batchmean") -> Tensor:
    """Compute KL(p || uniform), where uniform has num_classes.

    Args:
        p: (B, K) probabilities.
        num_classes: K.
    """
    if p.dim() != 2:
        raise ValueError(f"Expected p as (B,K), got {tuple(p.shape)}")
    if p.size(1) != num_classes:
        raise ValueError(f"Expected K={num_classes}, got {p.size(1)}")
    p = p.clamp_min(eps)
    log_q = -math.log(num_classes)  # log(1/K)
    kl = (p * (p.log() - log_q)).sum(dim=1)  # (B,)
    if reduction == "sum":
        return kl.sum()
    if reduction == "mean":
        return kl.mean()
    if reduction == "batchmean":
        return kl.sum() / max(1, p.size(0))
    raise ValueError(f"Unknown reduction={reduction}")


# ---------------------------
# 1) Deterministic EOT Transform
# ---------------------------

class DeterministicTransform:
    """A transform with frozen parameters for one batch so it can be reused on x and x'."""

    def apply(self, x_bar: Tensor) -> Tensor:
        """Apply on (B,C,T). Must be deterministic."""
        raise NotImplementedError


@dataclass
class IdentityTransform(DeterministicTransform):
    def apply(self, x_bar: Tensor) -> Tensor:
        return x_bar


@dataclass
class TimeShiftTransform(DeterministicTransform):
    shift: int

    def apply(self, x_bar: Tensor) -> Tensor:
        if self.shift == 0:
            return x_bar
        return torch.roll(x_bar, shifts=self.shift, dims=-1)


@dataclass
class ScaleTransform(DeterministicTransform):
    scale: float

    def apply(self, x_bar: Tensor) -> Tensor:
        return x_bar * self.scale


@dataclass
class ChannelDropoutTransform(DeterministicTransform):
    drop_prob: float
    mask: Optional[Tensor] = None

    def apply(self, x_bar: Tensor) -> Tensor:
        if self.drop_prob <= 0:
            return x_bar
        if self.mask is None:
            broadcast_shape = (x_bar.size(0), x_bar.size(1), *([1] * (x_bar.dim() - 2)))
            keep = torch.bernoulli((1 - self.drop_prob) * torch.ones(broadcast_shape, device=x_bar.device))
            self.mask = keep
        return x_bar * self.mask


@dataclass
class ResampleJitterTransform(DeterministicTransform):
    rate: float

    def apply(self, x_bar: Tensor) -> Tensor:
        if math.isclose(self.rate, 1.0):
            return x_bar
        batch, channels, time_steps = x_bar.shape
        new_length = max(1, int(math.ceil(time_steps * self.rate)))
        flat = x_bar.reshape(batch * channels, 1, time_steps)
        up = F.interpolate(flat, size=new_length, mode="linear", align_corners=False)
        back = F.interpolate(up, size=time_steps, mode="linear", align_corners=False)
        return back.reshape(batch, channels, time_steps)


class EOTDistribution:
    """Sample one deterministic transform per batch.

    Add more transforms as needed; key requirement is determinism for reuse.
    """

    def __init__(
        self,
        *,
        enable_shift: bool = False,
        max_shift: int = 0,
        shift_prob: float = 1.0,
        enable_scale: bool = False,
        scale_low: float = 0.9,
        scale_high: float = 1.1,
        scale_prob: float = 1.0,
        enable_channel_dropout: bool = False,
        channel_dropout: float = 0.0,
        channel_dropout_prob: float = 1.0,
        enable_resample: bool = False,
        resample_max_rate_delta: float = 0.0,
        resample_prob: float = 1.0,
    ) -> None:
        self.enable_shift = enable_shift
        self.max_shift = int(max_shift)
        self.shift_prob = float(shift_prob)
        self.enable_scale = enable_scale
        self.scale_low = float(scale_low)
        self.scale_high = float(scale_high)
        self.scale_prob = float(scale_prob)
        self.enable_channel_dropout = enable_channel_dropout
        self.channel_dropout = float(channel_dropout)
        self.channel_dropout_prob = float(channel_dropout_prob)
        self.enable_resample = enable_resample
        self.resample_max_rate_delta = float(resample_max_rate_delta)
        self.resample_prob = float(resample_prob)

    def sample(self, *, device: torch.device) -> DeterministicTransform:
        # Compose transforms by nesting (shift then scale).
        t: DeterministicTransform = IdentityTransform()

        if self.enable_shift and self.max_shift > 0 and self._should_apply(self.shift_prob, device):
            shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,), device=device).item())
            t = _Compose(t, TimeShiftTransform(shift=shift))

        if self.enable_scale and self._should_apply(self.scale_prob, device):
            scale = float(torch.empty(1, device=device).uniform_(self.scale_low, self.scale_high).item())
            t = _Compose(t, ScaleTransform(scale=scale))

        if self.enable_channel_dropout and self.channel_dropout > 0 and self._should_apply(self.channel_dropout_prob, device):
            t = _Compose(t, ChannelDropoutTransform(drop_prob=self.channel_dropout))

        if self.enable_resample and self.resample_max_rate_delta > 0 and self._should_apply(self.resample_prob, device):
            rate = 1.0 + float(
                torch.empty(1, device=device).uniform_(-self.resample_max_rate_delta, self.resample_max_rate_delta).item()
            )
            t = _Compose(t, ResampleJitterTransform(rate=rate))

        return t

    @staticmethod
    def _should_apply(prob: float, device: torch.device) -> bool:
        if prob >= 1.0:
            return True
        if prob <= 0.0:
            return False
        return bool(torch.rand(1, device=device) < prob)


@dataclass
class _Compose(DeterministicTransform):
    first: DeterministicTransform
    second: DeterministicTransform

    def apply(self, x_bar: Tensor) -> Tensor:
        return self.second.apply(self.first.apply(x_bar))


# ---------------------------
# 2) STFT Perturber (learn Δ only)
# ---------------------------

class STFTDeltaPerturber(nn.Module):
    """Implements Protect(x;Δ): STFT -> ReLU(|S|+Δ) -> iSTFT.

    Δ is shared across all samples (Simple Version A, M=1).
    """

    def __init__(
        self,
        *,
        channels: int,
        time_steps: int,
        n_fft: int = 256,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        epsilon_delta: float = 0.05,
        init_delta: float = 0.01,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.time_steps = int(time_steps)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length) if hop_length is not None else self.n_fft // 4
        self.win_length = int(win_length) if win_length is not None else self.n_fft
        self.epsilon_delta = float(epsilon_delta)
        self.init_delta = float(init_delta)
        self.device = device or torch.device("cpu")

        # Window buffer
        self.register_buffer("window", torch.hann_window(self.win_length, device=self.device))

        # Infer STFT frame count K exactly as PyTorch will produce.
        with torch.no_grad():
            dummy = torch.zeros(1, self.time_steps, device=self.device)
            stft = torch.stft(
                dummy,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                return_complex=True,
            )
            # stft: (1, F, K)
            self.freq_bins = int(stft.shape[-2])
            self.frames = int(stft.shape[-1])

        # Learnable \Delta
        delta = torch.empty(1, self.channels, self.freq_bins, self.frames, device=self.device)
        delta.uniform_(-self.init_delta, self.init_delta)
        self.delta = nn.Parameter(delta)

    @torch.no_grad()
    def clip_(self) -> None:
        """In-place clip of \Delta to satisfy the constraint."""
        self.delta.clamp_(-self.epsilon_delta, self.epsilon_delta)

    def forward(self, x: Tensor) -> Tensor:
        """Protect x via STFT magnitude perturbation.

        Args:
            x: (B, 1, C, T)

        Returns:
            x': (B, 1, C, T)
        """
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(f"Expected x shape (B,1,C,T), got {tuple(x.shape)}")
        B, _, C, T = x.shape
        if C != self.channels:
            raise ValueError(f"Channel mismatch: expected C={self.channels}, got {C}")
        if T != self.time_steps:
            raise ValueError(f"Time mismatch: expected T={self.time_steps}, got {T}")

        # Enforce constraint before use (algorithm step 1)
        self.clip_()

        x_bar = x.squeeze(1)              # (B,C,T)
        flat = x_bar.reshape(B * C, T)    # (B*C,T)

        # STFT
        S = torch.stft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )  # (B*C, F, K)

        # Reshape and split magnitude/phase
        S = S.reshape(B, C, S.shape[-2], S.shape[-1])  # (B, C, F, K)
        A = S.abs()
        Phi = torch.angle(S)

        if (A.shape[-2] != self.freq_bins) or (A.shape[-1] != self.frames):
            raise RuntimeError(
                f"STFT bins/frames mismatch: got (F,K)=({A.shape[-2]},{A.shape[-1]}), "
                f"expected ({self.freq_bins},{self.frames})."
            )

        # Magnitude perturbation (M=1) + ReLU
        A_prime = torch.relu(A + self.delta)  # broadcast over batch
        S_prime = torch.polar(A_prime, Phi)

        # iSTFT
        S_prime_flat = S_prime.reshape(B * C, self.freq_bins, self.frames)
        x_rec = torch.istft(
            S_prime_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=T,
        )  # (B*C, T)

        x_rec = x_rec.reshape(B, C, T)
        return x_rec.unsqueeze(1)

    def l2_regularizer(self) -> Tensor:
        """||Δ||_2^2 (mean)."""
        return (self.delta ** 2).mean()


# ---------------------------------
# 3) Stage 1: optimize \Delta only
# ---------------------------------

@torch.no_grad()
def _freeze(model: nn.Module) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def optimize_delta(
    *,
    perturber: STFTDeltaPerturber,
    teacher: nn.Module,
    uid_adv: nn.Module,
    dataloader: Iterable[Tuple[Tensor, Tensor, Tensor]],
    num_uid_classes: int,
    lambda_task: float,
    lambda_uid: float,
    lambda_reg: float,
    lr: float,
    epochs: int,
    eot: Optional[EOTDistribution] = None,
    device: Optional[torch.device] = None,
    log_every: int = 50,
) -> None:
    """Stage-1 optimizer: update \Delta only.

    dataloader must yield (x, y_task, u_uid). y_task is not used in Simple-A
    (distillation preserves task via teacher consistency), but is kept for
    compatibility.
    """
    if device is None:
        device = next(perturber.parameters()).device

    _freeze(teacher)
    _freeze(uid_adv)

    perturber.train()
    optimizer = torch.optim.Adam([perturber.delta], lr=lr)

    step = 0
    for epoch in range(1, epochs + 1):
        for batch in dataloader:
            if len(batch) != 3:
                raise ValueError("Expected batches as (x, y, u)")
            x, _, _u = batch
            x = x.to(device)

            # Sample one deterministic transform for this batch.
            if eot is None:
                t = IdentityTransform()
            else:
                t = eot.sample(device=device)

            # Apply transform to get x_t
            x_t_bar = t.apply(x.squeeze(1))
            x_t = x_t_bar.unsqueeze(1)

            # Protect in freq-domain based on x_t
            x_prime = perturber(x_t)

            # Apply the same transform to protected signal to get x'_t
            x_prime_t_bar = t.apply(x_prime.squeeze(1))
            x_prime_t = x_prime_t_bar.unsqueeze(1)

            with torch.no_grad():
                p_T = F.softmax(teacher(x_t), dim=1)

            p_T_prime = F.softmax(teacher(x_prime_t), dim=1)
            p_D_prime = F.softmax(uid_adv(x_prime_t), dim=1)

            # Losses
            loss_task = kl_pq(p_T, p_T_prime, reduction="batchmean")
            loss_uid = uniform_kl(p_D_prime, num_uid_classes, reduction="batchmean")
            loss_reg = perturber.l2_regularizer()

            loss = lambda_task * loss_task + lambda_uid * loss_uid + lambda_reg * loss_reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Clip after update (algorithm step 7)
            perturber.clip_()

            step += 1
            if log_every > 0 and step % log_every == 0:
                print(
                    f"[epoch {epoch}/{epochs} | step {step}] "
                    f"L={loss.item():.6f} (task={loss_task.item():.6f}, uid={loss_uid.item():.6f}, reg={loss_reg.item():.6f}) "
                    f"| max|Δ|={perturber.delta.detach().abs().max().item():.5f}"
                )


# ---------------------------------
# 4) Stage 2: protect data (inference)
# ---------------------------------

@torch.no_grad()
def protect_batch(
    *,
    perturber: STFTDeltaPerturber,
    x: Tensor,
    eot: Optional[EOTDistribution] = None,
) -> Tensor:
    """Apply Protect(x;Δ*) exactly as defined (with optional EOT)."""
    perturber.eval()
    device = next(perturber.parameters()).device
    x = x.to(device)

    if eot is None:
        t = IdentityTransform()
    else:
        t = eot.sample(device=device)

    x_t_bar = t.apply(x.squeeze(1))
    x_t = x_t_bar.unsqueeze(1)
    x_prime = perturber(x_t)
    return x_prime


# ---------------------------
# 5) Lightweight unit tests
# ---------------------------

class _TinyNet(nn.Module):
    def __init__(self, channels: int, time_steps: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=(channels, 9), padding=(0, 4))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B,1,C,T)
        z = self.conv(x)  # (B,8,1,T)
        z = F.relu(z)
        z = self.pool(z).squeeze(-1).squeeze(-1)  # (B,8)
        return self.fc(z)


def _make_sine_batch(B: int, C: int, T: int, device: torch.device) -> Tensor:
    t = torch.linspace(0, 1, T, device=device)
    xs = []
    for _ in range(B):
        chans = []
        for c in range(C):
            f = 5.0 + 2.0 * c
            chans.append(torch.sin(2 * math.pi * f * t))
        xs.append(torch.stack(chans, dim=0))  # (C,T)
    x_bar = torch.stack(xs, dim=0)  # (B,C,T)
    return x_bar.unsqueeze(1)  # (B,1,C,T)


def _run_unit_tests() -> None:
    """Lightweight sanity tests.

    Note: Backprop through STFT/iSTFT on CPU can be slow. These tests keep the
    tensor sizes tiny so they finish quickly.
    """
    device = torch.device("cpu")
    torch.manual_seed(0)

    # Tiny shapes
    B, C, T = 2, 2, 64
    x = _make_sine_batch(B, C, T, device)

    pert = STFTDeltaPerturber(
        channels=C,
        time_steps=T,
        n_fft=16,
        hop_length=4,
        win_length=16,
        epsilon_delta=0.1,
        init_delta=0.0,
        device=device,
    )

    # 1) Identity when Δ=0 (approximately)
    with torch.no_grad():
        pert.delta.zero_()
        x_rec = pert(x)
        mse = (x_rec - x).pow(2).mean().item()
    assert mse < 1e-6, f"Expected near-perfect recon when Δ=0, got MSE={mse}"

    # 2) Clip works (allow tiny fp residue)
    with torch.no_grad():
        pert.delta.fill_(1.0)
        pert.clip_()
        assert pert.delta.abs().max().item() <= pert.epsilon_delta + 1e-6

    # 3) KL(p||uniform) = 0 when p is uniform
    p = torch.full((B, 5), 1.0 / 5)
    kl0 = uniform_kl(p, 5).item()
    assert abs(kl0) < 1e-8

    # 4) Gradient flows to Δ through one backward
    teacher = nn.Sequential(nn.Flatten(), nn.Linear(1 * C * T, 4)).to(device)
    uid_adv = nn.Sequential(nn.Flatten(), nn.Linear(1 * C * T, 6)).to(device)
    _freeze(teacher)
    _freeze(uid_adv)

    # Simple forward/backward
    x_t = x
    x_prime = pert(x_t)
    p_T = F.softmax(teacher(x_t), dim=1)
    p_T_prime = F.softmax(teacher(x_prime), dim=1)
    p_D_prime = F.softmax(uid_adv(x_prime), dim=1)

    loss = kl_pq(p_T, p_T_prime) + uniform_kl(p_D_prime, 6) + 1e-3 * pert.l2_regularizer()
    pert.zero_grad(set_to_none=True)
    loss.backward()
    assert pert.delta.grad is not None
    assert torch.isfinite(pert.delta.grad).all()

    print("All unit tests passed.")


if __name__ == "__main__":
    _run_unit_tests()

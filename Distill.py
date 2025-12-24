"""Distill-STFT-UD 

Implements the algorithm you provided with log-magnitude perturbations:
- Only \Delta is learnable.
- STFT -> log-magnitude + \Delta (tanh-bounded) -> exp -> iSTFT.
- Task distillation: KL(p_T || p_T') with frozen teacher T.
- UID uniformization: maximize entropy H(p_D').
- Regularizer on time-domain perturbation energy.

This is a **self-contained** reference implementation (PyTorch only) designed for:
- correctness (shape checks, correct KL direction),
- reproducibility (deterministic EOT transform objects),
- easy integration into an existing training pipeline.

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


def entropy_from_logits(logits: Tensor, tau: float = 1.0) -> Tensor:
    """Compute the mean entropy H(p) from logits.

    Args:
        logits: (..., K) pre-softmax scores.
        tau: temperature to soften the distribution if desired.
    """
    logp = F.log_softmax(logits / tau, dim=-1)
    p = logp.exp()
    ent = -(p * logp).sum(dim=-1)
    return ent.mean()


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
        max_shift: int = 0,
        shift_prob: float = 1.0,
        scale_low: float = 0.9,
        scale_high: float = 1.1,
        scale_prob: float = 1.0,
        channel_dropout: float = 0.0,
        channel_dropout_prob: float = 1.0,
        resample_max_rate_delta: float = 0.0,
        resample_prob: float = 1.0,
    ) -> None:
        self.max_shift = int(max_shift)
        self.shift_prob = float(shift_prob)
        self.scale_low = float(scale_low)
        self.scale_high = float(scale_high)
        self.scale_prob = float(scale_prob)
        self.channel_dropout = float(channel_dropout)
        self.channel_dropout_prob = float(channel_dropout_prob)
        self.resample_max_rate_delta = float(resample_max_rate_delta)
        self.resample_prob = float(resample_prob)

    def sample(self, *, device: torch.device) -> DeterministicTransform:
        # Compose transforms by nesting (shift then scale).
        t: DeterministicTransform = IdentityTransform()

        if self.max_shift > 0 and self._should_apply(self.shift_prob, device):
            shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,), device=device).item())
            t = _Compose(t, TimeShiftTransform(shift=shift))

        if self._should_apply(self.scale_prob, device):
            scale = float(torch.empty(1, device=device).uniform_(self.scale_low, self.scale_high).item())
            t = _Compose(t, ScaleTransform(scale=scale))

        if self.channel_dropout > 0 and self._should_apply(self.channel_dropout_prob, device):
            t = _Compose(t, ChannelDropoutTransform(drop_prob=self.channel_dropout))

        if self.resample_max_rate_delta > 0 and self._should_apply(self.resample_prob, device):
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

        # Learnable \Delta parameterized via tanh for bounded perturbations in log-magnitude space
        delta_raw = torch.full((1, self.channels, self.freq_bins, self.frames), 0.0, device=self.device)
        if self.init_delta != 0:
            init_scaled = max(min(self.init_delta / (self.epsilon_delta + 1e-12), 0.99), -0.99)
            delta_raw.fill_(math.atanh(init_scaled))
        self.delta_raw = nn.Parameter(delta_raw)

    def get_delta(self) -> Tensor:
        """Return bounded Δ in the same shape as STFT magnitudes."""
        return self.epsilon_delta * torch.tanh(self.delta_raw)

    @torch.no_grad()
    def magnitude_stats(self, x: Tensor) -> dict[str, float]:
        """Compute basic STFT magnitude stats for x and |Δ| for logging/debugging.

        Args:
            x: (B,1,C,T) batch input in the same layout expected by ``forward``.
        """
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(f"Expected x shape (B,1,C,T), got {tuple(x.shape)}")
        B, _, C, T = x.shape
        if C != self.channels or T != self.time_steps:
            raise ValueError(
                f"Input mismatch for magnitude stats: got C={C}, T={T}; expected C={self.channels}, T={self.time_steps}"
            )

        x_bar = x.squeeze(1)
        flat = x_bar.reshape(B * C, T)
        S = torch.stft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        mag = S.abs()

        delta_abs = self.get_delta().detach().abs()
        mag_mean = mag.mean().item()
        return {
            "stft_mean": mag_mean,
            "stft_max": mag.max().item(),
            "delta_mean": delta_abs.mean().item(),
            "delta_max": delta_abs.max().item(),
            "delta_to_mag_ratio": delta_abs.mean().item() / (mag_mean + 1e-12),
        }

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

        eps_mag = 1e-6
        A_log = torch.log(A + eps_mag)
        Delta = self.get_delta()  # broadcast over batch
        A_log_prime = A_log + Delta
        A_prime = torch.exp(A_log_prime) - eps_mag
        A_prime = torch.relu(A_prime)
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
        """||Δ||_2^2 (mean) in the bounded space."""
        return (self.get_delta() ** 2).mean()


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
    log_every: int = 10,
    early_stop_patience: int = 50,
    task_loss_threshold: float = 1e-3,
    plateau_tolerance: float = 1e-5,
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
    optimizer = torch.optim.Adam([perturber.delta_raw], lr=lr)

    step = 0
    plateau_steps = 0
    prev_task_loss: Optional[float] = None
    prev_uid_loss: Optional[float] = None
    prev_reg_loss: Optional[float] = None
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

            t_logits_prime = teacher(x_prime_t)
            p_T_prime = F.softmax(t_logits_prime, dim=1)
            uid_logits_prime = uid_adv(x_prime_t)

            # Losses
            loss_task = kl_pq(p_T, p_T_prime, reduction="batchmean")
            H_uid = entropy_from_logits(uid_logits_prime, tau=1.0)
            loss_uid = -H_uid
            loss_reg = ((x_prime_t - x_t) ** 2).mean() / (x_t.pow(2).mean() + 1e-12)

            loss = lambda_task * loss_task + lambda_uid * loss_uid + lambda_reg * loss_reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            with torch.no_grad():
                delta_grad_norm = (
                    perturber.delta_raw.grad.detach().norm().item()
                    if perturber.delta_raw.grad is not None
                    else 0.0
                )
                delta_eff = perturber.get_delta()
                delta_max = float(delta_eff.abs().max().item())
                delta_mean = float(delta_eff.abs().mean().item())
                H_val = float(H_uid.item())
                logK = math.log(num_uid_classes)
                gap = logK - H_val

            optimizer.step()

            step += 1
            if log_every > 0 and step % log_every == 0:
                mag_stats = perturber.magnitude_stats(x_t)
                print(
                    f"[epoch {epoch}/{epochs} | step {step}] "
                    f"L={loss.item():.6f} (task={loss_task.item():.6f}, -H_uid={loss_uid.item():.6f}, reg={loss_reg.item():.6f}) "
                    f"| H={H_val:.4f} (logK={logK:.4f}, gap={gap:.4f}) "
                    f"| grad||Δ||={delta_grad_norm:.3e} | max|Δ|={delta_max:.5f} | mean|Δ|={delta_mean:.5f} "
                    f"| STFT|A| mean={mag_stats['stft_mean']:.6f}, max={mag_stats['stft_max']:.6f}; "
                    f"|Δ| mean={mag_stats['delta_mean']:.6f}, max={mag_stats['delta_max']:.6f}, mean_ratio={mag_stats['delta_to_mag_ratio']:.6f}"
                )

            if early_stop_patience > 0 and loss_task.item() <= task_loss_threshold:
                if prev_task_loss is not None:
                    task_change = abs(loss_task.item() - prev_task_loss)
                    uid_change = abs(loss_uid.item() - (prev_uid_loss or 0.0))
                    reg_change = abs(loss_reg.item() - (prev_reg_loss or 0.0))
                    if (
                        task_change < plateau_tolerance
                        and uid_change < plateau_tolerance
                        and reg_change < plateau_tolerance
                    ):
                        plateau_steps += 1
                    else:
                        plateau_steps = 0
                prev_task_loss = loss_task.item()
                prev_uid_loss = loss_uid.item()
                prev_reg_loss = loss_reg.item()

                if plateau_steps >= early_stop_patience:
                    print(
                        f"Early stopping delta optimization at epoch {epoch}, step {step}: "
                        f"task/uid/reg losses plateaued with task loss {loss_task.item():.6f}."
                    )
                    return
            else:
                plateau_steps = 0
                prev_task_loss = None
                prev_uid_loss = None
                prev_reg_loss = None


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
        pert.delta_raw.zero_()
        x_rec = pert(x)
        mse = (x_rec - x).pow(2).mean().item()
    assert mse < 1e-6, f"Expected near-perfect recon when Δ=0, got MSE={mse}"

    # 2) Bounded Δ via tanh (allow tiny fp residue)
    with torch.no_grad():
        pert.delta_raw.fill_(10.0)
        assert pert.get_delta().abs().max().item() <= pert.epsilon_delta + 1e-6

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
    assert pert.delta_raw.grad is not None
    assert torch.isfinite(pert.delta_raw.grad).all()

    print("All unit tests passed.")


# if __name__ == "__main__":
#     _run_unit_tests()

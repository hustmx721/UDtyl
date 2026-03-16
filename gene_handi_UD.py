import torch
from typing import Dict, Optional


class RandTemplate:
    def __init__(
        self,
        n_channels: int,
        T: int,
        alpha: float,
        per_channel_std: Optional[torch.Tensor] = None,
        user_std: Optional[Dict[int, float]] = None,
        device: str = "cpu",
    ):
        """User-dependent uniform noise template."""

        self.C, self.T = n_channels, T
        self.alpha = alpha
        self.device = device
        self.per_channel_std = (
            per_channel_std.to(device) if per_channel_std is not None else None
        )
        self.user_std = user_std or {}
        self.templates: Dict[int, torch.Tensor] = {}

    def _scale(self, noise: torch.Tensor, user_id: int) -> torch.Tensor:
        if user_id in self.user_std:
            scale = self.alpha * float(self.user_std[user_id])
            return noise * scale
        if self.per_channel_std is not None:
            scale = self.alpha * self.per_channel_std.view(-1, 1)  # [C,1]
        else:
            scale = self.alpha
        return noise * scale

    def get(self, user_id: int, seed_base: int = 2024) -> torch.Tensor:
        if user_id not in self.templates:
            g = torch.Generator(device=self.device).manual_seed(seed_base + int(user_id))
            noise = 2.0 * torch.rand((self.C, self.T), generator=g, device=self.device) - 1.0
            self.templates[user_id] = self._scale(noise, user_id)
        return self.templates[user_id]

    def apply(self, x: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        """Add user-specific noise to batched input.

        Supports both ``[B, C, T]`` and ``[B, 1, C, T]`` layouts to align with
        the handcrafted UD pipeline in ``main_handi.py``.
        """

        if x.dim() == 3:  # [B, C, T]
            B, C, T = x.shape
            template_batch = torch.stack(
                [self.get(int(uid)) for uid in user_ids.tolist()], dim=0
            ).to(x.device)
            return x + template_batch

        if x.dim() == 4:  # [B, 1, C, T]
            B, _, C, T = x.shape
            template_batch = torch.stack(
                [self.get(int(uid)) for uid in user_ids.tolist()], dim=0
            ).to(x.device)
            return x + template_batch.unsqueeze(1)

        raise ValueError(f"Unexpected tensor shape {tuple(x.shape)} in RandTemplate.apply")


def _make_user_code_wave(
    user_code: int, T: int, bit_len: int = 10, reps_per_bit: int = 10
) -> torch.Tensor:
    """Generate SN square wave ``[1, T]`` from a decimal code (UE-Chen style)."""

    bits = [int(c) for c in f"{int(user_code):0>{bit_len}b}"[-bit_len:]]
    bits = torch.tensor(bits, dtype=torch.float32) * 2 - 1  # {-1, +1}
    wave = bits.repeat_interleave(reps_per_bit)
    if wave.numel() < T:
        n_rep = (T + wave.numel() - 1) // wave.numel()
        wave = wave.repeat(n_rep)
    wave = wave[:T]
    return wave.view(1, -1)  # [1, T]


class SNTemplate:
    def __init__(
        self,
        n_channels: int,
        T: int,
        alpha: float,
        per_channel_std: Optional[torch.Tensor] = None,
        user_std: Optional[Dict[int, float]] = None,
        user_codes: Optional[Dict[int, int]] = None,
        device: str = "cpu",
        a_low: float = 0.5,
        a_high: float = 1.5,
        reps_per_bit: int = 10,
        bit_len: int = 10,
    ):
        self.C, self.T = n_channels, T
        self.alpha = alpha
        self.device = device
        self.per_channel_std = (
            per_channel_std.to(device) if per_channel_std is not None else None
        )  # [C] or None
        self.user_std = user_std or {}
        self.user_codes = user_codes or {}
        self.a_low, self.a_high = a_low, a_high
        self.reps_per_bit = reps_per_bit
        self.bit_len = bit_len
        self.templates: Dict[int, torch.Tensor] = {}

    def _scale(self, base: torch.Tensor, user_id: int) -> torch.Tensor:
        if user_id in self.user_std:
            scale = self.alpha * float(self.user_std[user_id])
            return base * scale
        if self.per_channel_std is not None:
            scale = self.alpha * self.per_channel_std.view(-1, 1)  # [C,1]
        else:
            scale = self.alpha
        return base * scale

    def get(self, user_id: int, seed_base: int = 2025) -> torch.Tensor:
        if user_id not in self.templates:
            user_code = self.user_codes.get(user_id, user_id)
            w = _make_user_code_wave(
                user_code,
                self.T,
                bit_len=self.bit_len,
                reps_per_bit=self.reps_per_bit,
            ).to(self.device)  # [1,T]
            g = torch.Generator(device=self.device).manual_seed(seed_base + int(user_id))
            a = torch.empty(self.C, device=self.device).uniform_(
                self.a_low, self.a_high, generator=g
            )
            base = a.view(-1, 1) @ w  # [C,1] x [1,T] -> [C,T]
            self.templates[user_id] = self._scale(base, user_id)
        return self.templates[user_id]

    def apply(self, x: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:  # [B, C, T]
            B, C, T = x.shape
            template_batch = torch.stack(
                [self.get(int(uid)) for uid in user_ids.tolist()], dim=0
            ).to(x.device)
            return x + template_batch

        if x.dim() == 4:  # [B, 1, C, T]
            B, _, C, T = x.shape
            template_batch = torch.stack(
                [self.get(int(uid)) for uid in user_ids.tolist()], dim=0
            ).to(x.device)
            return x + template_batch.unsqueeze(1)

        raise ValueError(f"Unexpected tensor shape {tuple(x.shape)} in SNTemplate.apply")


class STFTRandTemplate:
    def __init__(
        self,
        n_channels: int,
        n_fft: int = 256,
        hop_length: Optional[int] = None,
        alpha: float = 0.1,
        per_channel_std: Optional[torch.Tensor] = None,  # [C] or None
        device: str = "cpu",
    ):
        """
        最简频域 user-wise STFT 扰动：
        - 为每个 user 生成一个 [C, F] 的频域噪声模板（作用在幅度上）
        - F = n_fft // 2 + 1
        - 在所有时间窗 K 上广播，加到 |STFT| 上，保持相位不变
        """

        self.C = n_channels
        self.n_fft = n_fft
        self.hop_length = hop_length or (n_fft // 4)
        self.F = n_fft // 2 + 1
        self.alpha = alpha
        self.device = device
        self.per_channel_std = (
            per_channel_std.to(device) if per_channel_std is not None else None
        )

        self.window = torch.hann_window(self.n_fft, device=device)
        self.templates: Dict[int, torch.Tensor] = {}

    def _scale(self, noise: torch.Tensor) -> torch.Tensor:
        if self.per_channel_std is not None:
            scale = self.alpha * self.per_channel_std.view(-1, 1).to(noise.device)
        else:
            scale = self.alpha
        return noise * scale

    def get(self, user_id: int, seed_base: int = 2024) -> torch.Tensor:
        if user_id not in self.templates:
            g = torch.Generator(device=self.device).manual_seed(seed_base + int(user_id))
            noise = 2.0 * torch.rand((self.C, self.F), generator=g, device=self.device) - 1.0
            self.templates[user_id] = self._scale(noise)
        return self.templates[user_id]

    def apply(self, x: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, "x should be [B, C, T]"
        B, C, T = x.shape
        assert C == self.C, f"Expected C={self.C}, got {C}"

        device = x.device
        window = self.window.to(device)

        x_flat = x.reshape(B * C, T)
        X = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )  # [B*C, F, K]

        _, F, K = X.shape
        assert F == self.F, f"STFT freq dim {F} != expected {self.F}"

        X = X.reshape(B, C, F, K)
        mag = X.abs()
        phase = X.angle()

        template_batch = torch.stack(
            [self.get(int(uid)) for uid in user_ids.tolist()], dim=0
        ).to(device)
        mag_pert = mag + template_batch.unsqueeze(-1)
        mag_pert = torch.clamp(mag_pert, min=0.0)

        X_pert = mag_pert * torch.exp(1j * phase)
        X_pert_flat = X_pert.reshape(B * C, F, K)

        x_rec = torch.istft(
            X_pert_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            length=T,
            return_complex=False,
        )

        return x_rec.reshape(B, C, T)

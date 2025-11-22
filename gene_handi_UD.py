import torch
from typing import Dict, Optional

class RandTemplate:
    def __init__(self, n_channels: int, T: int,
                 alpha: float,
                 per_channel_std: Optional[torch.Tensor] = None,
                 device: str = "cpu"):
        """
        per_channel_std: shape [C]，若提供则按通道缩放；否则用全局 alpha。
        """
        self.C, self.T = n_channels, T
        self.alpha = alpha
        self.per_channel_std = per_channel_std  # [C] or None
        self.device = device
        self.templates: Dict[int, torch.Tensor] = {}

    def _scale(self, noise: torch.Tensor) -> torch.Tensor:
        # noise: [C, T] ~ U(-1,1)
        if self.per_channel_std is not None:
            scale = self.alpha * self.per_channel_std.view(-1, 1)  # [C,1]
        else:
            scale = self.alpha
        return noise * scale

    def get(self, user_id: int, seed_base: int = 2024) -> torch.Tensor:
        if user_id not in self.templates:
            g = torch.Generator(device=self.device).manual_seed(seed_base + int(user_id))
            noise = (2.0 * torch.rand((self.C, self.T), generator=g, device=self.device) - 1.0)
            self.templates[user_id] = self._scale(noise)
        return self.templates[user_id]

    def apply(self, x: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T], user_ids: [B]
        返回 x + Δ_{user}
        """
        B, C, T = x.shape
        assert C == self.C and T == self.T
        out = x.clone()
        for i, uid in enumerate(user_ids.tolist()):
            out[i] = out[i] + self.get(uid)
        return out


import torch
from typing import Dict, Optional, Tuple

def _make_user_code_wave(user_id: int, T: int, bit_len: int = 10, reps_per_bit: int = 10) -> torch.Tensor:
    """
    生成用户方波 w_u: [1, T]，由 user_id 决定的 10-bit 码（也可用独立随机）。
    """
    # 用 user_id 的哈希/seed 生成 10-bit 码
    g = torch.Generator().manual_seed(1315423911 ^ user_id)  # 简单混合种子
    bits = torch.randint(low=0, high=2, size=(bit_len,), generator=g)  # [0,1]
    bits = bits.float() * 2 - 1  # -> {-1, +1}
    wave = bits.repeat_interleave(reps_per_bit)  # 长度 bit_len * reps_per_bit
    # 按需循环/裁剪到 T
    if wave.numel() < T:
        n_rep = (T + wave.numel() - 1) // wave.numel()
        wave = wave.repeat(n_rep)
    wave = wave[:T]
    return wave.view(1, -1)  # [1, T]

class SNTemplate:
    def __init__(self, n_channels: int, T: int,
                 alpha: float,
                 per_channel_std: Optional[torch.Tensor] = None,
                 device: str = "cpu",
                 a_low: float = 0.5, a_high: float = 1.5,
                 reps_per_bit: int = 10, bit_len: int = 10):
        self.C, self.T = n_channels, T
        self.alpha = alpha
        self.per_channel_std = per_channel_std  # [C] or None
        self.device = device
        self.a_low, self.a_high = a_low, a_high
        self.reps_per_bit = reps_per_bit
        self.bit_len = bit_len
        self.templates: Dict[int, torch.Tensor] = {}

    def _scale(self, base: torch.Tensor) -> torch.Tensor:
        # base: [C, T], 若给出 per_channel_std 则用通道尺度
        if self.per_channel_std is not None:
            scale = self.alpha * self.per_channel_std.view(-1, 1)  # [C,1]
        else:
            scale = self.alpha
        return base * scale

    def get(self, user_id: int, seed_base: int = 2025) -> torch.Tensor:
        if user_id not in self.templates:
            # 时域方波（用户码）
            w = _make_user_code_wave(user_id, self.T,
                                     bit_len=self.bit_len, reps_per_bit=self.reps_per_bit).to(self.device)  # [1,T]
            # 空域通道缩放
            g = torch.Generator(device=self.device).manual_seed(seed_base + int(user_id))
            a = torch.empty(self.C, device=self.device).uniform_(self.a_low, self.a_high).to(self.device)  # [C]
            base = a.view(-1, 1) @ w  # [C,1] x [1,T] -> [C,T]
            self.templates[user_id] = self._scale(base)
        return self.templates[user_id]

    def apply(self, x: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T], user_ids: [B]
        """
        B, C, T = x.shape
        assert C == self.C and T == self.T
        out = x.clone()
        for i, uid in enumerate(user_ids.tolist()):
            out[i] = out[i] + self.get(uid)
        return out

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

        参数:
        - n_channels: EEG 通道数 C
        - n_fft: STFT 的 n_fft
        - hop_length: STFT 的 hop_length，若为 None 默认 n_fft // 4
        - alpha: 噪声强度系数
        - per_channel_std: [C]，可选，对不同通道做幅度缩放（类似你 RandTemplate 里做法）
        """
        self.C = n_channels
        self.n_fft = n_fft
        self.hop_length = hop_length or (n_fft // 4)
        self.F = n_fft // 2 + 1
        self.alpha = alpha
        self.per_channel_std = per_channel_std  # [C] or None
        self.device = device

        # Hann 窗，注意要根据输入 x 的 device 移动
        self.window = torch.hann_window(self.n_fft)

        # 每个 user 一个 [C, F] 模板
        self.templates: Dict[int, torch.Tensor] = {}

    def _scale(self, noise: torch.Tensor) -> torch.Tensor:
        """
        noise: [C, F] ~ U(-1, 1)
        """
        if self.per_channel_std is not None:
            # [C,1] * [C,F] -> [C,F]
            scale = self.alpha * self.per_channel_std.view(-1, 1).to(noise.device)
        else:
            scale = self.alpha
        return noise * scale

    def get(self, user_id: int, seed_base: int = 2024) -> torch.Tensor:
        """
        返回该 user 的频域模板 Δ_u: [C, F]
        """
        if user_id not in self.templates:
            g = torch.Generator(device=self.device).manual_seed(seed_base + int(user_id))
            # U(-1,1) 噪声
            noise = (2.0 * torch.rand((self.C, self.F), generator=g, device=self.device) - 1.0)
            self.templates[user_id] = self._scale(noise)
        return self.templates[user_id]

    def apply(self, x: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T], user_ids: [B]
        输出: x' = ISTFT( STFT(x) + freq-Delta(user) ) 对应的时域信号
        """
        assert x.dim() == 3, "x should be [B, C, T]"
        B, C, T = x.shape
        assert C == self.C, f"Expected C={self.C}, got {C}"

        device = x.device
        window = self.window.to(device)

        # 1) 先把 [B, C, T] 展平为 [B*C, T] 做 STFT，方便调用 torch.stft
        x_flat = x.reshape(B * C, T)  # [B*C, T]

        X = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True
        )  # [B*C, F, K]

        _, F, K = X.shape
        assert F == self.F, f"STFT freq dim {F} != expected {self.F}"

        # 2) reshape 回 [B, C, F, K]
        X = X.reshape(B, C, F, K)  # [B, C, F, K]

        # 3) 分解为幅度和相位
        mag = X.abs()           # [B, C, F, K]
        phase = X.angle()       # [B, C, F, K]

        # 4) 对每个样本 i，添加对应 user 的 Δ_u: [C, F] -> [C, F, 1] -> 广播到 K
        mag_pert = mag.clone()
        for i, uid in enumerate(user_ids.tolist()):
            delta_u = self.get(uid).to(device)         # [C, F]
            delta_u = delta_u.unsqueeze(-1)            # [C, F, 1]
            # 广播到 [C, F, K]，在幅度上加噪
            mag_pert[i] = mag_pert[i] + delta_u

        # 5) 幅度不能为负，做个下界裁剪
        mag_pert = torch.clamp(mag_pert, min=0.0)

        # 6) 重新合成复数谱 X' = mag' * exp(i * phase)
        X_pert = mag_pert * torch.exp(1j * phase)      # [B, C, F, K]

        # 7) reshape 回 [B*C, F, K] 做 ISTFT
        X_pert_flat = X_pert.reshape(B * C, F, K)

        x_rec = torch.istft(
            X_pert_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            length=T,              # 保持原长度
            return_complex=False
        )  # [B*C, T]

        x_rec = x_rec.reshape(B, C, T)
        return x_rec
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

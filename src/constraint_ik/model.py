"""Conditional denoiser eps_theta(x_t, t, c): a FiLM-conditioned residual MLP, plus its EMA."""
from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn as nn

# Keys that older checkpoints store in model_cfg but the MLP does not use.
_UNUSED_CFG_KEYS = ("dim_mult", "kernel")


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class FiLMBlock(nn.Module):
    """h + MLP(LN(h) * (1 + scale) + shift), with (scale, shift) = Linear(emb)."""

    def __init__(self, dim: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.film = nn.Linear(emb_dim, 2 * dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.Mish(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(emb).chunk(2, dim=-1)
        return x + self.mlp(self.norm1(x) * (1 + scale) + shift)


class CondEndpointDDPM(nn.Module):
    """eps_theta: x (B, D), integer t (B,), cond (B, 15) -> eps (B, D). D = 2n (sincos)."""

    def __init__(self, in_dim: int, cond_dim: int = 15, base_channels: int = 512,
                 emb_dim: int = 384, depth: int = 10, dropout: float = 0.0):
        super().__init__()
        self.config = dict(in_dim=in_dim, cond_dim=cond_dim, base_channels=base_channels,
                           emb_dim=emb_dim, depth=depth, dropout=dropout)
        self.in_dim = in_dim
        self.time_emb = SinusoidalTimeEmb(emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.Mish(), nn.Linear(emb_dim, emb_dim))
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, emb_dim), nn.Mish(),
            nn.Linear(emb_dim, emb_dim), nn.Mish(),
            nn.Linear(emb_dim, emb_dim))
        self.input_proj = nn.Linear(in_dim, base_channels)
        self.blocks = nn.ModuleList(
            [FiLMBlock(base_channels, emb_dim, dropout) for _ in range(depth)])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(base_channels), nn.Mish(), nn.Linear(base_channels, in_dim))

    @classmethod
    def from_config(cls, cfg: dict) -> "CondEndpointDDPM":
        return cls(**{k: v for k, v in cfg.items() if k not in _UNUSED_CFG_KEYS})

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(self.time_emb(t)) + self.cond_mlp(cond)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, emb)
        return self.out_proj(h)


class EMA:
    """Parameter EMA: hard copy for step < warmup, then ema = decay * ema + (1 - decay) * p."""

    def __init__(self, model: nn.Module, decay: float = 0.9995, warmup: int = 1000):
        self.decay = decay
        self.warmup = warmup
        self.model = deepcopy(model).requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        for ep, p in zip(self.model.parameters(), model.parameters()):
            if step < self.warmup:
                ep.copy_(p)
            else:
                ep.mul_(self.decay).add_(p, alpha=1.0 - self.decay)
        for eb, b in zip(self.model.buffers(), model.buffers()):
            eb.copy_(b)

    def state_dict(self) -> dict:
        return self.model.state_dict()

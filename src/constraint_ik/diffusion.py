"""Noise schedules, forward noising and deterministic DDIM (eta = 0): sampling, partial inversion, decoding.

Tensors are (B, D). `alphas` is the cumulative product alpha_bar_t, t = 0..T-1 (float32).
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch

X0_CLAMP = 5.0


def make_schedule(name: str, T: int = 100) -> torch.Tensor:
    """alpha_bar (T,) for 'cosine', 'linear' or 'sigmoid' (paper models), computed in float64."""
    if name == "cosine":
        s = np.arange(T + 1, dtype=np.float64)
        ab = np.cos((s / T + 0.008) / 1.008 * np.pi / 2) ** 2
        ab = ab / ab[0]
        betas = np.clip(1 - ab[1:] / ab[:-1], 0, 0.999)
    elif name == "linear":
        betas = np.linspace(1e-4, 0.02, T, dtype=np.float64)
    elif name == "sigmoid":
        t = np.linspace(-6, 6, T, dtype=np.float64)
        betas = 1e-4 + (0.02 - 1e-4) * (1 / (1 + np.exp(-t)))
    else:
        raise ValueError(f"unknown schedule {name!r}")
    return torch.from_numpy(np.cumprod(1 - betas)).float()


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor,
             alphas: torch.Tensor) -> torch.Tensor:
    """x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps, with t (B,) long."""
    shape = (-1,) + (1,) * (x0.dim() - 1)
    a = alphas.sqrt()[t].view(shape)
    b = (1.0 - alphas).sqrt()[t].view(shape)
    return a * x0 + b * noise


def ddim_timesteps(t_start: int, t_end: int, n_steps: int = 100, T: int = 100) -> list[int]:
    """Levels from t_start to t_end with stride max(1, T // n_steps), always ending on t_end."""
    s = max(1, T // n_steps)
    if t_end >= t_start:
        ts = list(range(t_start, t_end + 1, s))
    else:
        ts = list(range(t_start, t_end - 1, -s))
    if ts[-1] != t_end:
        ts.append(t_end)
    return ts


def ddim_step(model, x: torch.Tensor, t_cur: int, t_next: int, cond: torch.Tensor,
              alphas: torch.Tensor) -> torch.Tensor:
    """Move x from level t_cur to t_next with eps evaluated at t_cur (either direction)."""
    t = torch.full((x.shape[0],), t_cur, device=x.device, dtype=torch.long)
    eps = model(x, t, cond)
    a_cur, a_next = alphas[t_cur], alphas[t_next]
    x0 = ((x - (1 - a_cur).sqrt() * eps) / a_cur.sqrt()).clamp(-X0_CLAMP, X0_CLAMP)
    return a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps


def ddim_run(model, x: torch.Tensor, cond: torch.Tensor, alphas: torch.Tensor,
             timesteps: list[int], grad: bool = False) -> torch.Tensor:
    """Apply ddim_step along consecutive pairs of `timesteps`. grad=False runs under no_grad."""
    with contextlib.nullcontext() if grad else torch.no_grad():
        for t_cur, t_next in zip(timesteps[:-1], timesteps[1:]):
            x = ddim_step(model, x, t_cur, t_next, cond, alphas)
    return x


def ddim_invert_partial(model, x0: torch.Tensor, cond: torch.Tensor, alphas: torch.Tensor,
                        t_bar: int, n_steps: int = 100) -> torch.Tensor:
    """x_0 -> x_{t_bar}. eps is evaluated at the lower level of each step; t_bar = T - 1 is full inversion."""
    return ddim_run(model, x0, cond, alphas, ddim_timesteps(0, t_bar, n_steps, len(alphas)))


def ddim_decode_partial(model, x: torch.Tensor, cond: torch.Tensor, alphas: torch.Tensor,
                        t_bar: int, n_steps: int = 100, grad: bool = False) -> torch.Tensor:
    """x_{t_bar} -> x_0 (the last step lands on level 0, not on the predicted x0)."""
    return ddim_run(model, x, cond, alphas, ddim_timesteps(t_bar, 0, n_steps, len(alphas)), grad)


def ddim_sample(model, x_T: torch.Tensor, cond: torch.Tensor, alphas: torch.Tensor,
                n_steps: int = 100) -> torch.Tensor:
    """Deterministic sampling from x_T at level T - 1."""
    return ddim_decode_partial(model, x_T, cond, alphas, len(alphas) - 1, n_steps)

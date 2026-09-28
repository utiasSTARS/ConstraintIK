"""Joint (sincos) and rotation (R6d) encodings, the 15-D conditioning vector, and a z-score normalizer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

if TYPE_CHECKING:
    from constraint_ik.constraints import Target

Array = np.ndarray | torch.Tensor


def q_to_sincos(q: Array) -> Array:
    """(..., n) -> (..., 2n) laid out as [sin q_1..sin q_n, cos q_1..cos q_n]."""
    if torch.is_tensor(q):
        return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)
    q = np.asarray(q)
    return np.concatenate([np.sin(q), np.cos(q)], axis=-1)


def sincos_to_q(sc: Array) -> Array:
    """(..., 2n) -> (..., n) via atan2, in (-pi, pi]."""
    n = sc.shape[-1] // 2
    if torch.is_tensor(sc):
        return torch.atan2(sc[..., :n], sc[..., n:])
    return np.arctan2(sc[..., :n], sc[..., n:])


def R_to_R6d(R: Array) -> Array:
    """(..., 3, 3) -> (..., 6): first two columns."""
    if torch.is_tensor(R):
        return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)
    R = np.asarray(R)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def R6d_to_R(r6d: Array) -> Array:
    """(..., 6) -> (..., 3, 3) by Gram-Schmidt on the two columns."""
    if torch.is_tensor(r6d):
        r1, r2 = r6d[..., :3], r6d[..., 3:6]
        r1 = r1 / r1.norm(dim=-1, keepdim=True)
        r2 = r2 - (r2 * r1).sum(-1, keepdim=True) * r1
        r2 = r2 / r2.norm(dim=-1, keepdim=True)
        return torch.stack([r1, r2, torch.cross(r1, r2, dim=-1)], dim=-1)
    r6d = np.asarray(r6d, dtype=np.float64)
    r1, r2 = r6d[..., :3], r6d[..., 3:6]
    r1 = r1 / np.linalg.norm(r1, axis=-1, keepdims=True)
    r2 = r2 - (r2 * r1).sum(-1, keepdims=True) * r1
    r2 = r2 / np.linalg.norm(r2, axis=-1, keepdims=True)
    return np.stack([r1, r2, np.cross(r1, r2, axis=-1)], axis=-1)


@dataclass
class Normalizer:
    """Per-coordinate z-score (x - mean) / std."""
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, x: torch.Tensor, eps: float = 1e-6) -> "Normalizer":
        """x (N, D); unbiased std clamped at eps."""
        return cls(mean=x.mean(dim=0), std=x.std(dim=0).clamp_min(eps))

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def invert(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std.to(z.device) + self.mean.to(z.device)

    def to(self, device: str | torch.device) -> "Normalizer":
        return Normalizer(self.mean.to(device), self.std.to(device))

    def to_dict(self) -> dict:
        return {"mean": self.mean.detach().cpu(), "std": self.std.detach().cpu()}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Normalizer | None":
        return None if d is None else cls(mean=d["mean"].clone(), std=d["std"].clone())


def build_cond(targets: "Target | Sequence[Target]", pos_normalizer: Normalizer,
               R6d_normalizer: Normalizer | None = None) -> torch.Tensor:
    """15-D conditioning [z-scored pos (3), R6d (6), mask (6)], float32 on CPU.

    One target gives (15,), a sequence gives (B, 15).
    """
    single = not isinstance(targets, (list, tuple))
    ts = [targets] if single else list(targets)
    pos = torch.as_tensor(np.stack([t.pos for t in ts]), dtype=torch.float32)
    r6d = torch.as_tensor(np.stack([R_to_R6d(t.R) for t in ts]), dtype=torch.float32)
    mask = torch.as_tensor(np.stack([t.mask for t in ts]), dtype=torch.float32)
    pos_n = (pos - pos_normalizer.mean.cpu()) / pos_normalizer.std.cpu()
    if R6d_normalizer is not None:
        r6d = (r6d - R6d_normalizer.mean.cpu()) / R6d_normalizer.std.cpu()
    cond = torch.cat([pos_n, r6d, mask], dim=-1)
    return cond[0] if single else cond

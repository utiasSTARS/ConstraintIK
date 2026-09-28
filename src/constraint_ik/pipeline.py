"""DiffusionIK: a loaded model with its schedule and normalizers, exposing encode/decode, eps/score,
sampling, partial inversion x_0 -> x_{t_bar} and decoding g_c: x_{t_bar} -> q."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from constraint_ik.constraints import Target
from constraint_ik.diffusion import ddim_decode_partial, ddim_invert_partial, ddim_sample
from constraint_ik.encoding import Normalizer, build_cond, q_to_sincos, sincos_to_q
from constraint_ik.model import CondEndpointDDPM

CondLike = torch.Tensor | Target | Sequence[Target]


@dataclass
class DiffusionIK:
    """Tensors are (B, D) in normalized sincos space (D = 2n); joint angles are (B, n)."""
    model: CondEndpointDDPM
    alphas: torch.Tensor
    x_normalizer: Normalizer
    pos_normalizer: Normalizer
    R6d_normalizer: Normalizer | None
    robot: str
    type_names: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def device(self) -> torch.device:
        return self.alphas.device

    @property
    def dim(self) -> int:
        return self.model.in_dim

    @property
    def n_joints(self) -> int:
        return self.model.in_dim // 2

    @property
    def T(self) -> int:
        return len(self.alphas)

    def to(self, device: str | torch.device) -> "DiffusionIK":
        self.model.to(device)
        self.alphas = self.alphas.to(device)
        self.x_normalizer = self.x_normalizer.to(device)
        return self

    # ---- conditioning and encodings

    def cond(self, targets: Target | Sequence[Target]) -> torch.Tensor:
        """(15,) for one target, (B, 15) for a sequence; on self.device."""
        return build_cond(targets, self.pos_normalizer, self.R6d_normalizer).to(self.device)

    def _cond(self, cond: CondLike, batch: int) -> torch.Tensor:
        if not torch.is_tensor(cond):
            cond = self.cond(cond)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        return cond.expand(batch, -1) if cond.shape[0] == 1 else cond

    def encode(self, q: np.ndarray | torch.Tensor) -> torch.Tensor:
        """q (..., n) -> x_0 (..., D): float32 sincos, z-scored."""
        q = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        return self.x_normalizer.apply(q_to_sincos(q))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """x (..., D) -> q (..., n) in (-pi, pi] via atan2 of the un-normalized sincos."""
        return sincos_to_q(self.x_normalizer.invert(x))

    # ---- network

    def eps(self, x: torch.Tensor, t: int | torch.Tensor, cond: CondLike) -> torch.Tensor:
        """eps_theta(x_t, t, c) for x (B, D)."""
        if not torch.is_tensor(t):
            t = torch.full((x.shape[0],), int(t), device=x.device, dtype=torch.long)
        return self.model(x, t, self._cond(cond, x.shape[0]))

    def score(self, x: torch.Tensor, t: int, cond: CondLike) -> torch.Tensor:
        """Score estimate -eps_theta / sqrt(1 - alpha_bar_t) at an integer level t."""
        sigma = float((1 - self.alphas[t]).sqrt().item())
        return -self.eps(x, t, cond) / max(sigma, 1e-8)

    # ---- DDIM

    def sample(self, target: Target, n: int, n_steps: int = 100, seed: int | None = None,
               generator: torch.Generator | None = None) -> np.ndarray:
        """n IK samples (n, n_joints) for one target. `seed` reseeds torch's global RNG."""
        if seed is not None:
            torch.manual_seed(seed)
        cond = self.cond(target).unsqueeze(0).expand(n, -1)
        x_T = torch.randn((n, self.dim), device=self.device, generator=generator)
        x0 = ddim_sample(self.model, x_T, cond, self.alphas, n_steps)
        return self.decode(x0).cpu().numpy()

    def invert(self, q: np.ndarray | torch.Tensor, t_bar: int, cond: CondLike,
               n_steps: int = 100) -> torch.Tensor:
        """Partial DDIM inversion of joint configurations q (B, n) to x_{t_bar} (B, D)."""
        x0 = self.encode(q)
        x0 = x0.unsqueeze(0) if x0.dim() == 1 else x0
        return ddim_invert_partial(self.model, x0, self._cond(cond, x0.shape[0]), self.alphas,
                                   t_bar, n_steps)

    def decode_from(self, x: torch.Tensor, t_bar: int, cond: CondLike, n_steps: int = 100,
                    grad: bool = False) -> torch.Tensor:
        """g_c: x_{t_bar} (B, D) -> q (B, n). grad=True keeps the autograd graph (for Dg_c)."""
        x0 = ddim_decode_partial(self.model, x, self._cond(cond, x.shape[0]), self.alphas,
                                 t_bar, n_steps, grad)
        return self.decode(x0)

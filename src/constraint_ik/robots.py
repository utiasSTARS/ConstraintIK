"""UR5 and Franka Panda kinematics: IKFast FK/IK (numpy) and a differentiable torch FK."""
from __future__ import annotations

import importlib
import math
from typing import NamedTuple

import numpy as np
import torch


class Shell(NamedTuple):
    """Cylindrical workspace shell used for target sampling (metres)."""
    r_min: float
    r_max: float
    z_min: float
    z_max: float

    def contains(self, pos: np.ndarray) -> np.ndarray:
        """Boolean mask over (..., 3) points, with r = hypot(x, y)."""
        pos = np.asarray(pos, dtype=np.float64)
        r = np.hypot(pos[..., 0], pos[..., 1])
        return ((r >= self.r_min) & (r <= self.r_max)
                & (pos[..., 2] >= self.z_min) & (pos[..., 2] <= self.z_max))


class Robot:
    """Common kinematics interface. Joint angles in radians, IKFast conventions."""

    name: str
    n_joints: int
    lo: np.ndarray
    hi: np.ndarray
    shell: Shell
    _ikfast_name: str

    def __init__(self) -> None:
        self._ikfast_mod = None

    @property
    def _ikfast(self):
        if self._ikfast_mod is None:
            self._ikfast_mod = importlib.import_module(self._ikfast_name)
        return self._ikfast_mod

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """IKFast FK. q (..., n) -> pos (..., 3), R (..., 3, 3)."""
        q = np.asarray(q, dtype=np.float64)
        flat = q.reshape(-1, self.n_joints)
        pos = np.empty((len(flat), 3))
        R = np.empty((len(flat), 3, 3))
        for i, qi in enumerate(flat):
            p, r = self._ikfast.get_fk(list(qi))
            pos[i] = np.asarray(p, dtype=np.float64)
            R[i] = np.asarray(r, dtype=np.float64).reshape(3, 3)
        batch = q.shape[:-1]
        return pos.reshape(*batch, 3), R.reshape(*batch, 3, 3)

    def in_limits(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        return np.all((q >= self.lo) & (q <= self.hi), axis=-1)

    def sample_q(self, n: int, rng: np.random.Generator,
                 margin: float = 0.05) -> np.ndarray:
        """Uniform joint configurations inside the limits shrunk by `margin` of the range."""
        span = self.hi - self.lo
        lo = self.lo + margin * span
        hi = self.hi - margin * span
        return lo + rng.random((n, self.n_joints)) * (hi - lo)

    def _ik(self, pos: np.ndarray, R: np.ndarray, free: float,
            check_limits: bool) -> list[np.ndarray]:
        try:
            raw = self._ikfast.get_ik(list(pos), [list(row) for row in R], [free])
        except Exception:
            return []
        sols = [np.asarray(s, dtype=np.float64) for s in raw]
        return [s for s in sols if not check_limits or self.in_limits(s)]

    def ik_all(self, pos: np.ndarray, R: np.ndarray,
               check_limits: bool = True) -> np.ndarray:
        """All IKFast solutions for one pose, shape (m, n); m may be 0."""
        raise NotImplementedError

    def fk_torch(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable FK matching IKFast. q (..., n) -> pos (..., 3), R (..., 3, 3)."""
        raise NotImplementedError

    def link_frames(self, q: torch.Tensor) -> torch.Tensor:
        """Frames after each joint (and the Franka flange) in the IKFast base frame: (..., m, 4, 4)."""
        raise NotImplementedError


def _stack_rows(rows: list[list[torch.Tensor]]) -> torch.Tensor:
    return torch.stack([torch.stack(r, dim=-1) for r in rows], dim=-2)


def _chain(mats: list[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    T = torch.eye(4, dtype=like.dtype, device=like.device).expand(*like.shape[:-1], 4, 4)
    for A in mats:
        T = T @ A
    return T


# UR5: standard DH; IKFast frame = Rz(pi) base rotation and a fixed tool permutation.
_UR5_D = (0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823)
_UR5_A = (0.0, -0.425, -0.39225, 0.0, 0.0, 0.0)
_UR5_ALPHA = (math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0)
_UR5_BASE = ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
_UR5_TOOL = ((0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0))


def _dh(theta: torch.Tensor, d: float, a: float, alpha: float) -> torch.Tensor:
    """Standard DH A = Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
    ct, st = torch.cos(theta), torch.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    zero, one = torch.zeros_like(ct), torch.ones_like(ct)
    return _stack_rows([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [zero, torch.full_like(ct, sa), torch.full_like(ct, ca), torch.full_like(ct, d)],
        [zero, zero, zero, one],
    ])


class UR5(Robot):
    name = "ur5"
    n_joints = 6
    lo = np.zeros(6)
    hi = np.full(6, 2 * math.pi)
    shell = Shell(0.20, 0.70, -0.10, 0.80)
    _ikfast_name = "ikfast_ur5"

    def ik_all(self, pos: np.ndarray, R: np.ndarray,
               check_limits: bool = True) -> np.ndarray:
        sols = self._ik(pos, R, 0.0, check_limits)
        return np.array(sols).reshape(-1, self.n_joints)

    def fk_torch(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T = _chain([_dh(q[..., i], _UR5_D[i], _UR5_A[i], _UR5_ALPHA[i])
                    for i in range(6)], q)
        base = torch.tensor(_UR5_BASE, dtype=q.dtype, device=q.device)
        tool = torch.tensor(_UR5_TOOL, dtype=q.dtype, device=q.device)
        pos = (base @ T[..., :3, 3:4]).squeeze(-1)
        R = base @ T[..., :3, :3] @ tool
        return pos, R

    def link_frames(self, q: torch.Tensor) -> torch.Tensor:
        T = torch.eye(4, dtype=q.dtype, device=q.device)
        T[:3, :3] = torch.tensor(_UR5_BASE, dtype=q.dtype, device=q.device)
        T = T.expand(*q.shape[:-1], 4, 4)
        frames = []
        for i in range(6):
            T = T @ _dh(q[..., i], _UR5_D[i], _UR5_A[i], _UR5_ALPHA[i])
            frames.append(T)
        return torch.stack(frames, dim=-3)


# Franka: modified DH (Craig) A = Rx(alpha) Tx(a) Rz(theta) Tz(d); last row is the 0.107 m flange.
_FR_A = (0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088, 0.0)
_FR_ALPHA = (0.0, -math.pi / 2, math.pi / 2, math.pi / 2,
             -math.pi / 2, math.pi / 2, math.pi / 2, 0.0)
_FR_D = (0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0, 0.107)
_FR_N_FREE = 50


def _mdh(theta: torch.Tensor, a: float, alpha: float, d: float) -> torch.Tensor:
    ct, st = torch.cos(theta), torch.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    zero, one = torch.zeros_like(ct), torch.ones_like(ct)
    return _stack_rows([
        [ct, -st, zero, torch.full_like(ct, a)],
        [st * ca, ct * ca, torch.full_like(ct, -sa), torch.full_like(ct, -d * sa)],
        [st * sa, ct * sa, torch.full_like(ct, ca), torch.full_like(ct, d * ca)],
        [zero, zero, zero, one],
    ])


class Franka(Robot):
    name = "franka"
    n_joints = 7
    lo = np.array([-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671])
    hi = np.array([2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671])
    shell = Shell(0.10, 0.85, -0.10, 1.10)
    _ikfast_name = "ikfast_franka_panda"

    def ik_all(self, pos: np.ndarray, R: np.ndarray,
               check_limits: bool = True) -> np.ndarray:
        """IKFast solutions over a sweep of the free joint 7 (50 values across its limits)."""
        sols: list[np.ndarray] = []
        for free in np.linspace(float(self.lo[6]), float(self.hi[6]), _FR_N_FREE):
            sols.extend(self._ik(pos, R, free, check_limits))
        return np.array(sols).reshape(-1, self.n_joints)

    def fk_torch(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        thetas = [q[..., i] for i in range(7)] + [torch.zeros_like(q[..., 0])]
        T = _chain([_mdh(th, _FR_A[i], _FR_ALPHA[i], _FR_D[i])
                    for i, th in enumerate(thetas)], q)
        return T[..., :3, 3], T[..., :3, :3]

    def link_frames(self, q: torch.Tensor) -> torch.Tensor:
        thetas = [q[..., i] for i in range(7)] + [torch.zeros_like(q[..., 0])]
        T = torch.eye(4, dtype=q.dtype, device=q.device).expand(*q.shape[:-1], 4, 4)
        frames = []
        for i, th in enumerate(thetas):
            T = T @ _mdh(th, _FR_A[i], _FR_ALPHA[i], _FR_D[i])
            frames.append(T)
        return torch.stack(frames, dim=-3)


ROBOTS: dict[str, type[Robot]] = {"ur5": UR5, "franka": Franka}


def get_robot(name: str) -> Robot:
    if name not in ROBOTS:
        raise ValueError(f"unknown robot {name!r}; choose from {sorted(ROBOTS)}")
    return ROBOTS[name]()

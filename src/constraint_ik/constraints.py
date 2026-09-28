"""Constraint families, targets, constraint errors and validity, and the differentiable map h_c."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from constraint_ik.encoding import R6d_to_R
from constraint_ik.robots import Robot, get_robot

# Checkpoint type order (also the stratified-sampler order).
CONSTRAINTS = ("se3", "posonly", "plane", "line", "orient", "plane_orient", "line_orient")
# Figure and table order (increasing codimension within position / orientation groups).
PAPER_ORDER = ("plane", "line", "posonly", "orient", "plane_orient", "line_orient", "se3")

# [px, py, pz, rx, ry, rz]; 1 = constrained. With two rotation bits the body axis
# whose bit is 0 has its direction pinned (body z for line_orient).
MASKS: dict[str, tuple[int, ...]] = {
    "se3":          (1, 1, 1, 1, 1, 1),
    "posonly":      (1, 1, 1, 0, 0, 0),
    "plane":        (0, 0, 1, 0, 0, 0),
    "line":         (0, 1, 1, 0, 0, 0),
    "orient":       (0, 0, 0, 1, 1, 1),
    "plane_orient": (0, 0, 1, 1, 1, 1),
    "line_orient":  (0, 1, 1, 1, 1, 0),
}

DISPLAY = {
    "se3": "SE(3)",
    "posonly": "Position",
    "plane": "Plane",
    "line": "Line",
    "orient": "Orientation",
    "plane_orient": "Plane+Orientation",
    "line_orient": "Line+Orientation",
}


def codim(c: str) -> int:
    """Codimension k of M_c (number of constrained task coordinates)."""
    return int(sum(MASKS[c]))


def intrinsic_dim(c: str, n: int) -> int:
    """dim M_c = n - k for an n-joint arm."""
    return n - codim(c)


def pos_axes(c: str) -> list[int]:
    return [i for i in range(3) if MASKS[c][i]]


def n_orient_bits(c: str) -> int:
    return int(sum(MASKS[c][3:]))


def pinned_axis(c: str) -> int | None:
    """Body axis whose direction is pinned by a two-bit orientation constraint."""
    if n_orient_bits(c) != 2:
        return None
    return MASKS[c][3:].index(0)


@dataclass(frozen=True, eq=False)
class Target:
    """One constraint instance: family, target position (free axes 0) and rotation (I if free)."""
    constraint: str
    pos: np.ndarray
    R: np.ndarray

    @classmethod
    def from_pose(cls, constraint: str, pos: np.ndarray, R: np.ndarray) -> "Target":
        """Canonical fill: zero the free position axes, identity rotation for position-only families."""
        pos = np.asarray(pos, dtype=np.float64).copy()
        free = [i for i in range(3) if not MASKS[constraint][i]]
        pos[free] = 0.0
        R = np.asarray(R, dtype=np.float64) if n_orient_bits(constraint) else np.eye(3)
        return cls(constraint, pos, R)

    @classmethod
    def from_entry(cls, entry: dict) -> "Target":
        """From a ground-truth target dict with constraint_type, cond_pos, cond_R6d and optionally R."""
        c = entry["constraint_type"]
        R = entry.get("R") if n_orient_bits(c) else None
        if R is None:
            R = R6d_to_R(np.asarray(entry["cond_R6d"], dtype=np.float64))
        return cls.from_pose(c, entry["cond_pos"], R)

    @property
    def mask(self) -> np.ndarray:
        return np.asarray(MASKS[self.constraint], dtype=np.float64)


# ---------------------------------------------------------------- errors and validity

def _angle_rad(R: np.ndarray, R_tgt: np.ndarray, c: str) -> float:
    """SO(3) geodesic angle (3 bits) or angle between pinned body axes (2 bits)."""
    if n_orient_bits(c) == 3:
        cos = 0.5 * (np.trace(R @ R_tgt.T) - 1.0)
    else:
        j = pinned_axis(c)
        cos = R[:, j] @ R_tgt[:, j]
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def errors(robot: Robot, q: np.ndarray, target: Target) -> tuple[np.ndarray, np.ndarray]:
    """Constraint-relevant errors of q (..., n): position L2 over constrained axes in mm,
    orientation in degrees. NaN where the family has no such component."""
    q = np.asarray(q)
    pos, R = robot.fk(q.reshape(-1, q.shape[-1]))
    axes, nb = pos_axes(target.constraint), n_orient_bits(target.constraint)
    pos_mm = np.full(len(pos), np.nan)
    orn_deg = np.full(len(pos), np.nan)
    for i in range(len(pos)):
        if axes:
            pos_mm[i] = 1000 * np.linalg.norm(pos[i][axes] - target.pos[axes])
        if nb:
            orn_deg[i] = np.degrees(_angle_rad(R[i], target.R, target.constraint))
    return pos_mm.reshape(q.shape[:-1]), orn_deg.reshape(q.shape[:-1])


@dataclass(frozen=True)
class Tolerance:
    """Validity thresholds per family. per_axis=True checks |p_i - p*_i| on each constrained
    axis (L-inf); per_axis=False checks the L2 position error."""
    name: str
    pos_mm: dict[str, float]
    orn_deg: dict[str, float]
    per_axis: bool


# Per-family tolerances (t_bar picker, valid-sample filters, size sweep, Score-SVD anchors).
_LOOSE_ORN_DEG = float(np.degrees(0.3))   # 0.3 rad
LOOSE = Tolerance(
    "loose",
    pos_mm={"se3": 50.0, "posonly": 100.0, "plane": 50.0, "line": 80.0,
            "plane_orient": 50.0, "line_orient": 80.0},
    orn_deg={c: _LOOSE_ORN_DEG for c in ("se3", "orient", "plane_orient", "line_orient")},
    per_axis=True,
)
# Reported validity: 10 mm (L2) and 5 degrees.
STRICT = Tolerance(
    "strict",
    pos_mm={c: 10.0 for c in CONSTRAINTS if pos_axes(c)},
    orn_deg={c: 5.0 for c in CONSTRAINTS if n_orient_bits(c)},
    per_axis=False,
)


def is_valid(robot: Robot, q: np.ndarray, target: Target,
             tol: Tolerance = STRICT) -> np.ndarray:
    """Boolean validity of q (..., n) for target under `tol`; NaN errors count as invalid."""
    q = np.asarray(q)
    c = target.constraint
    if not tol.per_axis:
        pos_mm, orn_deg = errors(robot, q, target)
        ok = np.ones(pos_mm.shape, dtype=bool)
        with np.errstate(invalid="ignore"):
            if pos_axes(c):
                ok &= np.isfinite(pos_mm) & (pos_mm <= tol.pos_mm[c])
            if n_orient_bits(c):
                ok &= np.isfinite(orn_deg) & (orn_deg <= tol.orn_deg[c])
        return ok
    pos, R = robot.fk(q.reshape(-1, q.shape[-1]))
    ok = np.ones(len(pos), dtype=bool)
    for i in range(len(pos)):
        for a in pos_axes(c):
            if abs(pos[i][a] - target.pos[a]) > tol.pos_mm[c] / 1000:
                ok[i] = False
        if ok[i] and n_orient_bits(c):
            ok[i] = _angle_rad(R[i], target.R, c) <= np.radians(tol.orn_deg[c])
    return ok.reshape(q.shape[:-1])


# ---------------------------------------------------------------- differentiable h_c

def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Rotation vector of R (..., 3, 3), differentiable away from angle pi."""
    cos = (0.5 * (R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0)).clamp(-1.0 + 1e-9, 1.0 - 1e-9)
    theta = torch.acos(cos)
    vee = 0.5 * torch.stack([R[..., 2, 1] - R[..., 1, 2],
                             R[..., 0, 2] - R[..., 2, 0],
                             R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    scale = torch.where(theta < 1e-4, 1.0 + theta ** 2 / 6.0, theta / torch.sin(theta))
    return scale.unsqueeze(-1) * vee


class ConstraintMap:
    """h_c(q) = 0 exactly on M_c, with Dh_c of full row rank k (float64 autograd through fk_torch).

    Rows: p_i(q) - p*_i per constrained axis (m); so3_log(R(q)^T R*) for three rotation bits;
    E^T (R(q)[:, j] - R*[:, j]) for two bits, with E a fixed orthonormal basis orthogonal to R*[:, j].
    """

    def __init__(self, robot: Robot | str, target: Target, device: str = "cpu",
                 dtype: torch.dtype = torch.float64):
        self.robot = get_robot(robot) if isinstance(robot, str) else robot
        self.target = target
        self.dtype = dtype
        c = target.constraint
        self.axes = pos_axes(c)
        self.nb = n_orient_bits(c)
        self.k = codim(c)
        self.p_tgt = torch.as_tensor(target.pos, dtype=dtype, device=device)
        self.R_tgt = torch.as_tensor(target.R, dtype=dtype, device=device)
        self.j = pinned_axis(c)
        self.E = None
        if self.j is not None:
            v = self.R_tgt[:, self.j]
            ref = torch.zeros(3, dtype=dtype, device=device)
            ref[int(torch.argmin(v.abs()))] = 1.0
            e1 = torch.linalg.cross(v, ref)
            e1 = e1 / e1.norm()
            self.E = torch.stack([e1, torch.linalg.cross(v, e1)], dim=1)
        self._h_batched = torch.func.vmap(self._h)
        self._jac_batched = torch.func.vmap(torch.func.jacrev(self._h))

    def _h(self, q: torch.Tensor) -> torch.Tensor:
        pos, R = self.robot.fk_torch(q)
        parts = []
        if self.axes:
            parts.append(pos[self.axes] - self.p_tgt[self.axes])
        if self.nb == 3:
            parts.append(so3_log(R.T @ self.R_tgt))
        elif self.nb == 2:
            parts.append(self.E.T @ (R[:, self.j] - self.R_tgt[:, self.j]))
        return torch.cat(parts)

    def h(self, q: torch.Tensor) -> torch.Tensor:
        """(B, n) -> (B, k)."""
        return self._h_batched(q.to(self.dtype))

    def jac(self, q: torch.Tensor) -> torch.Tensor:
        """Dh_c: (B, n) -> (B, k, n)."""
        return self._jac_batched(q.to(self.dtype))

    def tangent_basis(self, q: torch.Tensor) -> torch.Tensor:
        """Orthonormal basis of ker Dh_c: (B, n) -> (B, n, n - k)."""
        _, _, Vh = torch.linalg.svd(self.jac(q), full_matrices=True)
        return Vh[:, self.k:, :].transpose(1, 2)

    def tangent_projector(self, q: torch.Tensor) -> torch.Tensor:
        """Orthogonal projector onto T_q M_c: (B, n) -> (B, n, n)."""
        T = self.tangent_basis(q)
        return T @ T.transpose(1, 2)

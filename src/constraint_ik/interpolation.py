"""Latent interpolation between two IK solutions: invert both to x_{t_bar}, interpolate (lerp or slerp),
decode the interpolants with g_c, and score the decoded path; plus the rule that picks t_bar."""
from __future__ import annotations

import numpy as np
import torch

from constraint_ik.constraints import LOOSE, STRICT, Target, errors, is_valid, n_orient_bits, pos_axes
from constraint_ik.diffusion import ddim_step
from constraint_ik.pipeline import DiffusionIK
from constraint_ik.robots import Robot

T_BARS = (0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95)
METHODS = ("linear", "slerp")
N_FRAMES = 24
LAMBDA_POS, LAMBDA_ORN, LAMBDA_STEP = 0.5, 1.0, 0.1
ORN_SCORED = ("orient", "plane_orient", "line_orient")   # families whose orientation error enters the t_bar score


def invert_levels(ik: DiffusionIK, q: np.ndarray, t_bars, cond: torch.Tensor,
                  n_steps: int = 100) -> dict[int, torch.Tensor]:
    """{t: x_t (1, D)} for one configuration q (n,) and cond (15,). With a stride-1 grid every partial
    inversion is a prefix of the one to max(t_bars), so one pass gives the tensors of separate inversions."""
    if max(1, ik.T // n_steps) != 1:
        return {t: ik.invert(q, t, cond, n_steps) for t in t_bars}
    x = ik.encode(q)[None]
    c = cond.reshape(1, -1)
    out = {0: x} if 0 in t_bars else {}
    with torch.no_grad():
        for t in range(max(t_bars)):
            x = ddim_step(ik.model, x, t, t + 1, c, ik.alphas)
            if t + 1 in t_bars:
                out[t + 1] = x
    return out


def lerp(z_a: torch.Tensor, z_b: torch.Tensor, n: int) -> torch.Tensor:
    """(1 - a) z_a + a z_b for a = linspace(0, 1, n); (n, D)."""
    a = torch.linspace(0.0, 1.0, n, device=z_a.device)[:, None]
    return (1 - a) * z_a.reshape(1, -1) + a * z_b.reshape(1, -1)


def slerp(z_a: torch.Tensor, z_b: torch.Tensor, n: int) -> torch.Tensor:
    """Great-circle interpolation of the directions of z_a, z_b on the sphere of radius
    (|z_a| + |z_b|) / 2; lerp if the directions are (anti)parallel. (n, D)."""
    z_a, z_b = z_a.reshape(-1), z_b.reshape(-1)
    r_a, r_b = z_a.norm(), z_b.norm()
    r = 0.5 * (r_a + r_b)
    u_a, u_b = z_a / (r_a + 1e-12), z_b / (r_b + 1e-12)
    theta = torch.arccos((u_a * u_b).sum().clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    s = torch.sin(theta)
    path = []
    for a in torch.linspace(0.0, 1.0, n, device=z_a.device):
        if s.abs().item() < 1e-4:
            path.append((1 - a) * z_a + a * z_b)
        else:
            d = (torch.sin((1 - a) * theta) * u_a + torch.sin(a * theta) * u_b) / s
            path.append(r * d / (d.norm() + 1e-12))
    return torch.stack(path)


INTERPOLANTS = {"linear": lerp, "slerp": slerp}


def interpolate(ik: DiffusionIK, q_a: np.ndarray, q_b: np.ndarray, target: Target, t_bar: int,
                method: str = "linear", n_frames: int = N_FRAMES, n_steps: int = 100) -> np.ndarray:
    """Decoded path (n_frames, n) from q_a to q_b through the latent at t_bar."""
    cond = ik.cond(target)
    z_a, z_b = (ik.invert(q, t_bar, cond, n_steps) for q in (q_a, q_b))
    z = INTERPOLANTS[method](z_a, z_b, n_frames)
    return ik.decode_from(z, t_bar, cond, n_steps).cpu().numpy()


def _nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if np.any(np.isfinite(x)) else float("nan")


def path_metrics(robot: Robot, q_path: np.ndarray, target: Target) -> dict:
    """Per-frame constraint errors and the per-path summary (means over all frames, endpoints included)."""
    q = np.asarray(q_path, dtype=np.float64)
    pos_mm, orn_deg = errors(robot, q, target)
    step_mm = 1000.0 * np.linalg.norm(np.diff(robot.fk(q)[0], axis=0), axis=-1)
    return {
        "pos_err_mm": pos_mm,
        "orn_err_deg": orn_deg,
        "valid_pct_loose": float(100 * is_valid(robot, q, target, LOOSE).mean()),
        "valid_pct_strict": float(100 * is_valid(robot, q, target, STRICT).mean()),
        "mean_pos_err_mm": _nanmean(pos_mm),
        "mean_orn_err_deg": _nanmean(orn_deg),
        "max_ee_step_mm": float(step_mm.max()),
        "p95_ee_step_mm": float(np.percentile(step_mm, 95)),
    }


def t_bar_scores(rows: list[dict], constraint: str) -> dict[int, float]:
    """score(t) = valid% (LOOSE) - 0.5 pos_mm - lambda_orn orn_deg - 0.1 p95 step_mm, each averaged over
    the paths at t; lambda_orn = 1 for orient, plane_orient, line_orient and 0 otherwise."""
    lam_pos = LAMBDA_POS if pos_axes(constraint) else 0.0
    lam_orn = LAMBDA_ORN if constraint in ORN_SCORED and n_orient_bits(constraint) else 0.0
    scores = {}
    for t in sorted({r["t_bar"] for r in rows}):
        sub = [r for r in rows if r["t_bar"] == t]
        mean = lambda k: float(np.mean([r[k] for r in sub]))
        s = mean("valid_pct_loose")
        if lam_pos:
            s -= lam_pos * mean("mean_pos_err_mm")
        if lam_orn:
            s -= lam_orn * mean("mean_orn_err_deg")
        scores[t] = s - LAMBDA_STEP * mean("p95_ee_step_mm")
    return scores


def pick_t_bar(rows: list[dict], constraint: str) -> int:
    """argmax of t_bar_scores (smallest t_bar on ties) over one method's sweep rows."""
    scores = t_bar_scores(rows, constraint)
    return max(scores, key=lambda t: (scores[t], -t))

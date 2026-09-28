"""Local flattening diagnostics along a straight latent path x(a) = (1 - a) x_a + a x_b:
directional sensitivity rho, joint-space speed nu, tangent overlap tau^q, and random-direction baselines."""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import beta

from constraint_ik.constraints import ConstraintMap, Target, codim
from constraint_ik.pipeline import DiffusionIK

N_ALPHA = 10


def directional_derivative(ik: DiffusionIK, z: torch.Tensor, v: torch.Tensor, t_bar: int,
                           cond: torch.Tensor, n_steps: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """q = g_c(z) and u = Dg_c(z) v for a batch of latents z (B, D) and one direction v (D,)."""
    f = lambda x: ik.decode_from(x, t_bar, cond, n_steps, grad=True)
    q, u = torch.autograd.functional.jvp(f, z, v.expand_as(z).contiguous())
    return q.detach(), u.detach()


def pair_diagnostics(ik: DiffusionIK, cmap: ConstraintMap, q_a: np.ndarray, q_b: np.ndarray,
                     target: Target, t_bar: int, n_alpha: int = N_ALPHA, n_steps: int = 100) -> dict:
    """rho, nu, tau^q at n_alpha evenly spaced points of the straight latent path between q_a and q_b."""
    cond = ik.cond(target)
    z_a, z_b = (ik.invert(q, t_bar, cond, n_steps)[0] for q in (q_a, q_b))
    dz = z_b - z_a
    v = dz / dz.norm()
    a = torch.linspace(0.0, 1.0, n_alpha, device=z_a.device)[:, None]
    q, u = directional_derivative(ik, (1 - a) * z_a + a * z_b, v, t_bar, cond, n_steps)
    P = cmap.tangent_projector(q.to(cmap.dtype))
    u64 = u.to(cmap.dtype)
    rho = u64.norm(dim=1)
    tau = torch.einsum("bij,bj->bi", P, u64).norm(dim=1) / rho.clamp_min(1e-15)
    return {"alpha": a[:, 0].cpu().numpy(), "rho": rho.cpu().numpy(),
            "nu": float(dz.norm()) * rho.cpu().numpy(), "tau": tau.cpu().numpy()}


def random_baseline(n: int, k: int) -> dict:
    """tau of a uniform random unit direction in R^n against a rank-(n - k) projector:
    tau^2 ~ Beta((n - k) / 2, k / 2). RMS, median and 5th percentile."""
    if k >= n:
        return {"rms": 0.0, "median": 0.0, "p5": 0.0}
    dist = beta((n - k) / 2, k / 2)
    return {"rms": float(np.sqrt((n - k) / n)), "median": float(np.sqrt(dist.ppf(0.5))),
            "p5": float(np.sqrt(dist.ppf(0.05)))}


def summarize(rows: list[dict], n: int, constraint: str) -> dict:
    """Medians over all (pair, alpha) samples, tau percentiles and the relative gain over the baseline."""
    rho = np.array([r["rho"] for r in rows])
    nu = np.array([r["nu"] for r in rows])
    tau = np.array([r["tau"] for r in rows])
    base = random_baseline(n, codim(constraint))
    tau_med = float(np.median(tau))
    return {"rho_med": float(np.median(rho)), "nu_med": float(np.median(nu)),
            "tau_med": tau_med, "tau_p5": float(np.percentile(tau, 5)),
            "tau_rand_rms": base["rms"], "tau_rand_med": base["median"], "tau_rand_p5": base["p5"],
            "delta_rel": (tau_med - base["median"]) / base["median"] if base["median"] > 0 else None,
            "n_samples": len(rows)}

"""How far the Score-SVD spectral gap survives as the anchor is pushed off M_c.

Anchors are those of eval_id.py (GT solutions nearest to valid model samples). Each anchor q0 is moved
along one fixed random unit joint-space direction v, either normal (uniform on the unit sphere of the
normal space row(Dh_c) of M_c) or isotropic (uniform on the unit sphere of R^n, i.e. a normal move plus
a tangent one), by the step s that makes the scaled constraint residual ||W h_c(q0 + s v)|| equal delta.
W converts h_c to mm (position rows) and degrees (rotation rows), so delta is the constraint violation
in mm / deg. Score-SVD(+dagger) then runs at t, K with the same noise draws for every delta.

    python scripts/eval_perturbation.py --robot ur5   # -> results/id/ur5_perturbation.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from constraint_ik import ConstraintMap, Target, errors, get_robot, load_checkpoint
from constraint_ik import id_estimation as ide
from constraint_ik.constraints import intrinsic_dim, pos_axes

KINDS = ("normal", "isotropic")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="ur5", choices=["ur5", "franka"])
    p.add_argument("--ckpt", default=None, help="default: checkpoints/{robot}.pt")
    p.add_argument("--gt", default=None, help="default: data/ground_truth_{robot}.pt")
    p.add_argument("--deltas", type=float, nargs="+", default=[0, 1, 2, 5, 10, 20, 50],
                   help="constraint violation, mm / deg")
    p.add_argument("--n_targets", type=int, default=15)
    p.add_argument("--n_samples", type=int, default=512)
    p.add_argument("--n_anchors", type=int, default=5)
    p.add_argument("--t", type=int, default=15)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="default: results/id/{robot}_perturbation.json")
    a = p.parse_args()
    a.ckpt = a.ckpt or ide.CKPT_PATH.format(robot=a.robot)
    a.gt = a.gt or ide.GT_PATH[a.robot]
    a.out = a.out or f"results/id/{a.robot}_perturbation.json"
    return a


def residual_weights(c: str, k: int) -> torch.Tensor:
    """h_c rows are positions (m) then rotations (rad): scale to mm and degrees."""
    n_pos = len(pos_axes(c))
    return torch.tensor([1000.0] * n_pos + [180.0 / np.pi] * (k - n_pos), dtype=torch.float64)


def step_to_violation(cmap: ConstraintMap, W, q0, v, delta, iters: int = 50) -> torch.Tensor:
    """s >= 0 with ||W h_c(q0 + s v)|| = delta, per row (bracket by doubling, then bisection)."""
    viol = lambda s: (cmap.h(q0 + s[:, None] * v) * W).norm(dim=1)
    s_lin = delta / (cmap.jac(q0) @ v.unsqueeze(-1)).squeeze(-1).mul(W).norm(dim=1)
    lo, hi = torch.zeros_like(s_lin), s_lin.clone()
    for _ in range(30):
        short = viol(hi) < delta
        if not short.any():
            break
        lo, hi = torch.where(short, hi, lo), torch.where(short, 2 * hi, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = viol(mid) < delta
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    return 0.5 * (lo + hi)


@torch.no_grad()
def spectra(ik, x0, cond, noise, t) -> np.ndarray:
    """Batched Score-SVD: x0 (A, D), cond (A, 15), noise (A, K, D) -> (A, D) descending spectra."""
    A, K, D = noise.shape
    a = ik.alphas[t]
    x_t = float(a.sqrt()) * x0[:, None, :] + float((1 - a).sqrt()) * noise
    S = ik.score(x_t.reshape(A * K, D), t, cond.repeat_interleave(K, dim=0))
    return ide.singular_values(S.reshape(A, K, D))


def main():
    args = parse_args()
    t0 = time.time()
    ik = load_checkpoint(args.ckpt, args.device)
    robot = get_robot(ik.robot)
    gt = ide.load_gt(args.gt, args.n_targets)
    D, n = ik.dim, ik.n_joints
    rng = np.random.default_rng(args.seed)          # anchor fallback, as eval_id.py
    dir_rng = np.random.default_rng(args.seed + 1)  # perturbation directions
    gen = torch.Generator(device=ik.device).manual_seed(args.seed)   # score noise
    deltas = torch.tensor(args.deltas, dtype=torch.float64)
    res = {"args": vars(args), "robot": ik.robot, "in_dim": D, "deltas": args.deltas,
           "sigma_t": float((1 - ik.alphas[args.t]).sqrt()), "by_constraint": {}}

    for c, entries in gt.items():
        d_true = intrinsic_dim(c, n)
        k_star = D - d_true
        q_pert = {kind: [] for kind in KINDS}             # [target] -> (n_delta, A, n)
        targets, q0_all = [], []
        for ii, e in enumerate(entries):
            target = Target.from_entry(e)
            sols = np.asarray(e["sols_dedup"], dtype=np.float64)
            q_valid = ide.valid_samples(ik, robot, target, args.n_samples, args.seed + ii)
            q0 = torch.as_tensor(sols[ide.model_anchors(sols, q_valid, args.n_anchors, rng)])
            cmap = ConstraintMap(robot, target)
            W = residual_weights(c, cmap.k)
            N = torch.linalg.qr(cmap.jac(q0).transpose(1, 2))[0]           # (A, n, k) basis of row(Dh_c)
            u = torch.as_tensor(dir_rng.standard_normal((len(q0), cmap.k, 1)))
            w = torch.as_tensor(dir_rng.standard_normal((len(q0), n)))
            V = torch.cat([(N @ u).squeeze(-1), w])
            V = V / V.norm(dim=1, keepdim=True)                            # rows ordered as KINDS
            # one bisection over all (delta, kind, anchor) rows
            Q0, VV = q0.repeat(2 * len(deltas), 1), V.repeat(len(deltas), 1)
            dd = deltas.repeat_interleave(len(V))
            s = torch.zeros_like(dd)
            s[dd > 0] = step_to_violation(cmap, W, Q0[dd > 0], VV[dd > 0], dd[dd > 0])
            Qp = (Q0 + s[:, None] * VV).reshape(len(deltas), 2, len(q0), n)
            for j, kind in enumerate(KINDS):
                q_pert[kind].append(Qp[:, j])
            targets += [target] * len(q0)
            q0_all.append(q0)

        q0_all = torch.cat(q0_all)
        cond = torch.stack([ik.cond(tg) for tg in targets])
        noise = torch.randn(len(q0_all), args.K, D, device=ik.device, generator=gen)
        blk = {"id": d_true, "k_star": k_star, "n_anchors": len(q0_all)}
        for kind in KINDS:
            Q = torch.cat(q_pert[kind], dim=1)           # (n_delta, A, n)
            out = {k: [] for k in ("frac_dagger", "frac_svd", "gap_ratio_median", "gap_ratio_q25",
                                   "gap_ratio_q75", "pos_mm", "orn_deg", "dq_norm", "dx_norm")}
            x_ref = ik.encode(q0_all)
            for Qd in Q:
                x0 = ik.encode(Qd)
                sv = spectra(ik, x0, cond, noise, args.t).astype(np.float64)
                d_hat = np.array([ide.id_from_spectrum(s, D) for s in sv])
                out["frac_svd"].append(float(np.mean(d_hat[:, 0] == d_true)))
                out["frac_dagger"].append(float(np.mean(d_hat[:, 1] == d_true)))
                ratio = sv[:, k_star - 1] / sv[:, k_star] if d_true > 0 else None
                for key, qq in zip(("median", "q25", "q75"), (0.5, 0.25, 0.75)):
                    out[f"gap_ratio_{key}"].append(None if ratio is None else float(np.quantile(ratio, qq)))
                err = [errors(robot, q.numpy(), tg) for q, tg in zip(Qd, targets)]
                pos, orn = np.array([e[0] for e in err]), np.array([e[1] for e in err])
                out["pos_mm"].append(None if np.isnan(pos).all() else float(np.median(pos)))
                out["orn_deg"].append(None if np.isnan(orn).all() else float(np.median(orn)))
                out["dq_norm"].append(float((Qd - q0_all).norm(dim=1).median()))
                out["dx_norm"].append(float((x0 - x_ref).norm(dim=1).median()))
            blk[kind] = out
        res["by_constraint"][c] = blk
        f = blk["normal"]["frac_dagger"], blk["isotropic"]["frac_dagger"]
        print(f"  {c:<13} ID={d_true}  frac d_hat=ID (dagger) normal {np.round(f[0], 2).tolist()} "
              f"isotropic {np.round(f[1], 2).tolist()}", flush=True)

    res["runtime_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {args.out} ({res['runtime_s']} s)")


if __name__ == "__main__":
    main()

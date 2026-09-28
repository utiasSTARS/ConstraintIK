"""Training-set size ablation (paper App. C.4, Fig. size_sweep, Table size-sweep-compare).

    python scripts/size_sweep.py build                     # data/size_sweep/{robot}/{c}/nNNNNNN.pt
    bash scripts/train_size_sweep.sh                        # checkpoints/size_sweep/{robot}/{c}/nNNNNNN.pt
    python scripts/size_sweep.py eval --model per_family    # results/size_sweep/per_family.json
    python scripts/size_sweep.py eval --model joint         # results/size_sweep/joint.json (paper models)

Each per-family model trains on a nested, target-stratified subset of one family's shard: UR5 from the
"2x" shards, Franka from the "baseline" shards and, for Orientation and Line+Orient., the "grid" shards
(scripts/generate_data.py --recipe). Evaluation of a family: its first n GT targets, m DDIM samples per
target (seed + target index), LOOSE validity, and the mean constraint error over the valid samples.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from constraint_ik.checkpoint import load_checkpoint
from constraint_ik.constraints import CONSTRAINTS, LOOSE, Target, errors, intrinsic_dim, is_valid
from constraint_ik.data import build_bundle
from constraint_ik.id_estimation import CKPT_PATH, GT_PATH, load_gt
from constraint_ik.robots import get_robot
from constraint_ik.targets import wrap_dist

SIZES = (500, 1000, 2000, 5000, 10000, 20000, 40000)
SOURCE = {"ur5": {c: "2x" for c in CONSTRAINTS},
          "franka": {c: "grid" if c in ("orient", "line_orient") else "baseline" for c in CONSTRAINTS}}
BRANCH_TOL = 0.5   # rad; a valid SE(3) sample covers the nearest GT branch within this (wrapped) distance


def budget(robot: str, c: str) -> tuple[int, int]:
    """(GT targets, samples per target) of the curve the family belongs to."""
    if robot == "ur5":
        return 50, 512
    return (20, 256) if SOURCE[robot][c] == "grid" else (10, 32)


def nested_order(target_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Round robin over per-target permutations: every prefix is nested in the longer ones and
    spreads evenly over the targets."""
    perms = [rng.permutation(np.flatnonzero(target_ids == a)) for a in np.unique(target_ids)]
    return np.array([p[r] for r in range(max(map(len, perms))) for p in perms if r < len(p)],
                    dtype=np.int64)


def build(args: argparse.Namespace) -> None:
    for robot in args.robots:
        for c in CONSTRAINTS:
            src = torch.load(f"data/{robot}_{SOURCE[robot][c]}/demos_{c}.pt", weights_only=False)
            order = nested_order(src["cluster_ids"].numpy(), np.random.default_rng(args.seed))
            out_dir = os.path.join(args.data_root, robot, c)
            os.makedirs(out_dir, exist_ok=True)
            for n in sorted({min(N, len(order)) for N in args.sizes}):
                idx = torch.from_numpy(order[:n])
                sub = {k: src[k][idx] for k in ("q_ends", "cond_pos", "cond_R6d", "cond_mask")}
                torch.save(build_bundle([{**sub, "constraint_type": c}], robot),
                           os.path.join(out_dir, f"n{n:06d}.pt"))
            print(f"{robot} {c}: {len(order)} configurations, {len(np.unique(src['cluster_ids']))} targets")


def evaluate(ik, robot, entries: list[dict], n_samples: int, seed: int) -> dict:
    """Validity and mean constraint error of model samples for GT entries of one family."""
    c = entries[0]["constraint_type"]
    n_valid, frac, cov, pos_err, orn_err = 0, [], [], [], []
    for i, e in enumerate(entries):
        t = Target.from_entry(e)
        q = ik.sample(t, n_samples, seed=seed + i)
        v = is_valid(robot, q, t, LOOSE)
        n_valid += int(v.sum())
        frac.append(float(v.mean()))
        pos, orn = errors(robot, q[v], t)
        pos_err += pos[np.isfinite(pos)].tolist()
        orn_err += orn[np.isfinite(orn)].tolist()
        if c == "se3":
            d = wrap_dist(q[v][:, None], e["sols_dedup"][None]) if v.any() else np.zeros((0, 1))
            hit = d.argmin(1)[d.min(1) <= BRANCH_TOL] if len(d) else []
            cov.append(len(np.unique(hit)) / len(e["sols_dedup"]))
    rec = {"intrinsic_dim": intrinsic_dim(c, robot.n_joints), "n_instances": len(entries),
           "n_samples_per_instance": n_samples,
           "valid_pct": 100.0 * n_valid / (len(entries) * n_samples),
           "per_instance_valid_mean": float(np.mean(frac)), "per_instance_valid_std": float(np.std(frac)),
           "mode_coverage_mean": float(np.mean(cov)) if cov else None}
    for name, vals in (("pos_err_mm", pos_err), ("orn_err_deg", orn_err)):
        if vals:
            rec[f"{name}_mean"], rec[f"{name}_std"] = float(np.mean(vals)), float(np.std(vals))
    return rec


def run_eval(args: argparse.Namespace) -> None:
    cells = []
    for robot in args.robots:
        for c in CONSTRAINTS:
            if args.model == "joint":
                cells.append((robot, c, None, CKPT_PATH.format(robot=robot)))
            else:
                cells += [(robot, c, N, os.path.join(args.ckpt_root, robot, c, f"n{N:06d}.pt"))
                          for N in args.sizes]
    records, t0, ik = [], time.time(), {}
    gt = {r: load_gt(GT_PATH[r]) for r in args.robots}
    for robot, c, N, path in cells:
        if not os.path.exists(path):
            print(f"missing {path}")
            continue
        if path not in ik:
            ik = {path: load_checkpoint(path, args.device)}
        n_targets, n_samples = budget(robot, c)
        rec = evaluate(ik[path], get_robot(robot), gt[robot][c][:n_targets], n_samples, args.seed)
        records.append({"robot": robot, "constraint": c, "N": N, **rec})
        print(f"{robot} {c} N={N}: valid {rec['valid_pct']:.1f}%  pos {rec.get('pos_err_mm_mean', float('nan')):.2f} mm"
              f"  orn {rec.get('orn_err_deg_mean', float('nan')):.2f} deg", flush=True)
    if not records:
        raise SystemExit(f"no checkpoints found under {args.ckpt_root}; see scripts/download.py --size_sweep")
    out = args.out or f"results/size_sweep/{args.model}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"args": vars(args), "budget": {r: {c: budget(r, c) for c in CONSTRAINTS} for r in args.robots},
                   "validity": LOOSE.name, "seconds": round(time.time() - t0, 1), "records": records}, f, indent=2)
    print("wrote", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["build", "eval"])
    p.add_argument("--robots", nargs="+", default=["ur5", "franka"], choices=["ur5", "franka"])
    p.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_root", default="data/size_sweep")
    p.add_argument("--model", default="per_family", choices=["per_family", "joint"])
    p.add_argument("--ckpt_root", default="checkpoints/size_sweep")
    p.add_argument("--out", default=None, help="default: results/size_sweep/{model}.json")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.stage == "build":
        build(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()

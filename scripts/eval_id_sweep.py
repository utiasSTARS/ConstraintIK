"""Score-SVD(+dagger) estimate vs noise level t or perturbation count K.

  --sweep t                 App. Fig. id_noise_sensitivity   -> results/id/{robot}_vs_t.json
  --sweep k                 App. Fig. id_k_sensitivity       -> results/id/{robot}_vs_k.json
  --sweep t_model_anchored  main Fig. 3 top row              -> results/id/{robot}_vs_t_model_anchored.json

Each (target, anchor, x) gets n_repeats independent K-noise batches; both gap rules are read off the
same spectra. t / k: uniform-random GT anchors, summary = median and IQR over targets of the per-target
median (over anchors) of the per-anchor median (over repeats). t_model_anchored: both anchor rules
("uniform" gives the paper's dotted Score-SVD curve, "model" the solid Score-SVD-dagger curve),
summary = median over anchors of the per-anchor mean (over repeats) with a 95% bootstrap CI.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from constraint_ik import Target, get_robot, load_checkpoint
from constraint_ik import id_estimation as ide
from constraint_ik.constraints import intrinsic_dim
from constraint_ik.stats import median_ci

PRESETS = {
    "t": dict(values=list(range(0, 100, 5)), n_targets=10, n_anchors=5, n_repeats=3, K=50, t=None,
              anchors=["uniform"]),
    "k": dict(values=[5, 10, 20, 30, 50, 75, 100, 150, 200], n_targets=10, n_anchors=5, n_repeats=3,
              K=None, t=15, anchors=["uniform"]),
    "t_model_anchored": dict(values=[1, *range(5, 100, 5), 99], n_targets=5, n_anchors=3, n_repeats=5,
                             K=50, t=None, anchors=["uniform", "model"]),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", required=True, choices=["ur5", "franka"])
    p.add_argument("--sweep", required=True, choices=list(PRESETS))
    p.add_argument("--ckpt", default=None, help="default: checkpoints/{robot}.pt")
    p.add_argument("--gt", default=None, help="default: data/ground_truth_{robot}.pt")
    p.add_argument("--values", type=int, nargs="+", default=None, help="t or K grid (default: preset)")
    p.add_argument("--n_targets", type=int, default=None)
    p.add_argument("--n_anchors", type=int, default=None, help="anchors per target")
    p.add_argument("--n_repeats", type=int, default=None, help="noise batches per (anchor, value)")
    p.add_argument("--K", type=int, default=None, help="perturbations for a t sweep")
    p.add_argument("--t", type=int, default=None, help="noise level for a K sweep")
    p.add_argument("--n_samples", type=int, default=256, help="model samples per target for 'model' anchors")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--boot_seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="default: results/id/{robot}_vs_{sweep}.json")
    a = p.parse_args()
    for k, v in PRESETS[a.sweep].items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    a.ckpt = a.ckpt or ide.CKPT_PATH.format(robot=a.robot)
    a.gt = a.gt or ide.GT_PATH[a.robot]
    a.out = a.out or f"results/id/{a.robot}_vs_{a.sweep}.json"
    return a


def run(ik, robot, gt, args, rule: str) -> dict:
    """{constraint: (anchor indices, raw)} with raw[value][target][anchor][repeat] = (d_svd, d_dagger)."""
    torch.manual_seed(args.seed)
    unif = np.random.RandomState(args.seed)     # uniform anchors
    rng = np.random.default_rng(args.seed)      # fallback for 'model' anchors
    sweep_t = args.sweep != "k"
    out = {}
    for c, entries in gt.items():
        raw = [[[] for _ in entries] for _ in args.values]
        anchors = []
        for ii, e in enumerate(entries):
            target = Target.from_entry(e)
            sols = np.asarray(e["sols_dedup"], dtype=np.float64)
            if rule == "model":
                q_valid = ide.valid_samples(ik, robot, target, args.n_samples, args.seed + ii)
                idx = ide.model_anchors(sols, q_valid, args.n_anchors, rng)
            else:
                idx = ide.uniform_anchors(len(sols), args.n_anchors, unif)
            anchors.append([int(i) for i in idx])
            cond = ik.cond(target)
            for x0 in ik.encode(sols[idx]):
                for vi, v in enumerate(args.values):
                    t, K = (v, args.K) if sweep_t else (args.t, v)
                    raw[vi][ii].append([ide.id_from_spectrum(
                        ide.singular_values(ide.score_matrix(ik, x0, cond, t, K)), ik.dim)
                        for _ in range(args.n_repeats)])
        out[c] = (anchors, raw)
        print(f"  {rule:<7} {c:<13} done", flush=True)
    return out


def summarize(raw_one_rule: list, sweep: str, args) -> dict:
    """raw_one_rule[value][target][anchor][repeat] -> per-value summary for the paper figure."""
    s = {"median": [], "lo": [], "hi": []}
    for per_target in raw_one_rule:
        if sweep == "t_model_anchored":
            units = [float(np.mean(reps)) for anchors in per_target for reps in anchors]
            med, lo, hi = median_ci(units, n_boot=args.n_boot, seed=args.boot_seed)
        else:
            units = np.array([np.median([np.median(reps) for reps in anchors]) for anchors in per_target])
            med, lo, hi = np.median(units), np.quantile(units, 0.25), np.quantile(units, 0.75)
        for k, v in zip(s, (med, lo, hi)):
            s[k].append(float(v))
    return s


def main():
    args = parse_args()
    t0 = time.time()
    ik = load_checkpoint(args.ckpt, args.device)
    robot = get_robot(ik.robot)
    gt = ide.load_gt(args.gt, args.n_targets)
    band = "95% bootstrap CI of the median over anchors" if args.sweep == "t_model_anchored" \
        else "IQR over targets"
    res = {"args": vars(args), "robot": ik.robot, "in_dim": ik.dim, "sweep": args.sweep,
           "x": "K" if args.sweep == "k" else "t", "values": args.values, "band": band,
           "id": {c: intrinsic_dim(c, ik.n_joints) for c in gt}, "by_constraint": {c: {} for c in gt}}
    for rule in args.anchors:
        for c, (anchors, raw) in run(ik, robot, gt, args, rule).items():
            blk = {"anchors": anchors}
            for j, est in enumerate(("svd", "dagger")):
                r = [[[[int(d[j]) for d in reps] for reps in per_t] for per_t in per_v] for per_v in raw]
                blk[est] = {**summarize(r, args.sweep, args), "raw": r}
            res["by_constraint"][c][rule] = blk
    res["runtime_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f)
    print(f"wrote {args.out} ({res['runtime_s']} s)")


if __name__ == "__main__":
    main()

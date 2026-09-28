"""Intrinsic dimension of every constraint manifold (Table tab:id-full, Fig. 2 bars, Fig. id_score_spectra).

Per GT target: local MLE / local PCA / TwoNN on the valid model samples, and Score-SVD(+dagger) at the
GT solutions nearest to those samples.

    python scripts/eval_id.py --robot ur5            # -> results/id/ur5.json
    python scripts/eval_id.py --worked_example       # UR5 SE(3) and Line spectra of App. D.1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from constraint_ik import PAPER_ORDER, Target, get_robot, load_checkpoint
from constraint_ik import id_estimation as ide
from constraint_ik.constraints import intrinsic_dim
from constraint_ik.stats import median_ci

# Fig. 2 bars: estimator key -> legend label.
FIG_METHODS = {"mle_k10": "Local MLE", "pca90": "PCA-90", "score_svd": "Score-SVD",
               "score_svd_dagger": "Score-SVD-dagger"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="ur5", choices=["ur5", "franka"])
    p.add_argument("--ckpt", default=None, help="default: checkpoints/{robot}.pt")
    p.add_argument("--gt", default=None, help="default: data/ground_truth_{robot}.pt")
    p.add_argument("--n_targets", type=int, default=15, help="GT targets per constraint")
    p.add_argument("--n_samples", type=int, default=512, help="model samples per target")
    p.add_argument("--n_anchors", type=int, default=5, help="Score-SVD anchors per target")
    p.add_argument("--t", type=int, default=15)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--mle_ks", type=int, nargs="+", default=[2, 10])
    p.add_argument("--pca_thresholds", type=float, nargs="+", default=[0.90, 0.99])
    p.add_argument("--nn_k", type=int, default=10, help="neighbourhood size for PCA and TwoNN")
    p.add_argument("--iso", type=float, default=ide.ISO_THRESHOLD)
    p.add_argument("--n_boot", type=int, default=2000, help="Fig. 2 bootstrap resamples")
    p.add_argument("--boot_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="default: results/id/{robot}.json")
    p.add_argument("--worked_example", action="store_true",
                   help="print the SE(3) and Line spectra nearest to their mean spectrum and exit")
    a = p.parse_args()
    a.ckpt = a.ckpt or ide.CKPT_PATH.format(robot=a.robot)
    a.gt = a.gt or ide.GT_PATH[a.robot]
    a.out = a.out or f"results/id/{a.robot}.json"
    return a


def eval_constraint(ik, robot, entries, args, rng) -> tuple[dict, list, list]:
    """Per-target estimates, anchor indices and all anchor spectra for one constraint."""
    per = defaultdict(list)
    anchors, spectra = [], []
    for ii, e in enumerate(entries):
        target = Target.from_entry(e)
        q_valid = ide.valid_samples(ik, robot, target, args.n_samples, args.seed + ii)
        if len(q_valid) < 4:
            continue
        per["valid_frac"].append(len(q_valid) / args.n_samples)
        for k in args.mle_ks:
            per[f"mle_k{k}"].append(ide.local_mle(q_valid, k, args.iso))
        for v in args.pca_thresholds:
            per[f"pca{round(v * 100)}"].append(ide.local_pca(q_valid, args.nn_k, v, args.iso))
        per["twonn"].append(ide.twonn(q_valid, args.nn_k, args.iso))

        sols = np.asarray(e["sols_dedup"], dtype=np.float64)
        idx = ide.model_anchors(sols, q_valid, args.n_anchors, rng)
        cond = ik.cond(target)
        d_svd, d_dag = [], []
        for x0 in ik.encode(sols[idx]):
            sv, a, b = ide.score_svd(ik, x0, cond, args.t, args.K)
            spectra.append(sv)
            d_svd.append(a)
            d_dag.append(b)
        per["score_svd"].append(float(np.median(d_svd)))
        per["score_svd_dagger"].append(float(np.median(d_dag)))
        anchors.append([int(i) for i in idx])
    return dict(per), anchors, spectra


def summarize(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(a)), "q25": float(np.quantile(a, 0.25)),
            "q75": float(np.quantile(a, 0.75)), "n": int(a.size)}


def worked_example(ik, robot, gt, args):
    """Spectrum nearest (L2) to the pooled mean spectrum, for SE(3) and Line, with both gap positions."""
    D = ik.dim
    for c in ("se3", "line"):
        _, _, spectra = eval_constraint(ik, robot, gt[c], args, np.random.default_rng(args.seed))
        S = np.asarray(spectra, dtype=np.float64)
        i = int(np.argmin(np.linalg.norm(S - S.mean(axis=0), axis=1)))
        sv = S[i]
        k_svd, k_dag = ide.gap_rank(sv), ide.gap_rank(sv, dagger=True)
        drops = np.append(sv, 0.0)
        drops = drops[:-1] - drops[1:]
        print(f"{args.robot} {c}: anchor {i} of {len(S)} (target {i // args.n_anchors}), "
              f"ID {intrinsic_dim(c, ik.n_joints)}, t={args.t}, K={args.K}, seed={args.seed}")
        print("  sigma:", ", ".join(f"{s:.1f}" for s in sv))
        print("  drops sigma_i - sigma_{i+1} (last vs appended 0):", ", ".join(f"{g:.1f}" for g in drops))
        print(f"  Score-SVD:        k_hat={k_svd:2d}  drop={drops[k_svd - 1]:.1f}  d_hat={D - k_svd}")
        print(f"  Score-SVD-dagger: k_hat={k_dag:2d}  drop={drops[k_dag - 1]:.1f}  d_hat={D - k_dag}")


def main():
    args = parse_args()
    t0 = time.time()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ik = load_checkpoint(args.ckpt, args.device)
    robot = get_robot(ik.robot)
    gt = ide.load_gt(args.gt, args.n_targets)
    if args.worked_example:
        worked_example(ik, robot, gt, args)
        return

    D = ik.dim
    out = {"args": vars(args), "robot": ik.robot, "in_dim": D, "n_joints": ik.n_joints, "by_constraint": {}}
    for c, entries in gt.items():
        per, anchors, spectra = eval_constraint(ik, robot, entries, args, rng)
        S = np.vstack([np.asarray(s, dtype=np.float64)[:D] for s in spectra])
        d = intrinsic_dim(c, ik.n_joints)
        summary = {m: summarize(v) for m, v in per.items() if m != "valid_frac"}
        out["by_constraint"][c] = {
            "id": d,
            "summary": summary,
            "per_target": per,
            "anchors": anchors,
            "spectrum": {"mean": S.mean(axis=0).tolist(), "std": S.std(axis=0).tolist(),
                         "n_spectra": len(S)},
            # gap index n - d_hat in sincos space, from the analytical ID and the per-target medians
            "gap": {"true": D - d, "svd": D - summary["score_svd"]["median"],
                    "dagger": D - summary["score_svd_dagger"]["median"]},
        }
        print(f"  {c:<13} ID={d}  " + "  ".join(f"{m}={s['median']:.2f}" for m, s in summary.items()),
              flush=True)

    # Fig. 2: median over targets with a percentile-bootstrap CI; one generator shared across
    # (constraint, estimator) cells in paper order, as in the published figure.
    boot = np.random.default_rng(args.boot_seed)
    bars = {m: {"median": [], "lo": [], "hi": []} for m in FIG_METHODS}
    for c in PAPER_ORDER:
        for m in FIG_METHODS:
            med, lo, hi = median_ci(out["by_constraint"][c]["per_target"][m], n_boot=args.n_boot, seed=boot)
            for key, v in zip(("median", "lo", "hi"), (med, lo, hi)):
                bars[m][key].append(v)
    out["fig_bars"] = {"order": list(PAPER_ORDER), "labels": FIG_METHODS, **bars}
    out["runtime_s"] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out} ({out['runtime_s']} s)")


if __name__ == "__main__":
    main()

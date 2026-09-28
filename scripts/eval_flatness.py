"""Directional sensitivity rho, joint-space speed nu and tangent overlap tau^q along the straight latent
paths of results/interpolation/targets at the linear t_bar of configs/tbar.json.

Writes results/flatness/{robot}_{c}.csv (one row per pair and alpha) and results/flatness/summary.json.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import time

import numpy as np
import torch

from constraint_ik import ConstraintMap, Target, load_checkpoint
from constraint_ik.constraints import PAPER_ORDER
from constraint_ik.flatness import N_ALPHA, pair_diagnostics, summarize

COLUMNS = ["pair", "target", "pair_width", "alpha", "rho", "nu", "tau"]


def run_cell(ik, robot: str, c: str, args) -> tuple[list[dict], dict]:
    data = dict(np.load(os.path.join(args.targets_dir, f"{robot}_{c}.npz")))
    t_bar = json.load(open(args.tbar_config))[robot][c]["linear"]
    targets = [Target.from_pose(c, p, R) for p, R in zip(data["pos"], data["R"])]
    rows = []
    for k in range(min(len(data["q_a"]), args.max_pairs or len(data["q_a"]))):
        target = targets[data["pair_target"][k]]
        cmap = ConstraintMap(robot, target, device=args.device)
        d = pair_diagnostics(ik, cmap, data["q_a"][k], data["q_b"][k], target, t_bar, args.n_alpha,
                             args.n_steps)
        for i in range(args.n_alpha):
            rows.append({"pair": k, "target": int(data["pair_target"][k]),
                         "pair_width": int(data["pair_width"][k]), "alpha": float(d["alpha"][i]),
                         "rho": float(d["rho"][i]), "nu": float(d["nu"][i]), "tau": float(d["tau"][i])})
    return rows, {"t_bar": t_bar, **summarize(rows, ik.n_joints, c)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", required=True, choices=["ur5", "franka"])
    p.add_argument("--constraint", default="all", choices=["all", *PAPER_ORDER])
    p.add_argument("--ckpt", default=None, help="default checkpoints/{robot}.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_alpha", type=int, default=N_ALPHA)
    p.add_argument("--n_steps", type=int, default=100, help="DDIM steps of g_c (inversion, decoding, Dg_c)")
    p.add_argument("--targets_dir", default="results/interpolation/targets")
    p.add_argument("--tbar_config", default="configs/tbar.json")
    p.add_argument("--out_dir", default="results/flatness")
    p.add_argument("--max_pairs", type=int, default=None, help="evaluate only the first pairs (quick checks)")
    args = p.parse_args()
    args.ckpt = args.ckpt or f"checkpoints/{args.robot}.pt"

    ik = load_checkpoint(args.ckpt, args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "summary.json")
    for c in PAPER_ORDER if args.constraint == "all" else (args.constraint,):
        t0 = time.time()
        rows, summ = run_cell(ik, args.robot, c, args)
        with open(os.path.join(args.out_dir, f"{args.robot}_{c}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        with open(summary_path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            text = f.read()
            summary = json.loads(text) if text.strip() else {}
            summary.setdefault(args.robot, {})[c] = {**summ, "seconds": round(time.time() - t0, 1),
                                                     "args": vars(args)}
            f.seek(0)
            f.truncate()
            json.dump(summary, f, indent=1, sort_keys=True)
        print(f"{args.robot}_{c:13s} t_bar {summ['t_bar']:3d}  rho {summ['rho_med']:.3f}  nu {summ['nu_med']:.2f}"
              f"  tau {summ['tau_med']:.3f}/{summ['tau_p5']:.3f}  base {summ['tau_rand_med']:.3f}/"
              f"{summ['tau_rand_p5']:.3f}  {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

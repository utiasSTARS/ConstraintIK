"""Latent interpolation (lin and slerp) over the t_bar grid, selection of t_bar, and the per-path results.

For every endpoint pair of results/interpolation/targets/{robot}_{c}.npz both endpoints are inverted to
each t_bar, interpolated with n_frames frames and decoded (100-step DDIM). Per cell it writes
  results/interpolation/sweep/{robot}_{c}.csv       one row per (method, t_bar, pair)
  results/interpolation/canonical/{robot}_{c}.csv   the rows at the selected t_bar of each method
  results/interpolation/runs/{robot}_{c}.json       args, checkpoint, t_bar scores, runtime
  outputs/interpolation/{robot}_{c}.npz             decoded paths and per-frame errors
and records the selected t_bar in configs/tbar.json.
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

from constraint_ik import Target, get_robot, load_checkpoint
from constraint_ik.constraints import PAPER_ORDER
from constraint_ik.interpolation import INTERPOLANTS, invert_levels, path_metrics, pick_t_bar, t_bar_scores

COLUMNS = ["robot", "constraint", "method", "t_bar", "target", "pair_width", "valid_pct_loose",
           "valid_pct_strict", "mean_pos_err_mm", "mean_orn_err_deg", "max_ee_step_mm", "p95_ee_step_mm"]


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def update_tbar(path: str, robot: str, constraint: str, t_bar: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        text = f.read()
        cfg = json.loads(text) if text.strip() else {}
        cfg.setdefault(robot, {})[constraint] = t_bar
        f.seek(0)
        f.truncate()
        json.dump(cfg, f, indent=1, sort_keys=True)
        f.write("\n")


def sweep(ik, robot, constraint: str, data: dict, t_bars, methods, n_frames: int, n_steps: int):
    targets = [Target.from_pose(constraint, p, R) for p, R in zip(data["pos"], data["R"])]
    P, n = data["q_a"].shape
    q_out = np.zeros((len(methods), len(t_bars), P, n_frames, n), dtype=np.float32)
    pos_err = np.full((len(methods), len(t_bars), P, n_frames), np.nan)
    orn_err = np.full_like(pos_err, np.nan)
    rows = []
    for k in range(P):
        target = targets[data["pair_target"][k]]
        cond = ik.cond(target)
        z_a = invert_levels(ik, data["q_a"][k], t_bars, cond, n_steps)
        z_b = invert_levels(ik, data["q_b"][k], t_bars, cond, n_steps)
        for i, m in enumerate(methods):
            for j, t in enumerate(t_bars):
                z = INTERPOLANTS[m](z_a[t], z_b[t], n_frames)
                q = ik.decode_from(z, t, cond, n_steps).cpu().numpy()
                met = path_metrics(robot, q, target)
                q_out[i, j, k], pos_err[i, j, k], orn_err[i, j, k] = q, met["pos_err_mm"], met["orn_err_deg"]
                rows.append({"robot": robot.name, "constraint": constraint, "method": m, "t_bar": t,
                             "target": int(data["pair_target"][k]), "pair_width": int(data["pair_width"][k]),
                             **{c: met[c] for c in COLUMNS[6:]}})
    return rows, q_out, pos_err, orn_err


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", required=True, choices=["ur5", "franka"])
    p.add_argument("--constraint", default="all", choices=["all", *PAPER_ORDER])
    p.add_argument("--ckpt", default=None, help="default checkpoints/{robot}.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--t_bars", default="0,10,20,30,40,50,60,70,80,85,90,95")
    p.add_argument("--methods", default="linear,slerp")
    p.add_argument("--n_frames", type=int, default=24)
    p.add_argument("--n_steps", type=int, default=100, help="DDIM steps over the full schedule")
    p.add_argument("--seed", type=int, default=0, help="unused by the deterministic DDIM paths; recorded")
    p.add_argument("--targets_dir", default="results/interpolation/targets")
    p.add_argument("--results_dir", default="results/interpolation")
    p.add_argument("--outputs_dir", default="outputs/interpolation")
    p.add_argument("--tbar_config", default="configs/tbar.json")
    args = p.parse_args()
    args.ckpt = args.ckpt or f"checkpoints/{args.robot}.pt"

    torch.manual_seed(args.seed)
    t_bars = tuple(int(t) for t in args.t_bars.split(","))
    methods = tuple(args.methods.split(","))
    ik = load_checkpoint(args.ckpt, args.device)
    robot = get_robot(ik.robot)
    constraints = PAPER_ORDER if args.constraint == "all" else (args.constraint,)
    for c in constraints:
        t0 = time.time()
        cell = f"{args.robot}_{c}"
        data = dict(np.load(os.path.join(args.targets_dir, f"{cell}.npz")))
        rows, q, pos_err, orn_err = sweep(ik, robot, c, data, t_bars, methods, args.n_frames, args.n_steps)
        write_csv(os.path.join(args.results_dir, "sweep", f"{cell}.csv"), rows)

        by_method = {m: [r for r in rows if r["method"] == m] for m in methods}
        scores = {m: t_bar_scores(by_method[m], c) for m in methods}
        t_bar = {m: pick_t_bar(by_method[m], c) for m in methods}
        write_csv(os.path.join(args.results_dir, "canonical", f"{cell}.csv"),
                  [r for m in methods for r in by_method[m] if r["t_bar"] == t_bar[m]])
        update_tbar(args.tbar_config, args.robot, c, t_bar)

        seconds = round(time.time() - t0, 1)
        run = {"robot": args.robot, "constraint": c, "args": vars(args), "t_bar": t_bar,
               "scores": {m: {str(t): s for t, s in scores[m].items()} for m in methods},
               "n_pairs": len(data["q_a"]), "seconds": seconds}
        os.makedirs(os.path.join(args.results_dir, "runs"), exist_ok=True)
        with open(os.path.join(args.results_dir, "runs", f"{cell}.json"), "w") as f:
            json.dump(run, f, indent=1)
        os.makedirs(args.outputs_dir, exist_ok=True)
        np.savez(os.path.join(args.outputs_dir, f"{cell}.npz"), q=q, pos_err_mm=pos_err, orn_err_deg=orn_err,
                 methods=np.array(methods), t_bars=np.array(t_bars), pair_target=data["pair_target"],
                 pair_width=data["pair_width"], meta=np.array(json.dumps(run)))
        print(f"{cell:20s} t_bar {t_bar}  {seconds:.0f} s", flush=True)


if __name__ == "__main__":
    main()

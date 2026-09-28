"""Synthesize the latent-interpolation targets and endpoint pairs (30 targets x 5 pair widths per cell).

Writes results/interpolation/targets/{robot}_{constraint}.npz with the target seed poses (pos, R; the
constrained target is Target.from_pose(constraint, pos, R)), the pairs (pair_target, pair_width, q_a,
q_b), a third solution q_c per pair, q_dist = |q_b - q_a| (wrapped), and meta (args, recipe, runtime).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import numpy as np

from constraint_ik.constraints import PAPER_ORDER
from constraint_ik.targets import build_targets, recipe


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="all", choices=["all", "ur5", "franka"])
    p.add_argument("--constraint", default="all", choices=["all", *PAPER_ORDER])
    p.add_argument("--n_targets", type=int, default=30)
    p.add_argument("--pair_widths", default="10,25,50,75,100")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4, help="processes for target synthesis (results do not depend on it)")
    p.add_argument("--out_dir", default="results/interpolation/targets")
    args = p.parse_args()

    widths = tuple(int(w) for w in args.pair_widths.split(","))
    robots = ("franka", "ur5") if args.robot == "all" else (args.robot,)
    constraints = PAPER_ORDER if args.constraint == "all" else (args.constraint,)
    os.makedirs(args.out_dir, exist_ok=True)
    for robot in robots:
        for c in constraints:
            t0 = time.time()
            data, _ = build_targets(robot, c, args.n_targets, args.seed, widths, args.workers)
            meta = {"robot": robot, "constraint": c, "args": vars(args), "recipe": asdict(recipe(robot, c)),
                    "seconds": round(time.time() - t0, 1)}
            path = os.path.join(args.out_dir, f"{robot}_{c}.npz")
            np.savez(path, **data, meta=np.array(json.dumps(meta)))
            print(f"{robot:6s} {c:12s} {len(data['q_a'])} pairs  {meta['seconds']:7.1f} s  -> {path}", flush=True)


if __name__ == "__main__":
    main()

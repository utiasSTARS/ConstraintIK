"""Generate the training data (CPU).

    python scripts/generate_data.py --robot ur5                     # data/ur5_2x/demos_*.pt + data/train_ur5.pt
    python scripts/generate_data.py --robot franka --constraint posonly
    python scripts/generate_data.py --robot franka --recipe baseline  # size-sweep sources: data/franka_baseline/
    python scripts/generate_data.py --robot franka --recipe grid      # size-sweep sources: data/franka_grid/
    python scripts/generate_data.py --robot franka --gt               # data/ground_truth_franka.pt

Families run in parallel processes; each has its own RNG (seed), so results do not depend on --workers.
The unified bundle is (re)built whenever all seven "2x" shards exist in --out.
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

import torch

from constraint_ik.constraints import CONSTRAINTS
from constraint_ik.data import RECIPES, build_bundle, generate_gt, generate_shard
from constraint_ik.robots import get_robot


def _shard_job(job: tuple) -> tuple[str, int, float]:
    robot, c, recipe, seed, path = job
    t0 = time.time()
    shard = generate_shard(get_robot(robot), c, recipe, seed)
    torch.save(shard, path)
    return c, len(shard["q_ends"]), time.time() - t0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", required=True, choices=["ur5", "franka"])
    p.add_argument("--recipe", default="2x", choices=list(RECIPES))
    p.add_argument("--constraint", nargs="+", default=["all"], help="families of the recipe (default: all)")
    p.add_argument("--out", default=None, help="shard directory (default: data/{robot}_{recipe})")
    p.add_argument("--bundle", default=None, help="default: data/train_{robot}.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--gt", action="store_true", help="write the Franka GT targets instead (50 per family)")
    args = p.parse_args()

    if args.gt:
        if args.robot != "franka":
            p.error("--gt: the UR5 GT targets are distributed as data/ground_truth_ur5.pt")
        out = args.out or "data/ground_truth_franka.pt"
        t0 = time.time()
        gt = generate_gt(get_robot("franka"), 50, args.seed)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        torch.save(gt, out)
        print(f"wrote {out}: {len(gt)} targets ({time.time() - t0:.0f} s)")
        return

    families = list(RECIPES[args.recipe]) if args.constraint == ["all"] else args.constraint
    unknown = set(families) - set(RECIPES[args.recipe])
    if unknown:
        p.error(f"recipe {args.recipe} has no {sorted(unknown)}")
    out_dir = args.out or f"data/{args.robot}_{args.recipe}"
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(args.robot, c, args.recipe, args.seed, os.path.join(out_dir, f"demos_{c}.pt")) for c in families]
    with Pool(min(args.workers, len(jobs))) as pool:
        for c, n, dt in pool.imap_unordered(_shard_job, jobs):
            print(f"{args.robot} {args.recipe} {c}: {n} configurations ({dt:.0f} s)", flush=True)

    shard_paths = [os.path.join(out_dir, f"demos_{c}.pt") for c in CONSTRAINTS]
    if args.recipe == "2x" and all(os.path.exists(s) for s in shard_paths):
        bundle = build_bundle([torch.load(s, weights_only=False) for s in shard_paths], args.robot)
        path = args.bundle or f"data/train_{args.robot}.pt"
        torch.save(bundle, path)
        print(f"wrote {path}: {len(bundle['q_ends'])} configurations")


if __name__ == "__main__":
    main()

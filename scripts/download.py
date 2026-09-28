"""Download the paper checkpoints and data from the Hugging Face Hub.

    python scripts/download.py                 # models + evaluation targets (enough for every eval script)
    python scripts/download.py --training      # also the training sets (for scripts/train.py)
    python scripts/download.py --size_sweep    # also the 14 per-family models at N = 40k
"""
from __future__ import annotations

import argparse
import os

from huggingface_hub import hf_hub_download

REPO_ID = "mlsdpk/ConstraintIK"
CORE = ["checkpoints/ur5.pt", "checkpoints/franka.pt",
        "data/ground_truth_ur5.pt", "data/ground_truth_franka.pt"]
TRAINING = ["data/train_ur5.pt", "data/train_franka.pt"]
SIZE_SWEEP = [f"checkpoints/size_sweep/{r}/{c}/n040000.pt" for r in ("ur5", "franka")
              for c in ("plane", "line", "posonly", "orient", "plane_orient", "line_orient", "se3")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", default=REPO_ID)
    p.add_argument("--training", action="store_true")
    p.add_argument("--size_sweep", action="store_true")
    args = p.parse_args()
    files = CORE + TRAINING * args.training + SIZE_SWEEP * args.size_sweep
    for f in files:
        if os.path.exists(f):
            print("exists  ", f)
            continue
        os.makedirs(os.path.dirname(f), exist_ok=True)
        hf_hub_download(args.repo_id, f, local_dir=".")
        print("fetched ", f)


if __name__ == "__main__":
    main()

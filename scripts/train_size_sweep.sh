#!/usr/bin/env bash
# Per-family models of the training-set size ablation (App. C.4): one model per (robot, family, N),
# trained with the cosine schedule for 100k steps; all other settings as the main models.
#
#   python scripts/size_sweep.py build && bash scripts/train_size_sweep.sh
#   ROBOTS=ur5 CONSTRAINTS="plane line" SIZES=40000 DEVICE=cuda:1 bash scripts/train_size_sweep.sh
#
# A cell is skipped when its sidecar records the requested number of steps.
set -euo pipefail
cd "$(dirname "$0")/.."

ROBOTS=${ROBOTS:-"ur5 franka"}
CONSTRAINTS=${CONSTRAINTS:-"plane line posonly orient plane_orient line_orient se3"}
SIZES=${SIZES:-"500 1000 2000 5000 10000 20000 40000"}
STEPS=${STEPS:-100000}
DEVICE=${DEVICE:-cuda:0}
PYTHON=${PYTHON:-python}

for robot in $ROBOTS; do
  for c in $CONSTRAINTS; do
    for n in $SIZES; do
      cell=$robot/$c/$(printf "n%06d" "$n")
      out=checkpoints/size_sweep/$cell.pt
      if grep -qs "\"steps\": $STEPS," "checkpoints/size_sweep/$cell.config.json"; then
        echo "skip $out"
        continue
      fi
      "$PYTHON" scripts/train.py --robot "$robot" --data "data/size_sweep/$cell.pt" --out "$out" \
        --schedule cosine --steps "$STEPS" --device "$DEVICE"
    done
  done
done

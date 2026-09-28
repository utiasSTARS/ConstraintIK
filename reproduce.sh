#!/usr/bin/env bash
# Reproduce the paper. Stages run in order; pass stage names to run a subset, e.g. `bash reproduce.sh tables figures`.
#   tables figures   CPU only, from the committed results/
#   eval             GPU, regenerates results/ from the downloaded checkpoints
#   data train       regenerate the training sets and retrain both models into checkpoints/retrained/
#                    (about 8 h per model; evaluate them with --ckpt)
#   media            README media (needs the viz extra; run after figures)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
DEVICE=${DEVICE:-cuda}
STAGES=${*:-tables figures}

for stage in $STAGES; do
  case $stage in
    data)
      for r in ur5 franka; do $PY scripts/generate_data.py --robot $r; done
      ;;
    train)
      for r in ur5 franka; do $PY scripts/train.py --robot $r --device $DEVICE; done
      ;;
    eval)
      for r in ur5 franka; do
        $PY scripts/eval_id.py --robot $r --device $DEVICE
        for s in t k t_model_anchored; do $PY scripts/eval_id_sweep.py --robot $r --sweep $s --device $DEVICE; done
        $PY scripts/eval_perturbation.py --robot $r --device $DEVICE
        $PY scripts/eval_interpolation.py --robot $r --device $DEVICE
        $PY scripts/eval_flatness.py --robot $r --device $DEVICE
      done
      $PY scripts/size_sweep.py eval --model joint --device $DEVICE
      ;;
    tables)
      $PY scripts/make_tables.py
      ;;
    figures)
      $PY scripts/make_figures.py
      ;;
    media)
      $PY scripts/render_media.py
      cp figures/readme_id.png docs/media/
      ;;
    *)
      echo "unknown stage: $stage" >&2; exit 1
      ;;
  esac
done

#!/bin/bash
# E138 (26.09): adaptive schedule-copy tolerance in the regime labels (SCHED_TOL_LONG=65 where D > 3600) — nested CV vs E106.
set -euo pipefail
cd /root/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6
out=data/cv_preds_e138_tol; mkdir -p "$out"
echo "[$(date +%H:%M)] nested CV"
SCHED_TOL_LONG=65 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 CV_OUT="$out" FEATURES=data/features_e106.parquet \
  DROP_PREFIX=day_,tq_,wxr_,month,te_ .venv/bin/python src/cv_fold_preds_nested.py >> data/e138.log 2>&1
echo "[$(date +%H:%M)] gates"
PYTHONPATH=src .venv/bin/python src/post_e14.py data/cv_preds_e106_air "$out" 2>&1 | tail -12 || true
echo "[$(date +%H:%M)] done"

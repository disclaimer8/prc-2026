#!/bin/bash
# E132 (25.09): neighbours' push-back delay features -> nested 6-fold CV against the E106 control (same DROP set).
set -euo pipefail
cd /root/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6
rm -f data/e132.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee data/e132.failed' ERR
echo "[$(date +%H:%M)] build features_e132"
FEATURES=data/features_e106.parquet FEATURES_OUT=data/features_e132.parquet .venv/bin/python src/e132_record_order.py > data/e132_build.log 2>&1
tail -3 data/e132_build.log
out=data/cv_preds_e132_ord; mkdir -p "$out"
echo "[$(date +%H:%M)] nested CV"
CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 CV_OUT="$out" FEATURES=data/features_e132.parquet \
  DROP_PREFIX=day_,tq_,wxr_,month,te_ .venv/bin/python src/cv_fold_preds_nested.py >> data/e132.log 2>&1
echo "[$(date +%H:%M)] gates"
PYTHONPATH=src .venv/bin/python src/fold_metrics.py data/cv_preds_e106_air "$out" 2>&1 | tail -15 || true
PYTHONPATH=src .venv/bin/python src/post_e14.py data/cv_preds_e106_air "$out" 2>&1 | tail -12 || true
echo "[$(date +%H:%M)] done"

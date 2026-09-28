#!/bin/bash
# E151 (journal 131): own-interval flows (Yuan et al. 2024) as base features -> nested 6-fold CV against the E106 control,
# exactly as E137 (src/hz_e137_overdue_cv.sh). Features are built at once; the CV waits for the e2e rebuild (#130) to free
# cores 6-11 (artifact: v927, or the failure marker).
set -euo pipefail
cd /root/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6
L=data/e151_chain.log
rm -f data/e151.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee -a $L data/e151.failed' ERR
echo "[$(date +%H:%M)] build features_e151" >> $L
POLARS_MAX_THREADS=2 FEATURES=data/features_e106.parquet FEATURES_OUT=data/features_e151.parquet .venv/bin/python src/e151_own_interval_flows.py > data/e151_build.log 2>&1
grep -v -i warn data/e151_build.log | tail -8 >> $L
until [[ -s /srv/prc-final-e2e/submissions/gentle-igloo_v927.parquet || -e data/final_any.failed ]]; do sleep 300; done
out=data/cv_preds_e151_flows; mkdir -p "$out"
echo "[$(date +%H:%M)] nested CV" >> $L
CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 CV_OUT="$out" FEATURES=data/features_e151.parquet \
  DROP_PREFIX=day_,tq_,wxr_,month,te_ .venv/bin/python src/cv_fold_preds_nested.py > data/e151_cv.log 2>&1
echo "[$(date +%H:%M)] gates" >> $L
PYTHONPATH=src .venv/bin/python src/fold_metrics.py data/cv_preds_e106_air "$out" 2>&1 | tail -15 >> $L || true
PYTHONPATH=src .venv/bin/python src/post_e14.py data/cv_preds_e106_air "$out" 2>&1 | tail -12 >> $L || true
echo "[$(date +%H:%M)] done" >> $L

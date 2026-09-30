#!/bin/bash
# E165 (journal 185): production nested CV with the filed-plan residuals (plan_*) in the base, against the hetzner
# production control cv_preds_e116prod_hz. After E163.
set -uo pipefail
cd /root/prc-2026
while pgrep -f "[h]z_builds_1oct.sh" > /dev/null || pgrep -f "[h]z_e161.sh" > /dev/null || pgrep -f "[h]z_e163.sh" > /dev/null; do sleep 120; done
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
L=data/e165.log
echo "[$(date +%H:%M)] E165 CV" > $L
mkdir -p data/cv_preds_e165_hz
CV_OUT=data/cv_preds_e165_hz FEATURES=data/features_e165.parquet .venv/bin/python src/cv_fold_preds_nested.py > data/e165_cv.log 2>&1
PYTHONPATH=src .venv/bin/python src/fold_metrics.py data/cv_preds_e116prod_hz data/cv_preds_e165_hz 2>&1 | tail -9 >> $L
PYTHONPATH=src .venv/bin/python src/post_e14.py data/cv_preds_e116prod_hz data/cv_preds_e165_hz 2>&1 | tail -8 >> $L
echo "[$(date +%H:%M)] done" >> $L

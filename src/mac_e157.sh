#!/bin/bash
# E157 (journal 152): production CV with the wake-ladder features (Hypothesis 6 + G7) against the Mac production control.
# Waits for the E156 LOMO to free the cores.
set -uo pipefail
cd ~/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=8 POLARS_MAX_THREADS=8
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
PY=.venv/bin/python
L=data/e157.log
until { grep -q "A2 vs A0" data/e156_lomo.log 2>/dev/null || grep -q "Traceback" data/e156_lomo.log 2>/dev/null; } && ! pgrep -f "[e]156_tabpfn" > /dev/null; do sleep 60; done
echo "[$(date +%H:%M)] E157 production CV" >> $L
mkdir -p data/cv_preds_e157
CV_OUT=data/cv_preds_e157 FEATURES=data/features_e157.parquet $PY src/cv_fold_preds_nested.py > data/e157_cv.log 2>&1 || echo "[$(date +%H:%M)] cv rc $?" >> $L
PYTHONPATH=src $PY src/fold_metrics.py data/cv_preds_e116prod data/cv_preds_e157 2>&1 | tail -9 >> $L
PYTHONPATH=src $PY src/post_e14.py data/cv_preds_e116prod data/cv_preds_e157 2>&1 | tail -8 >> $L
echo "[$(date +%H:%M)] done" >> $L

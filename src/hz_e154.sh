#!/bin/bash
# E154 (journal 142): OOF Huber corrector CV against the production OOF; waits for the no-E14 build (v588).
set -uo pipefail
cd /root/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6
export FEATURES=data/features_e106.parquet DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
PY=.venv/bin/python
L=data/e154.log
until { [[ -s submissions/gentle-igloo_v588.parquet ]] || [[ -e data/build_noe14.failed ]]; } && ! pgrep -f "[h]z_build_noe14" > /dev/null; do sleep 60; done
echo "[$(date +%H:%M)] E154 CV" >> $L
CTRL=data/cv_preds_e116prod CV_OUT=data/cv_preds_e154corr $PY src/e154_oof_corrector_cv.py >> $L 2>&1 || echo "[$(date +%H:%M)] E154 FAILED" >> $L
PYTHONPATH=src $PY src/fold_metrics.py data/cv_preds_e116prod data/cv_preds_e154corr 2>&1 | tail -9 >> $L
PYTHONPATH=src $PY src/post_e14.py data/cv_preds_e116prod data/cv_preds_e154corr 2>&1 | tail -8 >> $L
echo "[$(date +%H:%M)] done" >> $L

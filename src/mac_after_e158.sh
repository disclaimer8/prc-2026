#!/bin/bash
# Mac chain (journal 180): after the E158 CV -> E162 CV (same-operator inbound on the stand, classifier-only inb_*), gates,
# then the CatBoost seeds on features_e106_e51_e157 resume from 302.
cd ~/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
PY=.venv/bin/python
while pgrep -f "[m]ac_e158.sh" > /dev/null || pgrep -f "[e]130_rank.py" > /dev/null; do sleep 120; done
L=data/e162.log
echo "[$(date +%H:%M)] E162 CV start" >> $L
mkdir -p data/cv_preds_e162
CLF_ONLY_PREFIX=inb_ FOLDS=1,2,3 CV_OUT=data/cv_preds_e162 FEATURES=data/features_e162.parquet $PY src/cv_fold_preds_nested.py > data/e162_cv_a.log 2>&1 &
CLF_ONLY_PREFIX=inb_ FOLDS=4,5,6 CV_OUT=data/cv_preds_e162 FEATURES=data/features_e162.parquet $PY src/cv_fold_preds_nested.py > data/e162_cv_b.log 2>&1 &
wait
PYTHONPATH=src $PY src/fold_metrics.py data/cv_preds_e116prod data/cv_preds_e162 2>&1 | tail -9 >> $L
PYTHONPATH=src $PY src/post_e14.py data/cv_preds_e116prod data/cv_preds_e162 2>&1 | tail -8 >> $L
echo "[$(date +%H:%M)] done" >> $L
echo "[$(date +%H:%M)] resume after E158/E162" >> data/mac_catseeds_e157.log
START_SEED=302 bash src/mac_catseeds_e157.sh

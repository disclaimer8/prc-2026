#!/bin/bash
# E155 (journal 146): regularised LightGBM regressor (kind-mango v76 direction, capacity removal): min_data_in_leaf 50 -> 500,
# min_gain_to_split 0 -> 0.35, max_cat_threshold 9; classifier unchanged. Nested CV in the production config on the Mac,
# against the production control cv_preds_e116prod computed on this same machine with the same code.
cd ~/prc-2026
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 PROXY_SCALE=1 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config OMP_NUM_THREADS=10
export REG_OVERRIDE='{"min_data_in_leaf": 500, "min_gain_to_split": 0.35, "max_cat_threshold": 9}'
PY=.venv/bin/python
L=data/mac_e155_regcv.log
mkdir -p data/cv_preds_e155reg
echo "[$(date +%H:%M)] E155 nested CV, REG_OVERRIDE=$REG_OVERRIDE" >> $L
CV_OUT=data/cv_preds_e155reg FEATURES=data/features_e106.parquet $PY src/cv_fold_preds_nested.py > data/mac_cv_e155reg.log 2>&1 || echo "cv rc $?" >> $L
unset REG_OVERRIDE
PYTHONPATH=src $PY src/fold_metrics.py data/cv_preds_e116prod data/cv_preds_e155reg 2>&1 | tail -9 >> $L
PYTHONPATH=src $PY src/post_e14.py data/cv_preds_e116prod data/cv_preds_e155reg 2>&1 | tail -8 >> $L
echo "[$(date +%H:%M)] done" >> $L

#!/bin/bash
# U3 (journal 137, 139): E51 route geometry (taxiway-graph length, detour, extra metres; data/taxi_route.parquet) on the
# PRODUCTION feature set. Closed 16.09 as "absorbed by ap_stand / ap_runway", but the uploaded model drops STAND_mvt and ap_stand.
# Candidate = features_e106 + 3 route columns, E116 config, nested CV against the production control cv_preds_e116prod (same code).
cd ~/prc-2026
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 PROXY_SCALE=1 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config OMP_NUM_THREADS=10
PY=.venv/bin/python
L=data/mac_u3_e51prod.log
while pgrep -f "[m]ac_night_28sep.sh" > /dev/null; do sleep 120; done
echo "[$(date +%H:%M)] build features_e106_e51" >> $L
FEATURES=data/features_e106.parquet FEATURES_OUT=data/features_e106_e51.parquet $PY src/e51_add_route_features.py > data/mac_u3_build.log 2>&1 || { echo "build FAILED" >> $L; exit 1; }
tail -3 data/mac_u3_build.log >> $L
mkdir -p data/cv_preds_e51prod
echo "[$(date +%H:%M)] nested CV" >> $L
CV_OUT=data/cv_preds_e51prod FEATURES=data/features_e106_e51.parquet $PY src/cv_fold_preds_nested.py > data/mac_cv_e51prod.log 2>&1 || echo "cv rc $?" >> $L
PYTHONPATH=src $PY src/fold_metrics.py data/cv_preds_e116prod data/cv_preds_e51prod 2>&1 | tail -9 >> $L
PYTHONPATH=src $PY src/post_e14.py data/cv_preds_e116prod data/cv_preds_e51prod 2>&1 | tail -8 >> $L
echo "[$(date +%H:%M)] done" >> $L

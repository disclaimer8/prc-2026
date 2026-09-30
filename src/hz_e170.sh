#!/bin/bash
# E170 (journal 203): production nested CV of the E163 table + en-route ATFM along the route (ert_route_*, ert_dep, ert_dst),
# against the E163 CV (cv_preds_e163_hz). After E165; E169 waits for this one.
set -uo pipefail
cd /root/prc-2026
while pgrep -f "[h]z_e165.sh" > /dev/null; do sleep 120; done
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
L=data/e170.log
echo "[$(date +%H:%M)] E170 features" > $L
.venv/bin/python src/e170_build_features.py >> $L 2>&1 || { echo "[$(date +%H:%M)] features FAILED" >> $L; exit 1; }
echo "[$(date +%H:%M)] E170 CV" >> $L
mkdir -p data/cv_preds_e170_hz
CV_OUT=data/cv_preds_e170_hz FEATURES=data/features_e170.parquet .venv/bin/python src/cv_fold_preds_nested.py > data/e170_cv.log 2>&1
FEATURES=data/features_e106.parquet .venv/bin/python src/gate.py data/cv_preds_e163_hz data/cv_preds_e170_hz 2>&1 | grep -viE "warn|extension|Polars 2.0" >> $L
FEATURES=data/features_e106.parquet .venv/bin/python src/gate.py data/cv_preds_e116prod_hz data/cv_preds_e170_hz 2>&1 | grep SUMMARY >> $L
echo "[$(date +%H:%M)] done" >> $L

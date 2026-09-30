#!/bin/bash
# Final-table CV FIN1 (journal 209): e106 + E51 (taxi route) + E157 + E163 (destination ATFM) + E165 (plan residual) + E170
# (en-route ATFM), production nested CV, against the E163 CV and the production control. Starts when the E172 sched ranking
# model has freed cores 0-5 and the E170 table exists (E170 CV runs alongside on the other 6).
set -uo pipefail
cd /root/prc-2026
while pgrep -f "[h]z_e130sched_rank.sh" > /dev/null || ! grep -q "E170 CV" data/e170.log; do sleep 120; done
# memory: the E170 CV runs alongside; start only with >= 28 GB available (prod shares the box)
while [ "$(free -g | awk '/^Mem:/ {print $7}')" -lt 28 ]; do sleep 120; done
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
L=data/fin1.log
echo "[$(date +%H:%M)] FIN1 table" > $L
.venv/bin/python src/build_final_table.py data/features_fin1.parquet data/features_e170.parquet data/features_e106_e51.parquet \
  data/features_e157.parquet data/features_e165.parquet 2>&1 | grep -viE "warn|extension|Polars 2.0" >> $L || { echo "[$(date +%H:%M)] table FAILED" >> $L; exit 1; }
echo "[$(date +%H:%M)] FIN1 CV" >> $L
mkdir -p data/cv_preds_fin1_hz
CV_OUT=data/cv_preds_fin1_hz FEATURES=data/features_fin1.parquet nice -n 10 .venv/bin/python src/cv_fold_preds_nested.py > data/fin1_cv.log 2>&1
FEATURES=data/features_e106.parquet .venv/bin/python src/gate.py data/cv_preds_e163_hz data/cv_preds_fin1_hz 2>&1 | grep -viE "warn|extension|Polars 2.0" >> $L
FEATURES=data/features_e106.parquet .venv/bin/python src/gate.py data/cv_preds_e116prod_hz data/cv_preds_fin1_hz 2>&1 | grep SUMMARY >> $L
[[ -d data/cv_preds_e170_hz && $(ls data/cv_preds_e170_hz | wc -l) -eq 6 ]] && FEATURES=data/features_e106.parquet .venv/bin/python src/gate.py data/cv_preds_e170_hz data/cv_preds_fin1_hz 2>&1 | grep SUMMARY >> $L
echo "[$(date +%H:%M)] done" >> $L

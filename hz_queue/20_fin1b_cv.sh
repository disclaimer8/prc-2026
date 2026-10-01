#!/bin/bash
# FIN1b (journal 217): FIN1 without the E170 en-route ATFM columns (E170 rejected against E163: decided +0.563, 3/6; journal 209 said
# "if E170 fails, recompute FIN1 without ert_*") = E163 + E51 + E157 + E165. Production nested CV, gate against cv_preds_e163_hz.
set -uo pipefail
cd /root/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
L=data/fin1b.log
echo "[$(date +%H:%M)] FIN1b table" > $L
.venv/bin/python src/build_final_table.py data/features_fin1b.parquet data/features_e163.parquet data/features_e106_e51.parquet \
  data/features_e157.parquet data/features_e165.parquet 2>&1 | grep -viE "warn|extension|Polars 2.0" >> $L || { echo "[$(date +%H:%M)] table FAILED" >> $L; exit 1; }
echo "[$(date +%H:%M)] FIN1b CV" >> $L
mkdir -p data/cv_preds_fin1b_hz
CV_OUT=data/cv_preds_fin1b_hz FEATURES=data/features_fin1b.parquet nice -n 10 .venv/bin/python src/cv_fold_preds_nested.py > data/fin1b_cv.log 2>&1
FEATURES=data/features_e106.parquet .venv/bin/python src/gate.py data/cv_preds_e163_hz data/cv_preds_fin1b_hz 2>&1 | grep -viE "warn|extension|Polars 2.0" >> $L
FEATURES=data/features_e106.parquet .venv/bin/python src/gate.py data/cv_preds_fin1_hz data/cv_preds_fin1b_hz 2>&1 | grep SUMMARY >> $L
FEATURES=data/features_e106.parquet .venv/bin/python src/gate.py data/cv_preds_e116prod_hz data/cv_preds_fin1b_hz 2>&1 | grep SUMMARY >> $L
echo "[$(date +%H:%M)] done" >> $L

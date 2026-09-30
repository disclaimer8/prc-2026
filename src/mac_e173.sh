#!/bin/bash
# E173 (journal 210): two more anchored members after E172 (sched/eobt): AOBT_3 (the NM actual off-block, uncalibrated a = 1, where
# the production proxy is calibrated per airport) and IOBT (initial off-block). Regressor-only nested CV as E130.
cd ~/prc-2026
until grep -q "done" data/mac_e172.log; do sleep 60; done
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=12 POLARS_MAX_THREADS=12
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 FEATURES=data/features_e106.parquet
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
for M in iobt aobt; do
  echo "[$(date +%H:%M)] E173 $M" >> data/mac_e173.log
  E130_MODE=$M CV_OUT=data/cv_e130$M .venv/bin/python src/e130_raw_reg_cv.py >> data/mac_e173_$M.log 2>&1 || echo "[$(date +%H:%M)] $M FAILED" >> data/mac_e173.log
done
echo "[$(date +%H:%M)] done" >> data/mac_e173.log

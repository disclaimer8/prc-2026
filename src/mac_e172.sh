#!/bin/bash
# E172 (journal 205): third E130 member = the anchored regressor on SCHED, then EOBT_1 (regressor-only nested CV, as E130).
# Runs after the E163 base is shipped; stops the CatBoost E51+E157 seeds (the final table is decided after E170).
cd ~/prc-2026
until grep -q "shipped" data/mac_e163_base.log; do sleep 60; done
sleep 30; pkill -f mac_catseeds_e157.sh; pkill -f "catboost_seeds.py 5600"
echo "[$(date +%H:%M)] stopped for E172 (sched/eobt member CVs)" >> data/mac_catseeds_e157.log
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=12 POLARS_MAX_THREADS=12
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 FEATURES=data/features_e106.parquet
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
for M in sched eobt; do
  echo "[$(date +%H:%M)] E172 $M" >> data/mac_e172.log
  E130_MODE=$M CV_OUT=data/cv_e130$M .venv/bin/python src/e130_raw_reg_cv.py >> data/mac_e172_$M.log 2>&1 || echo "[$(date +%H:%M)] $M FAILED" >> data/mac_e172.log
done
echo "[$(date +%H:%M)] done" >> data/mac_e172.log

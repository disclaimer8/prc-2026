#!/bin/bash
# E174 (journal 218): member diversity by the model rather than the anchor. Two lobt-anchored regressor-only members with the
# nested CV of src/e130_raw_reg_cv.py, then each as the 6th member over raw/lobt/sched/eobt/aobt (box around the 5-member optimum).
cd ~/prc-2026
L=data/mac_e174.log
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=12 POLARS_MAX_THREADS=12
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 FEATURES=data/features_e106.parquet
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
PY=.venv/bin/python
ov () {  # bash 3.2 on the Mac: no associative arrays
  case $1 in
    lobt_s) echo '{"num_leaves": 31, "min_data_in_leaf": 500, "feature_fraction": 0.5, "lambda_l2": 50.0, "learning_rate": 0.08}' ;;
    lobt_h) echo '{"objective": "huber", "alpha": 300.0}' ;;
  esac
}
for M in lobt_s lobt_h; do
  echo "[$(date -u +%H:%MZ)] E174 $M CV" >> $L
  E130_MODE=lobt REG_OVERRIDE="$(ov $M)" CV_OUT=data/cv_e130$M $PY src/e130_raw_reg_cv.py > data/mac_e174_$M.log 2>&1 || echo "[$(date -u +%H:%MZ)] $M CV FAILED" >> $L
done
for M in lobt_s lobt_h; do
  [ "$(ls data/cv_e130$M 2>/dev/null | wc -l)" -eq 6 ] || { echo "[$(date -u +%H:%MZ)] grid $M skipped" >> $L; continue; }
  echo "[$(date -u +%H:%MZ)] grid 6th member $M" >> $L
  MEMBERS=raw,lobt,sched,eobt,aobt,$M E172_WMAX=0.95 E172_BOX=0.05-0.15,0.15-0.25,0.1-0.15,0.1-0.2,0.15-0.25,0-0.25 \
    $PY src/e172_members_cv.py > data/e174_grid_$M.log 2>&1 \
    && $PY src/gate.py data/cv_preds_e172_raw_lobt_sched_eobt_aobt_${M}_ctrl data/cv_preds_e172_raw_lobt_sched_eobt_aobt_$M 2>&1 \
       | grep -viE "warn|extension|Polars 2.0" >> data/e174_grid_$M.log \
    || echo "[$(date -u +%H:%MZ)] grid $M FAILED" >> $L
done
echo "[$(date -u +%H:%MZ)] done" >> $L

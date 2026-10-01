#!/bin/bash
# E180 (journal 238, external review item 5): a member with whole feature families removed -- the LOBT-anchored E130 regressor
# without the neighbour families (nb_, nbf_, nbs_, fw_) and the operator / stand identities (AIRCRAFT_OPERATOR_flt, airline_prefix,
# stand_prefix); NM times, geometry, weather, traffic counts and stand occupancy stay (86 of 153 features). Then the 7th member over
# the six (raw, lobt, sched, eobt, aobt, mlp). The Mac CatBoost E175 seed is paused (SIGSTOP) while this runs, resumed after.
cd ~/prc-2026
L=data/mac_e180.log
CB=$(pgrep -f "catboost_seeds.py 5600" | head -1)
[ -n "$CB" ] && kill -STOP $CB && echo "[$(date -u +%H:%MZ)] paused CatBoost E175 pid $CB" >> $L
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=12 POLARS_MAX_THREADS=12
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 FEATURES=data/features_e106.parquet
PY=.venv/bin/python
echo "[$(date -u +%H:%MZ)] E180 lobt member without neighbour families and identities" >> $L
E130_MODE=lobt CV_OUT=data/cv_e130lobt_nf \
  DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config,nb_,nbf_,nbs_,fw_,AIRCRAFT_OPERATOR_flt,airline_prefix,stand_prefix \
  $PY src/e130_raw_reg_cv.py > data/e180_cv.log 2>&1 || echo "[$(date -u +%H:%MZ)] CV FAILED" >> $L
[ -n "$CB" ] && kill -CONT $CB && echo "[$(date -u +%H:%MZ)] resumed CatBoost E175" >> $L
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config OMP_NUM_THREADS=2
if [ "$(ls data/cv_e130lobt_nf 2>/dev/null | wc -l)" -eq 6 ]; then
  echo "[$(date -u +%H:%MZ)] grid 7th member lobt_nf" >> $L
  MEMBERS=raw,lobt,sched,eobt,aobt,mlp,lobt_nf E172_WMAX=1.0 E172_BOX=0.05-0.1,0.1-0.2,0.1-0.15,0.1-0.15,0.15-0.2,0.2-0.25,0-0.25 \
    $PY src/e172_members_cv.py > data/e180_grid.log 2>&1 \
    && $PY src/gate.py data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlp_lobt_nf_ctrl data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlp_lobt_nf 2>&1 | grep -viE "warn|extension|Polars 2.0" >> data/e180_grid.log \
    || echo "[$(date -u +%H:%MZ)] grid FAILED" >> $L
fi
echo "[$(date -u +%H:%MZ)] done" >> $L

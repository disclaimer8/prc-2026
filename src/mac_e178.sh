#!/bin/bash
# E178 (journal 232): kNN member CV, then the 7th-member grid over the six (raw, lobt, sched, eobt, aobt, mlp) and the gate.
cd ~/prc-2026
L=data/mac_e178.log
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 FEATURES=data/features_e106.parquet
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config OMP_NUM_THREADS=4
PY=.venv/bin/python
echo "[$(date -u +%H:%MZ)] E178 kNN CV" >> $L
KNN_JOBS=4 $PY src/e178_knn_member_cv.py > data/e178_knn_cv.log 2>&1 || { echo "[$(date -u +%H:%MZ)] CV FAILED" >> $L; exit 1; }
echo "[$(date -u +%H:%MZ)] grid 7th member knn" >> $L
OMP_NUM_THREADS=2 MEMBERS=raw,lobt,sched,eobt,aobt,mlp,knn E172_WMAX=1.0 E172_BOX=0.05-0.1,0.15-0.2,0.1-0.15,0.1-0.15,0.15-0.2,0.2-0.25,0-0.25 \
  $PY src/e172_members_cv.py > data/e178_grid.log 2>&1 \
  && $PY src/gate.py data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlp_knn_ctrl data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlp_knn 2>&1 | grep -viE "warn|extension|Polars 2.0" >> data/e178_grid.log \
  || echo "[$(date -u +%H:%MZ)] grid FAILED" >> $L
echo "[$(date -u +%H:%MZ)] done" >> $L

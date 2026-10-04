#!/bin/bash
# E185 (journal 255) on home-pc — same commands as src/mac_e185.sh, without waiting for E182 (separate machine).
cd ~/prc-2026
L=data/hp_e185.log
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
PY=.venv/bin/python
echo "[$(date -u +%H:%MZ)] E185 OOF" >> $L
OMP_NUM_THREADS=12 POLARS_MAX_THREADS=12 FEATURES=data/features_fin1b.parquet CV_OUT=data/cv_e130motion \
  $PY src/e185_motion_only_cv.py > data/e185_cv.log 2>&1 || { echo "[$(date -u +%H:%MZ)] OOF FAILED" >> $L; date > data/hp_e185.failed; exit 1; }
echo "[$(date -u +%H:%MZ)] E185 grid + gate" >> $L
FEATURES=data/features_e106.parquet $PY src/e185_subset_grid_cv.py > data/e185_grid.log 2>&1 || { echo "[$(date -u +%H:%MZ)] grid FAILED" >> $L; date > data/hp_e185.failed; exit 1; }
FEATURES=data/features_e106.parquet $PY src/gate.py data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlpa_f1b data/e185_cv_sub \
  2>&1 | grep -viE "warn|extension|Polars 2.0" >> data/e185_grid.log
grep -h "SUMMARY" data/e185_grid.log | sed 's/^/  /' >> $L
R=$(grep -oE "^fold [0-9]+\+[0-9]+: rounds [0-9]+" data/e185_cv.log | awk '{s+=$4} END {printf "%d", s/NR}')
echo "[$(date -u +%H:%MZ)] E185 ranking member, 5 seeds, $R rounds" >> $L
OMP_NUM_THREADS=12 POLARS_MAX_THREADS=12 FEATURES=data/features_fin1b.parquet $PY src/e185_motion_rank.py 5 $R > data/e185_rank.log 2>&1 \
  || { echo "[$(date -u +%H:%MZ)] rank FAILED" >> $L; date > data/hp_e185.failed; exit 1; }
echo "[$(date -u +%H:%MZ)] done" >> $L
date > data/hp_e185.ready

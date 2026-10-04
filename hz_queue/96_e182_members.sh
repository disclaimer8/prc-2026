#!/bin/bash
# E182 (journal 248), hetzner half: the SCHED and raw members on features_fin1b, one CV at a time (two LightGBM CVs at once
# oversubscribe this box), after the 4.10 builds. The Mac fetches data/cv_e130{sched,raw}f once data/e182_hz.ready exists.
cd /root/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 FEATURES=data/features_fin1b.parquet
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
L=data/e182_hz.log
rm -f data/e182_hz.ready data/e182_hz.failed
until [[ -s data/builds_4oct.ready || -e data/builds_4oct.failed ]]; do sleep 60; done
for M in sched raw; do
  echo "[$(date +%H:%M)] ${M}f" >> $L
  E130_MODE=$M CV_OUT=data/cv_e130${M}f .venv/bin/python src/e130_raw_reg_cv.py > data/e182_${M}f.log 2>&1 \
    || { echo "[$(date +%H:%M)] ${M}f FAILED" >> $L; date > data/e182_hz.failed; exit 1; }
done
date > data/e182_hz.ready
echo "[$(date +%H:%M)] done" >> $L

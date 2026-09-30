#!/bin/bash
# 1.10 night chain on the Mac. The 30.09 Mac jobs died at 16:08 (after E173 iobt fold 3+9), so the eobt ranking model of
# journal 211 never ran. Order: 1) E172 eobt ranking model -> hetzner (needed by every 4-member build and the final recipe),
# 2) E173 iobt folds 4-6, 3) E173 aobt folds 1-6, 4) member grids: iobt, then aobt, each as the 5th member over
# raw/lobt/sched/eobt in a box around the 4-member optimum (0.15/0.20/0.15/0.20), gate of journal 210 (decided >= 4/6, trimmed < 0).
cd ~/prc-2026
L=data/mac_1oct.log
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=12 POLARS_MAX_THREADS=12
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 FEATURES=data/features_e106.parquet
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
PY=.venv/bin/python
echo "[$(date -u +%H:%MZ)] eobt rank" >> $L
E130_MODE=eobt $PY src/e130_rank.py 5 2551 > data/e130_eobt_rank.log 2>&1 \
  && rsync -a --timeout=120 submissions/e130_eobt_rank.npz hetzner:/root/prc-2026/submissions/ \
  && ssh hetzner "date > /root/prc-2026/data/e130_eobt_rank.ready" && echo "[$(date -u +%H:%MZ)] eobt rank shipped" >> $L \
  || echo "[$(date -u +%H:%MZ)] eobt rank FAILED" >> $L
echo "[$(date -u +%H:%MZ)] E173 iobt folds 4-6" >> $L
E130_MODE=iobt E130_FOLDS=4,5,6 CV_OUT=data/cv_e130iobt $PY src/e130_raw_reg_cv.py >> data/mac_e173_iobt.log 2>&1 || echo "[$(date -u +%H:%MZ)] iobt FAILED" >> $L
echo "[$(date -u +%H:%MZ)] E173 aobt" >> $L
E130_MODE=aobt CV_OUT=data/cv_e130aobt $PY src/e130_raw_reg_cv.py >> data/mac_e173_aobt.log 2>&1 || echo "[$(date -u +%H:%MZ)] aobt FAILED" >> $L
for M in iobt aobt; do
  [ "$(ls data/cv_e130$M 2>/dev/null | wc -l)" -eq 6 ] || { echo "[$(date -u +%H:%MZ)] grid $M skipped (folds missing)" >> $L; continue; }
  echo "[$(date -u +%H:%MZ)] grid 5th member $M" >> $L
  MEMBERS=raw,lobt,sched,eobt,$M E172_WMAX=0.9 E172_BOX=0.1-0.2,0.15-0.25,0.1-0.2,0.15-0.25,0-0.25 \
    $PY src/e172_members_cv.py > data/e173_grid_$M.log 2>&1 \
    && $PY src/gate.py data/cv_preds_e172_raw_lobt_sched_eobt_${M}_ctrl data/cv_preds_e172_raw_lobt_sched_eobt_$M 2>&1 \
       | grep -viE "warn|extension|Polars 2.0" >> data/e173_grid_$M.log \
    || echo "[$(date -u +%H:%MZ)] grid $M FAILED" >> $L
done
echo "[$(date -u +%H:%MZ)] done" >> $L

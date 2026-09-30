#!/bin/bash
# Journal 165: E157 passed the production CV (after E14 -0.133, trimmed -0.201, 5/6) and U3/E51 passed earlier (-0.165, 5/6).
# CatBoost (weight 0.65) sees neither, so its seeds move to the combined table features_e106_e51_e157 (seeds 300+,
# moved to submissions/e51e157_catseeds/). Runs until 25 seeds or until stopped; the E51-only seeds 200-204 stay as they are.
cd ~/prc-2026
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 PROXY_SCALE=1 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
export CATBOOST_THREADS=12 OMP_NUM_THREADS=12 POLARS_MAX_THREADS=12 FEATURES=data/features_e106_e51_e157.parquet
PY=.venv/bin/python
L=data/mac_catseeds_e157.log
mkdir -p submissions/e51e157_catseeds
s=${START_SEED:-300}
echo "[$(date +%H:%M)] start" >> $L
while [ $s -lt 325 ]; do
  [ -s submissions/e51e157_catseeds/cat_reg_rank_seed$s.npz ] && { s=$((s + 1)); continue; }  # already done (resume after a pause)
  echo "[$(date +%H:%M)] seed $s" >> $L
  nice -n 10 $PY src/catboost_seeds.py 5600 $s >> data/mac_catseeds_e157_run.log 2>&1 || { echo "[$(date +%H:%M)] seed $s FAILED" >> $L; break; }
  mv submissions/cat_reg_rank_seed$s.npz submissions/e51e157_catseeds/
  echo "[$(date +%H:%M)] seed $s done" >> $L
  s=$((s + 1))
done
echo "[$(date +%H:%M)] done" >> $L

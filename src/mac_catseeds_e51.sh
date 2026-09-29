#!/bin/bash
# After E155 (mac_e155_regcv.sh): CatBoost seeds on features_e106_e51 (U3 passed CV, journal 145; CatBoost at weight 0.65 does not
# see the route geometry yet). Seeds 200+, moved to submissions/e51_catseeds/; new seeds start until 07:45 next morning.
cd ~/prc-2026
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 PROXY_SCALE=1 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
export CATBOOST_THREADS=10 OMP_NUM_THREADS=10 POLARS_MAX_THREADS=10 FEATURES=data/features_e106_e51.parquet
PY=.venv/bin/python
L=data/mac_catseeds_e51.log
while pgrep -f "[m]ac_e155_regcv.sh" > /dev/null; do sleep 120; done
mkdir -p submissions/e51_catseeds
s=200
echo "[$(date +%H:%M)] start" >> $L
while :; do
  hm=$(date +%H%M)
  [[ $hm > 0745 && $hm < 1200 && $s -gt 200 && $(date +%d) != 29 ]] && break
  echo "[$(date +%H:%M)] seed $s" >> $L
  nice -n 10 $PY src/catboost_seeds.py 5600 $s >> data/mac_catseeds_e51_run.log 2>&1 || { echo "[$(date +%H:%M)] seed $s FAILED" >> $L; break; }
  mv submissions/cat_reg_rank_seed$s.npz submissions/e51_catseeds/
  echo "[$(date +%H:%M)] seed $s done" >> $L
  s=$((s + 1))
done
echo "[$(date +%H:%M)] done" >> $L

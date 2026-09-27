#!/bin/zsh
# E144 (journal 110): CatBoost seeds 5-9 on the E116 configuration, the same recipe as E121b seeds 1-4 (hetzner, v291).
# Runs on the Mac (seeds 5-7; hetzner does 8-9, src/hz_e144_catseeds.sh); each seed writes submissions/cat_reg_rank_seed<N>.npz and is copied to hetzner as soon as it exists.
set -uo pipefail
cd ~/prc-2026
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 PROXY_SCALE=1 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export FEATURES=data/features_e106.parquet DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
export CATBOOST_THREADS=10 OMP_NUM_THREADS=10 POLARS_MAX_THREADS=10
LOG=data/e144_catseeds.log
for s in ${=E144_SEEDS:-5 6 7}; do
  echo "[$(date +%H:%M)] seed $s" >> $LOG
  nice -n 10 .venv/bin/python src/catboost_seeds.py 5600 $s >> $LOG 2>&1 || { echo "[$(date +%H:%M)] FAILED seed $s" >> $LOG; exit 1; }
  scp -q submissions/cat_reg_rank_seed$s.npz hetzner:/root/prc-2026/submissions/e144_seeds/ && echo "[$(date +%H:%M)] seed $s copied" >> $LOG
done
echo "[$(date +%H:%M)] done" >> $LOG

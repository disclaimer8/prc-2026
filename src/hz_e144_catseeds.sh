#!/bin/bash
# E144 (journal 110): CatBoost seeds 8 and 9 on hetzner (the Mac does 5-7), E121b recipe, results in submissions/e144_seeds/.
set -euo pipefail
cd /root/prc-2026
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 PROXY_SCALE=1 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export FEATURES=data/features_e106.parquet DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
export CATBOOST_THREADS=6 OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6
rm -f data/e144.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee data/e144.failed' ERR
mkdir -p submissions/e144_seeds
for s in 9 8; do
  echo "[$(date +%H:%M)] seed $s"
  .venv/bin/python src/catboost_seeds.py 5600 $s > data/e144_seed$s.log 2>&1
  mv submissions/cat_reg_rank_seed$s.npz submissions/e144_seeds/
done
echo "[$(date +%H:%M)] done"

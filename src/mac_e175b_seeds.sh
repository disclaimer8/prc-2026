#!/bin/bash
# E175b (journal 243): CatBoost FIN1b seeds 10..14 on the Mac (idle since 00:35Z), merged with 0..9 into a SEPARATE file
# submissions/cat_fin1b_seeds15.npz -- the 10-seed file used by the 3.10 S5 build stays untouched. For the final if S5 wins.
cd ~/prc-2026
L=data/mac_e175.log
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=12 POLARS_MAX_THREADS=12 CATBOOST_THREADS=12
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 FEATURES=data/features_fin1b.parquet
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
export CAT_SEED_OUT=submissions/cat_fin1b_seed{}.npz CAT_MERGE_OUT=submissions/cat_fin1b_seeds15.npz
for s in 10 11 12 13 14; do
  echo "[$(date -u +%H:%MZ)] E175b seed $s" >> $L
  .venv/bin/python src/catboost_seeds.py 5600 $s >> data/mac_e175_seeds.log 2>&1 || { echo "[$(date -u +%H:%MZ)] seed $s FAILED" >> $L; continue; }
  .venv/bin/python src/catboost_seeds.py merge $(seq -s ' ' 0 14) >> data/mac_e175_seeds.log 2>&1 && echo "[$(date -u +%H:%MZ)] merged 15-file through seed $s" >> $L
done
echo "[$(date -u +%H:%MZ)] E175b done" >> $L

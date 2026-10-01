#!/bin/bash
# E175 (journal 221/226): CatBoost on the FIN1b table, seeds 5..9 on hetzner while the Mac does 0..4 (the Mac pulls these files,
# skips them in its own loop and merges all ten). Same settings as src/mac_e175_catfin1b.sh.
cd /root/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=10 POLARS_MAX_THREADS=10 CATBOOST_THREADS=10
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 FEATURES=data/features_fin1b.parquet
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
export CAT_SEED_OUT=submissions/cat_fin1b_seed{}.npz CAT_MERGE_OUT=submissions/cat_fin1b_seeds_hz.npz
for s in 5 6 7 8 9; do
  echo "[$(date +%H:%M)] seed $s" >> data/e175_hz.log
  .venv/bin/python src/catboost_seeds.py 5600 $s >> data/e175_hz_seeds.log 2>&1 || echo "[$(date +%H:%M)] seed $s FAILED" >> data/e175_hz.log
done
echo "[$(date +%H:%M)] done" >> data/e175_hz.log

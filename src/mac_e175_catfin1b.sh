#!/bin/bash
# E175 (journal 221): the CatBoost regressor (0.65 of the regular candidate) on the FIN1b table instead of E106. Ten seeds on the Mac,
# files submissions/cat_fin1b_seed<N>.npz (the E106 seeds stay untouched), merged and shipped to hetzner for a board build on v870.
cd ~/prc-2026
L=data/mac_e175.log
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=12 POLARS_MAX_THREADS=12 CATBOOST_THREADS=12
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 FEATURES=data/features_fin1b.parquet
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
export CAT_SEED_OUT=submissions/cat_fin1b_seed{}.npz CAT_MERGE_OUT=submissions/cat_fin1b_seeds.npz
PY=.venv/bin/python
[ -s data/features_fin1b.parquet ] || rsync -a --partial --timeout=300 hetzner:/root/prc-2026/data/features_fin1b.parquet data/ || { echo "[$(date -u +%H:%MZ)] table copy FAILED" >> $L; exit 1; }
for s in 0 1 2 3 4 5 6 7 8 9; do
  echo "[$(date -u +%H:%MZ)] seed $s" >> $L
  $PY src/catboost_seeds.py 5600 $s >> data/mac_e175_seeds.log 2>&1 || { echo "[$(date -u +%H:%MZ)] seed $s FAILED" >> $L; continue; }
  # after every seed: merge what exists and ship it, so the board build can use a partial average if time runs out
  $PY src/catboost_seeds.py merge 0 1 2 3 4 5 6 7 8 9 >> data/mac_e175_seeds.log 2>&1 \
    && rsync -a --timeout=120 submissions/cat_fin1b_seeds.npz hetzner:/root/prc-2026/submissions/ \
    && echo "[$(date -u +%H:%MZ)] merged and shipped through seed $s" >> $L
done
ssh hetzner "date > /root/prc-2026/data/cat_fin1b.ready"
echo "[$(date -u +%H:%MZ)] done" >> $L

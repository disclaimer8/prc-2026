#!/bin/zsh
# E32 (if accepted overnight 15-16.09): canonical pipeline on data/features_e32.parquet (E28 + ADS-B features in the model).
# v32 regimes+regressor, v33 blend with CatBoost, v34 TabPFN, v35 + E14.
set -euo pipefail
git -C "${0:a:h}/.." diff --quiet HEAD -- src/train.py src/features.py src/target_encoding.py src/blend.py src/catboost_reg.py src/seed_avg.py src/lirf_tabpfn.py src/e14_apply.py || { echo "ABORT: uncommitted changes in model code"; exit 1; }
cd "${0:a:h}/.."
export FEATURES=data/features_e32.parquet DROP_PREFIX=day_,tq_,wxr_,month
export TABPFN_ALLOW_CPU_LARGE_DATASET=1
V=32
echo "[$(date +%H:%M)] validate"
uv run python src/train.py validate > data/final_e32_validate.log 2>&1
grep -E "^best=" data/final_e32_validate.log
uv run python src/train.py submit $V | grep -E "^wrote"
REG=$(python3 -c "import json;print(int(json.load(open('data/last_validation.json'))['best']['reg']*1.1))")
echo "[$(date +%H:%M)] seeds"
uv run python src/seed_avg.py rank 5 $REG | grep -E "saved"
cp submissions/lgb_reg_seeds.npz submissions/lgb_reg_seeds_v32.npz
echo "[$(date +%H:%M)] catboost"
uv run python src/catboost_reg.py 5600 > data/final_e32_catboost.log 2>&1
cp submissions/cat_reg_rank.npz submissions/cat_reg_rank_v32.npz
LGB_SEEDS=submissions/lgb_reg_seeds_v32.npz uv run python src/blend.py $V $((V+1)) 0.65 | grep -E "^wrote"
uv run python src/lirf_tabpfn.py $((V+1)) $((V+2)) | grep -E "^wrote"
uv run python src/e14_apply.py $((V+2)) $((V+1)) $((V+3)) | grep -E "^wrote"
echo "[$(date +%H:%M)] FINAL submissions/gentle-igloo_v$((V+3)).parquet"

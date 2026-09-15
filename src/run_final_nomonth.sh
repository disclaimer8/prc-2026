#!/bin/zsh
# E24 accepted (15.09): final pipeline without the `month` feature, from the existing data/features_final.parquet
# (features unchanged; only the model inputs change). v19 regimes+regressor, v20 blend with CatBoost, v21 TabPFN, v22 + E14.
set -euo pipefail
cd "${0:a:h}/.."
export FEATURES=data/features_final.parquet DROP_PREFIX=day_,tq_,wxr_,month
export TABPFN_ALLOW_CPU_LARGE_DATASET=1
V=19
echo "[$(date +%H:%M)] validate"
uv run python src/train.py validate > data/final_nomonth_validate.log 2>&1
grep -E "^best=" data/final_nomonth_validate.log
uv run python src/train.py submit $V | grep -E "^wrote"
REG=$(python3 -c "import json;print(int(json.load(open('data/last_validation.json'))['best']['reg']*1.1))")
echo "[$(date +%H:%M)] seeds"
uv run python src/seed_avg.py rank 5 $REG | grep -E "saved"
cp submissions/lgb_reg_seeds.npz submissions/lgb_reg_seeds_v19.npz
echo "[$(date +%H:%M)] catboost"
uv run python src/catboost_reg.py 5600 > data/final_nomonth_catboost.log 2>&1
cp submissions/cat_reg_rank.npz submissions/cat_reg_rank_v19.npz
LGB_SEEDS=submissions/lgb_reg_seeds_v19.npz uv run python src/blend.py $V $((V+1)) 0.65 | grep -E "^wrote"
uv run python src/lirf_tabpfn.py $((V+1)) $((V+2)) | grep -E "^wrote"
uv run python src/e14_apply.py $((V+2)) $((V+1)) $((V+3)) | grep -E "^wrote"
echo "[$(date +%H:%M)] FINAL submissions/gentle-igloo_v$((V+3)).parquet"

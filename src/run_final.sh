#!/bin/zsh
# Final pipeline from raw challenge data to the submission file (reproducible, seeds fixed).
#   ./src/run_final.sh <V>      writes submissions/gentle-igloo_v<V+2>.parquet
# Prerequisites: uv sync; .env with PRC_ACCESS_KEY/PRC_SECRET_KEY; `uv run python src/download.py`; scripts/get_external.sh
# Components: regime mixture (LightGBM multiclass) + residual regressor (0.65 CatBoost + 0.35 five-seed LightGBM)
# + TabPFN-2 regime probabilities for LIRF departures without NM at 4-30 h delay.
set -euo pipefail
V=${1:?version number}
cd "${0:a:h}/.."
export FEATURES=data/features_final.parquet DROP_PREFIX=day_,tq_,wxr_
export TABPFN_ALLOW_CPU_LARGE_DATASET=1
FEATURES_OUT=$FEATURES uv run python src/features.py > data/final_features.log 2>&1
uv run python src/train.py validate > data/final_validate.log 2>&1
grep -E "^best=" data/final_validate.log
uv run python src/train.py submit $V | grep -E "^wrote"
REG=$(python3 -c "import json;print(int(json.load(open('data/last_validation.json'))['best']['reg']*1.1))")
uv run python src/seed_avg.py rank 5 $REG | grep -E "saved"
uv run python src/catboost_reg.py 5600 > data/final_catboost.log 2>&1
LGB_SEEDS=submissions/lgb_reg_seeds.npz uv run python src/blend.py $V $((V+1)) 0.65 | grep -E "^wrote"
uv run python src/lirf_tabpfn.py $((V+1)) $((V+2)) | grep -E "^wrote"
echo "FINAL submissions/gentle-igloo_v$((V+2)).parquet"

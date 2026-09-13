#!/bin/zsh
# v6: features v5 (neighbours), residual regressor, LOBT feasibility; v7 = v6 regime model + 0.65 CatBoost + 0.35 LightGBM 5-seed average.
set -e
cd ~/prc-2026
export FEATURES=data/features_v5.parquet
uv run python src/train.py validate > data/val_v6.log 2>&1
grep -E "^best=" data/val_v6.log
uv run python src/train.py submit 6 2>&1 | grep -v Warning | tail -1
REG=$(python3 -c "import json;print(int(json.load(open('data/last_validation.json'))['best']['reg']*1.1))")
uv run python src/seed_avg.py rank 5 $REG 2>&1 | grep -E "seed|saved"
uv run python src/catboost_reg.py 5600 > data/cat_rank_v5.log 2>&1
grep -v Warning data/cat_rank_v5.log | tail -1
LGB_SEEDS=submissions/lgb_reg_seeds.npz uv run python src/blend.py 6 7 0.65 2>&1 | grep -v Warning | tail -2
echo CHAIN_DONE

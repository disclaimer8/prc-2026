#!/bin/zsh
# v10-v12: hygiene bundle on features_v8 without day_* (E3 rejected):
# K2 labels (6 s, no planned regimes at EDDF/EHAM/LSZH), taxiing_at_ob fix, AOBT gate, B4 stale-event cap.
# v10 = regime model + LightGBM, v11 = + CatBoost 0.65 / 5-seed LightGBM 0.35, v12 = v11 + LIRF table.
set -e
set -o pipefail
cd ~/prc-2026
export FEATURES=data/features_v8.parquet DROP_PREFIX=day_
uv run python src/train.py validate > data/val_v10.log 2>&1
grep -E "^best=" data/val_v10.log
uv run python src/train.py submit 10 2>&1 | grep -E "^wrote"
REG=$(python3 -c "import json;print(int(json.load(open('data/last_validation.json'))['best']['reg']*1.1))")
uv run python src/seed_avg.py rank 5 $REG 2>&1 | grep -E "saved"
uv run python src/catboost_reg.py 5600 > data/cat_rank_v12.log 2>&1
tail -1 data/cat_rank_v12.log
LGB_SEEDS=submissions/lgb_reg_seeds.npz uv run python src/blend.py 10 11 0.65 2>&1 | grep -E "^wrote|seed average"
uv run python src/lirf_table.py 11 12 2>&1 | grep -E "overridden|wrote"
uv run python src/upload.py 12
n=0; until uv run python src/leaderboard.py | grep -q "_v12.parquet"; do n=$((n+1)); [ $n -ge 90 ] && { echo NO_SCORE; exit 1; }; sleep 30; done
uv run python src/leaderboard.py | tail -4
echo CHAIN_DONE

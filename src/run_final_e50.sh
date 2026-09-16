#!/bin/zsh
# E50 (only if its gate passes): the regime-rate features — how often a group records the block time from a planned
# time, out of fold, by airport / operator / stand / hour / flight number (REGIME_RATES=1 in src/target_encoding.py).
# v61 regimes+regressor, v62 blend, v63 TabPFN, v64 + E14, E36 stage -> v65. Built on data/features_e28.parquet with
# nothing else changed, so the leaderboard reads v65 against v30 as the effect of this one change.
set -euo pipefail
cd "${0:a:h}/.."
git diff --quiet HEAD -- src/train.py src/features.py src/target_encoding.py src/blend.py src/catboost_reg.py src/seed_avg.py src/lirf_tabpfn.py src/e14_apply.py src/e36_adsb_stand_geometry.py || { echo "ABORT: uncommitted model code"; exit 1; }
export FEATURES=data/features_e28.parquet DROP_PREFIX=day_,tq_,wxr_,month TABPFN_ALLOW_CPU_LARGE_DATASET=1 REGIME_RATES=1
V=61
echo "[$(date +%H:%M)] validate"
uv run python src/train.py validate > data/final_e50_validate.log 2>&1
grep -E "^best=" data/final_e50_validate.log
uv run python src/train.py submit $V | grep -E "^wrote"
REG=$(python3 -c "import json;print(int(json.load(open('data/last_validation.json'))['best']['reg']*1.1))")
echo "[$(date +%H:%M)] seeds"
uv run python src/seed_avg.py rank 5 $REG | grep -E "saved"
cp submissions/lgb_reg_seeds.npz submissions/lgb_reg_seeds_v$V.npz
echo "[$(date +%H:%M)] catboost"
uv run python src/catboost_reg.py 5600 > data/final_e50_catboost.log 2>&1
cp submissions/cat_reg_rank.npz submissions/cat_reg_rank_v$V.npz
LGB_SEEDS=submissions/lgb_reg_seeds_v$V.npz uv run python src/blend.py $V $((V+1)) 0.65 | grep -E "^wrote"
uv run python src/lirf_tabpfn.py $((V+1)) $((V+2)) | grep -E "^wrote"
uv run python src/e14_apply.py $((V+2)) $((V+1)) $((V+3)) | grep -E "^wrote"
PARTS=submissions/parts_v$((V+1)).npz .venv/bin/python src/e36_adsb_stand_geometry.py apply $((V+3)) 65 2>&1 | grep -E "^wrote|regime probabilities"
echo "[$(date +%H:%M)] FINAL submissions/gentle-igloo_v65.parquet"

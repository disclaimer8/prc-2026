#!/bin/zsh
# E47 (16.09): E32 re-decided under the corrected gate — ADS-B features inside the main model win 6/6 folds on the rows the
# base model actually decides (-1.41 s LB-equivalent), while the old total-SSE gate said 3/6 and rejected it on 15.09.
# Canonical chain on data/features_e32.parquet: v46 regimes+regressor, v47 blend with CatBoost, v48 TabPFN, v49 + E14,
# then the E36 stand-geometry stage (the one behind our best v30) -> v50, which is what gets uploaded.
set -euo pipefail
cd "${0:a:h}/.."
git diff --quiet HEAD -- src/train.py src/features.py src/target_encoding.py src/blend.py src/catboost_reg.py src/seed_avg.py src/lirf_tabpfn.py src/e14_apply.py src/e36_adsb_stand_geometry.py || { echo "ABORT: uncommitted model code"; exit 1; }
export FEATURES=data/features_e32.parquet DROP_PREFIX=day_,tq_,wxr_,month TABPFN_ALLOW_CPU_LARGE_DATASET=1
V=46
echo "[$(date +%H:%M)] validate"
uv run python src/train.py validate > data/final_e47_validate.log 2>&1
grep -E "^best=" data/final_e47_validate.log
uv run python src/train.py submit $V | grep -E "^wrote"
REG=$(python3 -c "import json;print(int(json.load(open('data/last_validation.json'))['best']['reg']*1.1))")
echo "[$(date +%H:%M)] seeds"
uv run python src/seed_avg.py rank 5 $REG | grep -E "saved"
cp submissions/lgb_reg_seeds.npz submissions/lgb_reg_seeds_v$V.npz
echo "[$(date +%H:%M)] catboost"
uv run python src/catboost_reg.py 5600 > data/final_e47_catboost.log 2>&1
cp submissions/cat_reg_rank.npz submissions/cat_reg_rank_v$V.npz
LGB_SEEDS=submissions/lgb_reg_seeds_v$V.npz uv run python src/blend.py $V $((V+1)) 0.65 | grep -E "^wrote"
uv run python src/lirf_tabpfn.py $((V+1)) $((V+2)) | grep -E "^wrote"
uv run python src/e14_apply.py $((V+2)) $((V+1)) $((V+3)) | grep -E "^wrote"
echo "[$(date +%H:%M)] stage E36 on the new base"
PARTS=submissions/parts_v$V.npz .venv/bin/python src/e36_adsb_stand_geometry.py apply $((V+3)) 50 2>&1 | grep -E "^wrote|regime probabilities"
echo "[$(date +%H:%M)] FINAL submissions/gentle-igloo_v50.parquet"

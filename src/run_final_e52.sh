#!/bin/zsh
# E52: both accepted changes at once — ADS-B features inside the model (data/features_e32.parquet, the re-decided
# E32 = E47) and the per-airport trust coefficient (PROXY_SCALE=1, E46): v56 regimes+regressor, v57 blend, v58 TabPFN,
# v59 + E14, E36 stage -> v60. Speculative, built overnight so the combination can go into the fourth slot of 17.09
# if both win separately, instead of waiting another day.
set -euo pipefail
cd "${0:a:h}/.."
# hold the ADS-B re-extraction off the offblock files for as long as this build runs (its stage step reads every day)
mkdir -p data && touch data/.adsb_lock
trap 'rm -f data/.adsb_lock' EXIT
git diff --quiet HEAD -- src/train.py src/features.py src/target_encoding.py src/blend.py src/catboost_reg.py src/seed_avg.py src/lirf_tabpfn.py src/e14_apply.py src/e36_adsb_stand_geometry.py || { echo "ABORT: uncommitted model code"; exit 1; }
export FEATURES=data/features_e32.parquet DROP_PREFIX=day_,tq_,wxr_,month TABPFN_ALLOW_CPU_LARGE_DATASET=1 PROXY_SCALE=1
V=56
echo "[$(date +%H:%M)] validate"
uv run python src/train.py validate > data/final_e52_validate.log 2>&1
grep -E "^best=" data/final_e52_validate.log
uv run python src/train.py submit $V | grep -E "^wrote"
REG=$(python3 -c "import json;print(int(json.load(open('data/last_validation.json'))['best']['reg']*1.1))")
echo "[$(date +%H:%M)] seeds"
uv run python src/seed_avg.py rank 5 $REG | grep -E "saved"
cp submissions/lgb_reg_seeds.npz submissions/lgb_reg_seeds_v$V.npz
echo "[$(date +%H:%M)] catboost"
uv run python src/catboost_reg.py 5600 > data/final_e52_catboost.log 2>&1
cp submissions/cat_reg_rank.npz submissions/cat_reg_rank_v$V.npz
LGB_SEEDS=submissions/lgb_reg_seeds_v$V.npz uv run python src/blend.py $V $((V+1)) 0.65 | grep -E "^wrote"
uv run python src/lirf_tabpfn.py $((V+1)) $((V+2)) | grep -E "^wrote"
uv run python src/e14_apply.py $((V+2)) $((V+1)) $((V+3)) | grep -E "^wrote"
PARTS=submissions/parts_v$((V+1)).npz .venv/bin/python src/e36_adsb_stand_geometry.py apply $((V+3)) 60 2>&1 | grep -E "^wrote|regime probabilities"
echo "[$(date +%H:%M)] FINAL submissions/gentle-igloo_v60.parquet"

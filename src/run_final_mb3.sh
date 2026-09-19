#!/bin/zsh
# E75 / MB3: the regressor trained only on the rows it serves (SERVED_ONLY=1), otherwise the accepted
# chain -- PROXY_SCALE=1 and the E74 stage on the 1 kt extraction, so v88 differs from v82 by MB3 alone.
# Our CV said -0.01 s and rejected it. We build it anyway because the instrument is known to be blind
# here: kind-mango measured -5.99 s on the board from the same change while their own hold-out showed
# nothing, and recorded why -- the benefit is entirely on 2026 out-of-distribution rows.
# Registered prediction, written before the score: -0.3 to +0.2 s, most likely nothing. Their -5.99 does
# not transfer to us: the LIRF tail table, E14 and TabPFN already own the rows MB3 removes.
set -euo pipefail
cd "${0:a:h}/.."
# hold the ADS-B re-extraction off the offblock files for as long as this build runs (its stage step reads every day)
mkdir -p data && touch data/.adsb_lock
trap 'rm -f data/.adsb_lock' EXIT
git diff --quiet HEAD -- src/train.py src/features.py src/target_encoding.py src/blend.py src/catboost_reg.py src/seed_avg.py src/lirf_tabpfn.py src/e14_apply.py src/e36_adsb_stand_geometry.py || { echo "ABORT: uncommitted model code"; exit 1; }
export FEATURES=data/features_e28.parquet DROP_PREFIX=day_,tq_,wxr_,month TABPFN_ALLOW_CPU_LARGE_DATASET=1 PROXY_SCALE=1 SERVED_ONLY=1
V=83
echo "[$(date +%H:%M)] validate"
uv run python src/train.py validate > data/final_mb3_validate.log 2>&1
grep -E "^best=" data/final_mb3_validate.log
uv run python src/train.py submit $V | grep -E "^wrote"
REG=$(python3 -c "import json;print(int(json.load(open('data/last_validation.json'))['best']['reg']*1.1))")
echo "[$(date +%H:%M)] seeds"
uv run python src/seed_avg.py rank 5 $REG | grep -E "saved"
cp submissions/lgb_reg_seeds.npz submissions/lgb_reg_seeds_v$V.npz
echo "[$(date +%H:%M)] catboost"
uv run python src/catboost_reg.py 5600 > data/final_mb3_catboost.log 2>&1
cp submissions/cat_reg_rank.npz submissions/cat_reg_rank_v$V.npz
LGB_SEEDS=submissions/lgb_reg_seeds_v$V.npz uv run python src/blend.py $V $((V+1)) 0.65 | grep -E "^wrote"
uv run python src/lirf_tabpfn.py $((V+1)) $((V+2)) | grep -E "^wrote"
uv run python src/e14_apply.py $((V+2)) $((V+1)) $((V+3)) | grep -E "^wrote"
ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v$((V+1)).npz \
  .venv/bin/python src/e70_apply.py $((V+3)) 87 2>&1 | grep -E "^wrote|regime probabilities"
ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v$((V+1)).npz \
  .venv/bin/python src/e64_apply.py 87 88 2>&1 | grep -E "^wrote|regime probabilities"
echo "[$(date +%H:%M)] FINAL submissions/gentle-igloo_v88.parquet"

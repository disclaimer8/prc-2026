#!/bin/zsh
# E90 chain (22.09): threshold 1800 chosen by the pre-registered E90 rule (vs E89: decided+trimmed -1.11, 4/6).
# Built only so that a board slot, if Denys chooses to spend one, has a file ready. Submits NOTHING.
# Only the classifier changes under CLF_COST_WEIGHT/CLF_COST_MIN_SPREAD, so the regressor seeds and CatBoost
# of the E86 build (v100, identical features and regressor) are reused as they are.
#   zsh src/run_final_e89.sh    log: data/final_e89.log, FINAL submissions/gentle-igloo_v125.parquet
set -euo pipefail
cd "${0:a:h}/.."
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=1800
export FEATURES=data/features_e77.parquet DROP_PREFIX=day_,tq_,wxr_,month TABPFN_ALLOW_CPU_LARGE_DATASET=1 PROXY_SCALE=1
git diff --quiet HEAD -- src/train.py src/features.py src/blend.py src/lirf_tabpfn.py src/e14_apply.py || { echo "ABORT: uncommitted model code"; exit 1; }
cmp -s submissions/cat_reg_rank.npz submissions/cat_reg_rank_v100.npz || { echo "ABORT: cat_reg_rank.npz is not v100's"; exit 1; }
V=120
echo "[$(date +%H:%M)] validate"
uv run python src/train.py validate > data/final_e90_validate.log 2>&1
grep -E "^best=" data/final_e90_validate.log
uv run python src/train.py submit $V | grep -E "^wrote"
LGB_SEEDS=submissions/lgb_reg_seeds_v100.npz uv run python src/blend.py $V $((V+1)) 0.65 | grep -E "^wrote"
uv run python src/lirf_tabpfn.py $((V+1)) $((V+2)) | grep -E "^wrote"
uv run python src/e14_apply.py $((V+2)) $((V+1)) $((V+3)) | grep -E "^wrote"
ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v$((V+1)).npz \
  .venv/bin/python src/e70_apply.py $((V+3)) $((V+4)) 2>&1 | grep -E "^wrote|regime probabilities"
ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v$((V+1)).npz \
  .venv/bin/python src/e64_apply.py $((V+4)) $((V+5)) 2>&1 | grep -E "^wrote|regime probabilities"
echo "[$(date +%H:%M)] FINAL submissions/gentle-igloo_v$((V+5)).parquet"

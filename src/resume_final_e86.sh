#!/bin/zsh
# Продолжение run_final_e86.sh после перезагрузки 22.09: validate/submit/seeds v100 уцелели (22:16, 22:49),
# погиб CatBoost. Отсюда та же цепочка, строка в строку.
set -euo pipefail
cd "${0:a:h}/.."
export CLF_COST_WEIGHT=1
export FEATURES=data/features_e77.parquet DROP_PREFIX=day_,tq_,wxr_,month TABPFN_ALLOW_CPU_LARGE_DATASET=1 PROXY_SCALE=1
git diff --quiet HEAD -- src/train.py src/features.py src/target_encoding.py src/blend.py src/catboost_reg.py src/seed_avg.py src/lirf_tabpfn.py src/e14_apply.py src/e36_adsb_stand_geometry.py || { echo "ABORT: uncommitted model code"; exit 1; }
V=100
[[ -s submissions/lgb_reg_seeds_v$V.npz && -s submissions/reg_v$V.txt && -s submissions/clf_v$V.txt ]] || { echo "ABORT: нет артефактов v$V"; exit 1; }
echo "[$(date +%H:%M)] catboost (resume)"
uv run python src/catboost_reg.py 5600 > data/final_e86_catboost.log 2>&1
cp submissions/cat_reg_rank.npz submissions/cat_reg_rank_v$V.npz
LGB_SEEDS=submissions/lgb_reg_seeds_v$V.npz uv run python src/blend.py $V $((V+1)) 0.65 | grep -E "^wrote"
uv run python src/lirf_tabpfn.py $((V+1)) $((V+2)) | grep -E "^wrote"
uv run python src/e14_apply.py $((V+2)) $((V+1)) $((V+3)) | grep -E "^wrote"
ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v$((V+1)).npz \
  .venv/bin/python src/e70_apply.py $((V+3)) $((V+4)) 2>&1 | grep -E "^wrote|regime probabilities"
ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v$((V+1)).npz \
  .venv/bin/python src/e64_apply.py $((V+4)) $((V+5)) 2>&1 | grep -E "^wrote|regime probabilities"
echo "[$(date +%H:%M)] FINAL submissions/gentle-igloo_v$((V+5)).parquet"

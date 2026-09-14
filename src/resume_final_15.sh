#!/bin/zsh
# Resume run_final.sh 15 after the 14.09 macOS-update reboot (killed inside CatBoost), then E14.
set -euo pipefail
cd "${0:a:h}/.."
export FEATURES=data/features_final.parquet DROP_PREFIX=day_,tq_,wxr_
export TABPFN_ALLOW_CPU_LARGE_DATASET=1
V=15
echo "[$(date +%H:%M)] catboost"
uv run python src/catboost_reg.py 5600 > data/final_catboost.log 2>&1
LGB_SEEDS=submissions/lgb_reg_seeds.npz uv run python src/blend.py $V $((V+1)) 0.65 | grep -E "^wrote"
uv run python src/lirf_tabpfn.py $((V+1)) $((V+2)) | grep -E "^wrote"
echo "FINAL submissions/gentle-igloo_v$((V+2)).parquet"
echo "[$(date +%H:%M)] E14: fold preds"
uv run python src/cv_fold_preds.py > data/cv_fold_preds.log 2>&1
echo "[$(date +%H:%M)] E14: rule"
uv run python src/e14_flight_history.py > data/e14.log 2>&1
echo "[$(date +%H:%M)] done"

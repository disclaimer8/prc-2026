#!/bin/zsh
# E93 partner: per-airport models with the E89 cost weight, ranking predictions -> v130.
# The OOF arm this file realises is data/cv_preds_e91_pa (PER_AIRPORT=1, same flags), so the blend weights
# fitted against data/cv_preds_e89_stage apply to it. Base only, no stage -- as the OOF arm is base only.
set -euo pipefail
cd "${0:a:h}/.."
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export FEATURES=data/features_e77.parquet DROP_PREFIX=day_,tq_,wxr_,month PROXY_SCALE=1
git diff --quiet HEAD -- src/train.py src/submit_per_airport.py || { echo "ABORT: uncommitted model code"; exit 1; }
echo "[$(date +%H:%M)] per-airport partner build"
.venv/bin/python src/submit_per_airport.py 130 2>&1 | grep -E "^wrote|rows |Error|Traceback"
echo "[$(date +%H:%M)] FINAL submissions/gentle-igloo_v130.parquet"

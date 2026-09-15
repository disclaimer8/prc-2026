#!/bin/zsh
# E38 per-airport models, then decision. Log: data/e38.log
cd "${0:a:h}/.."
git diff --quiet HEAD -- src/train.py src/features.py src/target_encoding.py || { echo "ABORT: uncommitted changes in model code"; exit 1; }
FEATURES=data/features_e28.parquet DROP_PREFIX=day_,tq_,wxr_,month .venv/bin/python src/e38_per_airport.py folds
.venv/bin/python src/e38_per_airport.py summary
echo "[$(date +%H:%M)] done"

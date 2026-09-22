#!/bin/zsh
# E92: партнёр NO_AOBT (как v90) с весом цены ошибки E89; OOF для подбора весов смеси поверх v115.
set -uo pipefail
cd /Users/denyskolomiiets/prc-2026
out=data/cv_preds_e92_noaobt_cw; mkdir -p $out
env NO_AOBT=1 PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 FEATURES=data/features_noaobt_e28.parquet \
  DROP_PREFIX=day_,tq_,wxr_,month CV_OUT=$out .venv/bin/python src/cv_fold_preds_nested.py >> data/e92.log 2>&1
echo "E92_DONE $(ls $out/fold_*.npz | wc -l | tr -d ' ')" >> data/e92.log

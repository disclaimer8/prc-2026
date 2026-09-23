#!/bin/bash
# Кросс-архитектурная перепроверка E89 (23.09): оба плеча фолдов 2 и 3 на x86, чтобы выигрыш
# -1.02 с по решаемым строкам не оказался свойством арифметики Apple Silicon. Ровно как проверяли E77.
cd /root/prc-2026
for arm in ctrl cand; do
  out=data/cv_preds_e89rep_$arm; mkdir -p "$out"
  if [ "$arm" = cand ]; then W="CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600"; else W=""; fi
  env $W FOLDS=2,3 CV_OUT="$out" FEATURES=data/features_e77.parquet DROP_PREFIX=day_,tq_,wxr_,month \
    OMP_NUM_THREADS=6 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
    nice -n 19 taskset -c 6-11 .venv/bin/python src/cv_fold_preds_nested.py >> data/e89rep.log 2>&1
done
echo E89REP_DONE >> data/e89rep.log

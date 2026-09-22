#!/bin/bash
cd /root/prc-2026
for t in 7200 1800; do
  out=data/cv_preds_e90_t$t; mkdir -p "$out"
  CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=$t FOLDS=4 CV_OUT="$out" FEATURES=data/features_e77.parquet \
    DROP_PREFIX=day_,tq_,wxr_,month OMP_NUM_THREADS=6 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
    nice -n 19 taskset -c 6-11 .venv/bin/python src/cv_fold_preds_nested.py >> data/e90.log 2>&1
done
echo E90_F4_DONE >> data/e90.log

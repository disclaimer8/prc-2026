#!/bin/bash
# E91 partner OOF: per-airport models with the E89 cost weight, six folds, on hetzner (both arms stay machine-consistent: partner only).
cd /root/prc-2026
out=data/cv_preds_e91_pa; mkdir -p "$out"
CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 PER_AIRPORT=1 CV_OUT="$out" FEATURES=data/features_e77.parquet \
  DROP_PREFIX=day_,tq_,wxr_,month OMP_NUM_THREADS=6 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
  nice -n 19 taskset -c 6-11 .venv/bin/python src/cv_fold_preds_nested.py >> data/e91.log 2>&1
echo E91_DONE >> data/e91.log

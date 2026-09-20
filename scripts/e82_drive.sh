#!/bin/zsh
# E82: both arms, folds 1-3 here; hetzner takes 4-6 (both arms there too, so no fold mixes machines).
set -uo pipefail
cd "${0:a:h}/.."
say() { print -r -- "[$(date +%H:%M)] $*" }
F=data/features_e82.parquet
for arm in ctrl cand; do
  out=data/cv_preds_e82_$arm
  if [[ $arm == ctrl ]]; then drop=day_,tq_,wxr_,month,nlf_,nls_; else drop=day_,tq_,wxr_,month; fi
  say "$arm: folds 1,2,3 -> $out"
  mkdir -p $out
  env FEATURES=$F DROP_PREFIX=$drop FOLDS=1,2,3 CV_OUT=$out .venv/bin/python src/cv_fold_preds_nested.py >> data/e82_${arm}.log 2>&1 \
    || { say "FAILED $arm"; exit 1; }
  say "$arm done: $(ls $out/fold_*.npz | wc -l | tr -d ' ') folds"
done
say "local arms finished"

#!/bin/zsh
# E84: оба плеча, фолды 1-3 здесь. hetzner берёт фолд 4, home-pc 5 и 6 — оба плеча на своей машине.
set -uo pipefail
cd "${0:a:h}/.."
say() { print -r -- "[$(date +%H:%M)] $*" }
F=data/features_e84.parquet
for arm in ctrl cand; do
  out=data/cv_preds_e84_$arm
  if [[ $arm == ctrl ]]; then drop=day_,tq_,wxr_,month,fwl_,fws_; else drop=day_,tq_,wxr_,month; fi
  say "$arm: фолды 1,2,3 -> $out"
  mkdir -p $out
  env FEATURES=$F DROP_PREFIX=$drop FOLDS=1,2,3 CV_OUT=$out .venv/bin/python src/cv_fold_preds_nested.py >> data/e84_${arm}.log 2>&1 \
    || { say "УПАЛО $arm"; exit 1; }
  say "$arm готово: $(ls $out/fold_*.npz | wc -l | tr -d ' ') фолда"
done
say "локальные плечи закончены"

#!/bin/zsh
# E86: классификатор с весами по цене ошибки. Мак берёт фолды 1,2,3,5,6; hetzner — фолд 4 (оба плеча там же).
set -uo pipefail
cd "${0:a:h}/.."
say() { print -r -- "[$(date +%H:%M)] $*" }
F=data/features_e77.parquet
for arm in ctrl cand; do
  out=data/cv_preds_e86_$arm
  mkdir -p $out
  if [[ $arm == cand ]]; then export CLF_COST_WEIGHT=1; else unset CLF_COST_WEIGHT; fi
  say "$arm: фолды 1,2,3,5,6 -> $out"
  env FEATURES=$F DROP_PREFIX=day_,tq_,wxr_,month FOLDS=1,2,3,5,6 CV_OUT=$out \
    .venv/bin/python src/cv_fold_preds_nested.py >> data/e86_${arm}.log 2>&1 || { say "УПАЛО $arm"; exit 1; }
  say "$arm готово: $(ls $out/fold_*.npz | wc -l | tr -d ' ')"
done
for i in $(seq 1 120); do
  scp -q hetzner:/root/prc-2026/data/cv_preds_e86_ctrl/fold_4_10.npz data/cv_preds_e86_ctrl/ 2>/dev/null
  scp -q hetzner:/root/prc-2026/data/cv_preds_e86_cand/fold_4_10.npz data/cv_preds_e86_cand/ 2>/dev/null
  c=$(ls data/cv_preds_e86_ctrl/fold_*.npz 2>/dev/null | wc -l | tr -d ' ')
  d=$(ls data/cv_preds_e86_cand/fold_*.npz 2>/dev/null | wc -l | tr -d ' ')
  say "собрано ctrl=$c cand=$d из 6"
  [[ "$c" -ge 6 && "$d" -ge 6 ]] && break
  sleep 120
done
c=$(ls data/cv_preds_e86_ctrl/fold_*.npz 2>/dev/null | wc -l | tr -d ' ')
d=$(ls data/cv_preds_e86_cand/fold_*.npz 2>/dev/null | wc -l | tr -d ' ')
[[ "$c" -ge 6 && "$d" -ge 6 ]] || { say "НЕ СУЖУ: ctrl=$c cand=$d"; exit 1; }
PYTHONPATH=src .venv/bin/python - <<'PY' | tee data/e86_cv.log
from fold_metrics import compare
dec, tot, small, trim, lower, n, strict = compare("data/cv_preds_e86_ctrl", "data/cv_preds_e86_cand", lb_rmse=243.8347)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3 and -trim >= 0.3
print(f"DECISION E86: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s)")
PY
say "готово"

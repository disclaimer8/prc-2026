#!/bin/zsh
set -uo pipefail
cd "${0:a:h}/.."
for arm in ctrl cand; do
  if [[ $arm == ctrl ]]; then drop=day_,tq_,wxr_,month,fwl_,fws_; else drop=day_,tq_,wxr_,month; fi
  env FEATURES=data/features_e84.parquet DROP_PREFIX=$drop FOLDS=5,6 CV_OUT=data/cv_preds_e84_$arm \
    .venv/bin/python src/cv_fold_preds_nested.py >> data/e84_${arm}.log 2>&1 || exit 1
done
PYTHONPATH=src .venv/bin/python - <<'PY' | tee data/e84_cv.log
from fold_metrics import compare
dec, tot, small, trim, lower, n, strict = compare("data/cv_preds_e84_ctrl", "data/cv_preds_e84_cand", lb_rmse=243.8347)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3 and -trim >= 0.3
print(f"DECISION E84: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s)")
PY

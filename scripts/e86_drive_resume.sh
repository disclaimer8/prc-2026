#!/bin/zsh
# E86 после перезагрузки 22.09 ~03:00: ctrl 5/5 и cand fold_1_7 уцелели, фолд 4 обоих плеч лежит на hetzner.
# Досчитывает cand на фолдах 2,3,5,6 и дальше идёт по e86_drive.sh без изменений (сбор, суд, те же ворота).
set -uo pipefail
cd "${0:a:h}/.."
say() { print -r -- "[$(date +%H:%M)] $*" }
F=data/features_e77.parquet
out=data/cv_preds_e86_cand
say "resume cand: фолды 2,3,5,6 -> $out"
env CLF_COST_WEIGHT=1 FEATURES=$F DROP_PREFIX=day_,tq_,wxr_,month FOLDS=2,3,5,6 CV_OUT=$out \
  .venv/bin/python src/cv_fold_preds_nested.py >> data/e86_cand.log 2>&1 || { say "УПАЛО cand"; exit 1; }
scp -q hetzner:/root/prc-2026/data/cv_preds_e86_ctrl/fold_4_10.npz data/cv_preds_e86_ctrl/
scp -q hetzner:/root/prc-2026/data/cv_preds_e86_cand/fold_4_10.npz data/cv_preds_e86_cand/
c=$(ls data/cv_preds_e86_ctrl/fold_*.npz | wc -l | tr -d ' '); d=$(ls $out/fold_*.npz | wc -l | tr -d ' ')
say "собрано ctrl=$c cand=$d из 6"
[[ "$c" -ge 6 && "$d" -ge 6 ]] || { say "НЕ СУЖУ: ctrl=$c cand=$d"; exit 1; }
PYTHONPATH=src .venv/bin/python - <<'PY' | tee data/e86_cv.log
from fold_metrics import compare
dec, tot, small, trim, lower, n, strict = compare("data/cv_preds_e86_ctrl", "data/cv_preds_e86_cand", lb_rmse=243.8347)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3 and -trim >= 0.3
print(f"DECISION E86: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s)")
PY
say "готово"

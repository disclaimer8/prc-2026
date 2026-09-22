#!/bin/zsh
# E95: ждёт конца E92 (по артефакту), затем кандидат с CLF_SEEDS=5 на всех 6 фолдах, суд против E89.
set -uo pipefail
cd /Users/denyskolomiiets/prc-2026
tg() { ssh -o BatchMode=yes hetzner "set -a; . /etc/flightfinder/telegram.env; set +a; curl -s -m 20 -X POST \"https://api.telegram.org/bot\$TELEGRAM_BOT_TOKEN/sendMessage\" -d chat_id=90206238 --data-urlencode text=\"$1\" >/dev/null" 2>/dev/null; }
until grep -q E92_DONE data/e92.log 2>/dev/null; do sleep 60; done
out=data/cv_preds_e95_cand; mkdir -p $out
env CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 CLF_SEEDS=5 FEATURES=data/features_e77.parquet \
  DROP_PREFIX=day_,tq_,wxr_,month CV_OUT=$out .venv/bin/python src/cv_fold_preds_nested.py >> data/e95.log 2>&1 \
  || { tg "PRC E95: плечо упало."; exit 1; }
PYTHONPATH=src .venv/bin/python - <<'PY' | tee data/e95_cv.log
from fold_metrics import compare
dec, tot, small, trim, lower, n, strict = compare("data/cv_preds_e89_cand", "data/cv_preds_e95_cand", lb_rmse=242.6556)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -trim >= 0.3
print(f"DECISION E95: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s)")
PY
tg "PRC E95 — сиды классификатора, вердикт:
$(grep 'DECISION E95' data/e95_cv.log)
Прогноз был -0.05..-0.50, центр -0.15."

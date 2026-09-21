#!/bin/zsh
# Ждёт конца CV по E86 (по артефакту), затем гонит оба плеча E88 и судит. Отчёт — в телеграм.
set -uo pipefail
cd /Users/denyskolomiiets/prc-2026
say() { print -r -- "[$(date +%H:%M)] $*" }
tg() { ssh -o BatchMode=yes hetzner "set -a; . /etc/flightfinder/telegram.env; set +a; curl -s -m 20 -X POST \"https://api.telegram.org/bot\$TELEGRAM_BOT_TOKEN/sendMessage\" -d chat_id=90206238 --data-urlencode text=\"$1\" >/dev/null" 2>/dev/null; }

for i in $(seq 1 300); do
  [[ -s data/e86_cv.log ]] && break
  pgrep -qf "e86_driv[e]" || break
  sleep 60
done
say "E86 освободил машину, стартую E88"
F=data/features_e88.parquet
for arm in ctrl cand; do
  out=data/cv_preds_e88_$arm
  mkdir -p $out
  if [[ $arm == ctrl ]]; then drop=day_,tq_,wxr_,month,rx_; else drop=day_,tq_,wxr_,month; fi
  say "$arm -> $out"
  env FEATURES=$F DROP_PREFIX=$drop CV_OUT=$out .venv/bin/python src/cv_fold_preds_nested.py >> data/e88_${arm}.log 2>&1 \
    || { tg "PRC E88: плечо $arm упало, вердикта нет."; exit 1; }
done
PYTHONPATH=src .venv/bin/python - <<'PY' | tee data/e88_cv.log
from fold_metrics import compare
dec, tot, small, trim, lower, n, strict = compare("data/cv_preds_e88_ctrl", "data/cv_preds_e88_cand", lb_rmse=243.8347)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3 and -trim >= 0.3
print(f"DECISION E88: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s)")
PY
tg "PRC E88 — здоровье приёма, вердикт:
$(grep 'DECISION E88' data/e88_cv.log)

Прогноз был -0.05..-0.60, центр -0.15. Потолок по диагностике 7.24 с."
say "готово"

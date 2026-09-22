#!/bin/zsh
# E90: два порога на маке (фолды 1,2,3,5,6), фолд 4 обоих — на hetzner. Сбор, суд против контроля и против E89.
set -uo pipefail
cd /Users/denyskolomiiets/prc-2026
say() { print -r -- "[$(date +%H:%M)] $*" }
tg() { ssh -o BatchMode=yes hetzner "set -a; . /etc/flightfinder/telegram.env; set +a; curl -s -m 20 -X POST \"https://api.telegram.org/bot\$TELEGRAM_BOT_TOKEN/sendMessage\" -d chat_id=90206238 --data-urlencode text=\"$1\" >/dev/null" 2>/dev/null; }
for t in 7200 1800; do
  out=data/cv_preds_e90_t$t; mkdir -p $out
  say "t=$t -> $out"
  env CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=$t FEATURES=data/features_e77.parquet DROP_PREFIX=day_,tq_,wxr_,month \
    FOLDS=1,2,3,5,6 CV_OUT=$out .venv/bin/python src/cv_fold_preds_nested.py >> data/e90_t$t.log 2>&1 \
    || { tg "PRC E90: t=$t упало."; exit 1; }
done
for i in $(seq 1 120); do
  ssh -o BatchMode=yes hetzner 'grep -q E90_F4_DONE /root/prc-2026/data/e90.log' && break; sleep 60
done
for t in 7200 1800; do scp -q hetzner:/root/prc-2026/data/cv_preds_e90_t$t/fold_4_10.npz data/cv_preds_e90_t$t/; done
PYTHONPATH=src .venv/bin/python - <<'PY' | tee data/e90_cv.log
from fold_metrics import compare
for t in (7200, 1800):
    for ref, name in (("data/cv_preds_e86_ctrl", "ctrl"), ("data/cv_preds_e89_cand", "E89")):
        dec, tot, small, trim, lower, n, strict = compare(ref, f"data/cv_preds_e90_t{t}", lb_rmse=243.8347)
        print(f"E90 t={t} vs {name}: folds {n}, lower {lower}; decided {dec:+.2f}, trimmed {trim:+.2f}, "
              f"sum {dec+trim:+.2f}, all {tot:+.2f}, y<=7200 {small:+.2f}, strict {strict:+.2f}")
PY
tg "PRC E90 — порог цены ошибки:
$(grep '^E90' data/e90_cv.log)
Правило: меняем 3600, если против E89 сумма decided+trimmed <= -0.3 и ниже в >=4/6."
say "готово"

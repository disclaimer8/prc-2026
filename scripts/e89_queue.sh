#!/bin/zsh
# E89: ждёт вердикт E88 (по артефакту), считает кандидата на фолдах 1,2,3,5,6, забирает фолд 4 с hetzner,
# судит против data/cv_preds_e86_ctrl. Отчёт в телеграм. Ничего не отправляет на доску.
set -uo pipefail
cd /Users/denyskolomiiets/prc-2026
say() { print -r -- "[$(date +%H:%M)] $*" }
tg() { ssh -o BatchMode=yes hetzner "set -a; . /etc/flightfinder/telegram.env; set +a; curl -s -m 20 -X POST \"https://api.telegram.org/bot\$TELEGRAM_BOT_TOKEN/sendMessage\" -d chat_id=90206238 --data-urlencode text=\"$1\" >/dev/null" 2>/dev/null; }
until grep -q "DECISION E88" data/e88_cv.log 2>/dev/null || ! pgrep -qf "e88_queu[e]"; do sleep 60; done
out=data/cv_preds_e89_cand; mkdir -p $out
say "E89 cand: фолды 1,2,3,5,6 -> $out"
env CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 FEATURES=data/features_e77.parquet DROP_PREFIX=day_,tq_,wxr_,month \
  FOLDS=1,2,3,5,6 CV_OUT=$out .venv/bin/python src/cv_fold_preds_nested.py >> data/e89_cand.log 2>&1 \
  || { tg "PRC E89: плечо упало на маке, вердикта нет."; exit 1; }
for i in $(seq 1 120); do
  ssh -o BatchMode=yes hetzner 'grep -q E89_F4_DONE /root/prc-2026/data/e89_cand.log' && break; sleep 60
done
scp -q hetzner:/root/prc-2026/data/cv_preds_e89_cand/fold_4_10.npz $out/
n=$(ls $out/fold_*.npz | wc -l | tr -d ' '); say "собрано $n из 6"
[[ $n -ge 6 ]] || { tg "PRC E89: собрано $n из 6, не сужу."; exit 1; }
PYTHONPATH=src .venv/bin/python - <<'PY' | tee data/e89_cv.log
from fold_metrics import compare
dec, tot, small, trim, lower, n, strict = compare("data/cv_preds_e86_ctrl", "data/cv_preds_e89_cand", lb_rmse=243.8347)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3 and -trim >= 0.3
print(f"DECISION E89: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s)")
PY
tg "PRC E89 — вес только за пределом часа, вердикт:
$(grep 'DECISION E89' data/e89_cv.log)

Прогноз: решаемые -0.05..-0.60 (центр -0.25), y<=7200 -0.3..+1.5. E86 было: -0.51 / y<=7200 +6.96."
say "готово"

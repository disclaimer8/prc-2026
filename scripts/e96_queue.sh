#!/bin/zsh
# E96: ждёт конца E95 (по артефакту), затем половина обучающих строк, две полосы, суд против E89.
set -uo pipefail
cd /Users/denyskolomiiets/prc-2026
say() { print -r -- "[$(date +%H:%M)] $*" }
tg() { ssh -o BatchMode=yes hetzner "set -a; . /etc/flightfinder/telegram.env; set +a; curl -s -m 20 -X POST \"https://api.telegram.org/bot\$TELEGRAM_BOT_TOKEN/sendMessage\" -d chat_id=90206238 --data-urlencode text=\"$1\" >/dev/null" 2>/dev/null; }
until [[ $(ls data/cv_preds_e95_cand/fold_*.npz 2>/dev/null | wc -l | tr -d ' ') -ge 6 ]]; do
  pgrep -qf "mac_lanes2.s[h]" || break
  sleep 60
done
until ! pgrep -qf "cv_fold_preds_neste[d]"; do sleep 30; done
out=data/cv_preds_e96_half; mkdir -p $out
say "E96: FIT_FRAC=0.5, фолды 1,2,3 | 4,5,6"
for f in 1,2,3 4,5,6; do
  env FIT_FRAC=0.5 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 FEATURES=data/features_e77.parquet \
    DROP_PREFIX=day_,tq_,wxr_,month FOLDS=$f CV_OUT=$out OMP_NUM_THREADS=6 \
    .venv/bin/python src/cv_fold_preds_nested.py >> data/e96.log 2>&1 &
done
wait
n=$(ls $out/fold_*.npz | wc -l | tr -d ' '); say "E96 собрано $n из 6"
[[ $n -ge 6 ]] || { tg "PRC E96: собрано $n из 6."; exit 1; }
PYTHONPATH=src .venv/bin/python - <<'PY' | tee data/e96_cv.log
from fold_metrics import compare
dec, tot, small, trim, lower, n, strict = compare("data/cv_preds_e89_cand", "data/cv_preds_e96_half", lb_rmse=242.6556)
print(f"E96 половина данных против всех: folds {n}, ниже в {lower}; decided {dec:+.2f} s, strict {strict:+.2f}, "
      f"all {tot:+.2f}, y<=7200 {small:+.2f}, trimmed {trim:+.2f}")
print("ЧТЕНИЕ: >=+3 крутая кривая (внешние данные имеют смысл); +0.5..+3 пограничная; <+0.5 плоская (линия закрыта)")
PY
tg "PRC E96 — кривая обучения (половина данных):
$(grep '^E96' data/e96_cv.log)
Прогноз был +0.8..+2.5, центр +1.4."
say "готово"

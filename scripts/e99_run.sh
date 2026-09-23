#!/bin/zsh
# E99 (23.09): снять целевые кодировки te_*/ter_* (агрегаты 2025 года) у цепочки E89. Контроль data/cv_preds_e89_cand.
set -uo pipefail
setopt NULL_GLOB
cd /Users/denyskolomiiets/prc-2026
tg() { ssh -o BatchMode=yes hetzner "set -a; . /etc/flightfinder/telegram.env; set +a; curl -s -m 20 -X POST \"https://api.telegram.org/bot\$TELEGRAM_BOT_TOKEN/sendMessage\" -d chat_id=90206238 --data-urlencode text=\"$1\" >/dev/null" 2>/dev/null; }
out=data/cv_preds_e99_note; mkdir -p $out
for f in 1,2,3 4,5,6; do
  env CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 FEATURES=data/features_e77.parquet \
    DROP_PREFIX=day_,tq_,wxr_,month,te_ FOLDS=$f CV_OUT=$out OMP_NUM_THREADS=6 \
    .venv/bin/python src/cv_fold_preds_nested.py >> data/e99.log 2>&1 &
done
wait
n=$(ls $out/fold_*.npz | wc -l | tr -d ' ')
[[ $n -ge 6 ]] || { tg "PRC E99: собрано $n из 6."; echo "E99_FAIL $n" >> data/e99.log; exit 1; }
PYTHONPATH=src .venv/bin/python - <<'PY' | tee data/e99_cv.log
from fold_metrics import compare
dec, tot, small, trim, lower, n, strict = compare("data/cv_preds_e89_cand", "data/cv_preds_e99_note", lb_rmse=242.6556)
print(f"E99 без te_/ter_ против E89: folds {n}, ниже в {lower}; decided {dec:+.2f} s, strict {strict:+.2f}, "
      f"all {tot:+.2f}, y<=7200 {small:+.2f}, trimmed {trim:+.2f}")
PY
tg "PRC E99 — без целевых кодировок:
$(grep '^E99' data/e99_cv.log)
Записано: decided <= +1.0 -> собирать полную цепочку для доски; > +1.0 -> закрыть."

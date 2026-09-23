#!/bin/zsh
set -uo pipefail
setopt NULL_GLOB
cd /Users/denyskolomiiets/prc-2026
tg() { ssh -o BatchMode=yes hetzner "set -a; . /etc/flightfinder/telegram.env; set +a; curl -s -m 20 -X POST \"https://api.telegram.org/bot\$TELEGRAM_BOT_TOKEN/sendMessage\" -d chat_id=90206238 --data-urlencode text=\"$1\" >/dev/null" 2>/dev/null; }
out=data/cv_preds_e96b_075; mkdir -p $out
for f in 1,2,3 4,5,6; do
  env FIT_FRAC=0.75 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 FEATURES=data/features_e77.parquet \
    DROP_PREFIX=day_,tq_,wxr_,month FOLDS=$f CV_OUT=$out OMP_NUM_THREADS=6 \
    .venv/bin/python src/cv_fold_preds_nested.py >> data/e96b.log 2>&1 &
done
wait
n=$(ls $out/fold_*.npz | wc -l | tr -d ' ')
[[ $n -ge 6 ]] || { tg "PRC E96b: собрано $n из 6."; exit 1; }
PYTHONPATH=src .venv/bin/python - <<'PY' | tee data/e96b_cv.log
from fold_metrics import compare
dec, tot, small, trim, lower, n, strict = compare("data/cv_preds_e89_cand", "data/cv_preds_e96b_075", lb_rmse=242.6556)
print(f"E96b три четверти данных против всех: folds {n}, ниже в {lower}; decided {dec:+.2f} s, strict {strict:+.2f}, "
      f"all {tot:+.2f}, y<=7200 {small:+.2f}, trimmed {trim:+.2f}")
PY
tg "PRC E96b — кривая обучения, три четверти данных:
$(grep '^E96b' data/e96b_cv.log)
Половина данных стоила +7.14 с."

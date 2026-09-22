#!/bin/zsh
set -uo pipefail
setopt NULL_GLOB
cd /Users/denyskolomiiets/prc-2026
tg() { ssh -o BatchMode=yes hetzner "set -a; . /etc/flightfinder/telegram.env; set +a; curl -s -m 20 -X POST \"https://api.telegram.org/bot\$TELEGRAM_BOT_TOKEN/sendMessage\" -d chat_id=90206238 --data-urlencode text=\"$1\" >/dev/null" 2>/dev/null; }
out=data/cv_preds_e94_regw; mkdir -p $out
for f in 1,2,3 4,5,6; do
  env REG_COST_WEIGHT=1 REG_COST_MIN_SPREAD=3600 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 \
    FEATURES=data/features_e77.parquet DROP_PREFIX=day_,tq_,wxr_,month FOLDS=$f CV_OUT=$out OMP_NUM_THREADS=6 \
    .venv/bin/python src/cv_fold_preds_nested.py >> data/e94.log 2>&1 &
done
wait
n=$(ls $out/fold_*.npz | wc -l | tr -d ' ')
[[ $n -ge 6 ]] || { tg "PRC E94: собрано $n из 6."; exit 1; }
PYTHONPATH=src .venv/bin/python - <<'PY' | tee data/e94_cv.log
from fold_metrics import compare
dec, tot, small, trim, lower, n, strict = compare("data/cv_preds_e89_cand", "data/cv_preds_e94_regw", lb_rmse=242.6556)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -trim >= 0.3
print(f"DECISION E94: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s)")
PY
tg "PRC E94 — цена ошибки у регрессора, вердикт:
$(grep 'DECISION E94' data/e94_cv.log)
Прогноз был -0.05..-0.40, центр -0.15. Слот только при -0.5 и лучше."

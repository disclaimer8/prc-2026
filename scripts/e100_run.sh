#!/bin/zsh
set -uo pipefail
setopt NULL_GLOB
cd /Users/denyskolomiiets/prc-2026
tg() { ssh -o BatchMode=yes hetzner "set -a; . /etc/flightfinder/telegram.env; set +a; curl -s -m 20 -X POST \"https://api.telegram.org/bot\$TELEGRAM_BOT_TOKEN/sendMessage\" -d chat_id=90206238 --data-urlencode text=\"$1\" >/dev/null" 2>/dev/null; }
NN_OUT=data/cv_preds_e100_nn POLARS_MAX_THREADS=4 .venv/bin/python src/e100_nn_partner.py >> data/e100.log 2>&1
n=$(ls data/cv_preds_e100_nn/fold_*.npz | wc -l | tr -d ' ')
[[ $n -ge 6 ]] || { tg "PRC E100: собрано $n из 6."; echo "E100_FAIL $n" >> data/e100.log; exit 1; }
PYTHONPATH=src .venv/bin/python src/e100_blend_eval.py 2>&1 | grep -v -i warn | tee data/e100_cv.log
tg "PRC E100 — NN-партнёр смеси:
$(grep 'DECISION' data/e100_cv.log)"

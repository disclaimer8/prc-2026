#!/bin/zsh
set -uo pipefail
setopt NULL_GLOB
cd /Users/denyskolomiiets/prc-2026
tg() { ssh -o BatchMode=yes hetzner "set -a; . /etc/flightfinder/telegram.env; set +a; curl -s -m 20 -X POST \"https://api.telegram.org/bot\$TELEGRAM_BOT_TOKEN/sendMessage\" -d chat_id=90206238 --data-urlencode text=\"$1\" >/dev/null" 2>/dev/null; }
NN_SEEDS=3 NN_OUT=data/cv_preds_e101_nn3 POLARS_MAX_THREADS=4 .venv/bin/python src/e100_nn_partner.py >> data/e101.log 2>&1
n=$(ls data/cv_preds_e101_nn3/fold_*.npz | wc -l | tr -d ' ')
[[ $n -ge 6 ]] || { tg "PRC E101: собрано $n из 6."; echo "E100_FAIL $n" >> data/e101.log; exit 1; }
E100_ARM=data/cv_preds_e101_nn3 PYTHONPATH=src .venv/bin/python src/e100_blend_eval.py 2>&1 | grep -v -i warn | tee data/e101_cv.log
tg "PRC E101 — три сида сети:
$(grep 'DECISION' data/e101_cv.log)"

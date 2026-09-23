#!/bin/zsh
set -uo pipefail
setopt NULL_GLOB
cd /Users/denyskolomiiets/prc-2026
tg() { ssh -o BatchMode=yes hetzner "set -a; . /etc/flightfinder/telegram.env; set +a; curl -s -m 20 -X POST \"https://api.telegram.org/bot\$TELEGRAM_BOT_TOKEN/sendMessage\" -d chat_id=90206238 --data-urlencode text=\"$1\" >/dev/null" 2>/dev/null; }
NO_AOBT=1 FEATURES=data/features_noaobt_e28.parquet NN_OUT=data/cv_preds_e102_nnnoaobt POLARS_MAX_THREADS=4 .venv/bin/python src/e100_nn_partner.py >> data/e102.log 2>&1
n=$(ls data/cv_preds_e102_nnnoaobt/fold_*.npz | wc -l | tr -d ' ')
[[ $n -ge 6 ]] || { tg "PRC E102: собрано $n из 6."; echo "E100_FAIL $n" >> data/e102.log; exit 1; }
E100_ARM=data/cv_preds_e102_nnnoaobt PYTHONPATH=src .venv/bin/python src/e100_blend_eval.py 2>&1 | grep -v -i warn | tee data/e102_cv.log
tg "PRC E102 — сеть без AOBT_3:
$(grep 'DECISION' data/e102_cv.log)"

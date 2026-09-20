#!/bin/zsh
# E78: train the NO_AOBT partner, then blend it with v44 into v82. Waits for E77 to free the machine.
#   zsh scripts/e78_drive.sh          log: data/e78_run.log
set -uo pipefail
cd "${0:a:h}/.."
say() { print -r -- "[$(date +%H:%M)] $*" }

while pgrep -qf cv_fold_preds_nested; do say "E77 still running, waiting 300 s"; sleep 300; done
say "machine free"

if [[ ! -s submissions/gentle-igloo_v90.parquet ]]; then
  say "training the NO_AOBT partner -> v90"
  env NO_AOBT=1 FEATURES=data/features_noaobt_e28.parquet DROP_PREFIX=day_,tq_,wxr_,month PROXY_SCALE=1 \
    .venv/bin/python src/train.py submit 90 > data/e78_partner.log 2>&1 || { say "FAILED partner fit"; exit 1; }
  grep -E "^wrote" data/e78_partner.log
fi

say "blending -> v91"
.venv/bin/python src/e78_partner_blend.py 91 2>&1 | grep -v Warning | tee data/e78_blend.log
say "done"

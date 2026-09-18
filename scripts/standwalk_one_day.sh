#!/bin/zsh
# E61: re-derive offblock2/3/5 for one day with the stand-dwell walk, into data/adsblol_standwalk.
set -euo pipefail
cd "${0:a:h}/.."
day=$1
for v in 2 3 5; do
  [[ -f ${ADSB_OUT_DIR:-data/adsblol_standwalk}/offblock${v}_${day}.parquet ]] && continue
  STAND_WALK=${STAND_WALK:-1} ADSB_OUT_DIR=${ADSB_OUT_DIR:-data/adsblol_standwalk} .venv/bin/python src/adsb_offblock_v${v}.py $day >/dev/null 2>&1 \
    || echo "FAILED v${v} ${day}"
done

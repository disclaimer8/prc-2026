#!/bin/zsh
# E61: re-derive the offblock features with STAND_WALK=1 into a separate directory, leaving the control intact.
# Reads the surface points already on disk, so nothing is downloaded.
#   scripts/reextract_standwalk.sh [parallelism]
set -euo pipefail
cd "${0:a:h}/.."
P=${1:-6}
OUT=${ADSB_OUT_DIR:-data/adsblol_standwalk}
mkdir -p $OUT
ls data/adsblol/surface_*.parquet | sed 's/.*surface_//;s/\.parquet//' | sort > /tmp/sw_days.txt
echo "[$(date +%H:%M)] $(wc -l < /tmp/sw_days.txt) days -> $OUT, parallelism $P"
xargs -P $P -n 1 scripts/standwalk_one_day.sh < /tmp/sw_days.txt
echo "[$(date +%H:%M)] done: $(ls $OUT/offblock2_*.parquet | wc -l | tr -d " ") of $(wc -l < /tmp/sw_days.txt) days have offblock2"

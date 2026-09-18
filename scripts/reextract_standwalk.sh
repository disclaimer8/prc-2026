#!/bin/zsh
# E61: re-derive the offblock features with STAND_WALK=1 into a separate directory, leaving the control intact.
# Reads the surface points already on disk, so nothing is downloaded.
#   scripts/reextract_standwalk.sh [parallelism]
set -euo pipefail
cd "${0:a:h}/.."
P=${1:-6}
OUT=data/adsblol_standwalk
mkdir -p $OUT
ls data/adsblol/surface_*.parquet | sed 's/.*surface_//;s/\.parquet//' | sort > /tmp/sw_days.txt
echo "[$(date +%H:%M)] $(wc -l < /tmp/sw_days.txt) days -> $OUT, parallelism $P"
cat /tmp/sw_days.txt | xargs -P $P -I{} zsh -c '
  for v in 2 3 5; do
    [[ -f data/adsblol_standwalk/offblock${v}_{}.parquet ]] && continue
    STAND_WALK=1 ADSB_OUT_DIR=data/adsblol_standwalk .venv/bin/python src/adsb_offblock_v${v}.py {} >/dev/null 2>&1 || echo "FAILED v${v} {}"
  done'
echo "[$(date +%H:%M)] done: $(ls $OUT/offblock2_*.parquet | wc -l | tr -d " ") of $(wc -l < /tmp/sw_days.txt) days have offblock2"

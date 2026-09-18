#!/bin/zsh
# E71: re-extract the ground features with the movement threshold at 1 kt instead of 3 (18.09).
# Writes to data/adsblol_kt1 and never touches data/adsblol, so the accepted path stays intact.
# The surface files are already downloaded; this only re-runs the three extractors over them.
# Resumable: a day whose offblock5 output already exists is skipped.
set -uo pipefail
cd "${0:a:h}/.."
out=data/adsblol_kt1
mkdir -p $out
n=0
for f in data/adsblol/surface_*.parquet; do
  d=${${f:t:r}#surface_}
  [[ -f $out/offblock5_$d.parquet ]] && continue
  while [ -e data/.adsb_lock ]; do echo "[$(date +%H:%M)] adsb lock present — holding"; sleep 60; done
  MOVING_KT=1 ADSB_OUT_DIR=$out .venv/bin/python src/adsb_offblock_v2.py $d >/dev/null 2>&1
  MOVING_KT=1 ADSB_OUT_DIR=$out .venv/bin/python src/adsb_offblock_v3.py $d >/dev/null 2>&1
  MOVING_KT=1 ADSB_OUT_DIR=$out .venv/bin/python src/adsb_offblock_v5.py $d >/dev/null 2>&1
  n=$((n+1))
  (( n % 25 == 0 )) && echo "[$(date +%H:%M)] $n days done, last $d"
done
echo "[$(date +%H:%M)] kt1 re-extraction finished: $(ls $out/offblock2_*.parquet 2>/dev/null | wc -l | tr -d ' ') days"

#!/bin/zsh
# E25 worker: surface extraction (streamed) and off-block features v2 for one day.
cd "${0:a:h}/.."
d=$1
./scripts/get_adsb_day.sh $d > data/adsblol/log_$d.txt 2>&1
echo "[$(date +%H:%M)] [$d] $(tail -1 data/adsblol/log_$d.txt)"
if [[ -f data/adsblol/surface_$d.parquet && ! -f data/adsblol/offblock2_$d.parquet ]]; then
  .venv/bin/python src/adsb_offblock_v2.py $d > /dev/null 2>&1 && echo "[$(date +%H:%M)] [$d] offblock2 done" || echo "[$d] offblock2 FAILED"
fi

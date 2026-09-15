#!/bin/zsh
# E25: extract surface points for the subset of days (July and January 2025 for validation, then July and January 2026
# for ranking), 4 days in parallel, then off-block features v2 per day. Log: data/adsb_batch.log
cd "${0:a:h}/.."
P=${P:-4}
days=()
for ym in 2025-07 2025-01 2026-07 2026-01; do
  for d in {01..31}; do days+=("$ym-$d"); done
done
print -l $days | xargs -P $P -I{} zsh -c './scripts/get_adsb_day.sh {} > data/adsblol/log_{}.txt 2>&1; tail -1 data/adsblol/log_{}.txt | sed "s/^/[{}] /"; [ -f data/adsblol/surface_{}.parquet ] && .venv/bin/python src/adsb_offblock_v2.py {} > /dev/null 2>&1 && echo "[{}] offblock2 done"'
echo "[$(date +%H:%M)] batch done"

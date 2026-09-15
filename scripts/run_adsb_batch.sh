#!/bin/zsh
# E25: extract surface points for July and January 2025 (validation), then July and January 2026 (ranking),
# P days in parallel (default 4). Log: data/adsb_batch.log
cd "${0:a:h}/.."
P=${P:-4}
for ym in 2025-07 2025-01 2026-07 2026-01; do for d in {01..31}; do echo "$ym-$d"; done; done | xargs -P $P -n 1 ./scripts/adsb_one_day.sh
echo "[$(date +%H:%M)] batch done"

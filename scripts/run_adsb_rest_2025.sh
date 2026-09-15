#!/bin/zsh
# E25: after the subset batch, the remaining months of 2025 (needed for the 6-fold gate). Log: data/adsb_rest.log
cd "${0:a:h}/.."
until grep -q "batch done" data/adsb_batch.log 2>/dev/null; do sleep 60; done
for ym in 2025-02 2025-03 2025-04 2025-05 2025-06 2025-08 2025-09 2025-10 2025-11 2025-12; do
  for d in {01..31}; do echo "$ym-$d"; done
done | xargs -P ${P:-4} -n 1 ./scripts/adsb_one_day.sh
echo "[$(date +%H:%M)] rest done"

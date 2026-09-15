#!/bin/zsh
# E18: download adsb.lol aircraft data links (ACARS/VDL2/HFDL, CC0) for given days into data/acars/<day>/.
#   scripts/get_acars_days.sh 2026-07-02 2026-07-20 2025-12-15
set -uo pipefail
cd "${0:a:h}/.."
for d in "$@"; do
  repo=adsblol/aircraft-data-links-2026
  [[ "$d" < "2026-05-14" ]] && repo=adsblol/aircraft-data-links-2025
  mkdir -p data/acars/$d
  echo "[$(date +%H:%M)] $d from $repo"
  gh release download adsblol-adl-$d -R $repo -D data/acars/$d --skip-existing || echo "FAILED $d"
  du -sh data/acars/$d
done
echo "[$(date +%H:%M)] done"

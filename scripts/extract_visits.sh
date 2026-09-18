#!/bin/zsh
# E64: visit features from the airborne points already on disk. Nothing is downloaded, nothing existing is touched.
set -euo pipefail
cd "${0:a:h}/.."
P=${1:-6}
ls data/adsblol/surface_*.parquet | sed 's/.*surface_//;s/\.parquet//' | sort > /tmp/visit_days.txt
echo "[$(date +%H:%M)] $(wc -l < /tmp/visit_days.txt) days, parallelism $P"
xargs -P $P -n 1 -I{} zsh -c '[[ -f data/adsblol/visit_{}.parquet ]] || .venv/bin/python src/adsb_visit.py {} >/dev/null 2>&1 || echo "FAILED {}"' < /tmp/visit_days.txt
echo "[$(date +%H:%M)] done: $(ls data/adsblol/visit_*.parquet | wc -l | tr -d " ") files"

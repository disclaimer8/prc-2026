#!/bin/zsh
set -euo pipefail
cd "${0:a:h}/.."
ls data/adsblol/surface_*.parquet | sed 's/.*surface_//;s/\.parquet//' | sort > /tmp/ob5_days.txt
echo "[$(date +%H:%M)] $(wc -l < /tmp/ob5_days.txt) days"
xargs -P ${1:-6} -n 1 -I{} .venv/bin/python src/adsb_offblock_v5.py {} < /tmp/ob5_days.txt >/dev/null 2>&1
echo "[$(date +%H:%M)] done"

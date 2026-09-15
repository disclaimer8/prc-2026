#!/bin/zsh
# E25: wait for the rest-of-2025 download, then the pre-registered 6-fold CV of the ADS-B second stage. Log: data/e25.log
cd "${0:a:h}/.."
until grep -q "rest done" data/adsb_rest.log 2>/dev/null; do sleep 120; done
echo "[$(date +%H:%M)] 2025 offblock files: $(ls data/adsblol/offblock2_2025-*.parquet | wc -l)"
.venv/bin/python src/e25_adsb_stage.py cv 2>&1 | grep -v Warning
echo "[$(date +%H:%M)] done"

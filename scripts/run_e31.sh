#!/bin/zsh
# E31: offblock4 (time-fallback matching) for every day with offblock2, then stage CV against E25. Log: data/e31.log
cd "${0:a:h}/.."
until grep -q "fill done" data/adsb_fill.log 2>/dev/null; do sleep 60; done
for f in data/adsblol/offblock2_*.parquet; do d=${${f:t:r}#offblock2_}; [[ -f data/adsblol/offblock4_$d.parquet ]] || echo $d; done | xargs -P 6 -n 1 .venv/bin/python src/adsb_offblock_v4.py 2>&1 | grep -E "Error|Traceback" | head
echo "[$(date +%H:%M)] offblock4 files: $(ls data/adsblol/offblock4_*.parquet | wc -l)"
.venv/bin/python src/e31_adsb_time_match.py cv 2>&1 | grep -v Warning
echo "[$(date +%H:%M)] done"

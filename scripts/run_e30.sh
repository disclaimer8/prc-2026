#!/bin/zsh
# E30: offblock3 features for every day with a surface file, then the stage CV against E25. Log: data/e30.log
cd "${0:a:h}/.."
until grep -q "fill done" data/adsb_fill.log 2>/dev/null; do sleep 60; done
for f in data/adsblol/surface_*.parquet; do d=${${f:t:r}#surface_}; [[ -f data/adsblol/offblock3_$d.parquet ]] || echo $d; done | xargs -P 6 -n 1 .venv/bin/python src/adsb_offblock_v3.py 2>&1 | grep -E "Error|Traceback" | head
echo "[$(date +%H:%M)] offblock3 files: $(ls data/adsblol/offblock3_*.parquet | wc -l)"
.venv/bin/python src/e30_adsb_stage_v2.py cv 2>&1 | grep -v Warning
echo "[$(date +%H:%M)] done"

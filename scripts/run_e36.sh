#!/bin/zsh
# E36: stand-geometry surface features for every day with a surface file (and offblock3 for days added by the fill), then CV.
cd "${0:a:h}/.."
until grep -q "fill done" data/adsb_fill2.log 2>/dev/null; do sleep 60; done
for f in data/adsblol/surface_*.parquet; do d=${${f:t:r}#surface_}; [[ -f data/adsblol/offblock3_$d.parquet ]] || echo $d; done | xargs -P 6 -n 1 .venv/bin/python src/adsb_offblock_v3.py 2>&1 | grep -E "Error|Traceback" | head
for f in data/adsblol/surface_*.parquet; do d=${${f:t:r}#surface_}; [[ -f data/adsblol/offblock5_$d.parquet ]] || echo $d; done | xargs -P 6 -n 1 .venv/bin/python src/adsb_offblock_v5.py 2>&1 | grep -E "Error|Traceback" | head
echo "[$(date +%H:%M)] offblock2/3/5 files: $(ls data/adsblol/offblock2_*.parquet | wc -l) $(ls data/adsblol/offblock3_*.parquet | wc -l) $(ls data/adsblol/offblock5_*.parquet | wc -l)"
.venv/bin/python src/e36_adsb_stand_geometry.py cv 2>&1 | grep -v Warning
echo "[$(date +%H:%M)] done"

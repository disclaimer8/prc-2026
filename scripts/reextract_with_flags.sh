#!/bin/zsh
# E44: re-extract surface points WITH the readsb staleness flag (src/adsb_extract.py, 16.09). One day at a time, streamed;
# the old flagless file is replaced only after the new one is written.  scripts/reextract_with_flags.sh 2025-01 2025-02 ...
cd "${0:a:h}/.."
# A build's ADS-B stage reads every offblock day at once, and this loop deletes a day's files for a minute or
# two while rebuilding them. Three times on 16.09 the two nearly overlapped. So: hold while data/.adsb_lock
# exists (a build creates it), and check before every day rather than once at the start.
wait_for_lock() {
  while [ -e data/.adsb_lock ]; do
    echo "[$(date +%H:%M)] data/.adsb_lock present — holding"
    sleep 60
  done
}

for ym in "$@"; do
  for f in data/adsblol/surface_$ym-*.parquet; do
    wait_for_lock
    [[ -e $f ]] || continue
    d=${${f:t:r}#surface_}
    .venv/bin/python -c "
import polars as pl,sys
print('has_flag' if 'stale' in pl.read_parquet_schema('$f') else 'old')" | grep -q has_flag && continue
    mv $f data/adsblol/.old_surface_$d.parquet
    if ./scripts/get_adsb_day.sh $d > data/adsblol/log_$d.txt 2>&1 && [[ -f data/adsblol/surface_$d.parquet ]]; then
      rm -f data/adsblol/.old_surface_$d.parquet data/adsblol/offblock2_$d.parquet data/adsblol/offblock3_$d.parquet data/adsblol/offblock5_$d.parquet
      .venv/bin/python src/adsb_offblock_v2.py $d >/dev/null 2>&1
      .venv/bin/python src/adsb_offblock_v3.py $d >/dev/null 2>&1
      .venv/bin/python src/adsb_offblock_v5.py $d >/dev/null 2>&1
      echo "[$(date +%H:%M)] $d re-extracted"
    else
      mv data/adsblol/.old_surface_$d.parquet $f
      echo "[$(date +%H:%M)] $d FAILED, old file kept"
    fi
  done
done
echo "[$(date +%H:%M)] reextract done"

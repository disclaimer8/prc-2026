#!/bin/zsh
# Build per-departure staleness features for every day whose surface file already carries the readsb `stale`
# flag, and keep pace with the re-extraction instead of waiting for it to finish. Idempotent: a day already
# done is skipped, a day not yet re-extracted is skipped until it is.
#   scripts/stale_features_loop.sh [passes]      default 200 passes with a minute between them
cd "${0:a:h}/.."
passes=${1:-200}
for i in $(seq 1 $passes); do
  built=0
  for f in data/adsblol/surface_*.parquet; do
    d=${${f:t:r}#surface_}
    [[ -f data/adsblol/stale_$d.parquet ]] && continue
    .venv/bin/python -c "
import polars as pl, sys
sys.exit(0 if 'stale' in pl.read_parquet_schema('$f') else 1)" || continue
    .venv/bin/python src/adsb_stale_features.py $d >/dev/null 2>&1 && built=$((built+1))
  done
  n=$(ls data/adsblol/stale_*.parquet 2>/dev/null | wc -l | tr -d " ")
  echo "[$(date +%H:%M)] pass $i: built $built, total $n"
  [[ $built -eq 0 ]] && sleep 60
done

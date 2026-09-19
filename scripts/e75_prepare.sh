#!/bin/zsh
# E75: the visit window is cut at midnight (adsb_visit.py reads only surface_<day>, the lookback is 12 h).
# Re-extract visit_* and visit_m_* with yesterday's points in the pool, into a separate directory so both
# arms stay runnable. Airframe identification stays on the current day -- that is a different experiment.
#
#   zsh scripts/e75_prepare.sh
#   VISIT_DIR=data/adsblol_visitprev  .venv/bin/python src/e74_onset_plus_stack.py     # candidate arm
#
# Waits for data/.adsb_lock: the MB3 build reads every day and must not race this.
set -u
cd "$(dirname "$0")/.."
OUT=data/adsblol_visitprev
say() { print -r -- "[$(date +%H:%M)] $*" }

while [[ -e data/.adsb_lock ]]; do say "data/.adsb_lock held, waiting 120 s"; sleep 120; done
say "lock clear"

mkdir -p "$OUT"
days=(${(f)"$(ls data/adsblol/surface_*.parquet | sed 's#.*/surface_##; s#\.parquet##' | sort)"})
say "${#days} days -> $OUT"

n=0
for d in $days; do
  n=$((n+1))
  if [[ -s "$OUT/visit_${d}.parquet" && -s "$OUT/visit_m_${d}.parquet" ]]; then continue; fi
  VISIT_PREV_DAY=1 ADSB_OUT_DIR=$OUT PYTHONPATH=src .venv/bin/python src/adsb_visit.py "$d" >> data/e75.log 2>&1 \
    || { say "FAILED visit $d"; exit 1; }
  VISIT_PREV_DAY=1 ADSB_OUT_DIR=$OUT PYTHONPATH=src .venv/bin/python src/e69_visit_merge.py "$d" >> data/e75.log 2>&1 \
    || { say "FAILED visit_m $d"; exit 1; }
  (( n % 25 == 0 )) && say "$n/${#days}"
done

got_v=$(ls "$OUT"/visit_2*.parquet 2>/dev/null | wc -l | tr -d ' ')
got_m=$(ls "$OUT"/visit_m_*.parquet 2>/dev/null | wc -l | tr -d ' ')
say "done: visit=$got_v visit_m=$got_m of ${#days}"
[[ "$got_v" -eq "${#days}" && "$got_m" -eq "${#days}" ]] || { say "REFUSING to judge: incomplete extraction"; exit 1; }

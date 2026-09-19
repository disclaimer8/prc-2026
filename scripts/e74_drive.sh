#!/bin/zsh
# Wait for the 1 kt stand-walk extraction, then run E74.
set -uo pipefail
cd "${0:a:h}/.."
for i in $(seq 1 120); do
  n=$(ls data/adsblol_sw_kt1/offblock5_*.parquet 2>/dev/null | wc -l | tr -d ' ')
  [[ $n -ge 420 ]] && break
  sleep 60
done
n=$(ls data/adsblol_sw_kt1/offblock5_*.parquet 2>/dev/null | wc -l | tr -d ' ')
if [[ $n -lt 420 ]]; then echo "[$(date +%H:%M)] stand-walk stuck at $n days — stopping"; exit 1; fi
echo "[$(date +%H:%M)] stand-walk done ($n days), running E74"
.venv/bin/python src/e74_onset_plus_stack.py 2>&1 | grep -v Warning | tee data/e74_cv.log

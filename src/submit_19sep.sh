#!/bin/zsh
# Upload queue for 19.09: waits for 2026-09-19 00:05 UTC and uploads the given versions one by one, waiting for each score.
#   src/submit_19sep.sh 75 ...          log: data/submit_19sep.log
# The counter reads only this day's log (src/submit_slot.sh counted every log ever written, i.e. all-time uploads).
cd "${0:a:h}/.."
target=$(date -u -j -f "%Y-%m-%d %H:%M" "2026-09-19 00:05" +%s)
while [ "$(date -u +%s)" -lt "$target" ]; do sleep 60; done
for v in "$@"; do
  n=$(grep -oE "^\[[0-9:]+Z\] uploaded v[0-9]+" data/submit_19sep.log 2>/dev/null | sort -u | wc -l | tr -d " ")
  if [ "$n" -ge 5 ]; then echo "[$(date -u +%H:%M)Z] limit reached, v$v NOT uploaded"; exit 1; fi
  [ -f submissions/gentle-igloo_v$v.parquet ] || { echo "[$(date -u +%H:%M)Z] v$v missing"; continue; }
  printf 'FINAL submissions/gentle-igloo_v%s.parquet\n' $v > data/ready_v$v.log
  ./src/submit_when_ready.sh $v data/ready_v$v.log
done
echo "[$(date -u +%H:%M)Z] queue finished"

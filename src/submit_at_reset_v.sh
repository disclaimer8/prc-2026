#!/bin/zsh
# Wait for the daily limit reset (00:05 UTC next day), then upload the given versions one by one and wait for scores.
#   src/submit_at_reset_v.sh 29 ...     log: data/submit_16sep.log
cd "${0:a:h}/.."
target=$(date -u -j -f "%Y-%m-%d %H:%M" "$(date -u -v+1d +%Y-%m-%d) 00:05" +%s)
while [ "$(date -u +%s)" -lt "$target" ]; do sleep 60; done
for v in "$@"; do
  printf 'FINAL submissions/gentle-igloo_v%s.parquet\n' $v > data/ready_v$v.log
  ./src/submit_when_ready.sh $v data/ready_v$v.log
done

#!/bin/zsh
# Upload one finished submission within the 16.09 daily limit: waits for 2026-09-16 00:05 UTC, serialises uploads with a lock,
# refuses if 5 uploads are already logged for 16.09, then waits for the score. Log: data/submit_16sep.log
#   src/submit_slot.sh 30
cd "${0:a:h}/.."
v=$1
target=$(date -u -j -f "%Y-%m-%d %H:%M" "2026-09-16 00:05" +%s)
while [ "$(date -u +%s)" -lt "$target" ]; do sleep 60; done
until mkdir data/.submit_lock 2>/dev/null; do sleep 20; done
trap 'rmdir data/.submit_lock' EXIT
n=$(grep -c "uploaded v" data/submit_16sep.log 2>/dev/null || echo 0)
if [ "$n" -ge 5 ]; then echo "[$(date -u +%H:%M)Z] limit reached, v$v NOT uploaded"; exit 1; fi
[ -f submissions/gentle-igloo_v$v.parquet ] || { echo "[$(date -u +%H:%M)Z] v$v missing"; exit 1; }
printf 'FINAL submissions/gentle-igloo_v%s.parquet\n' $v > data/ready_v$v.log
./src/submit_when_ready.sh $v data/ready_v$v.log

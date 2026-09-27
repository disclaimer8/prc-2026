#!/bin/zsh
# 28.09 slots: v404 (E141) -> v414 (E142) -> v447 (E144, + E143) -> v448 (E145) -> v314 (E81 pair with v316). Journal #109, #119.
set -uo pipefail
cd ~/prc-2026
LOG=data/submit_28sep.log
while [ "$(date -u +%F)" = "2026-09-27" ]; do sleep 60; done
sleep 300
upload_and_wait () {
  local v=$1
  echo "[$(date -u +%H:%MZ)] upload v$v" >> $LOG
  UPLOAD_REF=submissions/gentle-igloo_v366.parquet .venv/bin/python src/upload.py $v 2>&1 | grep -v -i warn | cut -c1-120 >> $LOG
  for i in {1..30}; do
    sleep 120
    s=$(.venv/bin/python src/leaderboard.py 2>/dev/null | grep "v$v\.parquet")
    if [ -n "$s" ]; then echo "[$(date -u +%H:%MZ)] SCORE $s" >> $LOG; break; fi
  done
}
for v in 404 414 447 448 314; do upload_and_wait $v; done
echo "[$(date -u +%H:%MZ)] end" >> $LOG

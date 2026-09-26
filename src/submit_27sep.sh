#!/bin/zsh
# 27.09 slots: v366 (prefix + CatBoost seeds) -> v376 (+ TabPFN 5-seed ensemble) -> v386 (+ local_change) -> v306 (prefix + local_change, no CatBoost seeds); 5th slot reserved.
set -uo pipefail
cd ~/prc-2026
LOG=data/submit_27sep.log
while [ "$(date -u +%F)" = "2026-09-26" ]; do sleep 60; done
sleep 300
upload_and_wait () {
  local v=$1
  echo "[$(date -u +%H:%MZ)] upload v$v" >> $LOG
  UPLOAD_REF=submissions/gentle-igloo_v316.parquet .venv/bin/python src/upload.py $v 2>&1 | grep -v -i warn | cut -c1-120 >> $LOG
  for i in {1..30}; do
    sleep 120
    s=$(.venv/bin/python src/leaderboard.py 2>/dev/null | grep "v$v\.parquet")
    if [ -n "$s" ]; then echo "[$(date -u +%H:%MZ)] SCORE $s" >> $LOG; break; fi
  done
}
for v in 366 376 386 306; do upload_and_wait $v; done
echo "[$(date -u +%H:%MZ)] end" >> $LOG

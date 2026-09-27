#!/bin/zsh
# 28.09 slots: v304 (both TabPFN) -> v314 (pfx TabPFN) -> v324 (local_change TabPFN) -> v334 (control TabPFN)
# Slot 5: E134/E140 candidate if built on hetzner (add manually before 00:05 UTC)
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
for v in 304 314 324 334; do upload_and_wait $v; done
echo "[$(date -u +%H:%MZ)] end" >> $LOG

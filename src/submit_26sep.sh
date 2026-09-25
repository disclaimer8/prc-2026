#!/bin/zsh
# 26.09 slot 1: wait for the UTC date change, upload v265 (E120), then poll the board for its score (<= 60 min).
set -uo pipefail
cd ~/prc-2026
while [ "$(date -u +%F)" = "2026-09-25" ]; do sleep 60; done
sleep 300
echo "[$(date -u +%H:%MZ)] upload v265" >> data/submit_26sep.log
UPLOAD_REF=submissions/gentle-igloo_v247.parquet .venv/bin/python src/upload.py 265 2>&1 | grep -v -i warn | cut -c1-120 >> data/submit_26sep.log
for i in {1..30}; do
  sleep 120
  s=$(.venv/bin/python src/leaderboard.py 2>/dev/null | grep 'v265\.parquet')
  if [ -n "$s" ]; then echo "[$(date -u +%H:%MZ)] SCORE $s" >> data/submit_26sep.log; break; fi
done
echo "[$(date -u +%H:%MZ)] end" >> data/submit_26sep.log

#!/bin/zsh
# 26.09 slot 2: after the v265 upload has been scored (or by 01:10 UTC at the latest), upload v248 = v247 + LOBT window clip.
set -uo pipefail
cd ~/prc-2026
until grep -qE 'SCORE|end' data/submit_26sep.log 2>/dev/null || [ "$(date -u +%F_%H%M)" \> "2026-09-26_0110" ]; do sleep 120; done
echo "[$(date -u +%H:%MZ)] upload v248" >> data/submit_26sep.log
UPLOAD_REF=submissions/gentle-igloo_v247.parquet .venv/bin/python src/upload.py 248 2>&1 | grep -v -i warn | cut -c1-120 >> data/submit_26sep.log
for i in {1..30}; do
  sleep 120
  s=$(.venv/bin/python src/leaderboard.py 2>/dev/null | grep 'v248\.parquet')
  if [ -n "$s" ]; then echo "[$(date -u +%H:%MZ)] SCORE $s" >> data/submit_26sep.log; break; fi
done
echo "[$(date -u +%H:%MZ)] end-b" >> data/submit_26sep.log

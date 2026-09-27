#!/bin/zsh
# 28.09 slot 5: v336 (control TabPFN + E81 + LOBT lift), pairs with v334; starts after the main queue writes "end".
set -uo pipefail
cd ~/prc-2026
LOG=data/submit_28sep.log
until grep -qE '^\[.*\] end$' $LOG 2>/dev/null; do sleep 120; done
v=336
echo "[$(date -u +%H:%MZ)] upload v$v (slot 5)" >> $LOG
UPLOAD_REF=submissions/gentle-igloo_v366.parquet .venv/bin/python src/upload.py $v 2>&1 | grep -v -i warn | cut -c1-120 >> $LOG
for i in {1..30}; do
  sleep 120
  s=$(.venv/bin/python src/leaderboard.py 2>/dev/null | grep "v$v\.parquet")
  if [ -n "$s" ]; then echo "[$(date -u +%H:%MZ)] SCORE $s" >> $LOG; break; fi
done
echo "[$(date -u +%H:%MZ)] end-5" >> $LOG

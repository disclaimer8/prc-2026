#!/bin/zsh
# 29.09 slots: v467 (15 CatBoost seeds) -> v477 -> v487 -> v497 (no-drift grid) -> v457 (no-drift, 10 seeds). Journal #126.
set -uo pipefail
cd ~/prc-2026
LOG=data/submit_29sep.log
while [ "$(date -u +%F)" = "2026-09-28" ]; do sleep 60; done
sleep 300
upload_and_wait () {
  local v=$1
  echo "[$(date -u +%H:%MZ)] upload v$v" >> $LOG
  UPLOAD_REF=submissions/gentle-igloo_v447.parquet .venv/bin/python src/upload.py $v 2>&1 | grep -v -i warn | cut -c1-120 >> $LOG
  for i in {1..30}; do
    sleep 120
    s=$(.venv/bin/python src/leaderboard.py 2>/dev/null | grep "v$v\.parquet")
    if [ -n "$s" ]; then echo "[$(date -u +%H:%MZ)] SCORE $s" >> $LOG; break; fi
  done
}
for v in 467 477 487 497 457; do upload_and_wait $v; done
echo "[$(date -u +%H:%MZ)] end" >> $LOG

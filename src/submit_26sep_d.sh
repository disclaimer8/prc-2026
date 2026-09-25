#!/bin/zsh
# 26.09 slots 2-5 after v265: v316 (E118 TabPFN prefix) -> v326 (E26a local-date change) -> v285 (E121 CatBoost seeds, if built) -> v249 (E133 rule + window clip).
# Each upload waits for the previous score (<= 60 min) so that the board attribution stays clean.
set -uo pipefail
cd ~/prc-2026
LOG=data/submit_26sep.log
until grep -qE 'SCORE .*v265|^\[.*\] end$' $LOG 2>/dev/null || [ "$(date -u +%F_%H%M)" \> "2026-09-26_0115" ]; do sleep 120; done
upload_and_wait () {
  local v=$1
  echo "[$(date -u +%H:%MZ)] upload v$v" >> $LOG
  UPLOAD_REF=submissions/gentle-igloo_v247.parquet .venv/bin/python src/upload.py $v 2>&1 | grep -v -i warn | cut -c1-120 >> $LOG
  for i in {1..30}; do
    sleep 120
    s=$(.venv/bin/python src/leaderboard.py 2>/dev/null | grep "v$v\.parquet")
    if [ -n "$s" ]; then echo "[$(date -u +%H:%MZ)] SCORE $s" >> $LOG; break; fi
  done
}
upload_and_wait 316
upload_and_wait 326
# v285 arrives from hetzner (E121); wait for it up to 04:30 UTC, then fall back to v248 first
until [ -f submissions/gentle-igloo_v285.parquet ] || [ "$(date -u +%F_%H%M)" \> "2026-09-26_0430" ]; do sleep 300; done
if [ -f submissions/gentle-igloo_v285.parquet ]; then upload_and_wait 285; fi
until [ -f submissions/gentle-igloo_v249.parquet ]; do sleep 60; done
upload_and_wait 249
echo "[$(date -u +%H:%MZ)] end-d" >> $LOG

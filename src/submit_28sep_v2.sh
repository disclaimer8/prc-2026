#!/bin/zsh
# 28.09 slots: v404 (E141 stage) -> v414 (E142, E141 x 5 stage seeds) -> v304 -> v314 -> v324 (E81 pairs with v306/v316/v326).
# If the hetzner build failed or a file is missing, v404 falls back to v334 and v414 to v336 (the fourth E81 pair).
set -uo pipefail
cd ~/prc-2026
LOG=data/submit_28sep.log
while [ "$(date -u +%F)" = "2026-09-27" ]; do sleep 60; done
sleep 300
built () {
  ssh hetzner 'cd /root/prc-2026 && [ ! -e data/e141b.failed ] && grep -q "\] done" data/e141b_build.log' 2>/dev/null || return 1
  [ -s submissions/gentle-igloo_v$1.parquet ] || scp -q hetzner:/root/prc-2026/submissions/gentle-igloo_v$1.parquet submissions/ 2>/dev/null
  [ -s submissions/gentle-igloo_v$1.parquet ]
}
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
if built 404; then upload_and_wait 404; else echo "[$(date -u +%H:%MZ)] v404 not built, fallback v334" >> $LOG; upload_and_wait 334; fi
if built 414; then upload_and_wait 414; else echo "[$(date -u +%H:%MZ)] v414 not built, fallback v336" >> $LOG; upload_and_wait 336; fi
for v in 304 314 324; do upload_and_wait $v; done
echo "[$(date -u +%H:%MZ)] end" >> $LOG

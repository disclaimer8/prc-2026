#!/bin/zsh
# 30.09 slots (journal 138): S1 v538 -> S2 v558 (E152) -> S3 v548 (E151 base) -> S4 v588 (S1 without E14; E153 v578 withdrawn, journal 141) -> S5 v927 (rebuilt E81 partners).
# Files come from hetzner once src/hz_builds_30sep.sh has finished (artifacts + process gone); a missing file is skipped and logged.
set -uo pipefail
cd ~/prc-2026
LOG=data/submit_30sep.log
until ssh hetzner "test -s /root/prc-2026/submissions/gentle-igloo_v588.parquet -o -e /root/prc-2026/data/build_noe14.failed" 2>/dev/null && ! ssh hetzner "pgrep -f \"[h]z_build_noe14\" > /dev/null" 2>/dev/null; do sleep 300; done
for v in 538 558 548 588; do
  scp -q hetzner:/root/prc-2026/submissions/gentle-igloo_v$v.parquet submissions/ 2>/dev/null && echo "[$(date -u +%H:%MZ)] got v$v" >> $LOG || echo "[$(date -u +%H:%MZ)] v$v missing on hetzner" >> $LOG
done
scp -q hetzner:/srv/prc-final-e2e/submissions/gentle-igloo_v927.parquet submissions/ 2>/dev/null && echo "[$(date -u +%H:%MZ)] got v927" >> $LOG
while [ "$(date -u +%F)" = "2026-09-29" ]; do sleep 60; done
sleep 300
upload_and_wait () {
  local v=$1 ref=$2
  [ -s submissions/gentle-igloo_v$v.parquet ] || { echo "[$(date -u +%H:%MZ)] skip v$v (missing)" >> $LOG; return; }
  echo "[$(date -u +%H:%MZ)] upload v$v" >> $LOG
  UPLOAD_REF=submissions/gentle-igloo_v$ref.parquet .venv/bin/python src/upload.py $v 2>&1 | grep -v -i warn | cut -c1-120 >> $LOG
  for i in {1..30}; do
    sleep 120
    s=$(.venv/bin/python src/leaderboard.py 2>/dev/null | grep "v$v\.parquet")
    if [ -n "$s" ]; then echo "[$(date -u +%H:%MZ)] SCORE $s" >> $LOG; break; fi
  done
}
for v in 538 558 548 588; do upload_and_wait $v 507; done
upload_and_wait 927 447
echo "[$(date -u +%H:%MZ)] end" >> $LOG

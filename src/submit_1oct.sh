#!/bin/zsh
# 1.10 slots (journal 166, 173): S1' v655 (T2 control) -> v729 (E159b LSZH on v588) -> v719 (E159b LSZH+EDDF on v588) -> S2' v695 (E130) -> S3' v685 (E157).
# Fallback for a missing hetzner file: E156 v665, then U3 v675. Files come from hetzner once src/hz_builds_1oct.sh has finished.
set -uo pipefail
cd ~/prc-2026
LOG=data/submit_1oct.log
until { ssh hetzner "test -s /root/prc-2026/submissions/gentle-igloo_v695.parquet -o -e /root/prc-2026/data/builds_1oct.failed" 2>/dev/null && ! ssh hetzner "pgrep -f \"[h]z_builds_1oct\" > /dev/null" 2>/dev/null; } || [ "$(date -u +%F)" != "2026-09-30" ]; do sleep 300; done
for v in 655 695 685 665 675; do
  scp -q hetzner:/root/prc-2026/submissions/gentle-igloo_v$v.parquet submissions/ 2>/dev/null && echo "[$(date -u +%H:%MZ)] got v$v" >> $LOG || echo "[$(date -u +%H:%MZ)] v$v missing on hetzner" >> $LOG
done
while [ "$(date -u +%F)" = "2026-09-30" ]; do sleep 60; done
sleep 300
upload_and_wait () {
  local v=$1 ref=$2
  [ -s submissions/gentle-igloo_v$v.parquet ] || { echo "[$(date -u +%H:%MZ)] skip v$v (missing)" >> $LOG; return 1; }
  echo "[$(date -u +%H:%MZ)] upload v$v" >> $LOG
  UPLOAD_REF=submissions/gentle-igloo_v$ref.parquet .venv/bin/python src/upload.py $v 2>&1 | grep -v -i warn | cut -c1-120 >> $LOG
  for i in {1..30}; do
    sleep 120
    s=$(.venv/bin/python src/leaderboard.py 2>/dev/null | grep "v$v\.parquet")
    if [ -n "$s" ]; then echo "[$(date -u +%H:%MZ)] SCORE $s" >> $LOG; break; fi
  done
}
n=0
for pair in "655 588" "729 588" "719 729" "695 655" "685 655"; do
  set -- ${=pair}
  upload_and_wait $1 $2 && n=$((n + 1))
done
[ $n -lt 5 ] && upload_and_wait 665 655; [ $n -lt 4 ] && upload_and_wait 675 655
echo "[$(date -u +%H:%MZ)] end ($n)" >> $LOG

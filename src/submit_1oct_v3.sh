#!/bin/zsh
# 1.10 slots v3 (journal 212, 30.09 23:05Z): v788 -> v808 (E171 + E172 sched member, CV -0.134 6/6 over v788), v788 becomes the first
# fallback. Previous header (journal 206): S1' v655 (T2 control) -> v729 (E159b LSZH on v588) -> v719 (E159b LSZH+EDDF on
# v588) -> v788 (E171: both E130 members, raw 0.17 + lobt 0.22, protected) -> v778 (E163 destination ATFM base, protected).
# Replaces src/submit_1oct.sh (v695 E130 raw -> v788; v665 E156 -> v778). Fallbacks: v695, then v665. All files are local.
set -uo pipefail
cd ~/prc-2026
LOG=data/submit_1oct.log
echo "[$(date -u +%H:%MZ)] queue v3 started (655 729 719 808 778; fallbacks 788 695)" >> $LOG
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
    if [ -n "$s" ]; then echo "[$(date -u +%H:%MZ)] SCORE $s" >> $LOG; return 0; fi
  done
  echo "[$(date -u +%H:%MZ)] v$v: no score after 60 min (counted as used)" >> $LOG
  return 0
}
n=0
for pair in "655 588" "729 588" "719 729" "808 655" "778 655"; do
  set -- ${=pair}
  upload_and_wait $1 $2 && n=$((n + 1))
done
[ $n -lt 5 ] && upload_and_wait 788 655 && n=$((n + 1))
[ $n -lt 5 ] && upload_and_wait 695 655 && n=$((n + 1))
echo "[$(date -u +%H:%MZ)] end ($n)" >> $LOG

#!/bin/zsh
# 2.10 slots (journal 216, 217): v819 (C2 = E163 + three members x2/3 + E167) -> v829 (C3: + eobt, four members x2/3) -> v839 (C3w:
# four members x1) -> v840 (C3w + E159b LSZH, built on the Mac) -> v859 (C4: five members x1 incl. aobt). Fallback: v830 (C3 + LSZH).
# hetzner files are fetched once its C4 build is ready (or failed), at the latest at 23:40Z on 1.10.
set -uo pipefail
cd ~/prc-2026
LOG=data/submit_2oct.log
echo "[$(date -u +%H:%MZ)] queue started (819 829 839 840 859; fallback 830)" >> $LOG
until ssh hetzner "test -s /root/prc-2026/data/build_2oct_c4.ready -o -e /root/prc-2026/data/build_2oct_c4.failed" 2>/dev/null || [ "$(date -u +%H%M)" -ge 2340 ] || [ "$(date -u +%F)" != "2026-10-01" ]; do sleep 120; done
for v in 829 839 859; do
  [ -s submissions/gentle-igloo_v$v.parquet ] || { scp -q hetzner:/root/prc-2026/submissions/gentle-igloo_v$v.parquet submissions/ 2>/dev/null && echo "[$(date -u +%H:%MZ)] got v$v" >> $LOG || echo "[$(date -u +%H:%MZ)] v$v missing on hetzner" >> $LOG; }
done
while [ "$(date -u +%F)" = "2026-10-01" ]; do sleep 60; done
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
for pair in "819 808" "829 819" "839 829" "840 839" "859 839"; do
  set -- ${=pair}
  upload_and_wait $1 $2 && n=$((n + 1))
done
[ $n -lt 5 ] && upload_and_wait 830 829 && n=$((n + 1))
echo "[$(date -u +%H:%MZ)] end ($n)" >> $LOG

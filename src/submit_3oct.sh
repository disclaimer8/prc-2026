#!/bin/zsh
# 3.10 slots (journal 242): v939 (S1, six members x2/3) -> v949 (S2, MLP .25) -> v959 (S3, FIN1b base) -> v969 (S4, E169) -> v979 (S5, CatBoost FIN1b).
# hetzner files are fetched once hz_queue/80_builds_3oct.sh is ready (or failed), at the latest at 23:40Z on 2.10.
set -uo pipefail
cd ~/prc-2026
LOG=data/submit_3oct.log
echo "[$(date -u +%H:%MZ)] queue started (939 949 959 969 979; fallback none)" >> $LOG
until ssh hetzner "test -s /root/prc-2026/data/builds_3oct.ready -o -e /root/prc-2026/data/builds_3oct.failed" 2>/dev/null || [ "$(date -u +%H%M)" -ge 2340 ] || [ "$(date -u +%F)" != "2026-10-02" ]; do sleep 120; done
for v in 939 949 959 969 979; do
  [ -s submissions/gentle-igloo_v$v.parquet ] || { scp -q hetzner:/root/prc-2026/submissions/gentle-igloo_v$v.parquet submissions/ 2>/dev/null && echo "[$(date -u +%H:%MZ)] got v$v" >> $LOG || echo "[$(date -u +%H:%MZ)] v$v missing on hetzner" >> $LOG; }
done
while [ "$(date -u +%F)" = "2026-10-02" ]; do sleep 60; done
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
for pair in "939 899" "949 939" "959 939" "969 939" "979 959"; do
  set -- ${=pair}
  upload_and_wait $1 $2 && n=$((n + 1))
done
echo "[$(date -u +%H:%MZ)] end ($n)" >> $LOG

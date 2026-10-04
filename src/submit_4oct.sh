#!/bin/zsh
# 4.10 slots (journal 247): v1109 (T1, E181a MLP x2/3) -> v1119 (T2, E181a MLP x1) -> v1129 (T3, CatBoost 15 seeds) -> v1139 (T4, E169) -> v1019 (e2e).
# Mac: waits for the E181a ranking MLP (src/e176_mlp_rank.py), hands it to hetzner, waits for hz_queue/95_builds_4oct.sh, uploads the same day.
set -uo pipefail
cd ~/prc-2026
LOG=data/submit_4oct.log
echo "[$(date -u +%H:%MZ)] queue started (1109 1119 1129 1139 1019; fallback none)" >> $LOG
until grep -q "^rc=" data/e181a_mlp_rank.log 2>/dev/null; do sleep 30; done
if grep -q "^rc=0" data/e181a_mlp_rank.log && [ -s submissions/e130_mlpa_rank.npz ]; then
  scp -q submissions/e130_mlpa_rank.npz hetzner:/root/prc-2026/submissions/ && ssh hetzner "date > /root/prc-2026/data/e130_mlpa_rank.ready" \
    && echo "[$(date -u +%H:%MZ)] E181a ranking MLP on hetzner: $(grep '^wrote' data/e181a_mlp_rank.log)" >> $LOG
else
  echo "[$(date -u +%H:%MZ)] E181a ranking MLP FAILED, stop" >> $LOG; exit 1
fi
until ssh hetzner "test -s /root/prc-2026/data/builds_4oct.ready -o -e /root/prc-2026/data/builds_4oct.failed" 2>/dev/null; do sleep 60; done
for v in 1109 1119 1129 1139 1019; do
  [ -s submissions/gentle-igloo_v$v.parquet ] || { scp -q hetzner:/root/prc-2026/submissions/gentle-igloo_v$v.parquet submissions/ 2>/dev/null && echo "[$(date -u +%H:%MZ)] got v$v" >> $LOG || echo "[$(date -u +%H:%MZ)] v$v missing on hetzner" >> $LOG; }
done
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
for pair in "1109 979" "1119 1109" "1129 1119" "1139 1129" "1019 808"; do
  set -- ${=pair}
  upload_and_wait $1 $2 && n=$((n + 1))
done
echo "[$(date -u +%H:%MZ)] end ($n)" >> $LOG

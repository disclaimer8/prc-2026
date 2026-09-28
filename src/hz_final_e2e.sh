#!/bin/bash
# End-to-end test of run_final_any_period.sh on the board period (journal 130): it must reproduce the v447 line.
#   base v900 = v240, blend v901 = v441 (the 10 CatBoost seeds are copied in from the mac: seeds_v440; training them
#   is not what is under test), tail v912..v917 = v442..v447 with E81=reuse, then a second tail v922..v927 with
#   E81=rebuild (v44/v60 partners rebuilt from scratch; TabPFN inside them is not deterministic run to run).
#   Two days (2026-01-15, 2026-07-20) go through the real download + extraction on hetzner; the other 60 are linked from MAIN.
# Run: nohup nice -n 19 taskset -c 6-11 bash /root/prc-2026/src/hz_final_e2e.sh > /root/prc-2026/data/final_e2e.log 2>&1 &
set -Eeuo pipefail
M=/root/prc-2026; W=/srv/prc-final-e2e
mkdir -p $W/submissions $W/data/adsblol $W/data/adsblol_kt1 $W/data/adsblol_sw_kt1
[[ -s $W/submissions/cat_reg_rank_seeds_v900.npz ]] || { echo "copy seeds_v440 from the mac first" >&2; exit 1; }
[[ -s $W/submissions/cat_reg_rank_v900.npz ]] || cp $M/submissions/cat_reg_rank_v240.npz $W/submissions/cat_reg_rank_v900.npz
for f in $M/data/adsblol/{surface,offblock2,offblock3,offblock5,visit,visit_m,head}_2026-*.parquet \
         $M/data/adsblol_kt1/offblock[235]_2026-*.parquet $M/data/adsblol_sw_kt1/offblock[235]_2026-*.parquet; do
  case $f in *2026-01-15*|*2026-07-20*) continue;; esac
  ln -sf $f $W/data/$(basename $(dirname $f))/
done
export RANKING=$M/data/prc-2026-datasets/ranking.parquet SUBMITTING=$M/data/prc-2026-datasets/submitting.parquet
export TAG=e2e WORK=$W VB=900 CAT_SEEDS=10 TABPFN_ROWS_FROM=361:291
echo "[$(date -u +%H:%MZ)] tail 1: E81=reuse"
VT=910 E81=reuse bash $M/src/run_final_any_period.sh
echo "[$(date -u +%H:%MZ)] tail 2: E81=rebuild"
VT=920 E81=rebuild bash $M/src/run_final_any_period.sh
echo "[$(date -u +%H:%MZ)] e2e done"

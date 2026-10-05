#!/usr/bin/env bash
# Registered E182 ranking, home shard: raw/lobt/sched, five seeds each.
set -euo pipefail
cd /home/a1/prc-2026
exec 9>data/e182_home_ranking_20261005.lock
flock -n 9 || { echo 'E182 ranking runner already active'; exit 2; }
test ! -e data/e182_home_ranking_20261005.ready
trap 'touch data/e182_home_ranking_20261005.failed' ERR
for spec in raw:2565 lobt:3065 sched:2958; do
  mode=${spec%:*}
  rounds=${spec#*:}
  echo "$(date -u +%FT%TZ) E182 ranking $mode, 5 seeds, $rounds rounds"
  env -i PATH=/usr/bin:/bin \
    POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
    OMP_NUM_THREADS=24 POLARS_MAX_THREADS=24 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    FEATURES=data/features_fin1b.parquet PROXY_SCALE=1 \
    CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 REGIME_RATES=0 \
    DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config \
    E130_MODE="$mode" E130_OUT="submissions/e130_${mode}f_rank.npz" \
    .venv/bin/python -u src/e130_rank.py 5 "$rounds" \
    >"data/e182_home_${mode}f_rank.log" 2>&1
done
touch data/e182_home_ranking_20261005.ready
echo "$(date -u +%FT%TZ) All three home E182 ranking members complete"

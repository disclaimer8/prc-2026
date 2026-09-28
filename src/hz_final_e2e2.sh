#!/bin/bash
# Journal 130, follow-up (external review 28.09 21:33): the first e2e reused CatBoost seeds and TabPFN rows. This run makes
# the chain compute both itself: July 2026 only (a subset ranking, template built from its DEP rows, no SUBMITTING),
# CAT_SEEDS=1 (catboost_reg.py inside the chain), TabPFN run in the chain (no TABPFN_ROWS_FROM), E81=drop (the rebuild branch
# was tested). Two July days downloaded for real, the rest linked. Starts when tonight's hetzner queue is done.
set -Eeuo pipefail
M=/root/prc-2026; W=/srv/prc-final-e2e2
until [[ ( -s $M/submissions/gentle-igloo_v527.parquet || -e $M/data/e151v2.failed ) && ( -s $M/submissions/gentle-igloo_v517.parquet || -e $M/data/seeds20_control.failed ) ]]; do sleep 300; done
mkdir -p $W/data/adsblol $W/data/adsblol_kt1 $W/data/adsblol_sw_kt1
cd $M
[[ -s $W/ranking_jul.parquet ]] || POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage .venv/bin/python -c "
import polars as pl
r = pl.read_parquet('data/prc-2026-datasets/ranking.parquet')
j = r.filter(pl.col('MVT_TIME_UTC_mvt').dt.month() == 7); j.write_parquet('$W/ranking_jul.parquet')
print('July rows', j.height, 'DEP', j.filter(pl.col('PHASE_mvt') == 'DEP').height)"
for f in $M/data/adsblol/{surface,offblock2,offblock3,offblock5,visit,visit_m,head}_2026-07-*.parquet \
         $M/data/adsblol_kt1/offblock[235]_2026-07-*.parquet $M/data/adsblol_sw_kt1/offblock[235]_2026-07-*.parquet; do
  case $f in *2026-07-10*|*2026-07-25*) continue;; esac
  ln -sf $f $W/data/$(basename $(dirname $f))/
done
export RANKING=$W/ranking_jul.parquet TAG=e2e2 WORK=$W VB=900 VT=910 CAT_SEEDS=1 E81=drop
unset TABPFN_ROWS_FROM SUBMITTING
echo "[$(date -u +%H:%MZ)] e2e2 start"
bash $M/src/run_final_any_period.sh
.venv/bin/python -c "
import numpy as np, polars as pl
T = 'TAXITIME_SEC_mvt'
a = pl.read_parquet('$W/submissions/gentle-igloo_v917.parquet'); b = pl.read_parquet('submissions/gentle-igloo_v447.parquet')
j = a.join(b, on='MVT_ID_mvt', suffix='_ref'); d = (j[T] - j[T + '_ref']).cast(float).to_numpy()
print(f'e2e2 v917 (July) vs v447 on the same rows: n {j.height} of {a.height}, RMS {np.sqrt((d ** 2).mean()):.2f} s, mean {d.mean():+.2f} s, max |d| {np.abs(d).max():.0f} s')"
echo "[$(date -u +%H:%MZ)] e2e2 done"

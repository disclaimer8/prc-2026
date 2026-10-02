#!/bin/bash
# Checklist "end-to-end run" (journal 244): src/run_final_any_period.sh FROM SCRATCH on the board period in a fresh WORK on /srv --
# METAR append, ADS-B reuse, all feature steps + parity, validate/submit, LightGBM seeds, CatBoost (2 seeds), three members (1 seed),
# blend, T2 tail, protect. Small seed counts to fit a night; the final is compared with v808 (same recipe, 25 + 5 seeds), so a
# few-second RMS difference is expected and a crash or a large difference is the finding.
cd /root/prc-2026
rm -rf /srv/prc-final-e2e1
MAIN=/root/prc-2026 WORK=/srv/prc-final-e2e1 RANKING=/root/prc-2026/data/prc-2026-datasets/ranking.parquet \
  SUBMITTING=/root/prc-2026/data/prc-2026-datasets/submitting.parquet TAG=e2e1 VB=1000 CAT_SEEDS=2 MEMBER_SEEDS=1 \
  MEMBERS=raw:2678:0.13,lobt:2678:0.17,sched:2826:0.13 E167=0 TABPFN_ROWS_FROM=361:291 E81=reuse ADSB=reuse \
  bash src/run_final_any_period.sh > data/e2e1.log 2>&1; rc=$?
echo "[$(date +%H:%M)] e2e1 rc=$rc" >> data/e2e1.log
if [[ -s /srv/prc-final-e2e1/submissions/gentle-igloo_v1019.parquet ]]; then
  .venv/bin/python -c "
import numpy as np, polars as pl
r = lambda p: pl.read_parquet(p).sort('MVT_ID_mvt')['TAXITIME_SEC_mvt'].cast(float).to_numpy()
d = r('/srv/prc-final-e2e1/submissions/gentle-igloo_v1019.parquet') - r('submissions/gentle-igloo_v808.parquet')
print(f'e2e1 final vs v808: changed {(d != 0).sum()}, RMS {np.sqrt((d ** 2).mean()):.3f} s, mean {d.mean():+.3f} s, max |d| {np.abs(d).max():.0f}')
" >> data/e2e1.log 2>&1
fi

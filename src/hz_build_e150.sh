#!/bin/bash
# E150 board build (journal 127) on the 15-seed line: v463 -> v504 stage x5 keep-all -> v505 E64 x5 -> v506 E81 -> v507 LOBT
set -Eeuo pipefail
cd /root/prc-2026
export OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export FEATURES=data/features_e106.parquet PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
PY=.venv/bin/python
L=data/build_e150.log
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" >> $L' ERR
echo "[$(date +%H:%M)] start" >> $L
STAGE_KEEP_ALL=1 STAGE_SEEDS=42,1,2,3,4 ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v461.npz $PY src/e141_apply.py 463 504 2>&1 | grep -E "seeds|^wrote" >> $L
STAGE_SEEDS=42,1,2,3,4 ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v461.npz $PY src/e64_apply.py 504 505 2>&1 | grep -E "^wrote" >> $L
BLEND_BASE=505 BLEND_PARTS="44:0.1214,60:0.0270" $PY src/blend_submissions.py 506 2>&1 | grep -viE "warn" | tail -1 >> $L
LOBT_CLIP=1 $PY src/lobt_edge_lift.py 506 507 2>&1 | grep -viE "warn" | tail -1 >> $L
$PY -c "
import numpy as np, polars as pl
r = lambda v: pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet').sort('MVT_ID_mvt')['TAXITIME_SEC_mvt'].cast(float).to_numpy()
d = r(507) - r(467); print(f'v507 vs v467: changed {(d != 0).sum()}, RMS {np.sqrt((d ** 2).mean()):.2f} s, mean {d.mean():+.2f} s, max |d| {np.abs(d).max():.0f}')
" >> $L
echo "[$(date +%H:%M)] done" >> $L

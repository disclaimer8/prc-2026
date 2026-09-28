#!/bin/bash
# 29.09 candidates on the 15-seed line (v461 -> v463 = TabPFN rows of v361 + E14; journal 120, 122, 123):
#   v467 = E70 x5 + E64 x5 with all features (built overnight)
#   v474..v477: E70 x5 no-drift + E64 x5 no-drift | v484..v487: E70 no-drift + E64 all | v494..v497: E70 all (v464) + E64 no-drift
set -Eeuo pipefail
cd /root/prc-2026
export OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export FEATURES=data/features_e106.parquet PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
PY=.venv/bin/python
L=data/builds_29sep.log
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" >> $L' ERR
D7=n_pts,sf_n_seg,vi_n_pts,vi_prev_pts,vi_alt_first,vm_n_pts,vm_prev_pts
D3=vi_n_pts,vi_prev_pts,vi_alt_first
e70 () { STAGE_SEEDS=42,1,2,3,4 DROP_FEATS=$1 ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v461.npz $PY src/e141_apply.py $2 $3 2>&1 | grep -E "^wrote" >> $L; }
e64 () { STAGE_SEEDS=42,1,2,3,4 DROP_FEATS=$1 ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v461.npz $PY src/e64_apply.py $2 $3 2>&1 | grep -E "^wrote" >> $L; }
tail2 () { BLEND_BASE=$1 BLEND_PARTS="44:0.1214,60:0.0270" $PY src/blend_submissions.py $(($1 + 1)) 2>&1 | grep -viE "warn" | tail -1 >> $L
           LOBT_CLIP=1 $PY src/lobt_edge_lift.py $(($1 + 1)) $(($1 + 2)) 2>&1 | grep -viE "warn" | tail -1 >> $L; }
echo "[$(date +%H:%M)] start" >> $L
e70 $D7 463 474; e64 $D3 474 475; tail2 475
e70 $D7 463 484; e64 "" 484 485; tail2 485
cp submissions/gentle-igloo_v464.parquet submissions/gentle-igloo_v494.parquet; e64 $D3 494 495; tail2 495
$PY -c "
import numpy as np, polars as pl
r = lambda v: pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet').sort('MVT_ID_mvt')['TAXITIME_SEC_mvt'].cast(float).to_numpy()
for a, b in ((467, 447), (477, 467), (487, 467), (497, 467), (457, 447)):
    d = r(a) - r(b); print(f'v{a} vs v{b}: changed {(d != 0).sum()}, RMS {np.sqrt((d ** 2).mean()):.2f} s, mean {d.mean():+.2f} s')
" >> $L
echo "[$(date +%H:%M)] done" >> $L

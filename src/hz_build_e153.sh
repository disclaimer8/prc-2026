#!/bin/bash
# E153 (journal 141): S1 with E14 replaced by E153 (prefix-aware E14, 256 rows instead of 43) -> v578, for the 30.09 slot S4.
# Waits for src/hz_builds_30sep.sh (artifact v568 + process gone). Chain from v532 (TabPFN rows on v531) exactly as S1.
set -Eeuo pipefail
cd /root/prc-2026
export OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 FEATURES=data/features_e106.parquet
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
PY=.venv/bin/python
L=data/build_e153.log
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee -a $L data/build_e153.failed' ERR
until { [[ -s submissions/gentle-igloo_v568.parquet ]] || [[ -e data/builds_30sep.failed ]]; } && ! pgrep -f "[h]z_builds_30sep" > /dev/null; do sleep 120; done
D7=n_pts,sf_n_seg,vi_n_pts,vi_prev_pts,vi_alt_first,vm_n_pts,vm_prev_pts
D3=vi_n_pts,vi_prev_pts,vi_alt_first
echo "[$(date +%H:%M)] E153 line: v532 -> v573..v578" >> $L
$PY src/e153_apply.py 532 531 573 2>&1 | grep -E "^wrote" >> $L
STAGE_KEEP_ALL=1 STAGE_SEEDS=42,1,2,3,4 DROP_FEATS=$D7 ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v531.npz $PY src/e141_apply.py 573 574 2>&1 | grep -E "^wrote" >> $L
E64_KEEP_ALL=1 STAGE_SEEDS=42,1,2,3,4 DROP_FEATS=$D3 ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v531.npz $PY src/e64_apply.py 574 575 2>&1 | grep -E "^wrote" >> $L
cp submissions/gentle-igloo_v575.parquet submissions/gentle-igloo_v576.parquet
BLEND_BASE=576 BLEND_PARTS="44:0.1214,60:0.0270" $PY src/blend_submissions.py 577 2>&1 | grep -viE "warn" | tail -1 >> $L
LOBT_CLIP=1 $PY src/lobt_edge_lift.py 577 578 2>&1 | grep -viE "warn" | tail -1 >> $L
$PY -c "
import numpy as np, polars as pl
r = lambda v: pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet').sort('MVT_ID_mvt')['TAXITIME_SEC_mvt'].cast(float).to_numpy()
d = r(578) - r(538); print(f'v578 vs v538 (S1): changed {(d != 0).sum()}, RMS {np.sqrt((d ** 2).mean()):.2f} s, mean {d.mean():+.2f} s, max |d| {np.abs(d).max():.0f}')
s = pl.read_parquet('submissions/gentle-igloo_v578.parquet'); assert s.height == 344841 and s['TAXITIME_SEC_mvt'].null_count() == 0; print('row checks ok')" >> $L 2>&1
echo "[$(date +%H:%M)] done" >> $L

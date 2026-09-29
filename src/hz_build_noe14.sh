#!/bin/bash
# S4 30.09 (journal 141): S1 WITHOUT E14 -> v588. On the production OOF (cv_preds_e116prod) E14 is net harmful: +0.376 s summed
# over 6 folds (lower 3/6); on the 14.09 control it was -7.9. The production base seems to have absorbed it. Capacity removal.
# Chain as S1 from v532 (TabPFN rows on v531), E14 skipped. Waits for src/hz_builds_30sep.sh (v568 + process gone).
set -Eeuo pipefail
cd /root/prc-2026
export OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 FEATURES=data/features_e106.parquet
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
PY=.venv/bin/python
L=data/build_noe14.log
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee -a $L data/build_noe14.failed' ERR
until { [[ -s submissions/gentle-igloo_v568.parquet ]] || [[ -e data/builds_30sep.failed ]]; } && ! pgrep -f "[h]z_builds_30sep" > /dev/null; do sleep 120; done
D7=n_pts,sf_n_seg,vi_n_pts,vi_prev_pts,vi_alt_first,vm_n_pts,vm_prev_pts
D3=vi_n_pts,vi_prev_pts,vi_alt_first
echo "[$(date +%H:%M)] no-E14 line: v532 -> v584..v588" >> $L
STAGE_KEEP_ALL=1 STAGE_SEEDS=42,1,2,3,4 DROP_FEATS=$D7 ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v531.npz $PY src/e141_apply.py 532 584 2>&1 | grep -E "^wrote" >> $L
E64_KEEP_ALL=1 STAGE_SEEDS=42,1,2,3,4 DROP_FEATS=$D3 ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v531.npz $PY src/e64_apply.py 584 585 2>&1 | grep -E "^wrote" >> $L
cp submissions/gentle-igloo_v585.parquet submissions/gentle-igloo_v586.parquet
BLEND_BASE=586 BLEND_PARTS="44:0.1214,60:0.0270" $PY src/blend_submissions.py 587 2>&1 | grep -viE "warn" | tail -1 >> $L
LOBT_CLIP=1 $PY src/lobt_edge_lift.py 587 588 2>&1 | grep -viE "warn" | tail -1 >> $L
$PY -c "
import numpy as np, polars as pl
r = lambda v: pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet').sort('MVT_ID_mvt')['TAXITIME_SEC_mvt'].cast(float).to_numpy()
d = r(588) - r(538); print(f'v588 vs v538 (S1): changed {(d != 0).sum()}, RMS {np.sqrt((d ** 2).mean()):.2f} s, mean {d.mean():+.2f} s, max |d| {np.abs(d).max():.0f}')
s = pl.read_parquet('submissions/gentle-igloo_v588.parquet'); assert s.height == 344841 and s['TAXITIME_SEC_mvt'].null_count() == 0; print('row checks ok')" >> $L 2>&1
echo "[$(date +%H:%M)] done" >> $L

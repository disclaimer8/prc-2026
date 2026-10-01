#!/bin/bash
# E169 on the board (journal 220): the E141 ADS-B stage learns the residual to the production OOF (data/cv_preds_e116prod) instead of
# cv_preds_nomonth (E36_BASE_DIR, read by e36's oof_base at import). C5 = C4 up to E167 (v852) -> T2 with that stage -> protected v869;
# FIN1b variant = v872 -> T2 -> protected v889. E64 keeps its hard-coded nomonth base (E169 measured the E141 stage only).
set -Eeuo pipefail
cd /root/prc-2026
export OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
PY=.venv/bin/python
L=data/build_e169.log
rm -f data/build_e169.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee -a $L data/build_e169.failed' ERR
D7=n_pts,sf_n_seg,vi_n_pts,vi_prev_pts,vi_alt_first,vm_n_pts,vm_prev_pts
D3=vi_n_pts,vi_prev_pts,vi_alt_first
tabpfn_rows () {  # tabpfn_rows <base> <out> <donor>: the 42 TabPFN rows (ids = v361 vs v291), values from the donor (361 or 601)
  B=$1 V=$2 DONOR=$3 $PY - >> $L 2>&1 <<'PY'
import os, polars as pl
Y = "TAXITIME_SEC_mvt"; B, V, D = int(os.environ["B"]), int(os.environ["V"]), int(os.environ["DONOR"])
r = lambda v: pl.read_parquet(f"submissions/gentle-igloo_v{v}.parquet")
ids = r(361).join(r(291), on="MVT_ID_mvt", suffix="_b").filter(pl.col(Y) != pl.col(f"{Y}_b")).select("MVT_ID_mvt")
chg = r(D).join(ids, on="MVT_ID_mvt").select("MVT_ID_mvt", pl.col(Y).alias("tab"))
assert chg.height == 42, chg.height
s = r(B)
s.join(chg, on="MVT_ID_mvt", how="left").with_columns(pl.coalesce("tab", Y).alias(Y)).select(s.columns).write_parquet(f"submissions/gentle-igloo_v{V}.parquet")
print(f"wrote v{V}: v{B} + the {chg.height} TabPFN rows of v{D}")
PY
}
t2 () {  # t2 <base> <V> <donor>: V TabPFN rows, V+1 E70, V+2 E64 keep-all, V+3 E152 gate, V+4 E81, V+5 LOBT = FINAL
  local B=$1 V=$2
  echo "[$(date +%H:%M)] T2 on v$B (TabPFN rows of v$3) -> v$V..v$((V + 5))" >> $L
  export FEATURES=data/features_e106.parquet   # tail steps read their inputs from the E106 table (as hz_builds_30sep.sh)
  tabpfn_rows $B $V $3
  STAGE_KEEP_ALL=1 STAGE_SEEDS=42,1,2,3,4 DROP_FEATS=$D7 ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v$B.npz \
    $PY src/e141_apply.py $V $((V + 1)) 2>&1 | grep -E "keep all|^wrote" >> $L
  E64_KEEP_ALL=1 STAGE_SEEDS=42,1,2,3,4 DROP_FEATS=$D3 ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v$B.npz \
    $PY src/e64_apply.py $((V + 1)) $((V + 2)) 2>&1 | grep -E "keep all|^wrote" >> $L
  $PY src/e152_stage_gate.py $V $((V + 2)) $B $((V + 3)) 2>&1 | grep -viE "warn" >> $L
  BLEND_BASE=$((V + 3)) BLEND_PARTS="44:0.1214,60:0.0270" $PY src/blend_submissions.py $((V + 4)) 2>&1 | grep -viE "warn" | tail -1 >> $L
  LOBT_CLIP=1 $PY src/lobt_edge_lift.py $((V + 4)) $((V + 5)) 2>&1 | grep -viE "warn" | tail -1 >> $L
}
cmp () {
  $PY -c "
import sys, numpy as np, polars as pl
r = lambda v: pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet').sort('MVT_ID_mvt')['TAXITIME_SEC_mvt'].cast(float).to_numpy()
a = sys.argv[1:]
for x, y in zip(a[::2], a[1::2]):
    d = r(int(x)) - r(int(y)); print(f'v{x} vs v{y}: changed {(d != 0).sum()}, RMS {np.sqrt((d ** 2).mean()):.3f} s, mean {d.mean():+.3f} s, max |d| {np.abs(d).max():.0f}')
    s = pl.read_parquet(f'submissions/gentle-igloo_v{x}.parquet'); assert s.height == 344841 and s['TAXITIME_SEC_mvt'].null_count() == 0
" "$@" >> $L 2>&1
}
export E36_BASE_DIR=data/cv_preds_e116prod
echo "[$(date +%H:%M)] C5: C4 (v852) -> T2 with the stage on the production OOF" >> $L
t2 852 863 361
$PY src/protect_tabpfn_rows.py 868 863 869 2>&1 | grep -viE "warn" >> $L
cmp 869 859
if [[ -s submissions/gentle-igloo_v872.parquet && -s submissions/parts_v872.npz ]]; then
  echo "[$(date +%H:%M)] FIN1b (v872) -> T2 with the stage on the production OOF" >> $L
  t2 872 883 361
  $PY src/protect_tabpfn_rows.py 888 883 889 2>&1 | grep -viE "warn" >> $L
  cmp 889 879 889 869
fi
date > data/build_e169.ready
echo "[$(date +%H:%M)] done" >> $L

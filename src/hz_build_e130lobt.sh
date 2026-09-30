#!/bin/bash
# E130 lobt board build (journal 199): blend 240 with the LOBT-anchored raw regressor (w 0.30), T2 tail, TabPFN rows protected.
# -0.22 against S1 v538 235.023. New tail T2 = TabPFN rows -> E70 x5 keep-all no-drift -> E64 x5 keep-all -> E152 gate
# (against the pre-stage file, E14 dropped) -> E81 -> LOBT. One change per slot against S1':
#   S1' v655  base v531 (control), v361 TabPFN rows                    = no-E14 + E152 (never measured together)
#   S2' v695  base v631 = blend 240 with the E130 raw regressor (w 0.30, submissions/e130_raw_rank.npz from the Mac)
#   S3' v685  base v621 = E157 (features_e157: v620 = E116 recipe + 5 LightGBM seeds, blend with the 25 CatBoost seeds)
#   S4' v675  base v591 = U3 (E51 route geometry, built 29.09)
#   S5' v665  base v531, A1 TabPFN rows of v601 (E156)
# Waits for the E130 CV (hz_e130.sh) to free cores 6-11.
set -Eeuo pipefail
cd /root/prc-2026
export OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
PY=.venv/bin/python
L=data/build_e130lobt.log
while pgrep -f "[h]z_build_2oct.sh" > /dev/null || [[ ! -s data/e130_lobt_rank.ready ]]; do sleep 120; done
rm -f data/build_e130lobt.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee -a $L data/build_e130lobt.failed' ERR
D7=n_pts,sf_n_seg,vi_n_pts,vi_prev_pts,vi_alt_first,vm_n_pts,vm_prev_pts
D3=vi_n_pts,vi_prev_pts,vi_alt_first
echo "[$(date +%H:%M)] start" >> $L

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
cat_blend () {  # cat_blend <base> <lgb seeds npz> <features> <out> [extra env]: blend with the 25 CatBoost seeds (seeds_v530)
  cp -p submissions/cat_reg_rank.npz submissions/cat_reg_rank_before_1oct.npz
  cp -f submissions/cat_reg_rank_seeds_v530.npz submissions/cat_reg_rank.npz
  env FEATURES=$3 LGB_SEEDS=$2 ${5:-} $PY src/blend.py "$1" "$4" 0.65 > data/blend_v$4.log 2>&1 || { cp -p submissions/cat_reg_rank_before_1oct.npz submissions/cat_reg_rank.npz; return 1; }
  cp -p submissions/cat_reg_rank_before_1oct.npz submissions/cat_reg_rank.npz
  grep -E "^wrote|^E130" data/blend_v$4.log >> $L
}
cat_blend 240 submissions/lgb_reg_seeds_v240.npz data/features_e106.parquet 761 "E130_REG2=submissions/e130_lobt_rank.npz E130_W=0.30"
t2 761 762 361
$PY src/protect_tabpfn_rows.py 767 762 768 2>&1 | grep -viE "warn" >> $L
cmp 761 531 768 655 768 695
echo "[$(date +%H:%M)] done" >> $L

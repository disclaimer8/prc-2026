#!/bin/bash
# E171 board build (journal 204): both E130 members in the regular candidate (raw w 0.17 + lobt w 0.22 = CV weights 0.25 / 0.325
# x 2/3 for the 0.65 CatBoost + 0.35 LightGBM blend) on base 240 -> v781 -> T2 -> protected v788 (vs v768 = lobt alone 0.30).
# Then composite C1 = E163 base v770 + both members -> v791 -> E167 rule A+B on its parts -> v792 -> T2 -> protected v799.
set -Eeuo pipefail
cd /root/prc-2026
export OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
PY=.venv/bin/python
L=data/build_e171.log
while pgrep -f "[h]z_build_e163.sh" > /dev/null || [[ ! -s data/e163_base.ready ]]; do sleep 60; done
rm -f data/build_e171.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee -a $L data/build_e171.failed' ERR
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
  grep -E "^wrote|^E130|^E171" data/blend_v$4.log >> $L
}
M2="E130_REG2=submissions/e130_raw_rank.npz E130_W=0.17 E130_REG3=submissions/e130_lobt_rank.npz E130_W3=0.22"
cat_blend 240 submissions/lgb_reg_seeds_v240.npz data/features_e106.parquet 781 "$M2"
t2 781 782 361
$PY src/protect_tabpfn_rows.py 787 782 788 2>&1 | grep -viE "warn" >> $L
cmp 781 761 788 768 788 655
echo "[$(date +%H:%M)] composite C1 (E163 + E171 + E167)" >> $L
cat_blend 770 submissions/lgb_reg_seeds_v770.npz data/features_e163.parquet 791 "$M2"
FEATURES=data/features_e106.parquet $PY src/e167_nm_window_rule.py 791 792 2>&1 | grep -viE "warn|extension|Polars 2.0" >> $L
t2 792 793 361
$PY src/protect_tabpfn_rows.py 798 793 799 2>&1 | grep -viE "warn" >> $L
cmp 791 771 799 778 799 788 799 655
echo "[$(date +%H:%M)] done" >> $L

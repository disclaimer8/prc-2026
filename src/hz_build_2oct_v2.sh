#!/bin/bash
# 2.10 board builds (journal 216), after the FIN1 CV on hetzner (one LightGBM job at a time, journal 213); E169 runs after them.
# Board 1.10 (journal 215): the three E130 members at x2/3 gave -0.983 on the board (v808 vs v655, prediction -0.39), E163 -0.174.
#   C3  = E163 base v770 + four members x2/3 (raw 0.10, lobt 0.13, sched 0.10, eobt 0.13) -> v821 -> E167 v822 -> T2 -> protected v829
#   C3w = the same with the CV weights x1   (raw 0.15, lobt 0.20, sched 0.15, eobt 0.20) -> v831 -> E167 v832 -> T2 -> protected v839
# (E159b LSZH on top is applied on the Mac: src/e159b_apply.py, OPDI data lives there.)
set -Eeuo pipefail
cd /root/prc-2026
export OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
PY=.venv/bin/python
L=data/build_2oct_v2.log
[[ -s data/e130_eobt_rank.ready && -s submissions/e130_eobt_rank.npz ]] || { echo "no eobt member" >> $L; exit 1; }
# after FIN1: its six folds written and its driver gone
until [[ $(ls data/cv_preds_fin1_hz 2>/dev/null | wc -l) -ge 6 ]] && ! pgrep -f "[h]z_fin1_cv.sh" > /dev/null; do sleep 120; done
rm -f data/build_2oct_v2.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee -a $L data/build_2oct_v2.failed' ERR
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
  grep -E "^wrote|^E130|^E171|^E172" data/blend_v$4.log >> $L
}
R=submissions/e130_raw_rank.npz; O=submissions/e130_lobt_rank.npz; S=submissions/e130_sched_rank.npz; E=submissions/e130_eobt_rank.npz
echo "[$(date +%H:%M)] C3: E163 + four members x2/3 + E167" >> $L
cat_blend 770 submissions/lgb_reg_seeds_v770.npz data/features_e163.parquet 821 "E130_MEMBERS=$R:0.10,$O:0.13,$S:0.10,$E:0.13"
FEATURES=data/features_e106.parquet $PY src/e167_nm_window_rule.py 821 822 2>&1 | grep -viE "warn|extension|Polars 2.0" >> $L
t2 822 823 361
$PY src/protect_tabpfn_rows.py 828 823 829 2>&1 | grep -viE "warn" >> $L
echo "[$(date +%H:%M)] C3w: E163 + four members x1 + E167" >> $L
cat_blend 770 submissions/lgb_reg_seeds_v770.npz data/features_e163.parquet 831 "E130_MEMBERS=$R:0.15,$O:0.20,$S:0.15,$E:0.20"
FEATURES=data/features_e106.parquet $PY src/e167_nm_window_rule.py 831 832 2>&1 | grep -viE "warn|extension|Polars 2.0" >> $L
t2 832 833 361
$PY src/protect_tabpfn_rows.py 838 833 839 2>&1 | grep -viE "warn" >> $L
cmp 829 819 839 829 839 808
date > data/build_2oct_v2.ready
echo "[$(date +%H:%M)] done" >> $L

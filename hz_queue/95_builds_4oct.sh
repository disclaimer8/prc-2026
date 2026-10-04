#!/bin/bash
# 4.10 board builds (journal 247). From the 3.10 board (journal 245): FIN1b base -0.22, CatBoost FIN1b -0.08 (v979 = 232.7516),
# MLP member at its full grid weight beats x2/3 (v949 -0.10); E169 -0.02. E181a (journal 246): the E100 MLP on features_fin1b
# replaces the E176 MLP, CV decided -0.487, lower in 6/6, grid weight .30.
#   T1 v1109 = v979 recipe (FIN1b base v870, CatBoost FIN1b 10 seeds, members x2/3) with the E181a MLP at .20 (x2/3 of .30)
#   T2 v1119 = T1 with the E181a MLP at its full grid weight .30 (LightGBM members x2/3)
#   T3 v1129 = T2 with CatBoost FIN1b 15 seeds (E175b, submissions/cat_fin1b_seeds15.npz)
#   T4 v1139 = T3 with the E141 stage on the production OOF (E169)
#   v1019 = the from-scratch end-to-end run (/srv/prc-final-e2e1, journal 244/246), copied as is
set -Eeuo pipefail
cd /root/prc-2026
export OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
PY=.venv/bin/python
L=data/builds_4oct.log
rm -f data/builds_4oct.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee -a $L data/builds_4oct.failed' ERR
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
cat_blend () {  # cat_blend <base> <lgb seeds npz> <features> <out> [extra env]; CAT_SEEDS_FILE (default seeds_v530)
  cp -p submissions/cat_reg_rank.npz submissions/cat_reg_rank_before_1oct.npz
  cp -f ${CAT_SEEDS_FILE:-submissions/cat_reg_rank_seeds_v530.npz} submissions/cat_reg_rank.npz
  env FEATURES=$3 LGB_SEEDS=$2 ${5:-} $PY src/blend.py "$1" "$4" 0.65 > data/blend_v$4.log 2>&1 || { cp -p submissions/cat_reg_rank_before_1oct.npz submissions/cat_reg_rank.npz; return 1; }
  cp -p submissions/cat_reg_rank_before_1oct.npz submissions/cat_reg_rank.npz
  grep -E "^wrote|^E130|^E171|^E172" data/blend_v$4.log >> $L
}
R=submissions/e130_raw_rank.npz; O=submissions/e130_lobt_rank.npz; S=submissions/e130_sched_rank.npz; E=submissions/e130_eobt_rank.npz
A=submissions/e130_aobt_rank.npz; MA=submissions/e130_mlpa_rank.npz
until [[ -s data/e130_mlpa_rank.ready && -s $MA ]]; do sleep 60; done
WA23="E130_MEMBERS=$R:0.033,$O:0.10,$S:0.067,$E:0.083,$A:0.10,$MA:0.20"
WA1="E130_MEMBERS=$R:0.033,$O:0.10,$S:0.067,$E:0.083,$A:0.10,$MA:0.30"
build () {  # build <base> <lgb seeds> <features> <blend out> <members env> -> E167 -> T2 -> protect at blend+8
  local B=$1 V=$4
  cat_blend $B $2 $3 $V "$5"
  FEATURES=data/features_e106.parquet $PY src/e167_nm_window_rule.py $V $((V + 1)) 2>&1 | grep -viE "warn|extension|Polars 2.0" >> $L
  t2 $((V + 1)) $((V + 2)) 361
  $PY src/protect_tabpfn_rows.py $((V + 7)) $((V + 2)) $((V + 8)) 2>&1 | grep -viE "warn" >> $L
}
echo "[$(date +%H:%M)] T1" >> $L; CAT_SEEDS_FILE=submissions/cat_fin1b_seeds.npz build 870 submissions/lgb_reg_seeds_v870.npz data/features_fin1b.parquet 1101 "$WA23"
echo "[$(date +%H:%M)] T2" >> $L; CAT_SEEDS_FILE=submissions/cat_fin1b_seeds.npz build 870 submissions/lgb_reg_seeds_v870.npz data/features_fin1b.parquet 1111 "$WA1"
echo "[$(date +%H:%M)] T3" >> $L; CAT_SEEDS_FILE=submissions/cat_fin1b_seeds15.npz build 870 submissions/lgb_reg_seeds_v870.npz data/features_fin1b.parquet 1121 "$WA1"
echo "[$(date +%H:%M)] T4 (E169 on T3)" >> $L
( export E36_BASE_DIR=data/cv_preds_e116prod; t2 1122 1133 361; $PY src/protect_tabpfn_rows.py 1138 1133 1139 2>&1 | grep -viE "warn" >> $L )
cp -p /srv/prc-final-e2e1/submissions/gentle-igloo_v1019.parquet submissions/gentle-igloo_v1019.parquet
cmp 1109 979 1119 1109 1129 1119 1139 1129 1019 808
date > data/builds_4oct.ready
echo "[$(date +%H:%M)] done" >> $L

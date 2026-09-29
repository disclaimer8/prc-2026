#!/bin/bash
# 30.09 board builds on hetzner (journal 138). Board 29.09 vs v467: E150 (E70 stage on all rows) -0.175 (v507 = 235.168, best),
# E147 no-drift E70 -0.027, E64 -0.035, both -0.062 (additive). One change per slot against S1:
#   25 CatBoost seeds (v510 per_seed 0-19 + seeds 20-24) -> seeds_v530; v531 = blend 240 (control base), v541 = blend 520 (E151 base)
#   S1 v538: v531 -> TabPFN rows of v361 -> E14 -> E70 x5 keep-all no-drift -> E64 x5 keep-all (E150c) no-drift -> E81 -> LOBT
#   S2 v558: S1 + E152 gate (confident filed-time rows keep the E14 output) before E81
#   S3 v548: S1 on the E151 base v541
#   S4 v568: S1 with the E64 stage filtered (no E150c)
#   (S5 is v927 of the e2e rebuild: v447 line with rebuilt E81 partners)
set -Eeuo pipefail
cd /root/prc-2026
export OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 TABPFN_ALLOW_CPU_LARGE_DATASET=1
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
PY=.venv/bin/python
L=data/builds_30sep.log
rm -f data/builds_30sep.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee -a $L data/builds_30sep.failed' ERR
D7=n_pts,sf_n_seg,vi_n_pts,vi_prev_pts,vi_alt_first,vm_n_pts,vm_prev_pts
D3=vi_n_pts,vi_prev_pts,vi_alt_first
echo "[$(date +%H:%M)] start" >> $L

# 25 CatBoost seeds
$PY - >> $L 2>&1 <<'PY'
import numpy as np
z = np.load("submissions/cat_reg_rank_seeds_v510.npz"); ids, P = z["mvt_id"], z["per_seed"]
assert P.shape[0] == 20, P.shape
extra = [np.load(f"submissions/e144_seeds/cat_reg_rank_seed{s}.npz") for s in range(20, 25)]
assert all(np.array_equal(e["mvt_id"], ids) for e in extra)
P2 = np.vstack([P] + [e["reg"][None, :] for e in extra])
np.savez("submissions/cat_reg_rank_seeds_v530.npz", mvt_id=ids, reg=P2.mean(axis=0), per_seed=P2)
print(f"merged {P2.shape[0]} CatBoost seeds; RMS change of the average vs 20 seeds {np.sqrt(((P2.mean(0) - P.mean(0)) ** 2).mean()):.2f} s")
PY
blend () {  # blend <base> <lgb seeds npz> <features> <out>
  cp -p submissions/cat_reg_rank.npz submissions/cat_reg_rank_before_30sep.npz
  cp -f submissions/cat_reg_rank_seeds_v530.npz submissions/cat_reg_rank.npz
  FEATURES=$3 LGB_SEEDS=$2 $PY src/blend.py "$1" "$4" 0.65 > data/blend_v$4.log 2>&1 || { cp -p submissions/cat_reg_rank_before_30sep.npz submissions/cat_reg_rank.npz; return 1; }
  cp -p submissions/cat_reg_rank_before_30sep.npz submissions/cat_reg_rank.npz
  grep -E "^wrote" data/blend_v$4.log >> $L
}
blend 240 submissions/lgb_reg_seeds_v240.npz data/features_e106.parquet 531
blend 520 submissions/lgb_reg_seeds_v520.npz data/features_e151.parquet 541

export FEATURES=data/features_e106.parquet   # the tail steps read d_mvt_lobt / E14 inputs from the E106 table (as hz_tail_on_base.sh)
tabpfn_rows () {  # tabpfn_rows <base> <out>: the 42 TabPFN rows of v361 (journal 118, 133)
  B=$1 V=$2 $PY - >> $L 2>&1 <<'PY'
import os, polars as pl
Y = "TAXITIME_SEC_mvt"; B, V = int(os.environ["B"]), int(os.environ["V"])
r = lambda v: pl.read_parquet(f"submissions/gentle-igloo_v{v}.parquet")
chg = r(361).join(r(291), on="MVT_ID_mvt", suffix="_b").filter(pl.col(Y) != pl.col(f"{Y}_b")).select("MVT_ID_mvt", pl.col(Y).alias("tab"))
s = r(B)
s.join(chg, on="MVT_ID_mvt", how="left").with_columns(pl.coalesce("tab", Y).alias(Y)).select(s.columns).write_parquet(f"submissions/gentle-igloo_v{V}.parquet")
print(f"wrote v{V}: v{B} + the {chg.height} TabPFN rows of v361")
PY
}
e70 () { STAGE_KEEP_ALL=1 STAGE_SEEDS=42,1,2,3,4 DROP_FEATS=$D7 ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v$1.npz $PY src/e141_apply.py $2 $3 2>&1 | grep -E "keep all|^wrote" >> $L; }
e64 () { E64_KEEP_ALL=$4 STAGE_SEEDS=42,1,2,3,4 DROP_FEATS=$D3 ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v$1.npz $PY src/e64_apply.py $2 $3 2>&1 | grep -E "keep all|^wrote" >> $L; }
e81lobt () {  # e81lobt <in> <e81 out> <final out>
  BLEND_BASE=$1 BLEND_PARTS="44:0.1214,60:0.0270" $PY src/blend_submissions.py $2 2>&1 | grep -viE "warn" | tail -1 >> $L
  LOBT_CLIP=1 $PY src/lobt_edge_lift.py $2 $3 2>&1 | grep -viE "warn" | tail -1 >> $L
}
line () {  # line <base> <V>: V TabPFN, V+1 E14, V+2 E70, V+3 E64 keep-all, V+4 (= V+3), V+5 E81, V+6 LOBT = FINAL
  local B=$1 V=$2
  echo "[$(date +%H:%M)] line on v$B -> v$V..v$((V + 6))" >> $L
  tabpfn_rows $B $V
  $PY src/e14_apply.py $V $B $((V + 1)) 2>&1 | grep -E "^wrote" >> $L
  e70 $B $((V + 1)) $((V + 2))
  e64 $B $((V + 2)) $((V + 3)) 1
  cp submissions/gentle-igloo_v$((V + 3)).parquet submissions/gentle-igloo_v$((V + 4)).parquet
  e81lobt $((V + 4)) $((V + 5)) $((V + 6))
}
line 531 532          # S1 -> v538
line 541 542          # S3 -> v548 (E151 base)
echo "[$(date +%H:%M)] S2: E152 gate on S1" >> $L
$PY src/e152_stage_gate.py 533 535 531 556 2>&1 | grep -viE "warn" >> $L
e81lobt 556 557 558   # S2 -> v558
echo "[$(date +%H:%M)] S4: S1 with the filtered E64 (no E150c)" >> $L
e64 531 534 565 0
e81lobt 565 567 568   # S4 -> v568
$PY -c "
import numpy as np, polars as pl
r = lambda v: pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet').sort('MVT_ID_mvt')['TAXITIME_SEC_mvt'].cast(float).to_numpy()
for a, b in ((538, 507), (538, 477), (558, 538), (548, 538), (568, 538)):
    d = r(a) - r(b); print(f'v{a} vs v{b}: changed {(d != 0).sum()}, RMS {np.sqrt((d ** 2).mean()):.2f} s, mean {d.mean():+.2f} s, max |d| {np.abs(d).max():.0f}')
for v in (538, 548, 558, 568):
    s = pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet'); assert s.height == 344841 and s['TAXITIME_SEC_mvt'].null_count() == 0
print('row checks ok')" >> $L 2>&1
echo "[$(date +%H:%M)] done" >> $L

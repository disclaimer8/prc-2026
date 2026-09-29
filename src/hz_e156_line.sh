#!/bin/bash
# E156 board build (journal 156): the LIRF-tail TabPFN rows from the 10-seed ensemble (A1, donor v601 = v361 with those 42 rows
# re-predicted) instead of the single-seed rows of v361, on the S1 tail recipe (hz_builds_30sep.sh). One change against each base:
#   v602..v608 = S1 v538 with the A1 rows (base v531);  v612..v618 = U3 v598 with the A1 rows (base v591).
# Waits for the E130 CV (hz_e130.sh) to free cores 6-11.
set -Eeuo pipefail
cd /root/prc-2026
export OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
export FEATURES=data/features_e106.parquet
PY=.venv/bin/python
L=data/e156_line.log
while pgrep -f "[h]z_e130.sh" > /dev/null; do sleep 120; done
rm -f data/e156_line.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee -a $L data/e156_line.failed' ERR
D7=n_pts,sf_n_seg,vi_n_pts,vi_prev_pts,vi_alt_first,vm_n_pts,vm_prev_pts
D3=vi_n_pts,vi_prev_pts,vi_alt_first
echo "[$(date +%H:%M)] start" >> $L
tabpfn_rows () {  # tabpfn_rows <base> <out>: the 42 TabPFN rows (ids = v361 vs v291), values from the A1 donor v601
  B=$1 V=$2 $PY - >> $L 2>&1 <<'PY'
import os, polars as pl
Y = "TAXITIME_SEC_mvt"; B, V = int(os.environ["B"]), int(os.environ["V"])
r = lambda v: pl.read_parquet(f"submissions/gentle-igloo_v{v}.parquet")
ids = r(361).join(r(291), on="MVT_ID_mvt", suffix="_b").filter(pl.col(Y) != pl.col(f"{Y}_b")).select("MVT_ID_mvt")
chg = r(601).join(ids, on="MVT_ID_mvt").select("MVT_ID_mvt", pl.col(Y).alias("tab"))
assert chg.height == 42, chg.height
s = r(B)
s.join(chg, on="MVT_ID_mvt", how="left").with_columns(pl.coalesce("tab", Y).alias(Y)).select(s.columns).write_parquet(f"submissions/gentle-igloo_v{V}.parquet")
print(f"wrote v{V}: v{B} + the {chg.height} A1 TabPFN rows of v601")
PY
}
e70 () { STAGE_KEEP_ALL=1 STAGE_SEEDS=42,1,2,3,4 DROP_FEATS=$D7 ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v$1.npz $PY src/e141_apply.py $2 $3 2>&1 | grep -E "keep all|^wrote" >> $L; }
e64 () { E64_KEEP_ALL=1 STAGE_SEEDS=42,1,2,3,4 DROP_FEATS=$D3 ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v$1.npz $PY src/e64_apply.py $2 $3 2>&1 | grep -E "keep all|^wrote" >> $L; }
line () {  # line <base> <V>: V TabPFN A1, V+1 E14, V+2 E70, V+3 E64 keep-all, V+4 (= V+3), V+5 E81, V+6 LOBT = FINAL
  local B=$1 V=$2
  echo "[$(date +%H:%M)] line on v$B -> v$V..v$((V + 6))" >> $L
  tabpfn_rows $B $V
  $PY src/e14_apply.py $V $B $((V + 1)) 2>&1 | grep -E "^wrote" >> $L
  e70 $B $((V + 1)) $((V + 2))
  e64 $B $((V + 2)) $((V + 3))
  cp submissions/gentle-igloo_v$((V + 3)).parquet submissions/gentle-igloo_v$((V + 4)).parquet
  BLEND_BASE=$((V + 4)) BLEND_PARTS="44:0.1214,60:0.0270" $PY src/blend_submissions.py $((V + 5)) 2>&1 | grep -viE "warn" | tail -1 >> $L
  LOBT_CLIP=1 $PY src/lobt_edge_lift.py $((V + 5)) $((V + 6)) 2>&1 | grep -viE "warn" | tail -1 >> $L
}
line 531 602   # S1 + E156 -> v608
line 591 612   # U3 + E156 -> v618
$PY -c "
import numpy as np, polars as pl
r = lambda v: pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet').sort('MVT_ID_mvt')['TAXITIME_SEC_mvt'].cast(float).to_numpy()
for a, b in ((608, 538), (618, 598)):
    d = r(a) - r(b); print(f'v{a} vs v{b}: changed {(d != 0).sum()}, RMS {np.sqrt((d ** 2).mean()):.3f} s, mean {d.mean():+.3f} s, max |d| {np.abs(d).max():.0f}')
for v in (608, 618):
    s = pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet'); assert s.height == 344841 and s['TAXITIME_SEC_mvt'].null_count() == 0
print('row checks ok')" >> $L 2>&1
echo "[$(date +%H:%M)] done" >> $L

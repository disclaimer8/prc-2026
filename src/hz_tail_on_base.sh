#!/bin/bash
# Tail of the v447 line on any blend base: hz_tail_on_base.sh <base> <first>  (journal 122)
#   base -> first TabPFN rows of v361 transplanted -> first+1 E14 -> +2 E141 x5 -> +3 E64 x5 -> +4 E81 -> +5 LOBT -> +6 E145
set -Eeuo pipefail
cd /root/prc-2026
B=$1; V=$2; L=data/tail_v$V.log
export OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export FEATURES=data/features_e106.parquet TABPFN_ALLOW_CPU_LARGE_DATASET=1 PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
PY=.venv/bin/python
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" >> $L' ERR
echo "[$(date +%H:%M)] tail on v$B -> v$V..v$((V + 6))" >> $L
B=$B V=$V $PY - >> $L 2>&1 <<'PY'
import os, polars as pl
Y = "TAXITIME_SEC_mvt"; B, V = int(os.environ["B"]), int(os.environ["V"])
r = lambda v: pl.read_parquet(f"submissions/gentle-igloo_v{v}.parquet")
chg = r(361).join(r(291), on="MVT_ID_mvt", suffix="_b").filter(pl.col(Y) != pl.col(f"{Y}_b")).select("MVT_ID_mvt", pl.col(Y).alias("tab"))
s = r(B)
s.join(chg, on="MVT_ID_mvt", how="left").with_columns(pl.coalesce("tab", Y).alias(Y)).select(s.columns).write_parquet(f"submissions/gentle-igloo_v{V}.parquet")
print(f"wrote v{V}: v{B} + the {chg.height} TabPFN rows of v361")
PY
$PY src/e14_apply.py $V $B $((V + 1)) 2>&1 | grep -E "^wrote" >> $L
STAGE_SEEDS=42,1,2,3,4 ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v$B.npz $PY src/e141_apply.py $((V + 1)) $((V + 2)) 2>&1 | grep -E "^wrote" >> $L
STAGE_SEEDS=42,1,2,3,4 ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v$B.npz $PY src/e64_apply.py $((V + 2)) $((V + 3)) 2>&1 | grep -E "^wrote" >> $L
BLEND_BASE=$((V + 3)) BLEND_PARTS="44:0.1214,60:0.0270" $PY src/blend_submissions.py $((V + 4)) 2>&1 | grep -viE "warn" | tail -1 >> $L
LOBT_CLIP=1 $PY src/lobt_edge_lift.py $((V + 4)) $((V + 5)) 2>&1 | grep -viE "warn" | tail -1 >> $L
$PY src/e145_pru_calib.py apply $((V + 5)) $((V + 6)) 2>&1 | grep -E "^wrote" >> $L
echo "[$(date +%H:%M)] done" >> $L

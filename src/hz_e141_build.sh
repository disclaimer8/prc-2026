#!/bin/bash
# E141 / E142 board builds on the v366 line (v291 base + pfx TabPFN = v361, E14 = v362), journal 98 / 103.
#   check: v362 -> E70 v400 must equal v363; v364 -> E81 v420 must equal v365 -> LOBT v421 must equal v366
#   E141 (1 seed):  v362 -> v401 stage+HEAD+E141 -> v402 E64 -> v403 E81 -> v404 LOBT  (FINAL)
#   E142 (5 seeds): v362 -> v411 ...              -> v412     -> v413     -> v414       (FINAL)
set -euo pipefail
cd /root/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6
PY=.venv/bin/python
rm -f data/e141b.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee data/e141b.failed' ERR
until grep -qE '\] done|FAILED' data/e142_chain.log 2>/dev/null; do sleep 120; done
same () { $PY - "$1" "$2" <<'PY'
import sys, polars as pl
a = pl.read_parquet(f"submissions/gentle-igloo_v{sys.argv[1]}.parquet").sort("MVT_ID_mvt")
b = pl.read_parquet(f"submissions/gentle-igloo_v{sys.argv[2]}.parquet").sort("MVT_ID_mvt")
print(f"v{sys.argv[1]} == v{sys.argv[2]}: {a.equals(b)}", flush=True)
sys.exit(0 if a.equals(b) else 1)
PY
}
tail4 () {  # $1 = stage output version: E64 -> E81 -> LOBT
  local V=$1
  ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v291.npz $PY src/e64_apply.py $V $((V + 1)) 2>&1 | grep -E "^wrote"
  BLEND_BASE=$((V + 1)) BLEND_PARTS="44:0.1214,60:0.0270" $PY src/blend_submissions.py $((V + 2)) 2>&1 | grep -viE "warn" | tail -2
  LOBT_CLIP=1 $PY src/lobt_edge_lift.py $((V + 2)) $((V + 3)) 2>&1 | grep -viE "warn" | tail -2
}
echo "[$(date +%H:%M)] check 1: E70 v362 -> v400 == v363"
ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v291.npz $PY src/e70_apply.py 362 400 2>&1 | grep -E "^wrote"
same 400 363
echo "[$(date +%H:%M)] check 2: E81 + LOBT v364 -> v420 == v365 -> v421 == v366"
BLEND_BASE=364 BLEND_PARTS="44:0.1214,60:0.0270" $PY src/blend_submissions.py 420 2>&1 | grep -viE "warn" | tail -2
same 420 365
LOBT_CLIP=1 $PY src/lobt_edge_lift.py 420 421 2>&1 | grep -viE "warn" | tail -2
same 421 366
echo "[$(date +%H:%M)] E141 1 seed"
ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v291.npz $PY src/e141_apply.py 362 401 2>&1 | grep -E "HEAD known|seeds|^wrote"
tail4 401
echo "[$(date +%H:%M)] E142: E141 with 5 stage seeds"
STAGE_SEEDS=42,1,2,3,4 ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v291.npz $PY src/e141_apply.py 362 411 2>&1 | grep -E "HEAD known|seeds|^wrote"
tail4 411
$PY - <<'PY'
import numpy as np, polars as pl
r = lambda v: pl.read_parquet(f"submissions/gentle-igloo_v{v}.parquet").sort("MVT_ID_mvt")["TAXITIME_SEC_mvt"].cast(float).to_numpy()
b = r(366)
for v in (404, 414):
    d = r(v) - b
    print(f"v{v} vs v366: changed {(d != 0).sum()}, RMS {np.sqrt((d ** 2).mean()):.2f} s, mean {d.mean():+.2f} s, max |d| {np.abs(d).max():.0f}")
PY
echo "[$(date +%H:%M)] done"

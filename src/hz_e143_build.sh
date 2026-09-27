#!/bin/bash
# E143 board builds for 29.09 (journal 107): E64 stage x 5 seeds on both E141 lines.
#   check: STAGE_SEEDS=42 v411 -> v430 must equal v412 (the edited e64_apply keeps its one-seed behaviour)
#   E141 x5 stage + E64 x5: v411 -> v422 E64 -> v423 E81 -> v424 LOBT (FINAL)
#   E141 x1 stage + E64 x5: v401 -> v432 E64 -> v433 E81 -> v434 LOBT (FINAL)
set -euo pipefail
cd /root/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6
PY=.venv/bin/python
rm -f data/e143b.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee data/e143b.failed' ERR
same () { $PY -c "import sys, polars as pl; a, b = (pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet').sort('MVT_ID_mvt') for v in sys.argv[1:3]); print(f'v{sys.argv[1]} == v{sys.argv[2]}:', a.equals(b)); sys.exit(0 if a.equals(b) else 1)" "$1" "$2"; }
e64 () { STAGE_SEEDS=$1 ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v291.npz $PY src/e64_apply.py $2 $3 2>&1 | grep -E "seeds|^wrote"; }
echo "[$(date +%H:%M)] check: one-seed E64 v411 -> v430 == v412"
e64 42 411 430
same 430 412
for pair in "411 422" "401 432"; do
  set -- $pair
  echo "[$(date +%H:%M)] E64 x5 on v$1"
  e64 42,1,2,3,4 $1 $2
  BLEND_BASE=$2 BLEND_PARTS="44:0.1214,60:0.0270" $PY src/blend_submissions.py $(($2 + 1)) 2>&1 | grep -viE "warn" | tail -1
  LOBT_CLIP=1 $PY src/lobt_edge_lift.py $(($2 + 1)) $(($2 + 2)) 2>&1 | grep -viE "warn" | tail -1
done
$PY -c "
import numpy as np, polars as pl
r = lambda v: pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet').sort('MVT_ID_mvt')['TAXITIME_SEC_mvt'].cast(float).to_numpy()
for a, b in ((424, 414), (434, 404)):
    d = r(a) - r(b); print(f'v{a} vs v{b}: changed {(d != 0).sum()}, RMS {np.sqrt((d ** 2).mean()):.2f} s, mean {d.mean():+.2f} s')
"
echo "[$(date +%H:%M)] done"

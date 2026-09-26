#!/bin/bash
# E136 (26.09): E12b re-opened on the board (rejected 14.09 by the "both months" rule; LOMO SSE -14.1 %).
# TabPFN (prefix feature, 5 seeds) also serves LIRF no-NM rows with D < 4 h, no one-day shift below 12 h; E14 runs after it
# and overwrites the rows with flight history >= 3, so TabPFN reaches the <4 h rows without history. Base v291 (CatBoost seeds).
#   all bins            -> v391..v394   (compare with v374 = same without the <4 h bins)
set -euo pipefail
cd /root/prc-2026
export OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export FEATURES=data/features_e106.parquet TABPFN_ALLOW_CPU_LARGE_DATASET=1 PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee data/e136.failed' ERR
B=291; V=390
echo "[$(date +%H:%M)] TabPFN on <2h,2-4h + tail bins"
TABPFN_EXTRA=pfx TABPFN_SEEDS=0,1,2,3,4 TABPFN_BINS="<2h,2-4h,4-8h,8-12h,12-30h" P_DAY_MIN_D=43200 \
  .venv/bin/python src/lirf_tabpfn.py $B $((V+1)) | grep -vi warn | grep -v "^\[{"
.venv/bin/python src/e14_apply.py $((V+1)) $B $((V+2)) | grep -E "^wrote"
ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v$B.npz .venv/bin/python src/e70_apply.py $((V+2)) $((V+3)) 2>&1 | grep -E "^wrote"
ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v$B.npz .venv/bin/python src/e64_apply.py $((V+3)) $((V+4)) 2>&1 | grep -E "^wrote"
.venv/bin/python - <<'PY'
import polars as pl, numpy as np
a=pl.read_parquet("submissions/gentle-igloo_v394.parquet").sort("MVT_ID_mvt")["TAXITIME_SEC_mvt"].to_numpy().astype(float)
b=pl.read_parquet("submissions/gentle-igloo_v374.parquet").sort("MVT_ID_mvt")["TAXITIME_SEC_mvt"].to_numpy().astype(float)
d=a-b; print(f"v394 vs v374: differing {int((d!=0).sum())}, RMS {np.sqrt((d**2).mean()):.3f} s, max {np.abs(d).max():.0f} s")
PY
echo "[$(date +%H:%M)] done"

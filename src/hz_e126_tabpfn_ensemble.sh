#!/bin/bash
# E126 (25.09, hypothesis 62, idea from zestful-fountain's 5-seed tail specialist): ensemble the LIRF tail TabPFN.
# From v241 (E116 blend): seeds 0-4 averaged -> v341..v344; n_estimators=16 (single seed) -> v351..v354. Control = v334 line.
set -euo pipefail
cd /root/prc-2026
export OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export FEATURES=data/features_e106.parquet TABPFN_ALLOW_CPU_LARGE_DATASET=1 PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
rm -f data/e126.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee data/e126.failed' ERR
run_tail () {  # $1 = env assignments, $2 = first version
  local V=$2
  env $1 .venv/bin/python src/lirf_tabpfn.py 241 $((V+1)) | grep -vi warn
  .venv/bin/python src/e14_apply.py $((V+1)) 241 $((V+2)) | grep -E "^wrote"
  ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v241.npz .venv/bin/python src/e70_apply.py $((V+2)) $((V+3)) 2>&1 | grep -E "^wrote"
  ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v241.npz .venv/bin/python src/e64_apply.py $((V+3)) $((V+4)) 2>&1 | grep -E "^wrote"
}
echo "[$(date +%H:%M)] seeds 0-4"; run_tail "TABPFN_SEEDS=0,1,2,3,4" 340
echo "[$(date +%H:%M)] n_estimators=16"; run_tail "TABPFN_N_EST=16" 350
.venv/bin/python - <<'PY'
import polars as pl, numpy as np
c=pl.read_parquet("submissions/gentle-igloo_v334.parquet").sort("MVT_ID_mvt")["TAXITIME_SEC_mvt"].to_numpy().astype(float)
for v in (344,354):
    p=pl.read_parquet(f"submissions/gentle-igloo_v{v}.parquet").sort("MVT_ID_mvt")["TAXITIME_SEC_mvt"].to_numpy().astype(float); d=p-c
    print(f"v{v} vs control v334: differing {int((d!=0).sum())}, RMS {np.sqrt((d**2).mean()):.3f} s, max {np.abs(d).max():.0f} s")
PY
echo "[$(date +%H:%M)] done"

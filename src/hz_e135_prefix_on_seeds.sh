#!/bin/bash
# E135 (26.09): combine the board winners — E118 prefix TabPFN (v316 = 236.32, -3.60) with the CatBoost seed average
# (E121b base v291, mac) and the TabPFN seed ensemble (E126). From v291 (+parts_v291):
#   pfx                      -> v361..v364
#   pfx + TabPFN seeds 0-4   -> v371..v374
#   pfx + local_change + seeds -> v381..v384
set -euo pipefail
cd /root/prc-2026
export OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export FEATURES=data/features_e106.parquet TABPFN_ALLOW_CPU_LARGE_DATASET=1 PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
rm -f data/e135.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee data/e135.failed' ERR
run_tail () {  # $1 = env assignments, $2 = base blend version, $3 = first output version
  local B=$2 V=$3
  env $1 .venv/bin/python src/lirf_tabpfn.py $B $((V+1)) | grep -vi warn | grep -v "^\[{"
  .venv/bin/python src/e14_apply.py $((V+1)) $B $((V+2)) | grep -E "^wrote"
  ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v$B.npz .venv/bin/python src/e70_apply.py $((V+2)) $((V+3)) 2>&1 | grep -E "^wrote"
  ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v$B.npz .venv/bin/python src/e64_apply.py $((V+3)) $((V+4)) 2>&1 | grep -E "^wrote"
}
echo "[$(date +%H:%M)] pfx on v291"; run_tail "TABPFN_EXTRA=pfx" 291 360
echo "[$(date +%H:%M)] pfx + 5 seeds on v291"; run_tail "TABPFN_EXTRA=pfx TABPFN_SEEDS=0,1,2,3,4" 291 370
echo "[$(date +%H:%M)] pfx + local_change + 5 seeds on v291"; run_tail "TABPFN_EXTRA=pfx,local_change TABPFN_SEEDS=0,1,2,3,4" 291 380
echo "[$(date +%H:%M)] done"

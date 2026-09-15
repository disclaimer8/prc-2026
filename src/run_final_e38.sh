#!/bin/zsh
# E38 accepted: per-airport models -> v40 (regimes+regressor), TabPFN -> v41, E14 -> v42, E36 ADS-B stage -> v43, upload v43.
set -euo pipefail
cd "${0:a:h}/.."
git diff --quiet HEAD -- src/train.py src/features.py src/target_encoding.py src/submit_per_airport.py src/e36_adsb_stand_geometry.py || { echo "ABORT: uncommitted model code"; exit 1; }
export FEATURES=data/features_e28.parquet DROP_PREFIX=day_,tq_,wxr_,month TABPFN_ALLOW_CPU_LARGE_DATASET=1
echo "[$(date +%H:%M)] per-airport submit"
.venv/bin/python src/submit_per_airport.py 40 2>&1 | grep -E "^wrote|rows |Error|Traceback"
.venv/bin/python src/lirf_tabpfn.py 40 41 | grep -E "^wrote"
.venv/bin/python src/e14_apply.py 41 40 42 | grep -E "^wrote"
PARTS=submissions/parts_v40.npz .venv/bin/python src/e36_adsb_stand_geometry.py apply 42 43 2>&1 | grep -E "^wrote|regime probabilities"
echo "[$(date +%H:%M)] FINAL submissions/gentle-igloo_v43.parquet"
./src/submit_slot.sh 43

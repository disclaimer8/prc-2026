#!/bin/zsh
# E42 accepted (16.09 00:51): after the per-airport chain writes v42 (E14), apply the E42 stage -> v44 and upload it;
# v37 (E40 stage on the old base) is uploaded last, if a slot is left.
cd "${0:a:h}/.."
until [[ -f submissions/gentle-igloo_v42.parquet ]] || grep -qE "ABORT|Traceback" data/run_final_e38.log; do sleep 60; done
grep -qE "ABORT|Traceback" data/run_final_e38.log && { echo "[$(date -u +%H:%M)Z] chain failed, v44 not built"; exit 1; }
PARTS=submissions/parts_v40.npz .venv/bin/python src/e42_stand_geometry_per_airport.py apply 42 44 2>&1 | grep -E "^wrote|regime probabilities"
./src/submit_slot.sh 44
./src/submit_slot.sh 37

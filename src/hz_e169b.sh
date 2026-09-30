#!/bin/bash
# E169 (journal 200) waiter, replaces hz_e169.sh (journal 213): start only after BOTH the E170 and the FIN1 CVs, so that no
# two num_threads=10 LightGBM jobs share the 12 cores again.
cd /root/prc-2026
while pgrep -f "[h]z_e170.sh" > /dev/null || pgrep -f "[h]z_fin1_cv.sh" > /dev/null; do sleep 120; done
export OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
echo "[$(date +%H:%M)] E169 stage-base CV" > data/e169.log
.venv/bin/python src/e169_stage_base_cv.py >> data/e169.log 2>&1
echo "[$(date +%H:%M)] done" >> data/e169.log

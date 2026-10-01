#!/bin/bash
# E169 (journal 200) waiter, replaces hz_e169b.sh (journal 216): after the E170 CV, the FIN1 CV and the 2.10 builds, one LightGBM job at a time.
cd /root/prc-2026
while pgrep -f "[h]z_e170.sh" > /dev/null || pgrep -f "[h]z_fin1_cv.sh" > /dev/null || pgrep -f "[h]z_build_2oct_v2.sh" > /dev/null; do sleep 120; done
export OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
echo "[$(date +%H:%M)] E169 stage-base CV" > data/e169.log
.venv/bin/python src/e169_stage_base_cv.py >> data/e169.log 2>&1
echo "[$(date +%H:%M)] done" >> data/e169.log

#!/bin/bash
# E161 step 2 (journal 179): monotone-constraint stage CV after the 1.10 builds free cores 6-11.
cd /root/prc-2026
while pgrep -f "[h]z_builds_1oct.sh" > /dev/null; do sleep 120; done
export OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
echo "[$(date +%H:%M)] E161 stage CV" > data/e161.log
.venv/bin/python src/e161_mono_stage_cv.py >> data/e161.log 2>&1
echo "[$(date +%H:%M)] done" >> data/e161.log

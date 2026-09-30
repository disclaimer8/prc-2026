#!/bin/bash
# E169 (journal 200): ADS-B stage trained on the production base vs the old base, both scored on the production base. After E165.
cd /root/prc-2026
while pgrep -f "[h]z_e163.sh" > /dev/null || pgrep -f "[h]z_e165.sh" > /dev/null; do sleep 120; done
export OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
echo "[$(date +%H:%M)] E169 stage-base CV" > data/e169.log
.venv/bin/python src/e169_stage_base_cv.py >> data/e169.log 2>&1
echo "[$(date +%H:%M)] done" >> data/e169.log

#!/bin/bash
# E143 (journal 107): E64 stage x 5 seeds CV; starts after the E141/E142 board build.
set -euo pipefail
cd /root/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6
rm -f data/e143.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee data/e143.failed' ERR
until grep -qE '\] done|FAILED' data/e141b_build.log 2>/dev/null; do sleep 120; done
echo "[$(date +%H:%M)] E143 CV"
PYTHONPATH=src .venv/bin/python src/e143_e64_seeds_cv.py 2>&1 | grep -viE "warn"
echo "[$(date +%H:%M)] done"

#!/bin/zsh
# After the E28 rebuild (v26): apply the accepted E25 stage on top (v28) and upload v28 as the 5th submission of 15.09.
# v28 vs v27 isolates E28 with E25 present in both. Log: data/submit_15sep.log
cd "${0:a:h}/.."
until grep -q "FINAL submissions/gentle-igloo_v26.parquet" data/run_final_e28.log 2>/dev/null || grep -qE "ABORT|Traceback|Error" data/run_final_e28.log 2>/dev/null; do sleep 60; done
grep -qE "ABORT|Traceback|Error" data/run_final_e28.log && { echo "[$(date -u +%H:%M)Z] v26 build failed"; exit 1; }
.venv/bin/python src/e25_adsb_stage.py apply 26 28 2>&1 | grep -v Warning
printf 'FINAL submissions/gentle-igloo_v28.parquet\n' > data/v28_ready.log
./src/submit_when_ready.sh 28 data/v28_ready.log

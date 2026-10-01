#!/bin/bash
# v808 reproduction test of src/run_final_any_period.sh (journal 214) on hetzner, where v802..v808 were built: files must be identical.
cd /root/prc-2026
rm -rf /srv/prc-final-test808
MAIN=/root/prc-2026 WORK=/srv/prc-final-test808 bash src/test_final_tail_v808.sh > data/test808.log 2>&1; rc=$?
echo "[$(date +%H:%M)] test808 rc=$rc" >> data/test808.log

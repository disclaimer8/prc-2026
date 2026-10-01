#!/bin/bash
# Rerun of the v808 reproduction test (journal 214) after the aobt ranking model has finished on the Mac.
cd ~/prc-2026
until grep -qE "aobt rank (shipped|FAILED)" data/mac_1oct.log; do sleep 60; done
echo "[$(date -u +%H:%MZ)] test808 rerun" >> data/mac_1oct.log
bash src/test_final_tail_v808.sh > data/test808.log 2>&1 && echo "[$(date -u +%H:%MZ)] test808 end" >> data/mac_1oct.log || echo "[$(date -u +%H:%MZ)] test808 FAILED" >> data/mac_1oct.log

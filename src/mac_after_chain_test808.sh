#!/bin/bash
# Journal 214: the v808 reproduction test of the final chain, after the Mac night chain (src/mac_1oct_chain.sh) has finished.
cd ~/prc-2026
until grep -q "\] done" data/mac_1oct.log; do sleep 60; done
echo "[$(date -u +%H:%MZ)] test808 start" >> data/mac_1oct.log
bash src/test_final_tail_v808.sh > data/test808.log 2>&1 && echo "[$(date -u +%H:%MZ)] test808 end" >> data/mac_1oct.log || echo "[$(date -u +%H:%MZ)] test808 FAILED" >> data/mac_1oct.log

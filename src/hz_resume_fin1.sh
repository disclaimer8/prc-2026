#!/bin/bash
# Journal 213 (1.10): two LightGBM CVs with num_threads=10 each (src/train.py) on 12 cores ran ~10x slow for 7-9 h. The FIN1 CV
# python (pid 1342063, fold 1+7) was paused with SIGSTOP so E170 runs alone. Resume FIN1 once hz_e170.sh has finished.
cd /root/prc-2026
while pgrep -f "[h]z_e170.sh" > /dev/null; do sleep 60; done
kill -CONT 1342063 && echo "[$(date +%H:%M)] FIN1 CV resumed (was paused 1.10 01:02 CEST for E170)" >> data/fin1.log

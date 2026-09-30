#!/bin/bash
# Journal 166: when the Mac's E130 ranking regressor is done, ship it to hetzner and release hz_builds_1oct.sh (S2').
cd ~/prc-2026
until grep -q "^done" data/e130_rank.log || grep -q "Traceback" data/e130_rank.log; do sleep 60; done
grep -q "^done" data/e130_rank.log || { echo "[$(date +%H:%M)] e130_rank failed" >> data/e130_rank.log; exit 1; }
for i in 1 2 3 4 5; do rsync -a --partial --timeout=120 submissions/e130_raw_rank.npz hetzner:/root/prc-2026/submissions/ && break; sleep 30; done
a=$(md5 -q submissions/e130_raw_rank.npz); b=$(ssh hetzner "md5sum /root/prc-2026/submissions/e130_raw_rank.npz | cut -d' ' -f1")
[ "$a" = "$b" ] && ssh hetzner "echo $a > /root/prc-2026/data/e130_rank.ready" && echo "[$(date +%H:%M)] shipped to hetzner ($a)" >> data/e130_rank.log

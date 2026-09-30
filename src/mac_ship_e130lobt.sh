#!/bin/bash
# Journal 199: ship the E130 lobt ranking model to hetzner when done, then resume the CatBoost E51+E157 seeds from 302.
cd ~/prc-2026
until grep -q "^done" data/e130_lobt_rank.log || grep -q "Traceback" data/e130_lobt_rank.log; do sleep 60; done
if grep -q "^done" data/e130_lobt_rank.log; then
  for i in 1 2 3 4 5; do rsync -a --partial --timeout=120 submissions/e130_lobt_rank.npz hetzner:/root/prc-2026/submissions/ && break; sleep 30; done
  a=$(md5 -q submissions/e130_lobt_rank.npz); b=$(ssh hetzner "md5sum /root/prc-2026/submissions/e130_lobt_rank.npz | cut -d' ' -f1")
  [ "$a" = "$b" ] && ssh hetzner "echo $a > /root/prc-2026/data/e130_lobt_rank.ready" && echo "[$(date +%H:%M)] shipped to hetzner ($a)" >> data/e130_lobt_rank.log
fi
echo "[$(date +%H:%M)] resume after E130 lobt" >> data/mac_catseeds_e157.log
START_SEED=302 bash src/mac_catseeds_e157.sh

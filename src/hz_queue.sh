#!/bin/bash
# The single job runner on hetzner (journal 217): one LightGBM job at a time (journal 213). Runs data/hz_queue/*.sh in name order
# after the 2.10 builds, moves each finished job to data/hz_queue/done/. New work = drop a numbered script into data/hz_queue/.
cd /root/prc-2026
mkdir -p data/hz_queue/done
while pgrep -f "[h]z_build_2oct_v2.sh" > /dev/null; do sleep 60; done
while true; do
  j=$(ls data/hz_queue/*.sh 2>/dev/null | sort | head -1)
  if [ -z "$j" ]; then sleep 120; continue; fi
  echo "[$(date +%H:%M)] start $j" >> data/hz_queue.log
  bash "$j" >> data/hz_queue.log 2>&1; rc=$?
  mv "$j" data/hz_queue/done/
  echo "[$(date +%H:%M)] end $j rc=$rc" >> data/hz_queue.log
done

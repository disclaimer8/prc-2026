#!/bin/bash
# After the E158 CV (mac_e158.sh) the Mac resumes the CatBoost seeds on features_e106_e51_e157 from seed 302 (journal 175).
cd ~/prc-2026
while pgrep -f "[m]ac_e158.sh" > /dev/null || pgrep -f "[e]130_rank.py" > /dev/null; do sleep 120; done
echo "[$(date +%H:%M)] resume after E158" >> data/mac_catseeds_e157.log
START_SEED=302 bash src/mac_catseeds_e157.sh

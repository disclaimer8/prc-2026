#!/bin/bash
# E172 eobt board member (journal 211): EOBT_1-anchored LightGBM on all 2025 -> ranking, 5 seeds, rounds = mean of the six nested
# fold choices (2551). Runs after the E173 member CVs; ships submissions/e130_eobt_rank.npz to hetzner.
cd ~/prc-2026
until grep -q "done" data/mac_e173.log; do sleep 60; done
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=12 POLARS_MAX_THREADS=12
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 FEATURES=data/features_e106.parquet
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
echo "[$(date +%H:%M)] eobt rank" >> data/mac_e172.log
E130_MODE=eobt .venv/bin/python src/e130_rank.py 5 2551 > data/e130_eobt_rank.log 2>&1 && rsync -a --timeout=120 submissions/e130_eobt_rank.npz hetzner:/root/prc-2026/submissions/ && ssh hetzner "date > /root/prc-2026/data/e130_eobt_rank.ready" && echo "[$(date +%H:%M)] eobt rank shipped" >> data/mac_e172.log

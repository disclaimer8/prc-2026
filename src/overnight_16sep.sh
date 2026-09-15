#!/bin/zsh
# Overnight queue 15->16.09 (Mac). Every branch acts only on a pre-registered DECISION line; uploads go through submit_slot.sh.
#   E37 (seeds of the ADS-B stage, Mac)      -> accept: apply on v26 -> v31 -> upload
#   E36 (stand geometry, home PC)            -> accept: offblock5 on the Mac for all days, apply on v26 -> v30 -> upload
#   E32 (ADS-B features in the model, Mac)   -> accept: full rebuild v32..v35, E30 stage -> v36 -> upload
#   refill of 2025 days                      -> offblock3/5 for the new days
# Log: data/overnight_16sep.log
cd "${0:a:h}/.."
log() { echo "[$(date +%H:%M)] $*"; }
feats_all() {  # offblock3 and offblock5 for every day with a surface file
  for f in data/adsblol/surface_*.parquet; do d=${${f:t:r}#surface_}; [[ -f data/adsblol/offblock3_$d.parquet ]] || echo $d; done | xargs -P 6 -n 1 .venv/bin/python src/adsb_offblock_v3.py >/dev/null 2>&1
  for f in data/adsblol/surface_*.parquet; do d=${${f:t:r}#surface_}; [[ -f data/adsblol/offblock5_$d.parquet ]] || echo $d; done | xargs -P 6 -n 1 .venv/bin/python src/adsb_offblock_v5.py >/dev/null 2>&1
  log "offblock2/3/5: $(ls data/adsblol/offblock2_*.parquet | wc -l) $(ls data/adsblol/offblock3_*.parquet | wc -l) $(ls data/adsblol/offblock5_*.parquet | wc -l)"
}

# --- E37 now (light) ---
(
  .venv/bin/python src/e37_adsb_stage_seeds.py cv 2>&1 | grep -v Warning > data/e37.log
  if grep -q "DECISION E37 (vs E30 stage): ACCEPT" data/e37.log; then
    log "E37 ACCEPT -> v31"
    PARTS=submissions/parts_v24.npz .venv/bin/python src/e37_adsb_stage_seeds.py apply 26 31 2>&1 | grep -v Warning
    ./src/submit_slot.sh 31
  else log "E37: $(grep DECISION data/e37.log)"; fi
) &

# --- refill -> features for new days ---
(
  until grep -q "fill done" data/adsb_fill2.log 2>/dev/null; do sleep 60; done
  log "refill done"; feats_all
) &

# --- E36 from the home PC ---
(
  until scp -q home-pc-lan:C:/Users/HiMaxym/prc-2026/data/e36.log data/e36.log 2>/dev/null && grep -q "DECISION E36" data/e36.log; do sleep 300; done
  log "$(grep 'DECISION E36' data/e36.log)"
  if grep -q "DECISION E36 (vs E30 stage): ACCEPT" data/e36.log; then
    until grep -q "fill done" data/adsb_fill2.log 2>/dev/null; do sleep 60; done
    feats_all
    PARTS=submissions/parts_v24.npz .venv/bin/python src/e36_adsb_stand_geometry.py apply 26 30 2>&1 | grep -v Warning
    ./src/submit_slot.sh 30
  fi
) &

# --- E32 ---
(
  until grep -qE "^\[..:..\] done|ABORT" data/e32.log 2>/dev/null; do sleep 120; done
  log "$(grep 'DECISION E32' data/e32.log)"
  if grep -q "DECISION E32: ACCEPT" data/e32.log; then
    ./src/run_final_e32.sh > data/run_final_e32.log 2>&1
    if grep -q "FINAL submissions/gentle-igloo_v35.parquet" data/run_final_e32.log; then
      PARTS=submissions/parts_v33.npz E30_BASE_DIR=data/cv_preds_e32 .venv/bin/python src/e30_adsb_stage_v2.py apply 35 36 2>&1 | grep -v Warning
      ./src/submit_slot.sh 36
    else log "E32 rebuild failed: $(tail -2 data/run_final_e32.log)"; fi
  fi
) &
wait
log "overnight queue finished"

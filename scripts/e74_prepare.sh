#!/bin/zsh
# E74: the accepted E70 stage, on the 1 kt extraction. E70 reads three sources -- offblock2/3/5 (data/adsblol),
# the stand-walk variant (data/adsblol_standwalk) and the visit files. The onset threshold changes the first
# two, so both must be re-derived at 1 kt or the arms are confounded: sw_extra_s is a difference between the
# two extractions and would otherwise straddle thresholds.
set -uo pipefail
cd "${0:a:h}/.."
say() { echo "[$(date +%H:%M)] $*" | tee -a data/e74.log }
say "stand-walk at 1 kt -> data/adsblol_sw_kt1"
MOVING_KT=1 ADSB_OUT_DIR=data/adsblol_sw_kt1 zsh scripts/reextract_standwalk.sh 6 >> data/e74.log 2>&1
say "done: $(ls data/adsblol_sw_kt1/offblock2_*.parquet 2>/dev/null | wc -l | tr -d ' ') days"

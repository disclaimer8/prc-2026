#!/bin/bash
# Hypothesis 56 (25.09): the LOMO "8 of 12 months" rule rejected TabPFN variants worth ~-0.3 s by row-weighted SSE.
# Slots are free (23.09 rule), so the board decides. From the E116 blend output v241 (+parts_v241):
#   control            241 -> 331..334  (must reproduce v245 exactly; otherwise stop)
#   both (pfx+local)   241 -> 301..304
#   pfx only (E118)    241 -> 311..314
#   local_change only  241 -> 321..324
# Then on the mac: blend E81 (BLEND_BASE=<v>4 BLEND_PARTS=44:0.1214,60:0.0270) -> +5, lobt_edge_lift (+clip) -> +6.
set -euo pipefail
cd /root/prc-2026
export OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export FEATURES=data/features_e106.parquet TABPFN_ALLOW_CPU_LARGE_DATASET=1 PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
rm -f data/e123.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee data/e123.failed' ERR
run_tail () {  # $1 = TABPFN_EXTRA, $2 = first version number
  local V=$2
  TABPFN_EXTRA=$1 .venv/bin/python src/lirf_tabpfn.py 241 $((V+1)) | grep -vi warn
  .venv/bin/python src/e14_apply.py $((V+1)) 241 $((V+2)) | grep -E "^wrote"
  ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v241.npz .venv/bin/python src/e70_apply.py $((V+2)) $((V+3)) 2>&1 | grep -E "^wrote"
  ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v241.npz .venv/bin/python src/e64_apply.py $((V+3)) $((V+4)) 2>&1 | grep -E "^wrote"
}
echo "[$(date +%H:%M)] control"
run_tail "" 330
.venv/bin/python - <<'PY'
import polars as pl
a=pl.read_parquet("submissions/gentle-igloo_v334.parquet").sort("MVT_ID_mvt"); b=pl.read_parquet("submissions/gentle-igloo_v245.parquet").sort("MVT_ID_mvt")
d=(a["TAXITIME_SEC_mvt"].to_numpy().astype(float)-b["TAXITIME_SEC_mvt"].to_numpy().astype(float))
n=int((d!=0).sum()); rms=float((d**2).mean()**0.5)
print(f"control v334 vs v245: differing rows {n}, RMS {rms:.3f} s (TabPFN on x86 vs Apple Silicon differs in the 42 TabPFN rows only; 25.09: 39 rows, RMS 0.609)")
assert n <= 42 and rms < 1.0, "divergence beyond the TabPFN rows — stop"
PY
echo "[$(date +%H:%M)] both"; run_tail "pfx,local_change" 300
echo "[$(date +%H:%M)] pfx"; run_tail "pfx" 310
echo "[$(date +%H:%M)] local_change"; run_tail "local_change" 320
echo "[$(date +%H:%M)] done"

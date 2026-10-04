#!/bin/bash
# E184 (04.10, idea 2 of analysis/competitor_repos/_UPDATE_2026-10-04.md): FIN1b + "overdue at the gate" counts
# ret_overdue_apt / ret_overdue_rwy / ret_atraso_15m (src/e184_ret_overdue.py), production nested CV,
# gate against cv_preds_fin1b_hz: decided >= 4/6 and trimmed < 0; added capacity, so the y <= 7200 leg decides (must be < 0).
# Waits for the E182 hetzner half (hz_queue/96_e182_members.sh): two LightGBM CVs at once run ~10x slower on this box.
# Needs data/features_fin1b_e184.parquet (built on the Mac by src/e184_build_table.py) or at least data/features_e184.parquet
# (then the table is built here by src/build_final_table.py, the way FIN1b itself was).
set -uo pipefail
cd /root/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
L=data/e184_cv.log
T=data/features_fin1b_e184.parquet
OUT=data/cv_preds_e184_hz
rm -f data/e184_hz.ready data/e184_hz.failed
fail() { echo "[$(date +%H:%M)] FAILED: $1" >> $L; date > data/e184_hz.failed; exit 1; }

echo "[$(date +%H:%M)] E184 waiting for data/e182_hz.ready or data/e182_hz.failed" > $L
until [[ -e data/e182_hz.ready || -e data/e182_hz.failed ]]; do sleep 60; done
echo "[$(date +%H:%M)] E182 marker: $(ls data/e182_hz.ready data/e182_hz.failed 2>/dev/null | tr '\n' ' ')" >> $L
# any other LightGBM CV still on the box? The pattern is anchored at the interpreter, so it cannot match a launching shell.
CVPAT='^(/root/prc-2026/)?\.venv/bin/python3? (-u )?src/(cv_fold_preds_nested|cv_fold_preds|e130_raw_reg_cv)\.py'
while pgrep -f "$CVPAT" > /dev/null; do
  echo "[$(date +%H:%M)] another CV is running: $(pgrep -af "$CVPAT" | head -2 | cut -c1-120 | tr '\n' ';')" >> $L
  sleep 300
done

if [[ ! -s $T ]]; then
  [[ -s data/features_e184.parquet ]] || fail "neither $T nor data/features_e184.parquet"
  echo "[$(date +%H:%M)] building $T from features_fin1b + features_e184" >> $L
  .venv/bin/python src/build_final_table.py $T data/features_fin1b.parquet data/features_e184.parquet 2>&1 \
    | grep -viE "warn|extension|Polars 2.0" >> $L || fail "table"
fi
.venv/bin/python -c "
import polars as pl
s = pl.read_parquet_schema('$T'); b = pl.read_parquet_schema('data/features_fin1b.parquet')
new = [c for c in s if c not in b]
assert new == ['ret_overdue_apt', 'ret_overdue_rwy', 'ret_atraso_15m'], new
n = pl.scan_parquet('$T').select(pl.len()).collect().item(); assert n == 2429888, n
print('table ok', n, len(s), new)
" >> $L 2>&1 || fail "table check"

echo "[$(date +%H:%M)] E184 CV" >> $L
mkdir -p $OUT
CV_OUT=$OUT FEATURES=$T nice -n 10 .venv/bin/python src/cv_fold_preds_nested.py > data/e184_cv_folds.log 2>&1 || fail "CV (see data/e184_cv_folds.log)"
FEATURES=data/features_e106.parquet .venv/bin/python src/gate.py data/cv_preds_fin1b_hz $OUT 2>&1 | grep -viE "warn|extension|Polars 2.0" >> $L || fail "gate"
date > data/e184_hz.ready
echo "[$(date +%H:%M)] done" >> $L

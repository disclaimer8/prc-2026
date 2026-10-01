#!/bin/bash
# H25 CV (journal 231): FIN1b + the destination delay wave dw_* (data/h25_dest_wave.parquet, built on the Mac), production nested CV,
# gate against cv_preds_fin1b_hz: decided >= 4/6 and trimmed < 0. Low priority: the probe put it at about -0.012 s on the board.
set -uo pipefail
cd /root/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
L=data/h25_cv.log
echo "[$(date +%H:%M)] H25 table" > $L
.venv/bin/python -c "
import polars as pl
f = pl.read_parquet('data/features_fin1b.parquet')
w = pl.read_parquet('data/h25_dest_wave.parquet').with_columns(pl.col('MVT_ID_mvt').cast(f.schema['MVT_ID_mvt']))
g = f.join(w, on='MVT_ID_mvt', how='left').with_columns(pl.col('dw_n').fill_null(0), pl.col('dw_nhub').fill_null(0))
assert g.height == f.height
g.write_parquet('data/features_h25.parquet'); print(g.shape, 'covered', int((g['dw_nhub'] >= 3).sum()))
" >> $L 2>&1 || { echo "[$(date +%H:%M)] table FAILED" >> $L; exit 1; }
mkdir -p data/cv_preds_h25_hz
CV_OUT=data/cv_preds_h25_hz FEATURES=data/features_h25.parquet nice -n 10 .venv/bin/python src/cv_fold_preds_nested.py > data/h25_cv_folds.log 2>&1
FEATURES=data/features_e106.parquet .venv/bin/python src/gate.py data/cv_preds_fin1b_hz data/cv_preds_h25_hz 2>&1 | grep -viE "warn|extension|Polars 2.0" >> $L
echo "[$(date +%H:%M)] done" >> $L

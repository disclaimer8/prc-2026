#!/usr/bin/env bash
# Operational runner for E186 preregistered in journal 259 / commit 176c394; RSS-only amendment journal 261.
set -euo pipefail
cd /home/a1/prc-2026
exec 9>data/e186_home_rss20_runner.lock
flock -n 9 || exit 2
PY=.venv-rocm/bin/python
SRC=gpu_e186/src
trap 'touch data/e186_home_rss20.failed' ERR
test ! -e data/e186_home_rss20.ready
test ! -e data/e186_home_rss20_cv.log
printf '[%s] E186 six-fold CV\n' "$(date -u +%FT%TZ)"
"$PY" -u "$SRC/e186_nn_cv.py" --device cuda:0 --cpu-threads 2 --max-rss-gib 20 --max-device-gib 8 --folds 1,2,3,4,5,6 --out data/cv_e130e186be_rss20 > data/e186_home_rss20_cv.log 2>&1
printf '[%s] Fixed-weight gate\n' "$(date -u +%FT%TZ)"
"$PY" -u "$SRC/e186_member_gate.py" --member data/cv_e130e186be_rss20 > data/e186_home_rss20_gate.log 2>&1
"$PY" "$SRC/strict_oof_gate.py" data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlpa_f1b data/cv_preds_e186be_w015_f1b --features data/features_fin1b.parquet --require-body --audit-only --json data/e186_home_rss20_strict.json >> data/e186_home_rss20_gate.log 2>&1
printf '[%s] All folds and gate complete; acceptance is in gate_result.json\n' "$(date -u +%FT%TZ)"
date -u +%FT%TZ > data/e186_home_rss20.ready

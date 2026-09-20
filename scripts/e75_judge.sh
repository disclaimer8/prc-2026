#!/bin/zsh
# E75 judgement: the accepted arm against itself, old visit files vs visits re-extracted across midnight.
#
# e74_onset_plus_stack.py builds both of ITS arms from whatever VISIT_DIR points at, so one run of it cannot
# answer this question -- its ctrl/cand pair is the 3 kt / 1 kt extraction, not the visit window. The visit
# change is isolated by holding everything else still and taking the SAME arm from two runs:
#
#   control    data/cv_preds_e74      the 19.09 17:43 run: 1 kt extraction + data/adsblol visits
#   candidate  this run with VISIT_DIR=data/adsblol_visitprev, same 1 kt extraction
#
# Both arms of this run are recomputed from scratch; only the control arm is reused, and it is reused because
# nothing under it moved -- the base visit files are untouched (a4a18da is byte-identical on them, verified on
# three files) and no model code changed since 17:43.
#
# Pre-registered gate (REVIEW_BRIEF.md 21l, written before the extraction started): decided rows, 6 folds,
# >= 0.3 s, lower in >= 4 of 6, the same on y <= 7200 and trimmed. Prediction: -0.5 .. +0.1 s, most likely
# -0.15 s -- i.e. the registered expectation sits BELOW the gate, and a reject at -0.15 is the predicted
# outcome, not a surprise to be explained away afterwards.
#
#   zsh scripts/e75_judge.sh          log: data/e75_cv.log
set -uo pipefail
cd "${0:a:h}/.."
say() { print -r -- "[$(date +%H:%M)] $*" }

for i in $(seq 1 90); do
  v=$(ls data/adsblol_visitprev/visit_2*.parquet 2>/dev/null | wc -l | tr -d ' ')
  m=$(ls data/adsblol_visitprev/visit_m_*.parquet 2>/dev/null | wc -l | tr -d ' ')
  [[ "$v" -ge 426 && "$m" -ge 426 ]] && break
  sleep 60
done
v=$(ls data/adsblol_visitprev/visit_2*.parquet 2>/dev/null | wc -l | tr -d ' ')
m=$(ls data/adsblol_visitprev/visit_m_*.parquet 2>/dev/null | wc -l | tr -d ' ')
if [[ "$v" -lt 426 || "$m" -lt 426 ]]; then say "REFUSING to judge: extraction incomplete (visit=$v visit_m=$m of 426)"; exit 1; fi
say "extraction complete (visit=$v visit_m=$m)"

rm -rf data/cv_preds_e75_ctrl data/cv_preds_e75_cand data/cv_preds_e74_keep data/cv_preds_e74_ctrl_keep
cp -r data/cv_preds_e74 data/cv_preds_e75_ctrl      # the 17:43 run, before the visit files moved
cp -r data/cv_preds_e74 data/cv_preds_e74_keep      # this run overwrites both E74 caches; put them back after
cp -r data/cv_preds_e74_ctrl data/cv_preds_e74_ctrl_keep
say "control preserved: $(ls data/cv_preds_e75_ctrl | wc -l | tr -d ' ') folds"

VISIT_DIR=data/adsblol_visitprev .venv/bin/python src/e74_onset_plus_stack.py 2>&1 | grep -v Warning | tee data/e75_cv.log
cp -r data/cv_preds_e74 data/cv_preds_e75_cand
rm -rf data/cv_preds_e74 data/cv_preds_e74_ctrl
mv data/cv_preds_e74_keep data/cv_preds_e74
mv data/cv_preds_e74_ctrl_keep data/cv_preds_e74_ctrl
say "E74 caches restored"

PYTHONPATH=src .venv/bin/python - <<'PY' | tee -a data/e75_cv.log
from fold_metrics import compare
dec, tot, small, trim, lower, n, strict = compare("data/cv_preds_e75_ctrl", "data/cv_preds_e75_cand", lb_rmse=247.7501)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3 and -trim >= 0.3
print(f"DECISION E75: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s)")
PY
say "done"

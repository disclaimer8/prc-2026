#!/bin/zsh
# E77: both arms of the filed-time neighbour twins, six folds each, then the registered gate.
#
# The two arms read the SAME feature file and differ only by dropping the 28 new columns, so nothing else
# can move between them -- src/e77_filed_twins.py refuses to write a file that is not a strict superset of
# features_e28 with every source column unchanged.
#
# Registered before the run (REVIEW_BRIEF.md 21r): gate = decided rows, 6 folds, >= 0.3 s, lower in >= 4 of
# 6, the same on y <= 7200 and on the trimmed variant. Prediction -0.15 .. -0.6 s, centre -0.30 s.
#
# home-pc is unreachable this morning (the ssh key is not in the agent after the reboot), so all six folds
# run here instead of three; scripts/cv_split.sh is the two-machine version.
#
#   zsh scripts/e77_drive.sh            log: data/e77_cv.log
set -uo pipefail
cd "${0:a:h}/.."
say() { print -r -- "[$(date +%H:%M)] $*" }

F=data/features_e77.parquet
[[ -s $F ]] || { say "building $F"; .venv/bin/python src/e77_filed_twins.py || exit 1; }

for arm in ctrl cand; do
  out=data/cv_preds_e77_$arm
  if [[ $(ls $out/fold_*.npz 2>/dev/null | wc -l | tr -d ' ') -eq 6 ]]; then say "$arm already complete"; continue; fi
  if [[ $arm == ctrl ]]; then drop=day_,tq_,wxr_,month,nbf_,nbs_; else drop=day_,tq_,wxr_,month; fi
  say "$arm: six folds -> $out"
  mkdir -p $out
  env FEATURES=$F DROP_PREFIX=$drop CV_OUT=$out .venv/bin/python src/cv_fold_preds_nested.py >> data/e77_${arm}.log 2>&1 \
    || { say "FAILED $arm"; exit 1; }
  say "$arm done: $(ls $out/fold_*.npz | wc -l | tr -d ' ') folds"
done

PYTHONPATH=src .venv/bin/python - <<'PY' | tee data/e77_cv.log
from fold_metrics import compare
dec, tot, small, trim, lower, n, strict = compare("data/cv_preds_e77_ctrl", "data/cv_preds_e77_cand", lb_rmse=246.9205)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3 and -trim >= 0.3
print(f"DECISION E77: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s)")
PY
say "done"

"""E50: does telling the model how often a group records the block time from a planned time help? (16.09)

    .venv/bin/python src/e50_regime_rates.py      compares data/cv_preds_nested (control) with data/cv_preds_e50

REGIME_RATES=1 adds out-of-fold shares of each planned regime per airport, airport x operator, airport x stand and
airport x hour (src/target_encoding.py). Ceiling for this direction, measured 16.09 on the fold predictions: perfect
detection of the rows whose truth is exactly a planned time (7% of rows) is worth -16.2 s on the rows the base model
decides; today the classifier gives those rows a mean max probability of 0.185 against 0.050 elsewhere.

Gate: SSE over the rows the base model decides (src/fold_metrics.py) lower by >= 0.3 s LB-equivalent and lower in
>= 4 of 6 folds. The strict, total and y <= 7200 numbers are printed alongside, as the protocol of 16.09 requires.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from fold_metrics import compare  # noqa: E402

dec, tot, small, trim, lower, n, strict = compare("data/cv_preds_nested", "data/cv_preds_e50")
ok = n == 6 and -dec >= 0.3 and lower >= 4
print(f"DECISION E50: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided {dec:+.2f} s, strict {strict:+.2f} s, "
      f"all {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s)")

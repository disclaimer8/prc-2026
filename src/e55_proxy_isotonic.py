"""E55: calibrate the shape of the proxy chain, not just its slope (17.09).

    .venv/bin/python src/e55_proxy_isotonic.py     compares data/cv_preds_e46 (the accepted E46, control) with data/cv_preds_e55

E46 multiplies each proxy by one coefficient per airport and source, which is a straight line through a
relationship that is clearly curved: measured on the training months, the AOBT proxy is accurate below 1200 s
(median error 0 to +4 s) and over-predicts by +580 s at 2400-3600 s and by +3281 s above 3600 s; LOBT and EOBT
over-predict by +1500 s at 2400-3600 s. PROXY_ISO=1 replaces the coefficient with an isotonic curve per airport
and source, falling back to the source-wide curve for small groups.

Measured on the proxy itself (fit on ten months, evaluated on two): RMSE 405 -> 378 (E46) -> 346 (curve) on
fold 1+7, 377 -> 355 -> 333 on 2+8, 364 -> 331 -> 299 on 3+9. How much of that survives into the final model is
what this run measures — the residual regressor already compensates part of the proxy error.

Gate: SSE over the rows the base model decides lower than E46 by >= 0.3 s LB-equivalent and lower in >= 4 of 6
folds, with strict, full and y <= 7200 reported alongside. Control is E46, the version on the leaderboard
(251.42), so the number is the increment over what already works.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from fold_metrics import compare  # noqa: E402

dec, tot, small, trim, lower, n, strict = compare("data/cv_preds_e46", "data/cv_preds_e55")
ok = n == 6 and -dec >= 0.3 and lower >= 4
print(f"DECISION E55: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided {dec:+.2f} s, strict {strict:+.2f} s, "
      f"all {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s)")

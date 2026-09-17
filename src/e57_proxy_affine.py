"""E57: an intercept in the proxy calibration - y ~ a(airport, source) * raw + b(airport, source) - instead of
E46's pure scale through the origin (17.09).

    .venv/bin/python src/e57_proxy_affine.py     compares data/cv_preds_e46 (control, on the leaderboard) with data/cv_preds_e57

Motive, measured before the run (scratchpad probe, six fit sets): forcing the line through the origin makes the slope
carry a fixed part of the taxi that does not scale with MVT - filed time. With an intercept the slopes drop to
0.29-0.93 and the intercepts land at +25 s (EDDF) to +592 s (EDDM), and the held-out proxy RMSE falls in all six folds
(358->339, 345->335, 323->300, 321->300, 322->301, 332->311). The bias of the pure scale by raw bin on fold 1+7 is
+194 s below 600 s and -628 s above 3600 s; the affine map cuts both ends (+59 s and -163 s).

Why this is not E55 again: E55 also made the proxy more accurate and lost 0.46 s, because an isotonic curve is a step
function that lies differently in different months, and the regressor learns a residual on top of the proxy - it needs a
predictable proxy, not an accurate one. An affine map is smooth and shape-preserving (one more parameter, same family as
E46), and its parameters are stable across fit sets: the intercept moves by at most 24 s at nine of ten airports over the
six fit sets (LTFM, the January-disruption airport, moves 159 s).

Type of change (REVIEW_BRIEF.md, "Read first" #2): calibration of a fixed transform - the family that has transferred to
the leaderboard (E46: -0.40 s CV -> -1.00 s LB). No learned capacity is added: two least-squares coefficients per
airport-source pair, ridge-shrunk to the global pair with PROXY_SCALE_RIDGE pseudo-rows, exactly as E46 shrinks its one.

Gate: SSE over the rows the base model actually decides (src/fold_metrics.py) lower than E46 by >= 0.3 s LB-equivalent
and lower in >= 4 of 6 folds; the same must hold on the rows with y <= 7200, so that a gain living in the tail the
mixture rewrites cannot pass. Strict, total and trimmed numbers are reported, not gated.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from fold_metrics import compare  # noqa: E402

dec, tot, small, trim, lower, n, strict = compare("data/cv_preds_e46", "data/cv_preds_e57")
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3
print(f"DECISION E57: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s, all LB-equivalent)")

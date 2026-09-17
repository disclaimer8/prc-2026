"""E60: re-judge E35 (neighbour ADS-B taxi times for departures without their own track) on the decided metric (17.09).

    E35_SAVE_DIR=data/cv_preds_e35 E35_CTRL_DIR=data/cv_preds_e35_ctrl .venv/bin/python src/e35_adsb_neighbours.py cv
    .venv/bin/python src/e60_e35_rejudge.py

E35 was rejected on 15.09 at -0.28 s against a -0.3 threshold, lower in 4 of 6 - decided by the TOTAL SSE, the metric
we replaced on 16.09 because it also scores the rows a submission overwrites with the LIRF tail table. That matters
more for E35 than for any other stage: E30/E36 correct rows with coverage >= 2, while the overwritten tail population
is LIRF departures with low coverage, so for them the two metrics nearly agree (-12.01 total against -11.91 decided).
E35 aims at exactly coverage < 2, so its whole number was computed on a population that overlaps the overwritten one.
The direction of the shift is not predictable: E28 went from -6.24 total to +0.37 decided, E32 from -1.74 to -1.41 and
flipped the decision the other way.

It is also the largest untouched part of the error map: rows with coverage 0 and 1 hold 35.23 + 25.91 = 61.1 s of the
budget and get no physical observation at all - no track of their own, and the stage does not touch them.

Nothing about the experiment changes: same script, same features, same control (the E30 stage on covered rows only),
same folds. Only the metric the result is read with, plus saving both arms so a fourth run is never needed.

Gate: SSE over the rows the base model actually decides lower than the control by >= 0.3 s LB-equivalent and lower in
>= 4 of 6 folds, and the same on the rows with y <= 7200. Strict, total and trimmed are reported, not gated.

If it passes, the neighbour features still have to be ported onto the E36 stage (stand geometry) that submissions
actually use, and that port is a separate experiment with its own gate - this one only says whether the signal exists.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from fold_metrics import compare  # noqa: E402

dec, tot, small, trim, lower, n, strict = compare("data/cv_preds_e35_ctrl", "data/cv_preds_e35")
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3
print(f"DECISION E60: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s, all LB-equivalent)")

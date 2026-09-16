"""E46: per-airport "trust in the filed time" coefficient instead of a median offset in the proxy chain (16.09).

    .venv/bin/python src/e46_proxy_scale.py       compares data/cv_preds_nested (offset, control) with data/cv_preds_e46
Idea from a competitor's public repo (elegant-alligator): they regress the target on MVT - LOBT with a coefficient per
airport (0.23 at LSZH to 0.86 at EGLL, ridge-shrunk to a global 0.63) instead of trusting the filed time as an off-block.
Here the same is applied to our calibrated chain: PROXY_SCALE=1 replaces "raw + median offset" with "raw x a(airport, source)".

Gate (REVIEW_BRIEF.md, E46): total SSE of the 6 leak-free folds lower by >= 0.3 s LB-equivalent and lower in >= 4 of 6,
plus January and July reported separately (the organisers call transfer across periods the main difficulty).
Amendment 16.09 11:30, written before any E46 fold finished (only the E43 numbers were known): the same must hold on the
rows with y <= 7200. E43 showed the per-airport model winning 8.85 s of leak-free CV entirely on 60-150 rows per fold with
y > 7200 (0.02-0.04% of rows, up to 36 h) and losing 3.79 s on the leaderboard: a gain that lives in that tail does not transfer.
Amendment 16.09 11:55, likewise before any E46 fold finished: the decisive number is the SSE over the rows the base model
actually decides (src/fold_metrics.py) — excluding the LIRF rows without NM and with D >= 4 h that the tail table overwrites
in every submission. On the per-airport experiment that single exclusion turns -8.85 s of CV gain into -1.40 s (leaderboard: +3.79 s).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from fold_metrics import compare  # noqa: E402

dec, tot, small, trim, lower, n = compare("data/cv_preds_nested", "data/cv_preds_e46")
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3
print(f"DECISION E46: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s, all LB-equivalent)")

"""E56: how tightly should the residual regressor's target be clipped? (17.09)

    REG_CLIP_HI=3600 scripts/cv_split.sh data/cv_preds_e56 PROXY_SCALE=1 REG_CLIP_HI=3600
    .venv/bin/python src/e56_reg_clip.py data/cv_preds_e56

The regressor predicts ordinary taxi-out; the regime mixture owns everything long. The clip at 7200 s is
therefore a calibration of a fixed transformation, not a fact about the data — and it barely binds: 0.028% of
training rows exceed 7200 s, 0.06% exceed 5400, 0.2% exceed 3600. A tighter clip spends less of the model's
capacity on rows the mixture will overwrite anyway.

Control is E46 (data/cv_preds_e46), the version on the leaderboard, and the candidate carries PROXY_SCALE=1 too,
so the only difference is the clip. Gate: SSE over the rows the base model decides lower by >= 0.3 s
LB-equivalent and lower in >= 4 of 6 folds.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from fold_metrics import compare  # noqa: E402

cand = sys.argv[1] if len(sys.argv) > 1 else "data/cv_preds_e56"
dec, tot, small, trim, lower, n, strict = compare("data/cv_preds_e46", cand)
ok = n == 6 and -dec >= 0.3 and lower >= 4
print(f"DECISION E56 ({cand}): {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s)")

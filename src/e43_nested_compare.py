"""E43: does the per-airport model still win when rounds are chosen without seeing the scored months? (16.09)

    .venv/bin/python src/e43_nested_compare.py
Compares data/cv_preds_nested (one model, nested rounds) with data/cv_preds_nested_ap (per airport, nested rounds) and
prints both against the leaky pair (data/cv_preds_e28 vs data/cv_preds_e38) for reference.
Leaderboard truth: per-airport was 3.79 s WORSE (v43 256.22 vs v30 252.43).
"""
import glob
import os

import numpy as np

LB = 6 * 2 * 258.18 * 344841


def compare(ctrl_dir, cand_dir):
    tot, lower, n = 0.0, 0, 0
    for p in sorted(glob.glob(f"{ctrl_dir}/fold_*.npz")):
        q = p.replace(ctrl_dir, cand_dir)
        if not os.path.exists(q):
            continue
        c, e = np.load(p), np.load(q)
        order = {m: i for i, m in enumerate(e["mvt_id"])}
        idx = np.array([order[m] for m in c["mvt_id"]])
        d = ((e["pred"][idx] - c["y"]) ** 2 - (c["pred"] - c["y"]) ** 2).sum()
        tot += d
        lower += d < 0
        n += 1
    return tot, lower, n


import sys

sys.path.insert(0, os.path.dirname(__file__))
from fold_metrics import compare as compare_full  # noqa: E402

for name, ctrl, cand in (("leaky rounds (as in the gates)", "data/cv_preds_e28", "data/cv_preds_e38"),
                         ("nested rounds", "data/cv_preds_nested", "data/cv_preds_nested_ap")):
    tot, lower, n = compare(ctrl, cand)
    if not n:
        print(f"{name}: not computed yet")
        continue
    print(f"{name}: per-airport vs one model over {n} folds — SSE {tot:+.3e} = {tot / LB:+.2f} s LB-equivalent, lower in {lower}/{n}")
    dec, all_, small, trim, low2, _ = compare_full(ctrl, cand, lb_rmse=258.18, quiet=True)
    print(f"    on the rows the base model decides {dec:+.2f} s (lower in {low2}/{n}); y<=7200 {small:+.2f} s; trimmed {trim:+.2f} s")

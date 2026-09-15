"""Check of the Gemini Deep Research claim (15.09): replacing the probability mixture by a hard choice of a regime when
p >= tau (0.5 or 0.35 for candidates more than 7200 s from the regular one) lowers RMSE by 4.5-6 s.
Out-of-fold, 6 folds, canonical model without month (data/cv_preds_nomonth); both variants recomputed from proba/values
without the LOBT clip, so they are comparable to each other.   .venv/bin/python src/gemini_threshold_check.py"""
import glob

import numpy as np

LB = 6 * 2 * 258.18 * 344841
tot = {"mixture": 0.0, "hard_0.50": 0.0, "hard_0.35": 0.0}
for p in sorted(glob.glob("data/cv_preds_nomonth/fold_*.npz")):
    z = np.load(p)
    pr, v, y = z["proba"], np.nan_to_num(z["values"]), z["y"]
    mix = (pr * v).sum(1)
    tot["mixture"] += ((mix - y) ** 2).sum()
    far = np.abs(v - v[:, [0]]) > 7200
    for tau in (0.5, 0.35):
        pick = np.where(far, pr, 0).argmax(1)
        pmax = np.where(far, pr, 0).max(1)
        hard = np.where(pmax >= tau, v[np.arange(len(y)), pick], v[:, 0] * pr[:, 0] / np.maximum(pr[:, 0] + np.where(~far, pr, 0)[:, 1:].sum(1), 1e-9) + np.where(~far, pr * v, 0)[:, 1:].sum(1) / np.maximum(pr[:, 0] + np.where(~far, pr, 0)[:, 1:].sum(1), 1e-9))
        tot[f"hard_{tau:.2f}"] += ((hard - y) ** 2).sum()
for k in ("hard_0.50", "hard_0.35"):
    print(f"{k}: SSE change vs mixture {tot[k] - tot['mixture']:+.3e} = {(tot[k] - tot['mixture']) / LB:+.2f} s LB-equivalent")

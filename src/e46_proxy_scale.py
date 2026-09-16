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
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np

LB = 6 * 2 * 252.43 * 344841
tot, tot_small, lower, n = 0.0, 0.0, 0, 0
for p in sorted(glob.glob("data/cv_preds_nested/fold_*.npz")):
    q = p.replace("cv_preds_nested", "cv_preds_e46")
    if not os.path.exists(q):
        continue
    c, e = np.load(p), np.load(q)
    order = {m: i for i, m in enumerate(e["mvt_id"])}
    idx = np.array([order[m] for m in c["mvt_id"]])
    dd = (e["pred"][idx] - c["y"]) ** 2 - (c["pred"] - c["y"]) ** 2
    d = dd.sum()
    reg_c, reg_e = c["reg"], e["reg"][idx]
    m = c["y"] <= 7200
    print(f"{os.path.basename(p)}: SSE {d:+.3e} = {d / LB:+.2f} s (y<=7200 {dd[m].sum() / LB:+.2f} s); regressor RMSE y<=7200 "
          f"{np.sqrt(np.mean((reg_c[m] - c['y'][m]) ** 2)):.2f} -> {np.sqrt(np.mean((reg_e[m] - c['y'][m]) ** 2)):.2f}")
    tot += d
    tot_small += dd[m].sum()
    lower += d < 0
    n += 1
ok = n == 6 and -tot / LB >= 0.3 and lower >= 4 and -tot_small / LB >= 0.3
print(f"DECISION E46: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}, total {tot:+.3e} = {tot / LB:+.2f} s "
      f"LB-equivalent, of it on y<=7200 {tot_small / LB:+.2f} s)")

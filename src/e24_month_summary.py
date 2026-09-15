"""E24 summary: full pipeline without `month` (data/cv_preds_nomonth) against the stored control with it (data/cv_preds).
    .venv/bin/python src/e24_month_summary.py"""
import glob
import os

import numpy as np

LB_SCALE = 6 * 2 * 274.28 * 344841  # 6 folds cover 12 months, ranking has 2
rows = []
for p in sorted(glob.glob("data/cv_preds/fold_*.npz")):
    q = p.replace("cv_preds", "cv_preds_nomonth")
    if not os.path.exists(q):
        continue
    c, e = np.load(p), np.load(q)
    assert np.array_equal(c["mvt_id"], e["mvt_id"])
    y = c["y"]
    se_c, se_e = ((c["pred"] - y) ** 2).sum(), ((e["pred"] - y) ** 2).sum()
    m = (y <= 7200)
    r_c, r_e = np.sqrt(np.mean((c["reg"][m] - y[m]) ** 2)), np.sqrt(np.mean((e["reg"][m] - y[m]) ** 2))
    rows.append((os.path.basename(p), se_c, se_e))
    print(f"{rows[-1][0]}: full SSE {se_c:.4e} -> {se_e:.4e} ({se_e - se_c:+.3e}); regressor RMSE y<=7200 {r_c:.2f} -> {r_e:.2f}")
d = sum(r[2] - r[1] for r in rows)
lower = sum(r[2] < r[1] for r in rows)
ok = len(rows) == 6 and -d / LB_SCALE >= 0.25 and lower >= 4
print(f"DECISION E24: {'ACCEPT' if ok else 'REJECT'} (folds {len(rows)}, lower in {lower}, total SSE change {d:+.3e} = {d / LB_SCALE:+.2f} s LB-equivalent RMSE)")

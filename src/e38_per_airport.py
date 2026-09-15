"""E38: separate model per airport (regime classifier + residual regressor trained only on that airport's rows), proposed by the
Gemini Deep Research report from NASA "Pushback to the Future" winners (15.09). Same features (E28, no month) and pipeline
(train.fit_predict with early stopping on the fold's own months, as the control).
    FEATURES=data/features_e28.parquet DROP_PREFIX=day_,tq_,wxr_,month .venv/bin/python src/e38_per_airport.py folds
    .venv/bin/python src/e38_per_airport.py summary
Control: data/cv_preds_e28 (one model for all airports). Candidate: data/cv_preds_e38.
Gate (REVIEW_BRIEF.md, E38): total SSE of 6 folds lower by >= 0.5 s LB-equivalent and lower in >= 4 of 6 folds.
"""
import glob
import os
import subprocess
import sys

import numpy as np

PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
RUNNER = r'''
import os, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np, polars as pl
sys.path.insert(0, "src")
import train as T
a, b = [int(x) for x in sys.argv[1].split(",")]
df = pl.read_parquet(T.FEATURES_PATH)
tr = df.filter(pl.col("split") == "train")
del df
ids, preds, regs, ys = [], [], [], []
for ap in sorted(tr["ADEP_mvt"].unique().to_list()):
    t = tr.filter(pl.col("ADEP_mvt") == ap)
    fit, val = t.filter(~pl.col("month").is_in([a, b])), t.filter(pl.col("month").is_in([a, b]))
    if val.height == 0:
        continue
    yv = val[T.TARGET].to_numpy()
    pred, parts, best, _ = T.fit_predict(fit, val, T.feature_columns(fit), None, val_y=yv)
    ids.append(val["MVT_ID_mvt"].to_numpy()); preds.append(pred); regs.append(parts["reg"]); ys.append(yv)
    print(ap, best, float(np.sqrt(np.mean((pred - yv) ** 2))), flush=True)
np.savez(f"data/cv_preds_e38/fold_{a}_{b}.npz", mvt_id=np.concatenate(ids), pred=np.concatenate(preds), reg=np.concatenate(regs), y=np.concatenate(ys))
print("saved", a, b)
'''
if sys.argv[1] == "folds":
    os.makedirs("data/cv_preds_e38", exist_ok=True)
    for a, b in PAIRS:
        if os.path.exists(f"data/cv_preds_e38/fold_{a}_{b}.npz"):
            continue
        proc = subprocess.run([sys.executable, "-c", RUNNER, f"{a},{b}"], env={**os.environ, "POLARS_MAX_THREADS": "10"}, capture_output=True, text=True)
        print(proc.stdout.strip()[-800:] or proc.stderr[-2000:], flush=True)
else:
    LB = 6 * 2 * 258.18 * 344841
    rows = []
    for p in sorted(glob.glob("data/cv_preds_e28/fold_*.npz")):
        q = p.replace("cv_preds_e28", "cv_preds_e38")
        if not os.path.exists(q):
            continue
        c, e = np.load(p), np.load(q)
        order = {m: i for i, m in enumerate(e["mvt_id"])}
        idx = np.array([order[m] for m in c["mvt_id"]])
        y = c["y"]
        dl = (e["pred"][idx] - y) ** 2 - (c["pred"] - y) ** 2
        rows.append((os.path.basename(p), dl.sum()))
        print(f"{rows[-1][0]}: SSE change {dl.sum():+.3e}")
    tot = sum(r[1] for r in rows)
    lower = sum(r[1] < 0 for r in rows)
    ok = len(rows) == 6 and -tot / LB >= 0.5 and lower >= 4
    print(f"DECISION E38: {'ACCEPT' if ok else 'REJECT'} (folds {len(rows)}, lower in {lower}, total {tot:+.3e} = {tot / LB:+.2f} s LB-equivalent)")

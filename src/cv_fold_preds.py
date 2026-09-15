"""Out-of-fold predictions of the full model on 6 month-pair folds, saved per row for rule experiments.

    FEATURES=data/features_final.parquet DROP_PREFIX=day_,tq_,wxr_ uv run python src/cv_fold_preds.py
Writes $CV_OUT (default data/cv_preds)/fold_<a>_<b>.npz (mvt_id, pred, reg, y, proba, values)."""
import os
import subprocess
import sys

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
fit, val = tr.filter(~pl.col("month").is_in([a, b])), tr.filter(pl.col("month").is_in([a, b]))
yv = val[T.TARGET].to_numpy()
pred, parts, best, _ = T.fit_predict(fit, val, T.feature_columns(fit), None, val_y=yv)
np.savez(os.environ.get("CV_OUT", "data/cv_preds") + f"/fold_{a}_{b}.npz", mvt_id=val["MVT_ID_mvt"].to_numpy(), pred=pred, reg=parts["reg"], y=yv, proba=parts["proba"], values=parts["values"])
print("saved", a, b, float(np.sqrt(np.mean((pred - yv) ** 2))))
'''
OUT = os.environ.get("CV_OUT", "data/cv_preds")
os.makedirs(OUT, exist_ok=True)
for a, b in PAIRS:
    if os.path.exists(f"{OUT}/fold_{a}_{b}.npz"):
        continue
    proc = subprocess.run([sys.executable, "-c", RUNNER, f"{a},{b}"], env={**os.environ, "POLARS_MAX_THREADS": "10"}, capture_output=True, text=True)
    print(proc.stdout.strip()[-200:] or proc.stderr[-2000:], flush=True)

"""E130 blend (journal 151): mix the second regressor into the regular candidate of the production control and rebuild the mixture.

    CTRL=data/cv_preds_e116prod E130=data/cv_e130raw CV_OUT=data/cv_preds_e130raw FEATURES=data/features_e106.parquet \\
        .venv/bin/python src/e130_blend_cv.py

For weight w: reg' = (1 - w) reg + w reg2; the regular candidate and the day-shift candidate (reg + 1 day) move, the control's
probabilities stay (the classifier does not see the regressor). The new prediction = control pred + [mix(reg') - mix(reg)], where
mix = clip(lobt_clip(sum proba * values), 0), so float32 storage of proba/values cannot move rows the change does not touch.
The weight for each fold is chosen on the other five folds (full SSE), never on the fold being scored. Output fold files go to the
usual gates: src/fold_metrics.py and src/post_e14.py against the control.
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import train as T  # noqa: E402

CTRL = os.environ.get("CTRL", "data/cv_preds_e116prod")
SRC = os.environ.get("E130", "data/cv_e130raw")
OUT = os.environ.get("CV_OUT", "data/cv_preds_e130raw")
GRID = np.round(np.arange(0.0, 0.61, 0.05), 2)
os.makedirs(OUT, exist_ok=True)
lobt = pl.read_parquet(T.FEATURES_PATH, columns=["MVT_ID_mvt", "split", "d_mvt_lobt"]).filter(pl.col("split") == "train").drop("split")


def mix(proba, values, reg, d):
    v = values.astype(float).copy()
    v[:, 0], v[:, -1] = reg, reg + T.DAY
    return np.clip(T.lobt_clip(np.sum(proba.astype(float) * np.nan_to_num(v), axis=1), d), 0, None)


folds = []
for p in sorted(glob.glob(f"{CTRL}/fold_*.npz")):
    z = dict(np.load(p))
    s = np.load(f"{SRC}/{os.path.basename(p)}")
    order = {k: j for j, k in enumerate(s["mvt_id"])}
    reg2 = s["reg2"][[order[k] for k in z["mvt_id"]]]
    d = pl.DataFrame({"MVT_ID_mvt": z["mvt_id"]}).join(lobt, on="MVT_ID_mvt", how="left", maintain_order="left")
    base = mix(z["proba"], z["values"], z["reg"], d)
    sse = {}
    for w in GRID:
        new = z["pred"] + mix(z["proba"], z["values"], (1 - w) * z["reg"] + w * reg2, d) - base
        sse[w] = float(((new - z["y"]) ** 2).sum())
    k = z["y"] <= 7200
    print(f"{os.path.basename(p)}: rebuilt control mixture vs stored pred RMS {np.sqrt(np.mean((base - z['pred']) ** 2)):.3f} s; reg vs reg2 corr {np.corrcoef(z['reg'], reg2)[0, 1]:.4f}, RMS diff {np.sqrt(np.mean((z['reg'] - reg2) ** 2)):.1f} s; "
          f"regular RMSE y<=7200 reg {np.sqrt(np.mean((z['reg'][k] - z['y'][k]) ** 2)):.2f} reg2 {np.sqrt(np.mean((reg2[k] - z['y'][k]) ** 2)):.2f}", flush=True)
    folds.append((p, z, reg2, d, base, sse))

n = sum(len(f[1]["y"]) for f in folds)
tot = {w: sum(f[5][w] for f in folds) for w in GRID}
print("pooled full RMSE by w (in-sample weight, for the curve only): "
      + ", ".join(f"{w:.2f} {np.sqrt(tot[w] / n):.3f}" for w in GRID), flush=True)
for i, (p, z, reg2, d, base, sse) in enumerate(folds):
    other = {w: tot[w] - sse[w] for w in GRID}
    w = min(GRID, key=lambda x: other[x])
    reg_new = (1 - w) * z["reg"] + w * reg2
    z["pred"] = z["pred"] + mix(z["proba"], z["values"], reg_new, d) - base
    z["reg"] = reg_new
    np.savez(f"{OUT}/{os.path.basename(p)}", **z)
    print(f"{os.path.basename(p)}: w chosen on the other folds {w:.2f}; fold SSE change {(sse[w] - sse[0.0]) / 1e9:+.3f}e9", flush=True)

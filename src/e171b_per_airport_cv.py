"""E171b (journal 205): E171 with the (raw, lobt) weights chosen per departure airport (10 groups) on the other five folds.
Control = E171 with one global pair (data/cv_preds_e171_both). Same grid and nested selection as src/e171_two_member_cv.py."""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import train as T  # noqa: E402

CTRL, OUT = "data/cv_preds_e116prod", os.environ.get("CV_OUT", "data/cv_preds_e171b_apt")
GROUP = os.environ.get("E171B_GROUP", "ADEP_mvt")
os.makedirs(OUT, exist_ok=True)
G = np.round(np.arange(0.0, 0.61, 0.05), 2)
GRID = [(a, b) for a in G for b in G if a + b <= 0.6 + 1e-9]
f = pl.read_parquet(T.FEATURES_PATH, columns=["MVT_ID_mvt", "split", "d_mvt_lobt", "ADEP_mvt", "FLIGHT_ID_mvt"]).filter(pl.col("split") == "train").drop("split")
f = f.with_columns(nm=pl.col("FLIGHT_ID_mvt").is_not_null().cast(pl.String), apt_nm=pl.col("ADEP_mvt") + "_" + pl.col("FLIGHT_ID_mvt").is_not_null().cast(pl.String))


def mix(proba, values, reg, d):
    v = values.astype(float).copy()
    v[:, 0], v[:, -1] = reg, reg + T.DAY
    return np.clip(T.lobt_clip(np.sum(proba.astype(float) * np.nan_to_num(v), axis=1), d), 0, None)


def load(src, name, ids):
    s = np.load(f"{src}/{name}")
    order = {k: j for j, k in enumerate(s["mvt_id"])}
    return s["reg2"][[order[k] for k in ids]]


folds = []
for p in sorted(glob.glob(f"{CTRL}/fold_*.npz")):
    z = dict(np.load(p)); nm = os.path.basename(p)
    r1, r2 = load("data/cv_e130raw", nm, z["mvt_id"]), load("data/cv_e130lobt", nm, z["mvt_id"])
    d = pl.DataFrame({"MVT_ID_mvt": z["mvt_id"]}).join(f, on="MVT_ID_mvt", how="left", maintain_order="left")
    g = d[GROUP].to_numpy()
    base = mix(z["proba"], z["values"], z["reg"], d)
    new = {w: z["pred"] + mix(z["proba"], z["values"], (1 - w[0] - w[1]) * z["reg"] + w[0] * r1 + w[1] * r2, d) - base for w in GRID}
    se = {w: (new[w] - z["y"]) ** 2 for w in GRID}
    folds.append((p, z, g, new, {(k, w): float(se[w][g == k].sum()) for k in np.unique(g) for w in GRID}))
    print(nm, flush=True)
keys = sorted(set(k for f_ in folds for (k, _) in f_[4]))
for p, z, g, new, sse in folds:
    pred = z["pred"].copy(); picks = {}
    for k in keys:
        other = {w: sum(f_[4].get((k, w), 0.0) for f_ in folds if f_[0] != p) for w in GRID}
        w = min(GRID, key=lambda x: other[x]); picks[k] = w
        pred[g == k] = new[w][g == k]
    z2 = dict(z); z2["pred"] = pred
    np.savez(f"{OUT}/{os.path.basename(p)}", **z2)
    print(os.path.basename(p), {k: tuple(float(x) for x in v) for k, v in picks.items()}, flush=True)

"""E171 (journal 204): both E130 members at once. E130 raw (CV -0.78, 5/6) and E130 lobt (-0.96, 6/6) were each blended alone
into the regular candidate; their errors differ (raw: no anchor; lobt: anchored at MVT - LOBT), so the pair may add.
reg' = (1 - a - b) reg + a reg2_raw + b reg2_lobt on the grid a, b in 0..0.6 step 0.05, a + b <= 0.6; (a, b) for each fold
chosen on the other five folds (full SSE), as src/e130_blend_cv.py. Output folds -> fold_metrics against the lobt-only CV.

    FEATURES=data/features_e106.parquet .venv/bin/python src/e171_two_member_cv.py
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
OUT = os.environ.get("CV_OUT", "data/cv_preds_e171_both")
os.makedirs(OUT, exist_ok=True)
G = np.round(np.arange(0.0, 0.61, 0.05), 2)
GRID = [(a, b) for a in G for b in G if a + b <= 0.6 + 1e-9]
lobt = pl.read_parquet(T.FEATURES_PATH, columns=["MVT_ID_mvt", "split", "d_mvt_lobt"]).filter(pl.col("split") == "train").drop("split")


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
    z = dict(np.load(p))
    nm = os.path.basename(p)
    r_raw, r_lobt = load("data/cv_e130raw", nm, z["mvt_id"]), load("data/cv_e130lobt", nm, z["mvt_id"])
    d = pl.DataFrame({"MVT_ID_mvt": z["mvt_id"]}).join(lobt, on="MVT_ID_mvt", how="left", maintain_order="left")
    base = mix(z["proba"], z["values"], z["reg"], d)
    sse = {}
    for a, b in GRID:
        new = z["pred"] + mix(z["proba"], z["values"], (1 - a - b) * z["reg"] + a * r_raw + b * r_lobt, d) - base
        sse[(a, b)] = float(((new - z["y"]) ** 2).sum())
    print(f"{nm}: corr(raw, lobt) {np.corrcoef(r_raw, r_lobt)[0, 1]:.4f}, corr(reg, lobt) {np.corrcoef(z['reg'], r_lobt)[0, 1]:.4f}", flush=True)
    folds.append((p, z, r_raw, r_lobt, d, base, sse))

n = sum(len(f[1]["y"]) for f in folds)
tot = {g: sum(f[6][g] for f in folds) for g in GRID}
best = sorted(GRID, key=lambda g: tot[g])[:5]
print("pooled RMSE (in-sample, curve only): control", f"{np.sqrt(tot[(0.0, 0.0)] / n):.3f}; lobt only best",
      f"{min(np.sqrt(tot[(0.0, b)] / n) for b in G):.3f}; raw only best {min(np.sqrt(tot[(a, 0.0)] / n) for a in G):.3f}; best pairs "
      + ", ".join(f"{g} {np.sqrt(tot[g] / n):.3f}" for g in best), flush=True)
for mode, cand in (("both", GRID), ("lobt_only", [g for g in GRID if g[0] == 0.0])):
    out = OUT if mode == "both" else OUT.replace("e171_both", "e171_lobtonly")
    os.makedirs(out, exist_ok=True)
    for p, z0, r_raw, r_lobt, d, base, sse in folds:
        z = dict(z0)
        other = {g: tot[g] - sse[g] for g in cand}
        a, b = min(cand, key=lambda g: other[g])
        reg_new = (1 - a - b) * z["reg"] + a * r_raw + b * r_lobt
        z["pred"] = z["pred"] + mix(z["proba"], z["values"], reg_new, d) - base
        z["reg"] = reg_new
        np.savez(f"{out}/{os.path.basename(p)}", **z)
        print(f"{mode} {os.path.basename(p)}: (raw, lobt) chosen on the other folds ({a:.2f}, {b:.2f}); fold SSE change {(sse[(a, b)] - sse[(0.0, 0.0)]) / 1e9:+.3f}e9", flush=True)

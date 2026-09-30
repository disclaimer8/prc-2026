"""E172 (journal 205): N anchored E130 members in the regular candidate. MEMBERS=raw,lobt,sched: reg' = (1 - sum w) reg + sum w_i reg2_i
on a grid (step 0.05, sum <= 0.6), weights for each fold chosen on the other five folds. Writes two CV dirs: the full set
(data/cv_preds_e172_<members>) and the same selection with the LAST member forced to 0 (data/cv_preds_e172_<members>_ctrl),
so src/gate.py compares exactly "with the last member" against "without it".

    FEATURES=data/features_e106.parquet MEMBERS=raw,lobt,sched .venv/bin/python src/e172_members_cv.py
"""
import glob
import itertools
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import train as T  # noqa: E402

CTRL = os.environ.get("CTRL", "data/cv_preds_e116prod")
MEM = os.environ.get("MEMBERS", "raw,lobt,sched").split(",")
TAG = "_".join(MEM)
SUF = os.environ.get("E172_SUFFIX", "")
OUT, OUTC = f"data/cv_preds_e172_{TAG}{SUF}", f"data/cv_preds_e172_{TAG}{SUF}_ctrl"
for o in (OUT, OUTC):
    os.makedirs(o, exist_ok=True)
G = np.round(np.arange(0.0, 0.61, 0.05), 2)
WMAX = float(os.environ.get("E172_WMAX", "0.6"))
G = np.round(np.arange(0.0, WMAX + 0.01, 0.05), 2)
GRID = [w for w in itertools.product(G, repeat=len(MEM)) if sum(w) <= WMAX + 1e-9]
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
    z = dict(np.load(p)); nm = os.path.basename(p)
    R = np.column_stack([load(f"data/cv_e130{m}", nm, z["mvt_id"]) for m in MEM])
    d = pl.DataFrame({"MVT_ID_mvt": z["mvt_id"]}).join(lobt, on="MVT_ID_mvt", how="left", maintain_order="left")
    base = mix(z["proba"], z["values"], z["reg"], d)
    sse = {}
    for w in GRID:
        wa = np.array(w)
        new = z["pred"] + mix(z["proba"], z["values"], (1 - wa.sum()) * z["reg"] + R @ wa, d) - base
        sse[w] = float(((new - z["y"]) ** 2).sum())
    print(f"{nm}: corr of the last member with the others {[round(float(np.corrcoef(R[:, -1], R[:, i])[0, 1]), 4) for i in range(len(MEM) - 1)]}", flush=True)
    folds.append((p, z, R, d, base, sse))
n = sum(len(f[1]["y"]) for f in folds)
tot = {w: sum(f[5][w] for f in folds) for w in GRID}
best = sorted(GRID, key=lambda w: tot[w])[:3]
bestc = min((w for w in GRID if w[-1] == 0), key=lambda w: tot[w])
print(f"pooled RMSE (in-sample, curve only): control {np.sqrt(tot[GRID[0]] / n):.3f}; without {MEM[-1]} {bestc} {np.sqrt(tot[bestc] / n):.3f}; best "
      + ", ".join(f"{tuple(float(x) for x in w)} {np.sqrt(tot[w] / n):.3f}" for w in best), flush=True)
for out, cand in ((OUT, GRID), (OUTC, [w for w in GRID if w[-1] == 0])):
    for p, z0, R, d, base, sse in folds:
        z = dict(z0)
        w = min(cand, key=lambda x: tot[x] - sse[x])
        wa = np.array(w)
        reg_new = (1 - wa.sum()) * z["reg"] + R @ wa
        z["pred"] = z["pred"] + mix(z["proba"], z["values"], reg_new, d) - base
        z["reg"] = reg_new
        np.savez(f"{out}/{os.path.basename(p)}", **z)
        print(f"{os.path.basename(out)} {os.path.basename(p)}: weights {tuple(float(x) for x in w)}", flush=True)

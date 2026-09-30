"""E160 (journal 176): tail blending (Panigrahi PRC 2025 fuel code, Stage_9_Fuel_predict_CV.py:1104-1127, default off there:
base > 800 kg -> 0.8 base + 0.2 model) as a weight that depends on the base prediction, on today's production OOF.

  a) E130 with a bucketed weight: reg' = (1 - w_k) reg + w_k reg2, k = bucket of the control's regular candidate reg
     ([0, 900), [900, 1800), [1800, 3600), >= 3600 s), w_k chosen on the other five folds (grid 0..0.8); against E130's single w.
  b) shrink to physics: reg' = proxy + (1 - l_k) (reg - proxy), proxy = calibrated NM proxy (train.proxy_base, PROXY_SCALE=1,
     fitted on the fold's 10 fit months), l_k per bucket on the other folds; and Panigrahi's fixed weights as a sign check
     (l = 0.5 for 1800-3600, 0.8 above).
Mixture rebuilt exactly as src/e130_blend_cv.py (regular and day-shift candidates move, probabilities stay). Output fold files
for the usual gates: CV_OUT/<variant>/fold_*.npz.
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("PROXY_SCALE", "1")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import train as T  # noqa: E402

CTRL = os.environ.get("CTRL", "data/cv_preds_e116prod")
SRC = os.environ.get("E130", "data/cv_e130raw")
OUT = os.environ.get("CV_OUT", "data/cv_preds_e160")
EDGES = [900, 1800, 3600]
GRID = np.round(np.arange(0.0, 0.81, 0.05), 2)
cols = ["MVT_ID_mvt", "split", "month", "ADEP_mvt", "d_mvt_aobt", "d_mvt_lobt", "d_mvt_eobt", T.TARGET, *T.CANDIDATES.values()]
tr = pl.read_parquet(T.FEATURES_PATH, columns=list(dict.fromkeys(cols))).filter(pl.col("split") == "train")


def mix(proba, values, reg, d):
    v = values.astype(float).copy()
    v[:, 0], v[:, -1] = reg, reg + T.DAY
    return np.clip(T.lobt_clip(np.sum(proba.astype(float) * np.nan_to_num(v), axis=1), d), 0, None)


folds = []
for p in sorted(glob.glob(f"{CTRL}/fold_*.npz")):
    z = dict(np.load(p))
    a, b = (int(x) for x in os.path.basename(p)[5:-4].split("_"))
    s = np.load(f"{SRC}/{os.path.basename(p)}")
    order = {k: j for j, k in enumerate(s["mvt_id"])}
    reg2 = s["reg2"][[order[k] for k in z["mvt_id"]]]
    v = pl.DataFrame({"MVT_ID_mvt": z["mvt_id"]}).join(tr, on="MVT_ID_mvt", how="left", maintain_order="left")
    fit = tr.filter(~pl.col("month").is_in([a, b]))
    fit_r = fit.filter(pl.Series(T.regime_label(fit) == 0))
    _, proxy = T.proxy_base(fit_r, v)
    base = mix(z["proba"], z["values"], z["reg"], v)
    k = np.digitize(z["reg"], EDGES)
    folds.append(dict(p=p, z=z, reg2=reg2, v=v, proxy=np.asarray(proxy, dtype=float), base=base, k=k))
    print(f"{os.path.basename(p)}: rebuilt control RMS {np.sqrt(np.mean((base - z['pred']) ** 2)):.3f} s; rows per bucket {np.bincount(k, minlength=4).tolist()}; "
          f"proxy vs reg RMS {np.sqrt(np.nanmean((folds[-1]['proxy'] - z['reg']) ** 2)):.1f} s", flush=True)


def sse_for(f, reg_new):
    new = f["z"]["pred"] + mix(f["z"]["proba"], f["z"]["values"], reg_new, f["v"]) - f["base"]
    return new, float(((new - f["z"]["y"]) ** 2).sum())


def run(name, make):  # make(f, bucket, w) -> reg' for rows of that bucket (whole array returned, used where k == bucket)
    tab = {}  # (fold, bucket, w) -> SSE contribution of that bucket's rows
    for i, f in enumerate(folds):
        for kk in range(4):
            m = f["k"] == kk
            for w in GRID:
                reg_new = np.where(m, make(f, w), f["z"]["reg"])
                new, _ = sse_for(f, reg_new)
                tab[(i, kk, w)] = float(((new[m] - f["z"]["y"][m]) ** 2).sum()) - float(((f["z"]["pred"][m] - f["z"]["y"][m]) ** 2).sum())
    os.makedirs(f"{OUT}/{name}", exist_ok=True)
    chosen = []
    for i, f in enumerate(folds):
        ws = [min(GRID, key=lambda w: sum(tab[(j, kk, w)] for j in range(len(folds)) if j != i)) for kk in range(4)]
        chosen.append(ws)
        reg_new = f["z"]["reg"].copy()
        for kk, w in enumerate(ws):
            m = f["k"] == kk
            reg_new = np.where(m, make(f, w), reg_new)
        new, s1 = sse_for(f, reg_new)
        zz = dict(f["z"]); zz["pred"] = new; zz["reg"] = reg_new
        np.savez(f"{OUT}/{name}/{os.path.basename(f['p'])}", **zz)
    print(f"{name}: weights per fold (buckets <15, 15-30, 30-60, >=60 min): {chosen}", flush=True)


run("a_e130_bucketed", lambda f, w: (1 - w) * f["z"]["reg"] + w * f["reg2"])
run("b_shrink_proxy", lambda f, w: np.where(np.isnan(f["proxy"]), f["z"]["reg"], f["proxy"] + (1 - w) * (f["z"]["reg"] - f["proxy"])))
# Panigrahi's fixed weights, no selection
os.makedirs(f"{OUT}/c_panigrahi_fixed", exist_ok=True)
for f in folds:
    lam = np.select([f["k"] == 2, f["k"] == 3], [0.5, 0.8], 0.0)
    reg_new = np.where(np.isnan(f["proxy"]), f["z"]["reg"], f["proxy"] + (1 - lam) * (f["z"]["reg"] - f["proxy"]))
    new, _ = sse_for(f, reg_new)
    zz = dict(f["z"]); zz["pred"] = new; zz["reg"] = reg_new
    np.savez(f"{OUT}/c_panigrahi_fixed/{os.path.basename(f['p'])}", **zz)
print("done", flush=True)

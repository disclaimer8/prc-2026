"""E164 (journal 184; reviewer ideas file 30.09, B2, lbragado 22.09 "prediction outliers look like training outliers"): monotone
remap of the final output inside groups airport x (NM / no NM), fitted on five folds of the staged OOF chain
(data/cv_preds_e150c_e64_keep), applied to the sixth. Two forms:
  (i) isotonic regression y ~ pred (the MSE-correct monotone recalibration),
  (ii) quantile mapping pred-quantile -> y-quantile (the literal B2 form; widens the output distribution).
Regular rows only (production classifier's max P of a filed-time regime < 0.95, from cv_preds_e116prod), as the proposal says.
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

st = pl.concat([pl.DataFrame({"MVT_ID_mvt": (z := np.load(p))["mvt_id"], "pred": z["pred"].astype(float), "y": z["y"].astype(float),
                              "fold": int(os.path.basename(p)[5:-4].split("_")[0])}) for p in sorted(glob.glob("data/cv_preds_e150c_e64_keep/fold_*.npz"))])
pp = pl.concat([pl.DataFrame({"MVT_ID_mvt": (z := np.load(p))["mvt_id"], "pmax": z["proba"][:, 1:5].max(1).astype(float)}) for p in sorted(glob.glob("data/cv_preds_e116prod/fold_*.npz"))])
f = pl.read_parquet("data/features_e106.parquet", columns=["MVT_ID_mvt", "split", "ADEP_mvt", "FLIGHT_ID_mvt"]).filter(pl.col("split") == "train")
d = st.join(pp, on="MVT_ID_mvt").join(f, on="MVT_ID_mvt").with_columns(g=pl.col("ADEP_mvt") + pl.when(pl.col("FLIGHT_ID_mvt").is_null()).then(pl.lit("_nonm")).otherwise(pl.lit("")))
reg = (d["pmax"] < 0.95).to_numpy()
print(f"rows {d.height}; regular (max P filed < 0.95) {reg.mean():.3f}; RMSE staged {np.sqrt(((d['pred'] - d['y']) ** 2).mean()):.3f}")
out = {"iso": d["pred"].to_numpy().copy(), "qmap": d["pred"].to_numpy().copy()}
pred, y, g, fold = d["pred"].to_numpy(), d["y"].to_numpy(), d["g"].to_numpy(), d["fold"].to_numpy()
for fo in range(1, 7):
    for grp in np.unique(g):
        tr = (fold != fo) & (g == grp) & reg
        te = (fold == fo) & (g == grp) & reg
        if tr.sum() < 500 or te.sum() == 0:
            continue
        iso = IsotonicRegression(out_of_bounds="clip").fit(pred[tr], y[tr])
        out["iso"][te] = iso.predict(pred[te])
        qs = np.linspace(0, 1, 201)
        pq, yq = np.quantile(pred[tr], qs), np.quantile(y[tr], qs)
        out["qmap"][te] = np.interp(pred[te], pq, yq)
n = len(y)
for k, v in out.items():
    per = []
    for fo in range(1, 7):
        m = fold == fo
        per.append(float(((v[m] - y[m]) ** 2).sum() - ((pred[m] - y[m]) ** 2).sum()))
    dd = np.sort((v - y) ** 2 - (pred - y) ** 2)
    print(f"{k:5s}: RMSE {np.sqrt(((pred - y) ** 2).mean()):.3f} -> {np.sqrt(((v - y) ** 2).mean()):.3f} "
          f"({np.sqrt(((v - y) ** 2).mean()) - np.sqrt(((pred - y) ** 2).mean()):+.3f} s); folds lower {sum(x < 0 for x in per)}/6; "
          f"trimmed (-20/+20 rows) {dd[20:-20].sum() / n / (2 * np.sqrt(((pred - y) ** 2).mean())):+.3f} s")

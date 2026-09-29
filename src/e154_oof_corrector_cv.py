"""E154 (journal 142): OOF corrector with a Huber loss on ALL rows the base is sure are regular (smart-jigsaw
src/post/corrector_v2.py: Huber alpha 800, only where base <= 7200, max(0, base + r_hat); board -4.05 on their weaker base).
Our ADS-B stages correct only rows with their own surface track; ~70k ranking rows (LTFM 47k) get no correction at all.

    CTRL=data/cv_preds_e116prod CV_OUT=data/cv_preds_e154corr FEATURES=data/features_e106.parquet DROP_PREFIX=<E116> \\
        .venv/bin/python src/e154_oof_corrector_cv.py

Control = the production OOF (journal 141 lesson). For each month-pair fold the corrector is trained on the OOF residuals of
the other 10 months (their predictions are out-of-fold for themselves), rows selected by the BASE prediction, never by the
label (their first version selected by the label and lost): pred <= 7200 and P(regular) >= P_REG_MIN (0.95; Huber pulls
bimodal rows to the median). Features = the production feature columns + base pred, P(regular), regressor output.
Output: the control's fold files with pred replaced on the selected rows; gates by src/fold_metrics.py and src/post_e14.py.
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import train as T  # noqa: E402

CTRL = os.environ.get("CTRL", "data/cv_preds_e116prod")
OUT = os.environ.get("CV_OUT", "data/cv_preds_e154corr")
P_REG_MIN = float(os.environ.get("P_REG_MIN", "0.95"))
PRED_MAX = 7200.0
ROUNDS = int(os.environ.get("ROUNDS", "400"))
PARAMS = dict(objective="huber", alpha=800.0, learning_rate=0.05, num_leaves=15, max_depth=4, min_data_in_leaf=2000,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=42,
              num_threads=int(os.environ.get("OMP_NUM_THREADS", "6")), deterministic=True)
os.makedirs(OUT, exist_ok=True)

feat = pl.read_parquet(os.environ.get("FEATURES", "data/features_e106.parquet")).filter(pl.col("split") == "train")
cols = T.feature_columns(feat)
files = sorted(glob.glob(f"{CTRL}/fold_*.npz"))
assert len(files) == 6, files
parts = []
for i, p in enumerate(files):
    z = np.load(p)
    parts.append(pl.DataFrame({"MVT_ID_mvt": z["mvt_id"], "pred": z["pred"], "reg": z["reg"], "y": z["y"], "p_reg": z["proba"][:, 0], "fold": np.full(len(z["y"]), i)}))
oof = pl.concat(parts)
d = oof.join(feat.select(["MVT_ID_mvt"] + cols), on="MVT_ID_mvt", how="left")
assert d.height == oof.height
X, cats = T.to_pandas(d, cols, None)
X["base_pred"], X["base_p_reg"], X["base_reg"] = d["pred"].to_numpy(), d["p_reg"].to_numpy(), d["reg"].to_numpy()
cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
sel = (d["pred"].to_numpy() <= PRED_MAX) & (d["p_reg"].to_numpy() >= P_REG_MIN)
r = d["y"].to_numpy() - d["pred"].to_numpy()
fold = d["fold"].to_numpy()
print(f"rows {d.height}, selected {int(sel.sum())} ({sel.mean():.3f}), features {X.shape[1]}, P_REG_MIN {P_REG_MIN}", flush=True)
for i, p in enumerate(files):
    tr, te = sel & (fold != i), sel & (fold == i)
    m = lgb.train(PARAMS, lgb.Dataset(X[tr], r[tr], categorical_feature=cat_cols), ROUNDS)
    z = dict(np.load(p))
    order = {k: j for j, k in enumerate(z["mvt_id"])}
    ids = d["MVT_ID_mvt"].to_numpy()[te]
    new = np.maximum(0.0, d["pred"].to_numpy()[te] + m.predict(X[te]))
    pred = z["pred"].copy()
    pred[[order[k] for k in ids]] = new
    y = z["y"]
    before, after = np.sqrt(((z["pred"] - y) ** 2).mean()), np.sqrt(((pred - y) ** 2).mean())
    k = y <= 7200
    print(f"{os.path.basename(p)}: corrected {int(te.sum())} rows, RMSE {before:.3f} -> {after:.3f} (y<=7200 "
          f"{np.sqrt(((z['pred'] - y)[k] ** 2).mean()):.3f} -> {np.sqrt(((pred - y)[k] ** 2).mean()):.3f})", flush=True)
    z["pred"] = pred
    np.savez(f"{OUT}/{os.path.basename(p)}", **z)

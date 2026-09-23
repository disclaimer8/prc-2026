"""Seed-averaged LightGBM regressor.

    FEATURES=data/features_v4.parquet uv run python src/seed_avg.py val 5 750
    FEATURES=data/features_v4.parquet uv run python src/seed_avg.py rank 5 823

val:  fit on 2025 without Jan/Jul, report single-seed vs averaged RMSE on regular Jan/Jul rows.
rank: fit on all 2025 regular rows, write submissions/lgb_reg_seeds.npz (mvt_id, reg, per_seed).
"""
import os
import sys
import time

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import lightgbm as lgb
import numpy as np
import polars as pl

import train as T
from features import CATEGORICAL, TARGET
from target_encoding import add_encodings

mode, n_seeds, rounds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
df = pl.read_parquet(T.FEATURES_PATH)
train = df.filter(pl.col("split") == "train")
if mode == "val":
    is_val = pl.col("month").is_in([1, 7])
    fit, pred = train.filter(~is_val), train.filter(is_val)
else:
    fit, pred = train, df.filter(pl.col("split") == "rank")

lab = T.regime_label(fit)
fit, pred = add_encodings(fit, pred, lab == 0, fit_labels=lab)
cols = T.feature_columns(fit)
fit_r = fit.filter(pl.Series(lab == 0))
Xf, cats = T.to_pandas(fit_r, cols, None)
Xp, _ = T.to_pandas(pred, cols, cats)
base_f, base_p = T.proxy_base(fit_r, pred)
yf = np.clip(np.clip(fit_r[TARGET].to_numpy(), *T.REG_CLIP) - base_f, -7200, 7200)
dtrain = lgb.Dataset(Xf, yf, weight=T.adv_weights(fit_r), categorical_feature=[c for c in CATEGORICAL if c in cols], free_raw_data=False)

preds = []
for s in range(n_seeds):
    t0 = time.time()
    params = {**T.REG_PARAMS, "seed": 1000 + s, "bagging_seed": 2000 + s, "feature_fraction_seed": 3000 + s}
    model = lgb.train(params, dtrain, num_boost_round=rounds)
    preds.append(model.predict(Xp) + base_p)
    msg = f"seed {s} done ({time.time() - t0:.0f}s)"
    if mode == "val":
        val_lab = T.regime_label(pred)
        y = pred[TARGET].to_numpy()
        m = (val_lab == 0) & (y <= T.REG_CLIP[1])
        msg += f" single={T.rmse(preds[-1][m], y[m]):.2f} avg{len(preds)}={T.rmse(np.mean(preds, axis=0)[m], y[m]):.2f}"
    print(msg, flush=True)

if mode == "rank":
    np.savez(
        "submissions/lgb_reg_seeds.npz",
        mvt_id=pred["MVT_ID_mvt"].to_numpy(),
        reg=np.mean(preds, axis=0),
        per_seed=np.array(preds),
    )
    print("saved submissions/lgb_reg_seeds.npz")

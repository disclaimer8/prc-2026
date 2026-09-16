"""CatBoost regressor on regular-regime departures of all 2025, predicting the ranking set.

    FEATURES=data/features_v4.parquet uv run python src/catboost_reg.py 5200
Writes submissions/cat_reg_rank.npz (mvt_id, reg). Iterations come from validation (best 5137).
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool

import train as T
from features import CATEGORICAL, TARGET
from target_encoding import add_encodings

iterations = int(sys.argv[1])
df = pl.read_parquet(T.FEATURES_PATH)
train, rank = df.filter(pl.col("split") == "train"), df.filter(pl.col("split") == "rank")
lab = T.regime_label(train)
train, rank = add_encodings(train, rank, lab == 0, fit_labels=lab)
cols = T.feature_columns(train)
cat_cols = [c for c in CATEGORICAL if c in cols]


def pool(d, y=None):
    pdf = d.select(cols).to_pandas()
    for c in cat_cols:
        pdf[c] = pdf[c].fillna("NA").astype(str)
    return Pool(pdf, y, cat_features=cat_cols)


fit = train.filter(pl.Series(lab == 0))
model = CatBoostRegressor(iterations=iterations, learning_rate=0.08, depth=8, loss_function="RMSE", thread_count=8, verbose=500, random_seed=0)
base_f, base_p = T.proxy_base(fit, rank)
model.fit(pool(fit, np.clip(np.clip(fit[TARGET].to_numpy(), *T.REG_CLIP) - base_f, -7200, 7200)))
reg = model.predict(pool(rank)) + base_p
np.savez("submissions/cat_reg_rank.npz", mvt_id=rank["MVT_ID_mvt"].to_numpy(), reg=reg)
model.save_model("submissions/cat_reg_rank.cbm")
print("done", len(reg), float(np.mean(reg)))

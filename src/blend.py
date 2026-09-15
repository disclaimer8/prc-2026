"""Rebuild a submission from saved models, blending LightGBM and CatBoost regressors.

    FEATURES=data/features_v4.parquet uv run python src/blend.py <base_version> <new_version> <cat_weight>

Loads submissions/clf_v<base>.txt and reg_v<base>.txt (LightGBM) and submissions/cat_reg_rank.npz,
recomputes the ranking feature matrix exactly as in train.py, and writes gentle-igloo_v<new>.parquet.
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import lightgbm as lgb
import numpy as np
import polars as pl

import train as T
from features import TARGET
from target_encoding import add_encodings

base, new, w_cat = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
df = pl.read_parquet(T.FEATURES_PATH)
train, rank = df.filter(pl.col("split") == "train"), df.filter(pl.col("split") == "rank")
lab = T.regime_label(train)
train, rank = add_encodings(train, rank, lab == 0)
cols = T.feature_columns(train)
_, cats = T.to_pandas(train, cols, None)
Xr, _ = T.to_pandas(rank, cols, cats)

clf = lgb.Booster(model_file=f"submissions/clf_v{base}.txt")
reg = lgb.Booster(model_file=f"submissions/reg_v{base}.txt")
assert clf.feature_name() == [c.replace(" ", "_") for c in cols], "feature list differs from the saved model"
proba = clf.predict(Xr)
_, base_p = T.proxy_base(train.filter(pl.Series(lab == 0)), rank)
r_lgb = reg.predict(Xr) + base_p  # the regressor predicts the residual against the NM proxy
if os.environ.get("LGB_SEEDS"):  # seed-averaged LightGBM regressor from src/seed_avg.py
    seeds = np.load(os.environ["LGB_SEEDS"])
    seed_map = dict(zip(seeds["mvt_id"], seeds["reg"]))
    r_single = r_lgb
    r_lgb = np.array([seed_map[m] for m in rank["MVT_ID_mvt"].to_numpy()])
    print(f"LightGBM regressor: {seeds['per_seed'].shape[0]}-seed average, mean |single-avg| = {np.mean(np.abs(r_single - r_lgb)):.1f}s")

if w_cat > 0:
    cat = np.load("submissions/cat_reg_rank.npz")
    cat_map = dict(zip(cat["mvt_id"], cat["reg"]))
    r_cat = np.array([cat_map[m] for m in rank["MVT_ID_mvt"].to_numpy()])
else:  # reproduces the base submission; used to check this script
    r_cat = r_lgb
r = w_cat * r_cat + (1 - w_cat) * r_lgb

values = [r] + [rank[c].to_numpy() for c in T.CANDIDATES.values()] + [r + T.DAY]
values = np.column_stack(values)
proba = np.where(np.isnan(values), 0.0, proba)
proba = T.mask_infeasible(proba, values, rank)
proba = proba / proba.sum(axis=1, keepdims=True)
pred = np.clip(T.lobt_clip(np.sum(proba * np.nan_to_num(values), axis=1), rank), 0, None)

template = pl.read_parquet("data/prc-2026-datasets/submitting.parquet")
preds = rank.select("MVT_ID_mvt").with_columns(pl.Series(TARGET, pred))
sub = template.select("MVT_ID_mvt").join(preds, on="MVT_ID_mvt", how="left")
assert sub.height == template.height and sub[TARGET].null_count() == 0
sub = sub.with_columns(pl.col(TARGET).round(0).cast(template.schema[TARGET]))
sub.write_parquet(f"submissions/gentle-igloo_v{new}.parquet")
np.savez(f"submissions/parts_v{new}.npz", mvt_id=rank["MVT_ID_mvt"].to_numpy(), reg=r, proba=proba, values=values, pred=pred)
old = pl.read_parquet(f"submissions/gentle-igloo_v{base}.parquet")
diff = old.join(sub, on="MVT_ID_mvt", suffix="_new").select((pl.col(TARGET) - pl.col(TARGET + "_new")).abs().mean()).item()
print(f"wrote v{new}: mean={sub[TARGET].mean():.0f}, mean |v{base}-v{new}| = {diff:.1f}s, corr lgb/cat reg = {np.corrcoef(r_lgb, r_cat)[0,1]:.3f}")

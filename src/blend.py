"""Rebuild a submission from saved models, blending LightGBM and CatBoost regressors.

    FEATURES=data/features_v4.parquet uv run python src/blend.py <base_version> <new_version> <cat_weight>

Loads submissions/clf_v<base>.txt and reg_v<base>.txt (LightGBM) and submissions/cat_reg_rank.npz,
recomputes the ranking feature matrix exactly as in train.py, and writes gentle-igloo_v<new>.parquet.
"""
import glob
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
train, rank = add_encodings(train, rank, lab == 0, fit_labels=lab)
cols = T.feature_columns(train)
_, cats = T.to_pandas(train, cols, None)
Xr, _ = T.to_pandas(rank, cols, cats)

clfs = [lgb.Booster(model_file=f"submissions/clf_v{base}.txt")]
clfs += [lgb.Booster(model_file=f) for f in sorted(glob.glob(f"submissions/clf_v{base}_s*.txt"))]  # E95 seeds
clf = clfs[0]
reg = lgb.Booster(model_file=f"submissions/reg_v{base}.txt")
assert clf.feature_name() == [c.replace(" ", "_") for c in cols], "feature list differs from the saved model"
proba = np.mean([c.predict(Xr) for c in clfs], axis=0)
print(f"classifier models averaged: {len(clfs)}")
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
if os.environ.get("E130_REG2"):  # E130 (journal 166): raw-target regressor mixed into the regular candidate at weight E130_W
    z2 = np.load(os.environ["E130_REG2"])
    reg2_map = dict(zip(z2["mvt_id"], z2["reg"]))
    reg2 = np.array([reg2_map[m] for m in rank["MVT_ID_mvt"].to_numpy()])
    w2 = float(os.environ["E130_W"])
    print(f"E130: raw regressor at w {w2}, corr with the blend {np.corrcoef(r, reg2)[0, 1]:.4f}, RMS diff {np.sqrt(np.mean((r - reg2) ** 2)):.1f} s")
    w3, reg3 = 0.0, 0.0
    if os.environ.get("E130_REG3"):  # E171 (journal 204): a second member (raw + lobt together), r = (1 - w2 - w3) r + w2 reg2 + w3 reg3
        z3 = np.load(os.environ["E130_REG3"])
        reg3_map = dict(zip(z3["mvt_id"], z3["reg"]))
        reg3 = np.array([reg3_map[m] for m in rank["MVT_ID_mvt"].to_numpy()])
        w3 = float(os.environ["E130_W3"])
        print(f"E171: second member at w {w3}, corr with reg2 {np.corrcoef(reg2, reg3)[0, 1]:.4f}")
    r = (1 - w2 - w3) * r + w2 * reg2 + w3 * reg3
elif os.environ.get("E130_MEMBERS"):  # E172 (journal 207): any number of members, "path:w,path:w,..."
    mem = [(p_, float(w_)) for p_, w_ in (x.rsplit(":", 1) for x in os.environ["E130_MEMBERS"].split(","))]
    regs = []
    for p_, w_ in mem:
        zm = np.load(p_)
        mm = dict(zip(zm["mvt_id"], zm["reg"]))
        regs.append(np.array([mm[m] for m in rank["MVT_ID_mvt"].to_numpy()]))
        print(f"E172 member {p_} at w {w_}, corr with the blend {np.corrcoef(r, regs[-1])[0, 1]:.4f}")
    r = (1 - sum(w_ for _, w_ in mem)) * r + sum(w_ * g for (_, w_), g in zip(mem, regs))

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

"""H19 (journal 188; external package item 2.1): observed stand geometry from ARRIVALS - median taxi-in of arrivals at the same
(airport, stand) from the OTHER half year (in ranking: the same month's arrivals, labels open) - as a feature for departures,
especially for rare/new stands (EDDF H/J in July 2026). Screen on the staged OOF residual (cv_preds_e150c_e64_keep), controls include
the existing per-row stand features; all rows and rare stands (< 100 departures in the training half).
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import lightgbm as lgb
import numpy as np
import polars as pl

cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
tr = pl.concat([pl.read_parquet(p, columns=cols) for p in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).with_columns(
    half=pl.when(pl.col("MVT_TIME_UTC_mvt").dt.month() <= 6).then(1).otherwise(2), month=pl.col("MVT_TIME_UTC_mvt").dt.month())
arr = tr.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("TAXITIME_SEC_mvt").is_between(0, 3600) & pl.col("STAND_mvt").is_not_null())
dep = tr.filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt", "half", "month", ap=pl.col("ADEP_mvt"), stand=pl.col("STAND_mvt"))
parts = []
for h in (1, 2):
    a = arr.filter(pl.col("half") != h).group_by(pl.col("ADES_mvt").alias("ap"), pl.col("STAND_mvt").alias("stand")).agg(st_ti_med=pl.col("TAXITIME_SEC_mvt").median(), st_ti_n=pl.len())
    apm = arr.filter(pl.col("half") != h).group_by(pl.col("ADES_mvt").alias("ap")).agg(ap_ti_med=pl.col("TAXITIME_SEC_mvt").median())
    nd = dep.filter(pl.col("half") != h).group_by("ap", "stand").agg(dep_n=pl.len())
    parts.append(dep.filter(pl.col("half") == h).join(a, on=["ap", "stand"], how="left").join(apm, on="ap", how="left").join(nd, on=["ap", "stand"], how="left")
                 .with_columns(st_ti_rel=pl.col("st_ti_med") - pl.col("ap_ti_med"), dep_n=pl.col("dep_n").fill_null(0)))
feat = pl.concat(parts)
NEW = ["st_ti_med", "st_ti_rel", "st_ti_n"]
schema = pl.read_parquet_schema("data/features_e106.parquet")
CTRL = [c for c in ("d_mvt_aobt", "d_mvt_lobt", "taxiing_at_ob", "rwy_dep_before_30", "dep_before_30", "offblocks_around_ob_30", "nbf_ap_30_0_mean",
                    "nb_rwy_30_30_mean", "stand_prev_arr_taxi_in", "stand_prev_arr_age", "nb_stand_4320_0_mean", "geo_dist_m") if c in schema]
f = pl.read_parquet("data/features_e106.parquet", columns=["MVT_ID_mvt", "split"] + CTRL).filter(pl.col("split") == "train")
oof = pl.concat([pl.DataFrame({k: np.load(p)[k] for k in ("mvt_id", "pred", "y")}) for p in sorted(glob.glob("data/cv_preds_e150c_e64_keep/fold_*.npz"))]).rename({"mvt_id": "MVT_ID_mvt"})
d = feat.join(f, on="MVT_ID_mvt").join(oof, on="MVT_ID_mvt").with_columns(r=pl.col("y") - pl.col("pred")).filter((pl.col("y") <= 7200) & (pl.col("r").abs() < 1800))
print("controls:", CTRL, "| st_ti known", round(float(d["st_ti_med"].is_not_null().mean()), 3), "| rare-stand rows", int((d["dep_n"] < 100).sum()))


def rank(x):
    return pl.Series(x).fill_nan(None).rank().fill_null(0).to_numpy().astype(float)


def partial(yv, x, controls):
    Z = np.column_stack([np.ones(len(yv))] + [rank(c) for c in controls])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return float(np.corrcoef(res(rank(yv)), res(rank(x)))[0, 1])


prm = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=300, feature_fraction=0.9, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=42, num_threads=4, deterministic=True)
for label, sub in (("all rows", d), ("rare stands (< 100 departures in the training half)", d.filter(pl.col("dep_n") < 100))):
    halves = {h: sub.filter(pl.col("half") == hh) for h, hh in (("H1", 1), ("H2", 2))}
    print(f"== {label}: rows {sub.height}; mean residual {sub['r'].mean():+.1f}")
    for c in NEW:
        p = [partial(g["r"].to_numpy(), g[c].to_numpy(), [g[k_].to_numpy() for k_ in CTRL]) for g in halves.values()]
        print(f"   {c:10s} partial H1 {p[0]:+.4f} / H2 {p[1]:+.4f}")
    for tr_h, te_h in (("H1", "H2"), ("H2", "H1")):
        a_, b_ = halves[tr_h], halves[te_h]
        line = f"   stage {tr_h} -> {te_h}: no stage {np.sqrt((b_['r'].to_numpy() ** 2).mean()):.3f}"
        for name, cs_ in (("controls", CTRL), ("+stand arrival geometry", CTRL + NEW)):
            m = lgb.train(prm, lgb.Dataset(a_.select(cs_).to_numpy(), a_["r"].to_numpy()), 300)
            line += f" | {name} {np.sqrt(((b_['r'].to_numpy() - m.predict(b_.select(cs_).to_numpy())) ** 2).mean()):.3f}"
        print(line)

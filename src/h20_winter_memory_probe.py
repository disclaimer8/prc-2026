"""H20 (journal 190, external package item 4): time-based winter memory. src/features.py:175-176 takes rolling_max(6) over METAR
ROWS (6 reports, ~1.5-3 h with half-hourly METAR + SPECI), not 6 hours. Here: hours since the last snow report and the last icing
proxy report, and snow / icing within 12, 24, 48 h (time windows), joined to each departure at its off-block proxy time as
features/add_weather does. Screen on the staged OOF residual, all rows and winter months (Jan, Feb, Dec), as H10.
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, "src")
import lightgbm as lgb
import numpy as np
import polars as pl

import features as FE

m = FE.load_metar().sort("ADEP_mvt", "metar_time")
m = m.with_columns(
    last_sn=pl.when(pl.col("wx_snow") == 1).then(pl.col("metar_time")).forward_fill().over("ADEP_mvt"),
    last_ic=pl.when(pl.col("wx_icing") == 1).then(pl.col("metar_time")).forward_fill().over("ADEP_mvt"))
m = m.with_columns(h_since_sn=((pl.col("metar_time") - pl.col("last_sn")).dt.total_minutes() / 60).clip(0, 24 * 30),
                   h_since_ic=((pl.col("metar_time") - pl.col("last_ic")).dt.total_minutes() / 60).clip(0, 24 * 30))
for hh in (12, 24, 48):
    m = m.with_columns(pl.col("wx_snow").rolling_max_by("metar_time", window_size=f"{hh}h").over("ADEP_mvt").alias(f"sn_{hh}h"),
                       pl.col("wx_icing").rolling_max_by("metar_time", window_size=f"{hh}h").over("ADEP_mvt").alias(f"ic_{hh}h"))
NEW = ["h_since_sn", "h_since_ic", "sn_12h", "sn_24h", "sn_48h", "ic_12h", "ic_24h", "ic_48h"]
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]
dep = pl.concat([pl.read_parquet(p, columns=cols) for p in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(pl.col("PHASE_mvt") == "DEP").with_columns(
    t=pl.col("AOBT_3_flt").fill_null(pl.col("MVT_TIME_UTC_mvt") - pl.duration(minutes=15)).dt.cast_time_unit("us"), month=pl.col("MVT_TIME_UTC_mvt").dt.month()).sort("t")
feat = dep.join_asof(m.select("ADEP_mvt", "metar_time", *NEW).sort("metar_time"), left_on="t", right_on="metar_time", by="ADEP_mvt", strategy="backward", tolerance="3h")
schema = pl.read_parquet_schema("data/features_e106.parquet")
CTRL = [c for c in ("d_mvt_aobt", "d_mvt_lobt", "taxiing_at_ob", "rwy_dep_before_30", "dep_before_30", "offblocks_around_ob_30", "nbf_ap_30_0_mean",
                    "nb_rwy_30_30_mean", "wx_temp", "wx_snow", "wx_snow_recent", "wx_icing", "wx_icing_recent", "wx_freezing", "wx_vis_mi") if c in schema]
f = pl.read_parquet("data/features_e106.parquet", columns=["MVT_ID_mvt", "split"] + CTRL).filter(pl.col("split") == "train")
oof = pl.concat([pl.DataFrame({k: np.load(p)[k] for k in ("mvt_id", "pred", "y")}) for p in sorted(glob.glob("data/cv_preds_e150c_e64_keep/fold_*.npz"))]).rename({"mvt_id": "MVT_ID_mvt"})
d = feat.select("MVT_ID_mvt", "month", *NEW).join(f, on="MVT_ID_mvt").join(oof, on="MVT_ID_mvt").with_columns(r=pl.col("y") - pl.col("pred")).filter(
    (pl.col("y") <= 7200) & (pl.col("r").abs() < 1800))
print("controls:", CTRL)
print("snow within 24 h at off-block:", round(float(d["sn_24h"].mean()), 4), "| icing within 24 h:", round(float(d["ic_24h"].mean()), 4))


def rank(x):
    return pl.Series(x).fill_nan(None).rank().fill_null(0).to_numpy().astype(float)


def partial(yv, x, controls):
    Z = np.column_stack([np.ones(len(yv))] + [rank(c) for c in controls])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return float(np.corrcoef(res(rank(yv)), res(rank(x)))[0, 1])


prm = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.9, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=42, num_threads=4, deterministic=True)
for label, sub, split in (("all rows, half -> half", d, ((1, 6), (7, 12))), ("winter (Jan, Feb, Dec), Jan+Feb -> Dec and back", d.filter(pl.col("month").is_in([1, 2, 12])), ((1, 2), (12, 12)))):
    halves = {"A": sub.filter(pl.col("month").is_between(*split[0])), "B": sub.filter(pl.col("month").is_between(*split[1]))}
    print(f"== {label}: rows {sub.height}")
    for c in NEW:
        p = [partial(g["r"].to_numpy(), g[c].to_numpy(), [g[k_].to_numpy() for k_ in CTRL]) for g in halves.values()]
        print(f"   {c:11s} partial A {p[0]:+.4f} / B {p[1]:+.4f}")
    for tr_h, te_h in (("A", "B"), ("B", "A")):
        a_, b_ = halves[tr_h], halves[te_h]
        line = f"   stage {tr_h} -> {te_h}: no stage {np.sqrt((b_['r'].to_numpy() ** 2).mean()):.3f}"
        for name, cs_ in (("controls", CTRL), ("+winter memory", CTRL + NEW)):
            mm = lgb.train(prm, lgb.Dataset(a_.select(cs_).to_numpy(), a_["r"].to_numpy()), 300)
            line += f" | {name} {np.sqrt(((b_['r'].to_numpy() - mm.predict(b_.select(cs_).to_numpy())) ** 2).mean()):.3f}"
        print(line)

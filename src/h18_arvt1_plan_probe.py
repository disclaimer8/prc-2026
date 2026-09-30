"""H18 (journal 185): kind-mango's filed-plan residuals on today's base (E72 tested only raw MVT - ARVT_1 on 19.09, +0.06, 3/6,
while route target encodings were still in the model; te_*/ter_* are dropped since E99/E110). Features per departure:
  plan_taxi_res = (MVT - ARVT_1) - its median per route (ADEP, ADES) from the other half year, clip +-3600
  plan_nm_taxi  = (ARVT_1 - EOBT_1) - its median per (ADEP, ADES, aircraft type) from the other half year, clip +-3600
Screen on the production OOF residual (cv_preds_e116prod), all rows and coverage < 2, as H17.
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import lightgbm as lgb
import numpy as np
import polars as pl

ep = lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt", "ARVT_1_flt", "EOBT_1_flt"]
raw = pl.concat([pl.read_parquet(p, columns=cols) for p in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(pl.col("PHASE_mvt") == "DEP")
d0 = raw.with_columns(month=pl.col("MVT_TIME_UTC_mvt").dt.month(), a=ep("MVT_TIME_UTC_mvt") - ep("ARVT_1_flt"), b=ep("ARVT_1_flt") - ep("EOBT_1_flt")).with_columns(
    half=pl.when(pl.col("month") <= 6).then(1).otherwise(2))
parts = []
for h in (1, 2):
    o = d0.filter(pl.col("half") != h)
    ma = o.group_by("ADEP_mvt", "ADES_mvt").agg(ma=pl.col("a").median())
    mb = o.group_by("ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt").agg(mb=pl.col("b").median())
    parts.append(d0.filter(pl.col("half") == h).join(ma, on=["ADEP_mvt", "ADES_mvt"], how="left").join(mb, on=["ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt"], how="left")
                 .select("MVT_ID_mvt", plan_taxi_res=(pl.col("a") - pl.col("ma")).clip(-3600, 3600), plan_nm_taxi=(pl.col("b") - pl.col("mb")).clip(-3600, 3600)))
feat = pl.concat(parts)
NEW = ["plan_taxi_res", "plan_nm_taxi"]
schema = pl.read_parquet_schema("data/features_e106.parquet")
CTRL = [c for c in ("d_mvt_aobt", "d_mvt_lobt", "d_mvt_eobt", "d_mvt_sched", "taxiing_at_ob", "rwy_dep_before_30", "dep_before_30", "arr_before_30",
                    "offblocks_around_ob_30", "nbf_ap_30_0_mean", "nb_rwy_30_30_mean") if c in schema]
f = pl.read_parquet("data/features_e106.parquet", columns=["MVT_ID_mvt", "split", "month"] + CTRL).filter(pl.col("split") == "train")
cov = pl.concat([pl.read_parquet(p, columns=["MVT_ID_mvt", "coverage"]) for p in sorted(glob.glob("data/adsblol_kt1/offblock2_2025-*.parquet"))]).unique("MVT_ID_mvt")
oof = pl.concat([pl.DataFrame({k: np.load(p)[k] for k in ("mvt_id", "pred", "y")}) for p in sorted(glob.glob("data/cv_preds_e116prod/fold_*.npz"))]).rename({"mvt_id": "MVT_ID_mvt"})
d = f.join(feat, on="MVT_ID_mvt").join(oof, on="MVT_ID_mvt").join(cov, on="MVT_ID_mvt", how="left").with_columns(pl.col("coverage").fill_null(0), r=pl.col("y") - pl.col("pred"))
d = d.filter((pl.col("y") <= 7200) & (pl.col("r").abs() < 1800))
print("known:", {c: round(float(d[c].is_not_null().mean()), 3) for c in NEW})


def rank(x):
    return pl.Series(x).fill_nan(None).rank().fill_null(0).to_numpy().astype(float)


def partial(yv, x, controls):
    Z = np.column_stack([np.ones(len(yv))] + [rank(c) for c in controls])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return float(np.corrcoef(res(rank(yv)), res(rank(x)))[0, 1])


prm = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.9, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=42, num_threads=4, deterministic=True)
for label, sub in (("all rows", d), ("coverage < 2", d.filter(pl.col("coverage") < 2))):
    halves = {h: sub.filter(pl.col("month").is_between(*rng)) for h, rng in (("H1", (1, 6)), ("H2", (7, 12)))}
    print(f"== {label}: rows {sub.height}")
    for c in NEW:
        p = [partial(g["r"].to_numpy(), g[c].to_numpy(), [g[k_].to_numpy() for k_ in CTRL]) for g in halves.values()]
        print(f"   {c:14s} partial H1 {p[0]:+.4f} / H2 {p[1]:+.4f}")
    for tr_h, te_h in (("H1", "H2"), ("H2", "H1")):
        tr, te = halves[tr_h], halves[te_h]
        line = f"   stage {tr_h} -> {te_h}: no stage {np.sqrt((te['r'].to_numpy() ** 2).mean()):.3f}"
        for name, cs_ in (("controls", CTRL), ("+plan residuals", CTRL + NEW)):
            m = lgb.train(prm, lgb.Dataset(tr.select(cs_).to_numpy(), tr["r"].to_numpy()), 300)
            line += f" | {name} {np.sqrt(((te['r'].to_numpy() - m.predict(te.select(cs_).to_numpy())) ** 2).mean()):.3f}"
        print(line)

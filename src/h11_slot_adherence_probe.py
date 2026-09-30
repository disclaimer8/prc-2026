"""EUROCONTROL AIU daily ATFM slot adherence (journal 167) as a day-level signal for the base model.

Source: https://www.eurocontrol.int/performance/data/download/csv/atfm_slot_adherence_{2025,2026}.csv (listed on
ansperformance.eu/data, "Daily ATFM slot adherence by airport", Jan 2019 - Aug 2026; organiser 17.09: "can be considered open
data for the challenge"). Per airport x UTC day: departures, ATFM-regulated departures (with a CTOT), and how many of those
left early / inside / late of the slot window. Features per departure (its airport, the UTC day of its take-off):
  h11_reg_share  regulated / departures,  h11_reg_n  regulated count,  h11_late_share / h11_early_share  of the regulated.
Screen as src/h10_corridor_lag_probe.py: production OOF residual, y <= 7200, |r| < 1800, partial Spearman given the model's
own windows per half year, LightGBM stage half -> half.
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import lightgbm as lgb
import numpy as np
import polars as pl

Y = "TAXITIME_SEC_mvt"
sa = pl.concat([pl.read_csv(f"data/ext_pru/atfm_slot_adherence_{y}.csv", infer_schema_length=0) for y in (2025, 2026)])
num = ["FLT_DEP_1", "FLT_DEP_REG_1", "FLT_DEP_OUT_EARLY_1", "FLT_DEP_IN_1", "FLT_DEP_OUT_LATE_1"]
sa = sa.with_columns([pl.col(c).cast(pl.Float64) for c in num]).select(
    ap=pl.col("APT_ICAO"), day=pl.col("FLT_DATE").str.to_date(),
    h11_reg_share=pl.col("FLT_DEP_REG_1") / pl.col("FLT_DEP_1"), h11_reg_n=pl.col("FLT_DEP_REG_1"),
    h11_late_share=pl.col("FLT_DEP_OUT_LATE_1") / pl.col("FLT_DEP_REG_1"),
    h11_early_share=pl.col("FLT_DEP_OUT_EARLY_1") / pl.col("FLT_DEP_REG_1"))
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", Y]
raw = pl.concat([pl.read_parquet(p, columns=cols) for p in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))])
dep = raw.filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt", ap=pl.col("ADEP_mvt"), day=pl.col("MVT_TIME_UTC_mvt").dt.date(),
                                                      month=pl.col("MVT_TIME_UTC_mvt").dt.month())
feat = dep.join(sa, on=["ap", "day"], how="left")
print(f"departures {feat.height}; no slot-adherence row {feat['h11_reg_share'].null_count()}")
print(sa.filter(pl.col("ap").is_in(["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]))
      .group_by("ap").agg(pl.col("h11_reg_share").mean().alias("mean_reg_share"), pl.col("h11_late_share").mean().alias("mean_late")).sort("ap"))
NEW = ["h11_reg_share", "h11_reg_n", "h11_late_share", "h11_early_share"]
feat = feat.select("MVT_ID_mvt", *NEW)

schema = pl.read_parquet_schema("data/features_e106.parquet")
CTRL = [c for c in ("d_mvt_aobt", "d_mvt_lobt", "taxiing_at_ob", "rwy_dep_before_10", "rwy_dep_before_30", "rwy_dep_after_10",
                    "dep_before_30", "arr_before_30", "offblocks_around_ob_30", "nbf_ap_30_0_mean", "nb_rwy_30_30_mean",
                    "nb_ap_30_30_mean", "taxi_in_mean_30") if c in schema]
print("controls:", CTRL)
base = pl.read_parquet("data/features_e106.parquet", columns=["MVT_ID_mvt", "split"] + CTRL).filter(pl.col("split") == "train").drop("split")
oof = pl.concat([pl.DataFrame({k: np.load(p)[k] for k in ("mvt_id", "pred", "y")}) for p in sorted(glob.glob("data/cv_preds_e116prod/fold_*.npz"))]).rename({"mvt_id": "MVT_ID_mvt"})
full = dep.select("MVT_ID_mvt", "month").join(feat, on="MVT_ID_mvt").join(base, on="MVT_ID_mvt").join(oof, on="MVT_ID_mvt") \
    .with_columns(r=pl.col("y") - pl.col("pred"))
d = full.filter((pl.col("y") <= 7200) & (pl.col("r").abs() < 1800))
print(f"rows {d.height}; residual sd {d['r'].std():.1f}")
def rank(x):
    return pl.Series(x).fill_nan(None).rank().fill_null(0).to_numpy().astype(float)


def partial(y, x, controls):
    Z = np.column_stack([np.ones(len(y))] + [rank(c) for c in controls])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return float(np.corrcoef(res(rank(y)), res(rank(x)))[0, 1])


halves = {h: d.filter(pl.col("month").is_between(*rng)) for h, rng in (("H1", (1, 6)), ("H2", (7, 12)))}
for c in NEW:
    p = [partial(g["r"].to_numpy(), g[c].to_numpy(), [g[k_].to_numpy() for k_ in CTRL]) for g in halves.values()]
    print(f"{c:15s} partial(r | controls) H1 {p[0]:+.4f} / H2 {p[1]:+.4f}")
params = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=42, num_threads=2, deterministic=True)
print("stage on the residual, half -> half (RMSE of r; lower = the features carry signal beyond the controls)")
for tr_h, te_h in (("H1", "H2"), ("H2", "H1")):
    tr, te = halves[tr_h], halves[te_h]
    line = f"  {tr_h} -> {te_h}: no stage {np.sqrt((te['r'].to_numpy() ** 2).mean()):.3f}"
    for name, cs_ in (("controls", CTRL), ("+slot adherence", CTRL + NEW)):
        m = lgb.train(params, lgb.Dataset(tr.select(cs_).to_numpy(), tr["r"].to_numpy()), 300)
        line += f" | {name} {np.sqrt(((te['r'].to_numpy() - m.predict(te.select(cs_).to_numpy())) ** 2).mean()):.3f}"
    print(line)

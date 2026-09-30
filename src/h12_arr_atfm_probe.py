"""EUROCONTROL AIU daily airport arrival ATFM delay by cause (journal 168) as a day-level signal for the base model.

Files apt_dly_{2024,2025,2026}.csv downloaded by Denys from ansperformance.eu/data ("Airport arrival ATFM delays - no post ops
adjustments", daily, Jan 2019 - Aug 2026; the site's robots.txt bars AI agents from /data/). Per airport x UTC day, per arrival:
total ATFM arrival delay, weather (W), de-icing (D), capacity (C + G), staffing (S), share of delayed arrivals. De-icing after
off-block is part of the taxi-out, and weather/de-icing regulation marks the bad days. Screen as src/h11_slot_adherence_probe.py.
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import lightgbm as lgb
import numpy as np
import polars as pl

Y = "TAXITIME_SEC_mvt"
AP10 = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]
raw_d = pl.concat([pl.read_csv(f"data/ext_pru/apt_dly_{y}.csv", infer_schema_length=0, encoding="utf8-lossy") for y in (2024, 2025, 2026)])
dc = [c for c in raw_d.columns if c.startswith("DLY_APT_ARR") or c.startswith("FLT_ARR")]
raw_d = raw_d.with_columns([pl.col(c).cast(pl.Float64) for c in dc]).filter(pl.col("APT_ICAO").is_in(AP10))
per = lambda c: (pl.col(c) / pl.col("FLT_ARR_1")).fill_nan(0.0)
sa = raw_d.select(ap=pl.col("APT_ICAO"), day=pl.col("FLT_DATE").str.slice(0, 10).str.to_date(),
                  h12_dly_per_arr=per("DLY_APT_ARR_1"), h12_w_per_arr=per("DLY_APT_ARR_W_1"), h12_d_per_arr=per("DLY_APT_ARR_D_1"),
                  h12_cap_per_arr=((pl.col("DLY_APT_ARR_C_1") + pl.col("DLY_APT_ARR_G_1")) / pl.col("FLT_ARR_1")).fill_nan(0.0),
                  h12_staff_per_arr=per("DLY_APT_ARR_S_1"), h12_share_dly=per("FLT_ARR_1_DLY"))
for c in ("h12_w_per_arr", "h12_d_per_arr"):
    k = sa.with_columns(y=pl.col("day").dt.year()).group_by("ap", "y").agg((pl.col(c) > 0).sum().alias("days")).sort("ap", "y")
    print(c, "days > 0 by airport and year:", {f"{r[0]}/{r[1]}": r[2] for r in k.iter_rows()})
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", Y]
raw = pl.concat([pl.read_parquet(p, columns=cols) for p in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))])
dep = raw.filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt", ap=pl.col("ADEP_mvt"), day=pl.col("MVT_TIME_UTC_mvt").dt.date(),
                                                      month=pl.col("MVT_TIME_UTC_mvt").dt.month())
feat = dep.join(sa, on=["ap", "day"], how="left")
print(f"departures {feat.height}; no slot-adherence row {feat['h12_dly_per_arr'].null_count()}")
print(sa.group_by("ap").agg(pl.col("h12_dly_per_arr").mean(), pl.col("h12_w_per_arr").mean(), pl.col("h12_d_per_arr").mean()).sort("ap"))
NEW = ["h12_dly_per_arr", "h12_w_per_arr", "h12_d_per_arr", "h12_cap_per_arr", "h12_staff_per_arr", "h12_share_dly"]
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
    for name, cs_ in (("controls", CTRL), ("+arrival ATFM delay", CTRL + NEW), ("+weather/de-icing only", CTRL + ["h12_w_per_arr", "h12_d_per_arr"])):
        m = lgb.train(params, lgb.Dataset(tr.select(cs_).to_numpy(), tr["r"].to_numpy()), 300)
        line += f" | {name} {np.sqrt(((te['r'].to_numpy() - m.predict(te.select(cs_).to_numpy())) ** 2).mean()):.3f}"
    print(line)

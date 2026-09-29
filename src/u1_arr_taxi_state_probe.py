"""U1 (journal 137, 150): arrivals' taxi-in residual as a live state sensor of the aerodrome. Arrivals in the ranking file carry
their taxi-in (block and taxi time are hidden only for departures), so "arrivals are taxiing 15 min slower than normal right now"
is observable at prediction time. The model already has raw taxi_in_mean_* means; this measures the residual form.

Normal taxi-in = median per (ADES, RUNWAY, STAND) with fallback (ADES, RUNWAY) and (ADES), from the OTHER months of 2025
(leave-month-out, so the normal never contains the row's own month). For each departure: mean clipped residual (+-1800 s) and
count of arrivals that went on block within (t_ob - W, t_ob], W = 30 / 60 min; t_ob = AOBT_3 (LOBT where missing).
Screen on the production OOF residual (data/cv_preds_e116prod), rows y <= 7200 and |r| < 1800: partial Spearman given the
taxi-in and traffic windows the model already has, by half year and airport, and a LightGBM stage half-year -> half-year
(controls vs controls + U1).
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import lightgbm as lgb
import numpy as np
import polars as pl

Y = "TAXITIME_SEC_mvt"
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt", Y]
raw = pl.concat([pl.read_parquet(p, columns=cols) for p in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))])
ep = lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)
arr = raw.filter((pl.col("PHASE_mvt") == "ARR") & pl.col(Y).is_between(0, 7200)).select(
    ap=pl.col("ADES_mvt"), rwy=pl.col("RUNWAY_mvt"), stand=pl.col("STAND_mvt"), month=pl.col("MVT_TIME_UTC_mvt").dt.month(),
    t_in=ep("MVT_TIME_UTC_mvt") + pl.col(Y).cast(pl.Float64), x=pl.col(Y).cast(pl.Float64))
# leave-month-out normals
parts = []
for m in range(1, 13):
    o = arr.filter(pl.col("month") != m)
    n3 = o.group_by("ap", "rwy", "stand").agg(n3=pl.col("x").median(), c3=pl.len())
    n2 = o.group_by("ap", "rwy").agg(n2=pl.col("x").median())
    n1 = o.group_by("ap").agg(n1=pl.col("x").median())
    a = arr.filter(pl.col("month") == m).join(n3, on=["ap", "rwy", "stand"], how="left").join(n2, on=["ap", "rwy"], how="left").join(n1, on="ap", how="left")
    parts.append(a.with_columns(normal=pl.when(pl.col("c3") >= 20).then(pl.col("n3")).otherwise(pl.coalesce("n2", "n1"))))
arr = pl.concat(parts).with_columns(res=(pl.col("x") - pl.col("normal")).clip(-1800, 1800)).drop_nulls("res")
print(f"arrivals {arr.height}, residual median {arr['res'].median():.0f}, p10/p90 {arr['res'].quantile(0.1):.0f}/{arr['res'].quantile(0.9):.0f}")

dep = raw.filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt", ap=pl.col("ADEP_mvt"), t_ob=pl.coalesce(ep("AOBT_3_flt"), ep("LOBT_flt")),
                                                      month=pl.col("MVT_TIME_UTC_mvt").dt.month()).drop_nulls("t_ob")
out = []
for (ap,), g in dep.group_by("ap"):
    a = arr.filter(pl.col("ap") == ap).sort("t_in")
    t, r = a["t_in"].to_numpy(), a["res"].to_numpy()
    cs = np.concatenate([[0.0], np.cumsum(r)])
    q = g["t_ob"].to_numpy()
    cols_out = {"MVT_ID_mvt": g["MVT_ID_mvt"]}
    for w in (30, 60):
        lo, hi = np.searchsorted(t, q - 60 * w, "right"), np.searchsorted(t, q, "right")
        n = hi - lo
        cols_out[f"u1_res_{w}"] = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
        cols_out[f"u1_n_{w}"] = n.astype(float)
    out.append(pl.DataFrame(cols_out))
u1 = pl.concat(out)
NEW = ["u1_res_30", "u1_res_60", "u1_n_30", "u1_n_60"]

schema = pl.read_parquet_schema("data/features_e106.parquet")
CTRL = [c for c in ("d_mvt_aobt", "d_mvt_lobt", "taxi_in_mean_30", "taxi_in_mean_60", "taxi_in_mean_180", "taxi_in_n_30", "dep_before_30",
                    "arr_before_30", "arr_around_ob_30", "offblocks_around_ob_30", "rwy_dep_before_30", "nbf_ap_30_0_mean") if c in schema]
print("controls:", CTRL)
base = pl.read_parquet("data/features_e106.parquet", columns=["MVT_ID_mvt", "split", "ADEP_mvt"] + CTRL).filter(pl.col("split") == "train").drop("split")
oof = pl.concat([pl.DataFrame({k: np.load(p)[k] for k in ("mvt_id", "pred", "y")}) for p in sorted(glob.glob("data/cv_preds_e116prod/fold_*.npz"))]).rename({"mvt_id": "MVT_ID_mvt"})
d = (dep.select("MVT_ID_mvt", "month").join(u1, on="MVT_ID_mvt").join(base, on="MVT_ID_mvt").join(oof, on="MVT_ID_mvt")
     .with_columns(r=pl.col("y") - pl.col("pred")).filter((pl.col("y") <= 7200) & (pl.col("r").abs() < 1800)))
print(f"rows {d.height}; residual sd {d['r'].std():.1f}")


def rank(x):
    return pl.Series(x).fill_nan(None).rank().fill_null(0).to_numpy().astype(float)


def partial(y, x, controls):
    Z = np.column_stack([np.ones(len(y))] + [rank(c) for c in controls])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return float(np.corrcoef(res(rank(y)), res(rank(x)))[0, 1])


halves = {h: d.filter(pl.col("month").is_between(*rng)) for h, rng in (("H1", (1, 6)), ("H2", (7, 12)))}
for c in NEW:
    p = [partial(g["r"].to_numpy(), g[c].to_numpy(), [g[k].to_numpy() for k in CTRL]) for g in halves.values()]
    print(f"{c:12s} partial(r | controls) H1 {p[0]:+.4f} / H2 {p[1]:+.4f}")
for ap in ("LTFM", "LFPG", "LIRF", "EGLL", "LEMD"):
    g = d.filter(pl.col("ADEP_mvt") == ap)
    print(f"  {ap}: u1_res_30 partial {partial(g['r'].to_numpy(), g['u1_res_30'].to_numpy(), [g[k].to_numpy() for k in CTRL]):+.4f}, n {g.height}")
params = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=42, num_threads=2, deterministic=True)
print("stage on the residual, half -> half (RMSE of r; lower = the features carry signal beyond the controls)")
for tr_h, te_h in (("H1", "H2"), ("H2", "H1")):
    tr, te = halves[tr_h], halves[te_h]
    line = f"  {tr_h} -> {te_h}: no stage {np.sqrt((te['r'].to_numpy() ** 2).mean()):.3f}"
    for name, cs_ in (("controls", CTRL), ("controls + U1", CTRL + NEW)):
        m = lgb.train(params, lgb.Dataset(tr.select(cs_).to_numpy(), tr["r"].to_numpy()), 300)
        line += f" | {name} {np.sqrt(((te['r'].to_numpy() - m.predict(te.select(cs_).to_numpy())) ** 2).mean()):.3f}"
    print(line)

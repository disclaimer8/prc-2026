"""External review 30.09 (journal 164), after the Canedo dyngraph2vec read:
  1. occupancy of the row's corridor 15 and 30 min before its off-block and the change to the current value (the lagged
     state is "the only transferable piece of dyngraph2vec");
  2. when the lags are empty: how many departures reach the holding point before the row by unimpeded taxi time
     (the plain neighbour count is already in E151).

Corridor proxies from the movement records only (the taxiway paths are not stored, data/taxi_route.parquet has lengths):
  taxiing interval of a departure = [AOBT_3, else LOBT; take-off], of an arrival = [landing; landing + taxi-in]
  (arrival taxi-in is known in ranking too). Counts at tau = t0 - 30 min, t0 - 15 min, t0 (t0 = the row's AOBT_3/LOBT):
  h10_rwy_*  departures taxiing to the same runway,  h10_ap_*  all movements taxiing at the airport,
  h10_*_d15 / _d30  = value at t0 minus value at t0 - 15 / - 30 min.
  h10_hold_q  departures on the same runway that reach the holding point (t_ob + unimpeded) no later than the row and
              have not taken off by the row's own holding-point time; unimpeded = P10 taxi-out of (airport, stand, runway),
              fallback (airport, runway), from the other half year (no own-label leak).
Screen as src/h6h7_wake_probe.py: production OOF residual (data/cv_preds_e116prod), y <= 7200, |r| < 1800, partial Spearman
given the model's own windows (incl. taxiing_at_ob) per half year, and a LightGBM stage half -> half.
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
mv = raw.with_columns(t=ep("MVT_TIME_UTC_mvt"), month=pl.col("MVT_TIME_UTC_mvt").dt.month())
dep = mv.filter(pl.col("PHASE_mvt") == "DEP").select(
    "MVT_ID_mvt", "month", ap=pl.col("ADEP_mvt"), rwy=pl.col("RUNWAY_mvt"), stand=pl.col("STAND_mvt"), y=pl.col(Y),
    s=pl.coalesce(ep("AOBT_3_flt"), ep("LOBT_flt")), e=pl.col("t"))
arr = mv.filter(pl.col("PHASE_mvt") == "ARR").select(
    ap=pl.col("ADES_mvt"), s=pl.col("t"), e=pl.col("t") + pl.col(Y).cast(pl.Float64)).drop_nulls()
print(f"departures {dep.height} (no off-block proxy {dep['s'].null_count()}), arrivals {arr.height}")

# unimpeded taxi = P10 of the label in the OTHER half year (a row never sees its own half's labels)
dep = dep.with_columns(half=pl.when(pl.col("month") <= 6).then(1).otherwise(2))
unimp = []
for h in (1, 2):
    o = dep.filter((pl.col("half") != h) & pl.col("y").is_between(60, 7200))
    a = o.group_by("ap", "stand", "rwy").agg(u1=pl.col("y").quantile(0.10), n1=pl.len()).filter(pl.col("n1") >= 20)
    b = o.group_by("ap", "rwy").agg(u2=pl.col("y").quantile(0.10))
    unimp.append(dep.filter(pl.col("half") == h).select("MVT_ID_mvt", "ap", "stand", "rwy")
                 .join(a, on=["ap", "stand", "rwy"], how="left").join(b, on=["ap", "rwy"], how="left")
                 .select("MVT_ID_mvt", u=pl.coalesce("u1", "u2")))
dep = dep.join(pl.concat(unimp), on="MVT_ID_mvt", how="left").with_columns(hp=pl.col("s") + pl.col("u"))


def taxiing(s_sorted, e_sorted, tau):
    """number of intervals [s, e] containing tau"""
    return np.searchsorted(s_sorted, tau, "right") - np.searchsorted(e_sorted, tau, "left")


out = []
for ap in dep["ap"].unique().to_list():
    d_ap = dep.filter(pl.col("ap") == ap)
    a_ap = arr.filter(pl.col("ap") == ap)
    ok = d_ap.drop_nulls("s")
    S_all = np.sort(np.concatenate([ok["s"].to_numpy(), a_ap["s"].to_numpy()]))
    E_all = np.sort(np.concatenate([ok["e"].to_numpy(), a_ap["e"].to_numpy()]))
    for rw in d_ap["rwy"].unique().to_list():
        g = d_ap.filter(pl.col("rwy") == rw)
        gk = g.drop_nulls("s")
        Sr, Er = np.sort(gk["s"].to_numpy()), np.sort(gk["e"].to_numpy())
        t0 = g["s"].to_numpy()
        valid = ~np.isnan(t0)
        t0f = np.where(valid, t0, 0.0)
        res = {"MVT_ID_mvt": g["MVT_ID_mvt"]}
        for name, S, E in (("rwy", Sr, Er), ("ap", S_all, E_all)):
            c0 = taxiing(S, E, t0f) - 1.0  # the row itself starts at t0
            c15, c30 = taxiing(S, E, t0f - 900).astype(float), taxiing(S, E, t0f - 1800).astype(float)
            for k, v in ((f"h10_{name}_0", c0), (f"h10_{name}_15", c15), (f"h10_{name}_30", c30),
                         (f"h10_{name}_d15", c0 - c15), (f"h10_{name}_d30", c0 - c30)):
                res[k] = np.where(valid, v, np.nan)
        # queue at the holding point: others with hp_j <= hp_i that are still on the ground (take-off after hp_i)
        # an aircraft cannot wait at the holding point after its take-off: hp' = min(hp, take-off), so every [hp', take-off]
        # is a proper interval and the queue at the row's hp' is the interval count there, minus the row itself
        hp = np.fmin(g["hp"].to_numpy(), g["e"].to_numpy())
        hk = gk.drop_nulls("hp")
        H = np.sort(np.fmin(hk["hp"].to_numpy(), hk["e"].to_numpy()))
        T = np.sort(hk["e"].to_numpy())
        res["h10_hold_q"] = np.where(np.isnan(hp), np.nan, taxiing(H, T, np.nan_to_num(hp)) - 1.0)
        out.append(pl.DataFrame(res))
feat = pl.concat(out)
NEW = [c for c in feat.columns if c.startswith("h10_")]

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
print(f"rows {d.height}; residual sd {d['r'].std():.1f}; null share of h10_rwy_0 {d['h10_rwy_0'].null_count() / d.height:.4f}")
print("already in the model? Spearman of h10_ap_0 with taxiing_at_ob:",
      f"{np.corrcoef(d['h10_ap_0'].fill_null(0).rank().to_numpy(), d['taxiing_at_ob'].fill_null(0).rank().to_numpy())[0, 1]:.3f}"
      if "taxiing_at_ob" in d.columns else "n/a")
print(d.select(NEW).describe().filter(pl.col("statistic").is_in(["mean", "std", "min", "max"])))


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
LAG = [c for c in NEW if c != "h10_hold_q"]
print("stage on the residual, half -> half (RMSE of r; lower = the features carry signal beyond the controls)")
for tr_h, te_h in (("H1", "H2"), ("H2", "H1")):
    tr, te = halves[tr_h], halves[te_h]
    line = f"  {tr_h} -> {te_h}: no stage {np.sqrt((te['r'].to_numpy() ** 2).mean()):.3f}"
    for name, cs_ in (("controls", CTRL), ("+lags", CTRL + LAG), ("+hold_q", CTRL + ["h10_hold_q"]), ("+both", CTRL + NEW)):
        m = lgb.train(params, lgb.Dataset(tr.select(cs_).to_numpy(), tr["r"].to_numpy()), 300)
        line += f" | {name} {np.sqrt(((te['r'].to_numpy() - m.predict(te.select(cs_).to_numpy())) ** 2).mean()):.3f}"
    print(line)

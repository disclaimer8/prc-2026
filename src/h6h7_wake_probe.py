"""Hypotheses 6 and 7 (journal 152): wake separation ladder on the runway; separate regressor heads by wake category.

H6. For every departure: the departure that took off just before it on the same runway (ADEP, RUNWAY), its wake category,
the gap between the two take-offs, the required gap = median gap of that (leader, follower) category pair at that airport
(GREKI, Discord 17.09: M behind M 113 s, M behind H 148 s, behind J 182-241 s), the reviewer's excess hold max(0, req - gap),
the saturation margin gap - req, and a ladder-weighted queue: sum of required gaps over the take-offs on the same runway
between the row's off-block (AOBT_3, LOBT where missing) and its own take-off, plus the heavy/super count ahead in 10 min (G7).
H7. Bias and RMSE of the production OOF residual by the row's own wake category (and airport x category): the part a separate
head could remove as a bias is sum n_c * mean_c^2.

Screen on the production OOF residual (data/cv_preds_e116prod), rows y <= 7200 and |r| < 1800, as src/u1_arr_taxi_state_probe.py:
partial Spearman given what the model already has, by half year, and a LightGBM stage half-year -> half-year.
Wake category = WK_TBL_CAT_flt, falling back to the most common category of the aircraft type (NM fields are null without NM).
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import lightgbm as lgb
import numpy as np
import polars as pl

Y = "TAXITIME_SEC_mvt"
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt", Y]
raw = pl.concat([pl.read_parquet(p, columns=cols) for p in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))])
ep = lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)
tmode = raw.drop_nulls("WK_TBL_CAT_flt").group_by("AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt").agg(n=pl.len()).sort("n", descending=True) \
    .unique("AIRCRAFT_TYPE_mvt", keep="first").select("AIRCRAFT_TYPE_mvt", tcat="WK_TBL_CAT_flt")
dep = (raw.filter(pl.col("PHASE_mvt") == "DEP").join(tmode, on="AIRCRAFT_TYPE_mvt", how="left")
       .select("MVT_ID_mvt", ap=pl.col("ADEP_mvt"), rwy=pl.col("RUNWAY_mvt"), cat=pl.coalesce("WK_TBL_CAT_flt", "tcat").fill_null("M"),
               t=ep("MVT_TIME_UTC_mvt"), t_ob=pl.coalesce(ep("AOBT_3_flt"), ep("LOBT_flt")), month=pl.col("MVT_TIME_UTC_mvt").dt.month())
       .sort("ap", "rwy", "t")
       .with_columns(prev_cat=pl.col("cat").shift(1).over("ap", "rwy"), gap=(pl.col("t") - pl.col("t").shift(1).over("ap", "rwy"))))
# required gap = median gap of the (leader, follower) pair at the airport, over pairs 30 s < gap <= 600 s (a queue, not a lull)
req = (dep.filter(pl.col("gap").is_between(30, 600)).group_by("ap", "prev_cat", "cat").agg(req=pl.col("gap").median(), n=pl.len())
       .filter(pl.col("n") >= 50))
print("ladder (median gap, s) pooled over airports:")
print(dep.filter(pl.col("gap").is_between(30, 600)).group_by("prev_cat", "cat").agg(med=pl.col("gap").median(), n=pl.len()).sort("n", descending=True).head(10))
dep = dep.join(req.drop("n"), on=["ap", "prev_cat", "cat"], how="left").with_columns(
    h6_gap=pl.col("gap").clip(0, 1800), h6_req=pl.col("req"),
    h6_excess_hold=(pl.col("req") - pl.col("gap")).clip(0, None), h6_margin=(pl.col("gap") - pl.col("req")).clip(-600, 1800),
    h6_prev_heavy=pl.col("prev_cat").is_in(["H", "J"]).cast(pl.Float64))
print(f"departures {dep.height}; excess hold > 0 on {(dep['h6_excess_hold'] > 0).mean():.3f} of rows, > 30 s on {(dep['h6_excess_hold'] > 30).mean():.3f}")
# ladder-weighted queue and heavy count ahead, per (airport, runway)
out = []
for (ap, rw), g in dep.group_by("ap", "rwy"):
    t, q = g["t"].to_numpy(), g["t_ob"].to_numpy()
    rq = np.nan_to_num(g["req"].to_numpy(), nan=120.0)
    hv = g["cat"].is_in(["H", "J"]).to_numpy().astype(float)
    cr, ch = np.concatenate([[0.0], np.cumsum(rq)]), np.concatenate([[0.0], np.cumsum(hv)])
    lo = np.searchsorted(t, np.where(np.isnan(q), t, q), "right")
    hi = np.searchsorted(t, t, "left")  # strictly before own take-off
    lo = np.minimum(lo, hi)
    l10 = np.searchsorted(t, t - 600, "right")
    out.append(pl.DataFrame({"MVT_ID_mvt": g["MVT_ID_mvt"], "h6_ladder_q": np.where(np.isnan(q), np.nan, cr[hi] - cr[lo]),
                             "h6_n_ahead": np.where(np.isnan(q), np.nan, (hi - lo).astype(float)),
                             "g7_heavy_10": ch[hi] - ch[np.minimum(l10, hi)]}))
feat = dep.select("MVT_ID_mvt", "month", "cat", "h6_gap", "h6_req", "h6_excess_hold", "h6_margin", "h6_prev_heavy").join(pl.concat(out), on="MVT_ID_mvt")
NEW = ["h6_gap", "h6_req", "h6_excess_hold", "h6_margin", "h6_prev_heavy", "h6_ladder_q", "g7_heavy_10"]

schema = pl.read_parquet_schema("data/features_e106.parquet")
CTRL = [c for c in ("d_mvt_aobt", "d_mvt_lobt", "rwy_dep_before_10", "rwy_dep_before_30", "rwy_dep_after_10", "dep_before_30",
                    "arr_before_30", "offblocks_around_ob_30", "nbf_ap_30_0_mean", "taxi_in_mean_30") if c in schema]
print("controls:", CTRL)
base = pl.read_parquet("data/features_e106.parquet", columns=["MVT_ID_mvt", "split", "ADEP_mvt"] + CTRL).filter(pl.col("split") == "train").drop("split")
oof = pl.concat([pl.DataFrame({k: np.load(p)[k] for k in ("mvt_id", "pred", "y")}) for p in sorted(glob.glob("data/cv_preds_e116prod/fold_*.npz"))]).rename({"mvt_id": "MVT_ID_mvt"})
full = feat.join(base, on="MVT_ID_mvt").join(oof, on="MVT_ID_mvt").with_columns(r=pl.col("y") - pl.col("pred"))
d = full.filter((pl.col("y") <= 7200) & (pl.col("r").abs() < 1800))
print(f"rows {d.height}; residual sd {d['r'].std():.1f}")

# H7: residual by own wake category (all rows y <= 7200, no |r| cut: a head would face the tails too)
k = full.filter(pl.col("y") <= 7200)
n_all = full.height
print("H7 residual by own wake category (y <= 7200):")
print(k.group_by("cat").agg(n=pl.len(), bias=pl.col("r").mean(), rmse=(pl.col("r") ** 2).mean().sqrt()).sort("n", descending=True))
ac = k.group_by("ADEP_mvt", "cat").agg(n=pl.len(), bias=pl.col("r").mean())
ceil = float((ac["n"] * ac["bias"] ** 2).sum()) / n_all
print(f"H7 ceiling if a head removed every airport x category bias: full-set MSE -{ceil:.2f} s^2 = RMSE about -{ceil / (2 * 360):.3f} s "
      f"(RMSE ~360); largest |bias| cells:")
print(ac.filter(pl.col("n") >= 500).with_columns(a=pl.col("bias").abs()).sort("a", descending=True).head(6).drop("a"))


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
print("margin buckets (saturated runway = gap near the ladder): mean residual")
print(d.with_columns(b=pl.col("h6_margin").cut([-30, 0, 30, 60, 120, 300])).group_by("b").agg(n=pl.len(), r=pl.col("r").mean()).sort("b"))
params = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=42, num_threads=2, deterministic=True)
print("stage on the residual, half -> half (RMSE of r; lower = the features carry signal beyond the controls)")
for tr_h, te_h in (("H1", "H2"), ("H2", "H1")):
    tr, te = halves[tr_h], halves[te_h]
    line = f"  {tr_h} -> {te_h}: no stage {np.sqrt((te['r'].to_numpy() ** 2).mean()):.3f}"
    for name, cs_ in (("controls", CTRL), ("controls + H6/G7", CTRL + NEW)):
        m = lgb.train(params, lgb.Dataset(tr.select(cs_).to_numpy(), tr["r"].to_numpy()), 300)
        line += f" | {name} {np.sqrt(((te['r'].to_numpy() - m.predict(te.select(cs_).to_numpy())) ** 2).mean()):.3f}"
    print(line)

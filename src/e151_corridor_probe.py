"""E151 probe (journal 131): Yuan, Liu & Chen 2024, Appl. Sci. 14(21) 9968, table 2 — flows over the flight's OWN taxi interval.

For a departure i with taxi interval [a_i, t_i] = [AOBT(i), ATOT(i)] and other intervals [s_j, e_j]
(departures: [AOBT, ATOT]; arrivals: [ALDT, AIBT]):
    D1/A1  s_j < a_i,      a_i < e_j < t_i        (ahead of me, gone during my taxi)
    D2/A2  s_j > a_i,      e_j < t_i              (nested: started after me and finished before me; paper: corr 0.81 / 0.74)
    D3/A3  s_j < a_i,      e_j > t_i              (spans my whole taxi)
    D4/A4  a_i < s_j < t_i, e_j > t_i             (started during my taxi, still going)
Levels: airport, runway (same departure runway; arrivals on the same runway), corridor (runway x stand zone;
stand zone = stand_prefix, else a 600 m cell of the runway-relative stand position; arrivals only where a prefix exists).
AOBT = NM AOBT_3 (LOBT where missing), ATOT/ALDT = MVT time, AIBT = ALDT + taxi-in; all known in ranking too.

Screen on the base OOF residual (data/cv_preds_nomonth), rows y <= 7200 and |r| < 1800, by half year:
raw Spearman with y (paper check), partial Spearman with r given the windows we already have, and a small
LightGBM stage on r trained on one half and scored on the other: controls only vs controls + the new counts.
"""
import glob

import lightgbm as lgb
import numpy as np
import polars as pl

CELL = 600.0
KMAX = 400

raw = pl.concat([pl.read_parquet(p, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                                            "MVT_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt", "TAXITIME_SEC_mvt"])
                 for p in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))])
ep = lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)
dep = raw.filter(pl.col("PHASE_mvt") == "DEP").select(
    "MVT_ID_mvt", ap=pl.col("ADEP_mvt"), rwy=pl.col("RUNWAY_mvt"), t=ep("MVT_TIME_UTC_mvt"),
    a=pl.coalesce(ep("AOBT_3_flt"), ep("LOBT_flt")), month=pl.col("MVT_TIME_UTC_mvt").dt.month())
dep = dep.filter(pl.col("a").is_not_null() & (pl.col("t") - pl.col("a")).is_between(0, 7200))
arr = raw.filter(pl.col("PHASE_mvt") == "ARR").select(
    ap=pl.col("ADES_mvt"), rwy=pl.col("RUNWAY_mvt"), zone=pl.col("STAND_mvt").str.extract(r"^([A-Za-z]+)", 1),
    s=ep("MVT_TIME_UTC_mvt"), e=ep("MVT_TIME_UTC_mvt") + pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)).drop_nulls(["s", "e"])
geo = pl.read_parquet("data/features_e106.parquet", columns=["MVT_ID_mvt", "stand_prefix", "geo_dist_along", "geo_dist_across", "split"]
                      ).filter(pl.col("split") == "train").drop("split")
cell = lambda c: (pl.col(c).fill_nan(None) / CELL).floor().cast(pl.Int32).cast(pl.Utf8)
dep = dep.join(geo, on="MVT_ID_mvt", how="left").with_columns(
    zone=pl.coalesce(pl.col("stand_prefix"), cell("geo_dist_along") + "_" + cell("geo_dist_across")))


def counts(a, t, s, e):
    """D1..D4 for queries [a, t] against objects [s, e] (numpy float arrays). Self is excluded by the strict inequalities."""
    o = np.argsort(s, kind="stable"); s, e = s[o], e[o]
    es = np.sort(e)
    n_s_lt_a = np.searchsorted(s, a, "left")
    n_e_lt_t = np.searchsorted(es, t, "left")
    start_in = np.searchsorted(s, t, "left") - np.searchsorted(s, a, "right")
    end_in = n_e_lt_t - np.searchsorted(es, a, "right")
    lo, hi = np.searchsorted(s, a, "right"), np.searchsorted(s, t, "left")
    nested = np.zeros(len(a))
    for k in range(min(KMAX, int((hi - lo).max(initial=0)))):
        j = lo + k
        ok = j < hi
        nested[ok] += e[j[ok]] < t[ok]
    d1, d2, d4 = end_in - nested, nested, start_in - nested
    d3 = n_s_lt_a - n_e_lt_t + nested
    return d1, d2, d3, d4


def level(name, dkeys, akeys):
    parts = []
    agroups = {k: g for k, g in arr.group_by(akeys)} if akeys else {}
    for key, g in dep.drop_nulls(dkeys).group_by(dkeys):
        a, t = g["a"].to_numpy(), g["t"].to_numpy()
        cols = {"MVT_ID_mvt": g["MVT_ID_mvt"]}
        for i, v in enumerate(counts(a, t, a.copy(), t.copy()), 1):
            cols[f"{name}_D{i}"] = v
        ag = agroups.get(tuple(key[:len(akeys)]) if len(akeys) < len(dkeys) else key) if akeys else None
        if name == "cor":   # arrivals into the same stand zone (any runway), only where the zone is a letter prefix
            ag = agroups.get((key[0], key[2])) if akeys else None
        for i in range(1, 5):
            cols[f"{name}_A{i}"] = np.full(len(a), np.nan)
        if ag is not None and ag.height:
            for i, v in enumerate(counts(a, t, ag["s"].to_numpy(), ag["e"].to_numpy()), 1):
                cols[f"{name}_A{i}"] = v
        parts.append(pl.DataFrame(cols))
    out = pl.concat(parts)
    D = sum(pl.col(f"{name}_D{i}") for i in range(1, 5)); A = sum(pl.col(f"{name}_A{i}") for i in range(1, 5))
    return out.with_columns((D / (D + A)).alias(f"{name}_dprop"))


feats = level("ap", ["ap"], ["ap"])
feats = feats.join(level("rw", ["ap", "rwy"], ["ap", "rwy"]), on="MVT_ID_mvt", how="left")
feats = feats.join(level("cor", ["ap", "rwy", "zone"], ["ap", "zone"]), on="MVT_ID_mvt", how="left")
NEW = [c for c in feats.columns if c != "MVT_ID_mvt"]

CTRL = ["d_mvt_aobt", "d_mvt_lobt", "dep_before_30", "dep_after_30", "arr_before_30", "arr_around_ob_30", "offblocks_around_ob_30",
        "rwy_dep_before_10", "rwy_dep_before_30", "rwy_dep_after_10", "rwy_arr_before_30", "nbf_ap_30_0_mean", "nbf_rwy_60_0_mean"]
base = pl.read_parquet("data/features_e106.parquet", columns=["MVT_ID_mvt", "split"] + CTRL).filter(pl.col("split") == "train").drop("split")
oof = pl.concat([pl.DataFrame({k: np.load(p)[k] for k in ("mvt_id", "pred", "y")}) for p in sorted(glob.glob("data/cv_preds_nomonth/fold_*.npz"))])
d = (dep.select("MVT_ID_mvt", "month").join(feats, on="MVT_ID_mvt").join(base, on="MVT_ID_mvt")
     .join(oof.rename({"mvt_id": "MVT_ID_mvt"}), on="MVT_ID_mvt").with_columns(r=pl.col("y") - pl.col("pred")))
d = d.filter((pl.col("y") <= 7200) & (pl.col("r").abs() < 1800))
print(f"rows {d.height}; residual sd {d['r'].std():.1f} s")
print(d.select(NEW).describe().filter(pl.col("statistic").is_in(["mean", "null_count", "max"])))


def rank(x):
    return pl.Series(x).fill_nan(None).rank().fill_null(0).to_numpy().astype(float)


def spearman(x, y):
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def partial(y, x, controls):
    Z = np.column_stack([np.ones(len(y))] + [rank(c) for c in controls])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return float(np.corrcoef(res(rank(y)), res(rank(x)))[0, 1])


print("\nfeature            spearman(y)  spearman(r)  partial(r | controls): months 1-6 / 7-12")
halves = {h: d.filter(pl.col("month").is_between(*rng)) for h, rng in (("H1", (1, 6)), ("H2", (7, 12)))}
for c in NEW:
    p = [partial(g["r"].to_numpy(), g[c].to_numpy(), [g[k].to_numpy() for k in CTRL]) for g in halves.values()]
    print(f"{c:16s} {spearman(d[c].to_numpy(), d['y'].to_numpy()):+.3f}       {spearman(d[c].to_numpy(), d['r'].to_numpy()):+.4f}      {p[0]:+.4f} / {p[1]:+.4f}")

params = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=42, num_threads=2, deterministic=True)
print("\nstage on the residual, train one half -> score the other (RMSE of r after the stage; lower = the features carry signal)")
for tr_h, te_h in (("H1", "H2"), ("H2", "H1")):
    tr, te = halves[tr_h], halves[te_h]
    r0 = float(np.sqrt((te["r"].to_numpy() ** 2).mean()))
    line = f"  {tr_h} -> {te_h}: no stage {r0:.3f}"
    for name, cols in (("controls", CTRL), ("controls + E151", CTRL + NEW)):
        m = lgb.train(params, lgb.Dataset(tr.select(cols).to_numpy(), tr["r"].to_numpy()), 300)
        rr = te["r"].to_numpy() - m.predict(te.select(cols).to_numpy())
        line += f" | {name} {np.sqrt((rr ** 2).mean()):.3f}"
    print(line)

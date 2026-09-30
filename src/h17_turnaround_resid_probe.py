"""H17 (journal 182): the same-operator stand-inbound features (inb_* of E162) against the REGRESSOR residual, overall and on rows
without our ADS-B ground coverage (coverage < 2: LTFM, LFPG, LIRF-July ...), as H10/H16. kind-mango v20 (-13.8 vs v19, a bundle with
disruption meters and a rule, built WITHOUT AOBT_3/LOBT) used them in the regressor. Controls include our existing stand features.
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import lightgbm as lgb
import numpy as np
import polars as pl

NEW = ["inb_found", "inb_delay", "inb_slack", "inb_age"]
schema = pl.read_parquet_schema("data/features_e162.parquet")
CTRL = [c for c in ("d_mvt_aobt", "d_mvt_lobt", "d_mvt_sched", "taxiing_at_ob", "rwy_dep_before_30", "dep_before_30", "arr_before_30",
                    "offblocks_around_ob_30", "nbf_ap_30_0_mean", "nb_rwy_30_30_mean", "stand_prev_arr_age", "stand_prev_arr_taxi_in",
                    "stand_next_arr_gap") if c in schema]
f = pl.read_parquet("data/features_e162.parquet", columns=["MVT_ID_mvt", "split", "month", "ADEP_mvt"] + CTRL + NEW).filter(pl.col("split") == "train")
cov = pl.concat([pl.read_parquet(p, columns=["MVT_ID_mvt", "coverage"]) for p in sorted(glob.glob("data/adsblol_kt1/offblock2_2025-*.parquet"))]).unique("MVT_ID_mvt")
oof = pl.concat([pl.DataFrame({k: np.load(p)[k] for k in ("mvt_id", "pred", "y")}) for p in sorted(glob.glob("data/cv_preds_e116prod/fold_*.npz"))]).rename({"mvt_id": "MVT_ID_mvt"})
d = f.join(oof, on="MVT_ID_mvt").join(cov, on="MVT_ID_mvt", how="left").with_columns(pl.col("coverage").fill_null(0), r=pl.col("y") - pl.col("pred"))
d = d.filter((pl.col("y") <= 7200) & (pl.col("r").abs() < 1800))
print("controls:", CTRL)


def rank(x):
    return pl.Series(x).fill_nan(None).rank().fill_null(0).to_numpy().astype(float)


def partial(yv, x, controls):
    Z = np.column_stack([np.ones(len(yv))] + [rank(c) for c in controls])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return float(np.corrcoef(res(rank(yv)), res(rank(x)))[0, 1])


prm = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.9, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=42, num_threads=4, deterministic=True)
for label, sub in (("all rows", d), ("coverage < 2 (no ADS-B ground)", d.filter(pl.col("coverage") < 2))):
    halves = {h: sub.filter(pl.col("month").is_between(*rng)) for h, rng in (("H1", (1, 6)), ("H2", (7, 12)))}
    print(f"== {label}: rows {sub.height}")
    for c in NEW:
        p = [partial(g["r"].to_numpy(), g[c].to_numpy(), [g[k_].to_numpy() for k_ in CTRL]) for g in halves.values()]
        print(f"   {c:12s} partial H1 {p[0]:+.4f} / H2 {p[1]:+.4f}")
    for tr_h, te_h in (("H1", "H2"), ("H2", "H1")):
        tr, te = halves[tr_h], halves[te_h]
        line = f"   stage {tr_h} -> {te_h}: no stage {np.sqrt((te['r'].to_numpy() ** 2).mean()):.3f}"
        for name, cs_ in (("controls", CTRL), ("+turnaround", CTRL + NEW)):
            m = lgb.train(prm, lgb.Dataset(tr.select(cs_).to_numpy(), tr["r"].to_numpy()), 300)
            line += f" | {name} {np.sqrt(((te['r'].to_numpy() - m.predict(te.select(cs_).to_numpy())) ** 2).mean()):.3f}"
        print(line)

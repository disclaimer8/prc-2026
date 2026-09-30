"""H16 (journal 181, Denys): ATFM not as an airport/day feature of the departure airport (H11, absorbed) but as a local marker of
"congestion in the sky" at the flight's DESTINATION: the destination airport's daily arrival ATFM delay (apt_dly_*, all European
airports, from ansperformance.eu/data, downloaded by Denys). A regulated destination means the flight likely has a CTOT:
waiting for the slot window at the runway (longer taxi) and/or a late but REAL block time (regime classifier, as H15).
Features per departure (destination ADES_mvt, UTC day of take-off): dst_dly_per_arr, dst_share_dly, dst_w_per_arr (weather),
dst_cap_per_arr (C + G), dst_known (destination present in the file). Two screens on the production OOF (cv_preds_e116prod):
(1) regressor residual: partial Spearman given the model's windows, LightGBM stage half -> half (as H10/H11);
(2) regime pocket: P(schedule) re-fitted with / without these features (control stage), as src/h15_turnaround_probe.py.
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, "src")
import lightgbm as lgb
import numpy as np
import polars as pl

import train as T
from fold_metrics import decided_map

raw_d = pl.concat([pl.read_csv(f"data/ext_pru/apt_dly_{y}.csv", infer_schema_length=0, encoding="utf8-lossy") for y in (2024, 2025, 2026)])
dc = [c for c in raw_d.columns if c.startswith("DLY_APT_ARR") or c.startswith("FLT_ARR")]
raw_d = raw_d.with_columns([pl.col(c).cast(pl.Float64) for c in dc])
per = lambda c: (pl.col(c) / pl.col("FLT_ARR_1")).fill_nan(0.0)
dst = raw_d.select(dest=pl.col("APT_ICAO"), day=pl.col("FLT_DATE").str.slice(0, 10).str.to_date(), dst_dly_per_arr=per("DLY_APT_ARR_1"),
                   dst_share_dly=per("FLT_ARR_1_DLY"), dst_w_per_arr=per("DLY_APT_ARR_W_1"),
                   dst_cap_per_arr=((pl.col("DLY_APT_ARR_C_1") + pl.col("DLY_APT_ARR_G_1")) / pl.col("FLT_ARR_1")).fill_nan(0.0), dst_known=pl.lit(1.0)).unique(["dest", "day"])
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt"]
dep = pl.concat([pl.read_parquet(p, columns=cols) for p in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(
    pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt", dest=pl.col("ADES_mvt"), day=pl.col("MVT_TIME_UTC_mvt").dt.date(), month=pl.col("MVT_TIME_UTC_mvt").dt.month())
feat = dep.join(dst, on=["dest", "day"], how="left").with_columns(pl.col("dst_known").fill_null(0.0))
NEW = ["dst_dly_per_arr", "dst_share_dly", "dst_w_per_arr", "dst_cap_per_arr", "dst_known"]
print(f"departures {feat.height}; destination in the ATFM file {feat['dst_known'].mean():.3f}; with any arrival ATFM delay that day "
      f"{(feat['dst_dly_per_arr'].fill_null(0) > 0).mean():.3f}; >= 1 min/arrival {(feat['dst_dly_per_arr'].fill_null(0) >= 1).mean():.3f}")

# (1) regressor residual
schema = pl.read_parquet_schema("data/features_e106.parquet")
CTRL = [c for c in ("d_mvt_aobt", "d_mvt_lobt", "taxiing_at_ob", "rwy_dep_before_10", "rwy_dep_before_30", "rwy_dep_after_10",
                    "dep_before_30", "arr_before_30", "offblocks_around_ob_30", "nbf_ap_30_0_mean", "nb_rwy_30_30_mean",
                    "nb_ap_30_30_mean", "taxi_in_mean_30") if c in schema]
base = pl.read_parquet("data/features_e106.parquet", columns=["MVT_ID_mvt", "split"] + CTRL).filter(pl.col("split") == "train").drop("split")
oof = pl.concat([pl.DataFrame({k: np.load(p)[k] for k in ("mvt_id", "pred", "y")}) for p in sorted(glob.glob("data/cv_preds_e116prod/fold_*.npz"))]).rename({"mvt_id": "MVT_ID_mvt"})
full = feat.select("MVT_ID_mvt", "month", *NEW).join(base, on="MVT_ID_mvt").join(oof, on="MVT_ID_mvt").with_columns(r=pl.col("y") - pl.col("pred"))
d = full.filter((pl.col("y") <= 7200) & (pl.col("r").abs() < 1800))


def rank(x):
    return pl.Series(x).fill_nan(None).rank().fill_null(0).to_numpy().astype(float)


def partial(yv, x, controls):
    Z = np.column_stack([np.ones(len(yv))] + [rank(c) for c in controls])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return float(np.corrcoef(res(rank(yv)), res(rank(x)))[0, 1])


halves = {h: d.filter(pl.col("month").is_between(*rng)) for h, rng in (("H1", (1, 6)), ("H2", (7, 12)))}
for c in NEW:
    p = [partial(g["r"].to_numpy(), g[c].to_numpy(), [g[k_].to_numpy() for k_ in CTRL]) for g in halves.values()]
    print(f"{c:18s} partial(r | controls) H1 {p[0]:+.4f} / H2 {p[1]:+.4f}")
prm = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.9, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=42, num_threads=4, deterministic=True)
print("(1) stage on the residual, half -> half")
for tr_h, te_h in (("H1", "H2"), ("H2", "H1")):
    tr, te = halves[tr_h], halves[te_h]
    line = f"  {tr_h} -> {te_h}: no stage {np.sqrt((te['r'].to_numpy() ** 2).mean()):.3f}"
    for name, cs_ in (("controls", CTRL), ("+destination ATFM", CTRL + NEW)):
        m = lgb.train(prm, lgb.Dataset(tr.select(cs_).to_numpy(), tr["r"].to_numpy()), 300)
        line += f" | {name} {np.sqrt(((te['r'].to_numpy() - m.predict(te.select(cs_).to_numpy())) ** 2).mean()):.3f}"
    print(line)

# (2) regime pocket (as H15)
Z = [dict(np.load(p)) for p in sorted(glob.glob("data/cv_preds_e116prod/fold_*.npz"))]
o = pl.DataFrame({"MVT_ID_mvt": np.concatenate([z["mvt_id"] for z in Z])}).with_row_index("i")
P = np.concatenate([z["proba"] for z in Z]).astype(float); V = np.concatenate([z["values"] for z in Z]).astype(float)
y = np.concatenate([z["y"] for z in Z]).astype(float); pred = np.concatenate([z["pred"] for z in Z]).astype(float)
f = pl.read_parquet("data/features_e106.parquet", columns=list(dict.fromkeys(["MVT_ID_mvt", "split", "FLIGHT_ID_mvt", "d_mvt_lobt", T.TARGET, "ADEP_mvt", *T.CANDIDATES.values()]))).filter(pl.col("split") == "train")
lab = pl.DataFrame({"MVT_ID_mvt": f["MVT_ID_mvt"], "label": T.regime_label(f)})
o = o.join(feat.select("MVT_ID_mvt", "month", *NEW), on="MVT_ID_mvt", how="left").join(lab, on="MVT_ID_mvt", how="left").join(
    f.select("MVT_ID_mvt", "FLIGHT_ID_mvt", "d_mvt_lobt", "ADEP_mvt"), on="MVT_ID_mvt", how="left").sort("i")
dm = decided_map("data/features_e106.parquet")
dec = np.array([dm.get(m, True) for m in o["MVT_ID_mvt"].to_list()])
pocket = dec & (np.abs(V[:, 1] - V[:, 0]) > 3600) & ~np.isnan(V[:, 1])
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
X0 = np.column_stack([lg(P[:, 1]), lg(P[:, 0]), V[:, 1] - V[:, 0], o["FLIGHT_ID_mvt"].is_not_null().to_numpy().astype(float),
                      o["ADEP_mvt"].cast(pl.Categorical).to_physical().to_numpy().astype(float)])
X1 = np.column_stack([X0, o.select(NEW).to_numpy().astype(float)])
tgt = (o["label"].to_numpy() == 1).astype(float)
half = np.where(o["month"].to_numpy() <= 6, 1, 2)
prm2 = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=200, lambda_l2=10.0, verbose=-1, seed=42, num_threads=4, deterministic=True)
d_lobt = o["d_mvt_lobt"].to_numpy()


def rebuild(ps, m):
    Pn = P[m].copy(); rest = np.clip(1 - P[m, 1], 1e-9, None)
    Pn = Pn * ((1 - ps) / rest)[:, None]; Pn[:, 1] = ps
    Pn = np.where(np.isnan(V[m]), 0.0, Pn); Pn = Pn / Pn.sum(1, keepdims=True)
    mixv = np.sum(Pn * np.nan_to_num(V[m]), axis=1)
    return np.clip(np.where(np.isnan(d_lobt[m]), mixv, np.clip(mixv, d_lobt[m] - 3606, d_lobt[m] + 3606)), 0, None)


print(f"(2) regime pocket rows {int(pocket.sum())}")
res = {}
for name, X in (("control stage", X0), ("+ destination ATFM", X1)):
    tot = 0.0
    for h_tr, h_te in ((1, 2), (2, 1)):
        mtr, mte = pocket & (half == h_tr), pocket & (half == h_te)
        m = lgb.train(prm2, lgb.Dataset(X[mtr], tgt[mtr], categorical_feature=[4]), 300)
        dd = (rebuild(m.predict(X[mte]), mte) - y[mte]) ** 2 - (pred[mte] - y[mte]) ** 2
        tot += dd.sum()
        print(f"  {name}: H{h_tr} -> H{h_te} dSSE {dd.sum() / 1e6:+.1f}e6, trimmed(5/5) {np.sort(dd)[5:-5].sum() / 1e6:+.1f}e6", flush=True)
    res[name] = tot
print(f"(2) candidate minus control: {(res['+ destination ATFM'] - res['control stage']) / 1e6:+.1f}e6 SSE -> "
      f"{np.sqrt((np.sum((pred - y) ** 2) + res['+ destination ATFM'] - res['control stage']) / len(y)) - np.sqrt(np.mean((pred - y) ** 2)):+.3f} s RMSE")

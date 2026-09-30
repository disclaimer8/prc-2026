"""H15 (journal 179, Denys): inbound delay of the previous arrival of the SAME OPERATOR on the SAME STAND, as a regime marker
(block = schedule vs a real off-block) for rows without a track. G5 (18.09) measured the same airframe's inbound delay through the
ADS-B link, so it only reached rows with a track (-0.077 s, 5/6, reversed sign: a late inbound means the block is real).

Features per departure: the latest arrival at (airport, stand) by the same operator (AIRCRAFT_OPERATOR_flt, else the call-sign
prefix), blocked in before the departure's take-off and within 24 h: inb_delay = BLOCK_arr - SCHED_arr, inb_slack = SCHED_dep -
BLOCK_arr, inb_age = MVT_dep - BLOCK_arr, found flag. Probe on the production OOF (cv_preds_e116prod): rows the base decides where
the schedule candidate is > 3600 s from the regular one (the oracle pocket, journal 169). Stage: LightGBM on
[logit P(sched), logit P(regular), D, NM flag, airport] (+ inbound features) predicting truth = schedule, half-year -> half-year;
P(sched) replaced, the other classes rescaled, mixture rebuilt; SSE of the pocket, candidate minus control stage.
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

ep = lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt", "FLIGHT_mvt", "AIRCRAFT_OPERATOR_flt", "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
raw = pl.concat([pl.read_parquet(p, columns=cols) for p in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).with_columns(
    op=pl.coalesce("AIRCRAFT_OPERATOR_flt", pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]{2,3})", 1)))
arr = raw.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("STAND_mvt").is_not_null() & pl.col("op").is_not_null()).select(
    ap=pl.col("ADES_mvt"), stand=pl.col("STAND_mvt"), op="op", t_in=ep("BLOCK_TIME_UTC_mvt"), s_in=ep("SCHED_TIME_UTC_mvt")).drop_nulls("t_in").sort("t_in")
dep = raw.filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt", ap=pl.col("ADEP_mvt"), stand=pl.col("STAND_mvt"), op="op",
                                                       mvt=ep("MVT_TIME_UTC_mvt"), s_dep=ep("SCHED_TIME_UTC_mvt"),
                                                       month=pl.col("MVT_TIME_UTC_mvt").dt.month()).with_columns(key=pl.col("mvt") - 120).sort("key")
j = dep.join_asof(arr, left_on="key", right_on="t_in", by=["ap", "stand", "op"], strategy="backward", tolerance=86400.0)
j = j.with_columns(inb_found=pl.col("t_in").is_not_null().cast(pl.Float64), inb_delay=pl.col("t_in") - pl.col("s_in"),
                   inb_slack=pl.col("s_dep") - pl.col("t_in"), inb_age=pl.col("mvt") - pl.col("t_in"))
print(f"departures {dep.height}; with a same-operator inbound on the stand within 24 h: {j['inb_found'].mean():.3f}")
NEW = ["inb_found", "inb_delay", "inb_slack", "inb_age"]
Z = [dict(np.load(p)) for p in sorted(glob.glob("data/cv_preds_e116prod/fold_*.npz"))]
o = pl.DataFrame({"MVT_ID_mvt": np.concatenate([z["mvt_id"] for z in Z])}).with_row_index("i")
P = np.concatenate([z["proba"] for z in Z]).astype(float); V = np.concatenate([z["values"] for z in Z]).astype(float)
y = np.concatenate([z["y"] for z in Z]).astype(float); pred = np.concatenate([z["pred"] for z in Z]).astype(float)
f = pl.read_parquet("data/features_e106.parquet", columns=list(dict.fromkeys(["MVT_ID_mvt", "split", "FLIGHT_ID_mvt", "d_mvt_lobt", T.TARGET, "ADEP_mvt", *T.CANDIDATES.values()]))).filter(pl.col("split") == "train")
lab = pl.DataFrame({"MVT_ID_mvt": f["MVT_ID_mvt"], "label": T.regime_label(f)})
o = o.join(j.select("MVT_ID_mvt", "month", *NEW), on="MVT_ID_mvt", how="left").join(lab, on="MVT_ID_mvt", how="left").join(
    f.select("MVT_ID_mvt", "FLIGHT_ID_mvt", "d_mvt_lobt", "ADEP_mvt"), on="MVT_ID_mvt", how="left").sort("i")
dm = decided_map("data/features_e106.parquet")
dec = np.array([dm.get(m, True) for m in o["MVT_ID_mvt"].to_list()])
pocket = dec & (np.abs(V[:, 1] - V[:, 0]) > 3600) & ~np.isnan(V[:, 1])
print(f"pocket rows {int(pocket.sum())}; truth schedule there {np.mean(o['label'].to_numpy()[pocket] == 1):.3f}; "
      f"inbound found there {np.nanmean(o['inb_found'].to_numpy()[pocket]):.3f}")
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
X0 = np.column_stack([lg(P[:, 1]), lg(P[:, 0]), V[:, 1] - V[:, 0], o["FLIGHT_ID_mvt"].is_not_null().to_numpy().astype(float),
                      o["ADEP_mvt"].cast(pl.Categorical).to_physical().to_numpy().astype(float)])
X1 = np.column_stack([X0, o.select(NEW).to_numpy().astype(float)])
tgt = (o["label"].to_numpy() == 1).astype(float)
half = np.where(o["month"].to_numpy() <= 6, 1, 2)
prm = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=200, lambda_l2=10.0, verbose=-1, seed=42, num_threads=4, deterministic=True)
d_lobt = o["d_mvt_lobt"].to_numpy()


def rebuild(ps, m):
    Pn = P[m].copy()
    rest = np.clip(1 - P[m, 1], 1e-9, None)
    Pn = Pn * ((1 - ps) / rest)[:, None]; Pn[:, 1] = ps
    Pn = np.where(np.isnan(V[m]), 0.0, Pn); Pn = Pn / Pn.sum(1, keepdims=True)
    mixv = np.sum(Pn * np.nan_to_num(V[m]), axis=1)
    lo, hi = d_lobt[m] - 3606, d_lobt[m] + 3606
    return np.clip(np.where(np.isnan(d_lobt[m]), mixv, np.clip(mixv, lo, hi)), 0, None)


for name, X in (("control stage (classifier outputs only)", X0), ("+ same-operator inbound on the stand", X1)):
    tot = 0.0
    for h_tr, h_te in ((1, 2), (2, 1)):
        mtr, mte = pocket & (half == h_tr), pocket & (half == h_te)
        m = lgb.train(prm, lgb.Dataset(X[mtr], tgt[mtr], categorical_feature=[4]), 300)
        new = rebuild(m.predict(X[mte]), mte)
        dd = (new - y[mte]) ** 2 - (pred[mte] - y[mte]) ** 2
        tot += dd.sum()
        print(f"  {name}: H{h_tr} -> H{h_te} rows {int(mte.sum())} dSSE {dd.sum() / 1e6:+.1f}e6, trimmed(5/5) {np.sort(dd)[5:-5].sum() / 1e6:+.1f}e6", flush=True)
    print(f"{name}: total dSSE {tot / 1e6:+.1f}e6 -> full-2025 RMSE {np.sqrt((np.sum((pred - y) ** 2) + tot) / len(y)) - np.sqrt(np.mean((pred - y) ** 2)):+.3f} s")

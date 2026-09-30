"""E168 (journal 198; pre-registered in review/ideas_round3_2026-09-30.md sec. 2): the OPDI stage on the UNION of ground events
exit-parking_position / exit-apron / entry-taxiway (E159c used exit-parking only). LSZH 2025, leave-one-month-out on top of the
staged OOF chain (E168_BASE, default data/cv_preds_e150c_e64_keep as E159c). Features: MVT - event and (MVT - event) - pred for each
of the three events (NaN where absent), presence flags, pred. Reports per coverage group, the subset "no exit-parking" (rows the
union adds) and, for comparison, E159c's single-event stage on the rows that have exit-parking.
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

BASE = os.environ.get("E168_BASE", "data/cv_preds_e150c_e64_keep")
EV = ["exit-parking_position", "exit-apron", "entry-taxiway"]
ANCH = ["take-off", "first-xing-fl50", "first_seen"]
ev = pl.concat([pl.read_parquet(p, columns=["flight_id", "flt_id", "adep", "type", "t"]) for p in sorted(glob.glob("data/opdi/ev_*.parquet"))])
ev = ev.filter(pl.col("type").is_in(EV + ANCH) & (pl.col("adep") == "LSZH")).with_columns(t=pl.col("t").dt.epoch("s").cast(pl.Float64))
w = ev.group_by("flight_id", "type").agg(pl.col("t").min(), pl.col("flt_id").first(), pl.col("adep").first()).pivot(on="type", index=["flight_id", "flt_id", "adep"], values="t")
for c in EV + ANCH:
    if c not in w.columns:
        w = w.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
w = w.with_columns(anchor=pl.coalesce(*ANCH)).filter(pl.col("anchor").is_not_null())
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt"]
tr = pl.concat([pl.read_parquet(p, columns=cols) for p in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))])
d = tr.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LSZH")).with_columns(
    mvt=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64), month=pl.col("MVT_TIME_UTC_mvt").dt.month(),
    cs=pl.coalesce("CALLSIGN_flt", "FLIGHT_mvt").str.replace_all(" ", "")).select("MVT_ID_mvt", "ADEP_mvt", "mvt", "month", "cs")
j = d.join(w, left_on=["cs", "ADEP_mvt"], right_on=["flt_id", "adep"], how="left").with_columns(dt=pl.col("anchor") - pl.col("mvt"))
j = j.with_columns(ok=pl.when(pl.col("dt").is_between(-900, 1800)).then(pl.col("dt").abs())).sort("MVT_ID_mvt", "ok", nulls_last=True).unique("MVT_ID_mvt", keep="first")
j = j.with_columns([pl.when(pl.col("ok").is_not_null()).then(pl.col(c)).alias(c) for c in EV]).filter(pl.any_horizontal([pl.col(c).is_not_null() for c in EV]))
cov = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage"]) for f in sorted(glob.glob("data/adsblol_kt1/offblock2_2025-*.parquet"))]).unique("MVT_ID_mvt")
st = pl.concat([pl.DataFrame({k: np.load(p)[k] for k in ("mvt_id", "pred", "y")}) for p in sorted(glob.glob(f"{BASE}/fold_*.npz"))]).rename({"mvt_id": "MVT_ID_mvt"})
j = j.join(cov, on="MVT_ID_mvt", how="left").with_columns(pl.col("coverage").fill_null(0)).join(st, on="MVT_ID_mvt", how="inner")
short = {"exit-parking_position": "park", "exit-apron": "apron", "entry-taxiway": "twy"}
j = j.with_columns([(pl.col("mvt") - pl.col(c)).alias(f"t_{s}") for c, s in short.items()]).with_columns(
    [(pl.col(f"t_{s}") - pl.col("pred")).alias(f"d_{s}") for s in short.values()] +
    [pl.col(c).is_not_null().cast(pl.Float64).alias(f"has_{s}") for c, s in short.items()] +
    [(pl.col("y") - pl.col("pred")).alias("r")])
F = [f"{p}_{s}" for s in short.values() for p in ("t", "d", "has")] + ["pred"]
F1 = ["d_park", "t_park", "pred"]  # E159c-like single event (without twy_s)
P = dict(objective="regression", learning_rate=0.05, num_leaves=15, min_data_in_leaf=100, lambda_l2=10.0, feature_fraction=1.0,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42, num_threads=4, deterministic=True)
months = sorted(j["month"].unique().to_list())
print(f"base {BASE}; LSZH rows with any of the three events {j.height}; with exit-parking {int(j['has_park'].sum())}, exit-apron {int(j['has_apron'].sum())}, "
      f"entry-taxiway {int(j['has_twy'].sum())}; without exit-parking {int((j['has_park'] == 0).sum())}")
out = []
for m_ in months:
    a, b = j.filter(pl.col("month") != m_), j.filter(pl.col("month") == m_)
    g = lgb.train(P, lgb.Dataset(a.select(F).to_numpy(), a["r"].to_numpy()), 300)
    ap_ = a.filter(pl.col("has_park") == 1)
    g1 = lgb.train(P, lgb.Dataset(ap_.select(F1).to_numpy(), ap_["r"].to_numpy()), 300)
    b = b.with_columns(new=pl.col("pred") + pl.Series(g.predict(b.select(F).to_numpy())),
                       new1=pl.when(pl.col("has_park") == 1).then(pl.col("pred") + pl.Series(g1.predict(b.select(F1).to_numpy()))).otherwise(pl.col("pred")))
    out.append(b)
o = pl.concat(out).with_columns(e0=(pl.col("y") - pl.col("pred")) ** 2, e1=(pl.col("y") - pl.col("new")) ** 2, e159=(pl.col("y") - pl.col("new1")) ** 2)


def rep(label, s, col="e1"):
    dd = np.sort((s[col] - s["e0"]).to_numpy())
    better = sum(1 for m_ in months if s.filter(pl.col("month") == m_)[col].sum() < s.filter(pl.col("month") == m_)["e0"].sum())
    print(f"{label:52s} n {s.height:6d} RMSE {np.sqrt(s['e0'].mean()):7.1f} -> {np.sqrt(s[col].mean()):7.1f} ({(np.sqrt(s[col].mean()) / np.sqrt(s['e0'].mean()) - 1) * 100:+.1f} %), "
          f"months better {better}/{len(months)}, trimmed(3/3) {dd[3:-3].sum() / 1e6:+.1f}e6")


rep("E168 union, coverage >= 2", o.filter(pl.col("coverage") >= 2))
rep("E168 union, coverage < 2", o.filter(pl.col("coverage") < 2))
rep("E168 union, rows WITHOUT exit-parking (added)", o.filter(pl.col("has_park") == 0))
rep("E168 union, rows with exit-parking", o.filter(pl.col("has_park") == 1))
rep("E159c-like single event, rows with exit-parking", o.filter(pl.col("has_park") == 1), "e159")

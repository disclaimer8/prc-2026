"""E159c (journal 175): the OPDI exit-parking stage (E159b) on all LSZH 2025 months, leave-one-month-out, on top of the staged
OOF chain (data/cv_preds_e150c_e64_keep). Reports per month: rows with OPDI, RMSE staged -> +OPDI on rows with / without our
coverage, and pooled; the final board model is then fitted on every month (E159_MONTHS=all in src/e159b_apply.py).
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from e159_opdi_stage_cv import match, opdi_wide  # noqa: E402

BASE = os.environ.get("E159_BASE", "data/cv_preds_e150c_e64_keep")
AP = os.environ.get("E159_AP", "LSZH")
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt"]
tr = pl.concat([pl.read_parquet(p, columns=cols) for p in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))])
d = tr.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == AP)).with_columns(
    mvt=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64), month=pl.col("MVT_TIME_UTC_mvt").dt.month(),
    cs=pl.coalesce("CALLSIGN_flt", "FLIGHT_mvt").str.replace_all(" ", "")).drop("CALLSIGN_flt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "PHASE_mvt")
j = match(d, opdi_wide()).filter(pl.col("exit").is_not_null())
cov = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage"]) for f in sorted(glob.glob("data/adsblol_kt1/offblock2_2025-*.parquet"))]).unique("MVT_ID_mvt")
st = pl.concat([pl.DataFrame({k: np.load(p)[k] for k in ("mvt_id", "pred", "y")}) for p in sorted(glob.glob(f"{BASE}/fold_*.npz"))]).rename({"mvt_id": "MVT_ID_mvt"})
j = j.join(cov, on="MVT_ID_mvt", how="left").with_columns(pl.col("coverage").fill_null(0)).join(st, on="MVT_ID_mvt", how="inner").with_columns(
    t_opdi=pl.col("mvt") - pl.col("exit"), twy_s=pl.col("twy") - pl.col("exit")).with_columns(d_opdi=pl.col("t_opdi") - pl.col("pred"), r=pl.col("y") - pl.col("pred"))
F = ["d_opdi", "t_opdi", "pred", "twy_s"]
P = dict(objective="regression", learning_rate=0.05, num_leaves=15, min_data_in_leaf=100, lambda_l2=10.0, feature_fraction=1.0,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42, num_threads=4, deterministic=True)
months = sorted(j["month"].unique().to_list())
print(f"{AP}: months with OPDI {months}; rows {j.height}; per month {j.group_by('month').len().sort('month').rows()}")
out = []
for m_ in months:
    a, b = j.filter(pl.col("month") != m_), j.filter(pl.col("month") == m_)
    g = lgb.train(P, lgb.Dataset(a.select(F).to_numpy(), a["r"].to_numpy()), 300)
    out.append(b.with_columns(new=pl.col("pred") + pl.Series(g.predict(b.select(F).to_numpy()))))
o = pl.concat(out).with_columns(e0=(pl.col("y") - pl.col("pred")) ** 2, e1=(pl.col("y") - pl.col("new")) ** 2, cv=pl.col("coverage") >= 2)
print(o.group_by("month", "cv").agg(n=pl.len(), rmse0=pl.col("e0").mean().sqrt(), rmse1=pl.col("e1").mean().sqrt()).with_columns(
    rel=(pl.col("rmse1") / pl.col("rmse0") - 1) * 100).sort("month", "cv").with_columns(pl.col(pl.Float64).round(1)))
for cvv in (True, False):
    s = o.filter(pl.col("cv") == cvv)
    dd = np.sort((s["e1"] - s["e0"]).to_numpy())
    print(f"pooled coverage>=2={cvv}: n {s.height}, RMSE {np.sqrt(s['e0'].mean()):.1f} -> {np.sqrt(s['e1'].mean()):.1f}, months better "
          f"{sum(1 for m_ in months if s.filter(pl.col('month') == m_)['e1'].sum() < s.filter(pl.col('month') == m_)['e0'].sum())}/{len(months)}, "
          f"dSSE {dd.sum() / 1e6:+.1f}e6, trimmed(3/3) {dd[3:-3].sum() / 1e6:+.1f}e6")

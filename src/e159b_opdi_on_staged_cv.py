"""E159b (journal 173): does OPDI exit-parking add anything on top of the ADS-B stages? LSZH 2025, Jan <-> Jul, base = the staged
OOF chain (data/cv_preds_e150c_e64_keep: E70 keep-all + E64 keep-all on the production-like base), split by our coverage.
Same stage model as E159 (src/e159_opdi_stage_cv.py), features computed against the staged prediction.
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
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt"]
tr = pl.concat([pl.read_parquet(p, columns=cols) for p in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))])
d = tr.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LSZH")).with_columns(
    mvt=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64), month=pl.col("MVT_TIME_UTC_mvt").dt.month(),
    cs=pl.coalesce("CALLSIGN_flt", "FLIGHT_mvt").str.replace_all(" ", "")).filter(pl.col("month").is_in([1, 7])).drop("CALLSIGN_flt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "PHASE_mvt")
j = match(d, opdi_wide()).filter(pl.col("exit").is_not_null())
cov = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage"]) for f in sorted(glob.glob("data/adsblol_kt1/offblock2_2025-0[17]-*.parquet"))]).unique("MVT_ID_mvt")
st = pl.concat([pl.DataFrame({k: np.load(p)[k] for k in ("mvt_id", "pred", "y")}) for p in sorted(glob.glob(f"{BASE}/fold_*.npz"))]).rename({"mvt_id": "MVT_ID_mvt"})
j = j.join(cov, on="MVT_ID_mvt", how="left").with_columns(pl.col("coverage").fill_null(0)).join(st, on="MVT_ID_mvt", how="inner").with_columns(
    t_opdi=pl.col("mvt") - pl.col("exit"), twy_s=pl.col("twy") - pl.col("exit")).with_columns(d_opdi=pl.col("t_opdi") - pl.col("pred"), r=pl.col("y") - pl.col("pred"))
F = ["d_opdi", "t_opdi", "pred", "twy_s"]
P = dict(objective="regression", learning_rate=0.05, num_leaves=15, min_data_in_leaf=100, lambda_l2=10.0, feature_fraction=1.0,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42, num_threads=4, deterministic=True)
print(f"base {BASE}; LSZH rows with OPDI exit {j.height}, coverage >= 2: {int((j['coverage'] >= 2).sum())}")
for tr_m, te_m in ((1, 7), (7, 1)):
    a, b = j.filter(pl.col("month") == tr_m), j.filter(pl.col("month") == te_m)
    m = lgb.train(P, lgb.Dataset(a.select(F).to_numpy(), a["r"].to_numpy()), 300)
    b = b.with_columns(new=pl.col("pred") + pl.Series(m.predict(b.select(F).to_numpy())))
    for name, sub in (("coverage >= 2", b.filter(pl.col("coverage") >= 2)), ("coverage < 2", b.filter(pl.col("coverage") < 2))):
        e0, e1 = (sub["y"] - sub["pred"]).to_numpy(), (sub["y"] - sub["new"]).to_numpy()
        dd = np.sort(e1 ** 2 - e0 ** 2)
        print(f"fit {tr_m} -> score {te_m} | {name:14s} n {sub.height:6d} RMSE staged {np.sqrt(np.mean(e0 ** 2)):8.2f} +OPDI {np.sqrt(np.mean(e1 ** 2)):8.2f} "
              f"({(np.sqrt(np.mean(e1 ** 2)) / np.sqrt(np.mean(e0 ** 2)) - 1) * 100:+.1f} %) dSSE {dd.sum() / 1e6:+.2f}e6 trimmed {dd[3:-3].sum() / 1e6:+.2f}e6")

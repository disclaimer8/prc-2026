"""E159 (journal 171-172) on a board file: OPDI exit-parking stage for EDDF/LSZH 2026 departures without adsb.lol coverage.

    .venv/bin/python src/e159_apply.py <final version> <pre-stage base version> <new version>

The stage g is fitted on LSZH 2025 (Jan + Jul, every departure with an OPDI exit-parking event; r = y - production OOF) exactly
as in src/e159_opdi_stage_cv.py, then applied to the ranking rows of E159_APS (default EDDF,LSZH) that have an OPDI exit-parking
event and adsb.lol coverage < 2: new = clip(base + g, LOBT window, >= 0), base = the pre-stage file (the CV residual was taken
against the pre-stage base). Every other row keeps the final file's value.
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from e159_opdi_stage_cv import airport_offsets, match, opdi_wide  # noqa: E402

final_v, base_v, new_v = (int(x) for x in sys.argv[1:4])
APS = os.environ.get("E159_APS", "EDDF,LSZH").split(",")
Y = "TAXITIME_SEC_mvt"
F = ["d_opdi", "t_opdi", "pred", "twy_s"]
w = opdi_wide()
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt"]
prep = lambda df: df.filter(pl.col("PHASE_mvt") == "DEP").with_columns(
    mvt=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64), month=pl.col("MVT_TIME_UTC_mvt").dt.month(),
    cs=pl.coalesce("CALLSIGN_flt", "FLIGHT_mvt").str.replace_all(" ", "")).drop("CALLSIGN_flt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "PHASE_mvt")
# fit on LSZH 2025
tr = prep(pl.concat([pl.read_parquet(p, columns=cols) for p in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]))
tr = tr.filter((pl.col("ADEP_mvt") == "LSZH") & pl.col("month").is_in([1, 7]))
oof = pl.concat([pl.DataFrame({k: np.load(p)[k] for k in ("mvt_id", "pred", "y")}) for p in sorted(glob.glob("data/cv_preds_e116prod/fold_*.npz"))]).rename({"mvt_id": "MVT_ID_mvt"})
a = match(tr, w).filter(pl.col("exit").is_not_null()).join(oof, on="MVT_ID_mvt", how="inner").with_columns(
    t_opdi=pl.col("mvt") - pl.col("exit"), twy_s=pl.col("twy") - pl.col("exit")).with_columns(d_opdi=pl.col("t_opdi") - pl.col("pred"), r=pl.col("y") - pl.col("pred"))
P = dict(objective="regression", learning_rate=0.05, num_leaves=15, min_data_in_leaf=100, lambda_l2=10.0, feature_fraction=1.0,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42, num_threads=4, deterministic=True)
g = lgb.train(P, lgb.Dataset(a.select(F).to_numpy(), a["r"].to_numpy()), 300)
print(f"stage fitted on LSZH 2025: {a.height} rows")
# apply to ranking
rk = prep(pl.read_parquet("data/prc-2026-datasets/ranking.parquet", columns=cols)).filter(pl.col("ADEP_mvt").is_in(APS))
cov = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage"]) for f in sorted(glob.glob("data/adsblol_kt1/offblock2_2026-*.parquet"))]).unique("MVT_ID_mvt")
base = pl.read_parquet(f"submissions/gentle-igloo_v{base_v}.parquet").rename({Y: "pred"}).with_columns(pl.col("pred").cast(pl.Float64))
lobt = pl.read_parquet("data/features_e106.parquet", columns=["MVT_ID_mvt", "split", "d_mvt_lobt"]).filter(pl.col("split") == "rank").drop("split")
SH = airport_offsets(APS) if os.environ.get("E159_SHIFT", "1") == "1" else {a_: 0.0 for a_ in APS}  # journal 173
b = match(rk, w).filter(pl.col("exit").is_not_null()).with_columns(exit=pl.col("exit") - pl.col("ADEP_mvt").replace_strict(SH, return_dtype=pl.Float64)).join(cov, on="MVT_ID_mvt", how="left").with_columns(pl.col("coverage").fill_null(0)) \
    .filter(pl.col("coverage") < 2).join(base, on="MVT_ID_mvt", how="inner").join(lobt, on="MVT_ID_mvt", how="left").with_columns(
    t_opdi=pl.col("mvt") - pl.col("exit"), twy_s=pl.col("twy") - pl.col("exit")).with_columns(d_opdi=pl.col("t_opdi") - pl.col("pred"))
new = b["pred"].to_numpy() + g.predict(b.select(F).to_numpy())
d = b["d_mvt_lobt"].to_numpy()
new = np.where(np.isnan(d), new, np.clip(new, d - 3606, d + 3606))
b = b.with_columns(e159=pl.Series(np.clip(new, 0, None)))
print("rows changed by airport and month:", b.group_by("ADEP_mvt", "month").len().sort("ADEP_mvt", "month").rows())
fin = pl.read_parquet(f"submissions/gentle-igloo_v{final_v}.parquet")
out = fin.join(b.select("MVT_ID_mvt", "e159"), on="MVT_ID_mvt", how="left").with_columns(
    pl.when(pl.col("e159").is_not_null()).then(pl.col("e159").round(0)).otherwise(pl.col(Y)).cast(fin.schema[Y]).alias(Y)).select(fin.columns)
assert out.height == fin.height == 344841 and out[Y].null_count() == 0
dd = (out.sort("MVT_ID_mvt")[Y].cast(pl.Float64) - fin.sort("MVT_ID_mvt")[Y].cast(pl.Float64)).to_numpy()
out.write_parquet(f"submissions/gentle-igloo_v{new_v}.parquet")
print(f"wrote v{new_v} = v{final_v} + E159 on {b.height} rows: changed {(dd != 0).sum()}, RMS {np.sqrt(np.mean(dd ** 2)):.3f} s, mean {dd.mean():+.3f} s, "
      f"row RMS change {np.sqrt(np.mean(dd[dd != 0] ** 2)):.1f} s")

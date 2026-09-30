"""E159b (journal 173) on a board file: OPDI exit-parking stage on top of the full chain, every EDDF/LSZH 2026 departure with an
OPDI exit-parking event (any adsb.lol coverage).

    E159_APS=EDDF,LSZH .venv/bin/python src/e159b_apply.py <final version> <new version>

g is fitted on LSZH 2025 (Jan + Jul, all departures with OPDI exit-parking) against the staged OOF chain
(data/cv_preds_e150c_e64_keep), as in src/e159b_opdi_on_staged_cv.py; on ranking, pred = the final file's value and
new = clip(pred + g, LOBT window, >= 0).
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

final_v, new_v = int(sys.argv[1]), int(sys.argv[2])
APS = os.environ.get("E159_APS", "EDDF,LSZH").split(",")
BASE = os.environ.get("E159_BASE", "data/cv_preds_e150c_e64_keep")
Y = "TAXITIME_SEC_mvt"
F = ["d_opdi", "t_opdi", "pred", "twy_s"]
w = opdi_wide()
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt"]
prep = lambda df: df.filter(pl.col("PHASE_mvt") == "DEP").with_columns(
    mvt=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64), month=pl.col("MVT_TIME_UTC_mvt").dt.month(),
    cs=pl.coalesce("CALLSIGN_flt", "FLIGHT_mvt").str.replace_all(" ", "")).drop("CALLSIGN_flt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "PHASE_mvt")
tr = prep(pl.concat([pl.read_parquet(p, columns=cols) for p in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]))
tr = tr.filter((pl.col("ADEP_mvt") == "LSZH") & pl.col("month").is_in([1, 7]))
st = pl.concat([pl.DataFrame({k: np.load(p)[k] for k in ("mvt_id", "pred", "y")}) for p in sorted(glob.glob(f"{BASE}/fold_*.npz"))]).rename({"mvt_id": "MVT_ID_mvt"})
a = match(tr, w).filter(pl.col("exit").is_not_null()).join(st, on="MVT_ID_mvt", how="inner").with_columns(
    t_opdi=pl.col("mvt") - pl.col("exit"), twy_s=pl.col("twy") - pl.col("exit")).with_columns(d_opdi=pl.col("t_opdi") - pl.col("pred"), r=pl.col("y") - pl.col("pred"))
P = dict(objective="regression", learning_rate=0.05, num_leaves=15, min_data_in_leaf=100, lambda_l2=10.0, feature_fraction=1.0,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42, num_threads=4, deterministic=True)
g = lgb.train(P, lgb.Dataset(a.select(F).to_numpy(), a["r"].to_numpy()), 300)
print(f"stage fitted on LSZH 2025 against {BASE}: {a.height} rows")
fin = pl.read_parquet(f"submissions/gentle-igloo_v{final_v}.parquet")
rk = prep(pl.read_parquet("data/prc-2026-datasets/ranking.parquet", columns=cols)).filter(pl.col("ADEP_mvt").is_in(APS))
lobt = pl.read_parquet("data/features_e106.parquet", columns=["MVT_ID_mvt", "split", "d_mvt_lobt"]).filter(pl.col("split") == "rank").drop("split")
SH = airport_offsets(APS) if os.environ.get("E159_SHIFT", "1") == "1" else {a_: 0.0 for a_ in APS}  # journal 173
b = match(rk, w).filter(pl.col("exit").is_not_null()).with_columns(exit=pl.col("exit") - pl.col("ADEP_mvt").replace_strict(SH, return_dtype=pl.Float64)).join(fin.rename({Y: "pred"}).with_columns(pl.col("pred").cast(pl.Float64)), on="MVT_ID_mvt", how="inner") \
    .join(lobt, on="MVT_ID_mvt", how="left").with_columns(t_opdi=pl.col("mvt") - pl.col("exit"), twy_s=pl.col("twy") - pl.col("exit")) \
    .with_columns(d_opdi=pl.col("t_opdi") - pl.col("pred"))
new = b["pred"].to_numpy() + g.predict(b.select(F).to_numpy())
d = b["d_mvt_lobt"].to_numpy()
new = np.where(np.isnan(d), new, np.clip(new, d - 3606, d + 3606))
b = b.with_columns(e159=pl.Series(np.clip(new, 0, None)))
print("rows by airport and month:", b.group_by("ADEP_mvt", "month").len().sort("ADEP_mvt", "month").rows())
out = fin.join(b.select("MVT_ID_mvt", "e159"), on="MVT_ID_mvt", how="left").with_columns(
    pl.when(pl.col("e159").is_not_null()).then(pl.col("e159").round(0)).otherwise(pl.col(Y)).cast(fin.schema[Y]).alias(Y)).select(fin.columns)
assert out.height == fin.height == 344841 and out[Y].null_count() == 0
dd = (out.sort("MVT_ID_mvt")[Y].cast(pl.Float64) - fin.sort("MVT_ID_mvt")[Y].cast(pl.Float64)).to_numpy()
out.write_parquet(f"submissions/gentle-igloo_v{new_v}.parquet")
print(f"wrote v{new_v} = v{final_v} + E159b: changed {(dd != 0).sum()}, RMS {np.sqrt(np.mean(dd ** 2)):.3f} s, mean {dd.mean():+.3f} s, "
      f"row RMS change {np.sqrt(np.mean(dd[dd != 0] ** 2)):.1f} s")

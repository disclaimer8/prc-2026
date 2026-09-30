"""E165 (journal 185): kind-mango's filed-plan residuals in the base. plan_taxi_res = (MVT - ARVT_1) - median per route (ADEP, ADES),
plan_nm_taxi = (ARVT_1 - EOBT_1) - median per (ADEP, ADES, aircraft type); both clipped +-3600. The medians use input columns only
(no label), so they are taken over 2025 and ranking together. Writes data/features_e165.parquet = features_e106 + plan_*.
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

ep = lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt", "ARVT_1_flt", "EOBT_1_flt"]
files = sorted(glob.glob("data/prc-2026-datasets/training_*.parquet")) + ["data/prc-2026-datasets/ranking.parquet"]
d = pl.concat([pl.read_parquet(p, columns=cols) for p in files]).filter(pl.col("PHASE_mvt") == "DEP").with_columns(
    a=ep("MVT_TIME_UTC_mvt") - ep("ARVT_1_flt"), b=ep("ARVT_1_flt") - ep("EOBT_1_flt"))
d = d.with_columns(plan_taxi_res=(pl.col("a") - pl.col("a").median().over("ADEP_mvt", "ADES_mvt")).clip(-3600, 3600),
                   plan_nm_taxi=(pl.col("b") - pl.col("b").median().over("ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt")).clip(-3600, 3600))
f = pl.read_parquet("data/features_e106.parquet").join(d.select("MVT_ID_mvt", "plan_taxi_res", "plan_nm_taxi"), on="MVT_ID_mvt", how="left")
for s in ("train", "rank"):
    x = f.filter(pl.col("split") == s)
    print(s, x.height, {c: round(float(x[c].is_not_null().mean()), 3) for c in ("plan_taxi_res", "plan_nm_taxi")})
f.write_parquet("data/features_e165.parquet")
print("wrote data/features_e165.parquet", f.shape)

"""E162 (journal 180): same-operator inbound on the same stand as classifier-only inputs (inb_* with CLF_ONLY_PREFIX=inb_).
Per departure (2025 and ranking): the latest arrival at (airport, stand) by the same operator (AIRCRAFT_OPERATOR_flt, else the
call-sign prefix) blocked in before take-off - 120 s and within 24 h. inb_found, inb_delay = BLOCK_arr - SCHED_arr,
inb_slack = SCHED_dep - BLOCK_arr, inb_age = MVT_dep - BLOCK_arr. Arrival block times are known in ranking. Writes
data/features_e162.parquet = data/features_e106.parquet + inb_*.
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

ep = lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt", "FLIGHT_mvt", "AIRCRAFT_OPERATOR_flt", "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
files = sorted(glob.glob("data/prc-2026-datasets/training_*.parquet")) + ["data/prc-2026-datasets/ranking.parquet"]
raw = pl.concat([pl.read_parquet(p, columns=cols) for p in files], how="diagonal_relaxed").with_columns(
    op=pl.coalesce("AIRCRAFT_OPERATOR_flt", pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]{2,3})", 1)))
arr = raw.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("STAND_mvt").is_not_null() & pl.col("op").is_not_null()).select(
    ap=pl.col("ADES_mvt"), stand=pl.col("STAND_mvt"), op="op", t_in=ep("BLOCK_TIME_UTC_mvt"), s_in=ep("SCHED_TIME_UTC_mvt")).drop_nulls("t_in").sort("t_in")
dep = raw.filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt", ap=pl.col("ADEP_mvt"), stand=pl.col("STAND_mvt"), op="op",
                                                       mvt=ep("MVT_TIME_UTC_mvt"), s_dep=ep("SCHED_TIME_UTC_mvt")).with_columns(key=pl.col("mvt") - 120).sort("key")
j = dep.join_asof(arr, left_on="key", right_on="t_in", by=["ap", "stand", "op"], strategy="backward", tolerance=86400.0)
j = j.select("MVT_ID_mvt", inb_found=pl.col("t_in").is_not_null().cast(pl.Float64), inb_delay=pl.col("t_in") - pl.col("s_in"),
             inb_slack=pl.col("s_dep") - pl.col("t_in"), inb_age=pl.col("mvt") - pl.col("t_in"))
f = pl.read_parquet("data/features_e106.parquet").join(j, on="MVT_ID_mvt", how="left")
for s in ("train", "rank"):
    x = f.filter(pl.col("split") == s)
    print(s, x.height, "inbound found", round(float(x["inb_found"].mean()), 3))
f.write_parquet("data/features_e162.parquet")
print("wrote data/features_e162.parquet", f.shape)

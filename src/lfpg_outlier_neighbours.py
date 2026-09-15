"""Review 15.09, item 3: neighbours by MVT_ID of the two LFPG outliers (182378789 EJU983W y=84240, 182377662 EJU42AY y=58206),
and whether their BLOCK equals a time of another record (neighbour, same flight on other days, any LFPG movement).
    .venv/bin/python src/lfpg_outlier_neighbours.py"""
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(250)
jan = pl.concat([pl.read_parquet("data/prc-2026-datasets/training_2024-12-01_2025-01-01.parquet") if os.path.exists("data/prc-2026-datasets/training_2024-12-01_2025-01-01.parquet") else pl.DataFrame(),
                 pl.read_parquet("data/prc-2026-datasets/training_2025-01-01_2025-02-01.parquet")], how="diagonal_relaxed")
lfpg = jan.filter((pl.col("ADEP_mvt") == "LFPG") | (pl.col("ADES_mvt") == "LFPG")).sort("MVT_ID_mvt")
show = ["MVT_ID_mvt", "PHASE_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "STAND_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "FLIGHT_ID_mvt"]
for mid in (182378789, 182377662):
    row = lfpg.filter(pl.col("MVT_ID_mvt") == mid)
    print(f"\n=== {mid}")
    print(lfpg.filter(pl.col("MVT_ID_mvt").is_between(mid - 4, mid + 4)).select(show))
    blk, fl = row["BLOCK_TIME_UTC_mvt"][0], row["FLIGHT_mvt"][0]
    print("same flight number, all January rows:")
    print(jan.filter(pl.col("FLIGHT_mvt") == fl).sort("MVT_TIME_UTC_mvt").select(show))
    near = lfpg.filter(((pl.col("BLOCK_TIME_UTC_mvt") - blk).dt.total_seconds().abs() <= 60) | ((pl.col("MVT_TIME_UTC_mvt") - blk).dt.total_seconds().abs() <= 60)
                       | ((pl.col("SCHED_TIME_UTC_mvt") - blk).dt.total_seconds().abs() <= 60))
    print("LFPG records with BLOCK, MVT or SCHED within 60 s of this BLOCK:", near.height)
    print(near.select(show).head(10))

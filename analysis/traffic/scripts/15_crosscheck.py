"""Independent recount of headline numbers straight from raw columns (no common.py helpers)."""
import polars as pl, os
D = os.path.expanduser("~/prc-2026/data/prc-2026-datasets")
c = ["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "MARKET_SEGMENT_flt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt"]
j25 = pl.read_parquet(f"{D}/training_2025-07-01_2025-08-01.parquet", columns=c)
a25 = pl.read_parquet(f"{D}/training_2025-01-01_2025-02-01.parquet", columns=c)
r = pl.read_parquet(f"{D}/ranking.parquet", columns=c)
j26 = r.filter(pl.col("MVT_TIME_UTC_mvt").dt.month() == 7); a26 = r.filter(pl.col("MVT_TIME_UTC_mvt").dt.month() == 1)
def at(df, h): return df.filter(((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == h)) | ((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == h)))
for h in ["LEMD", "LEBL", "EDDM", "EDDF", "EHAM"]:
    print(h, "Jul25", at(j25, h).height, "Jul26", at(j26, h).height, f"{(at(j26,h).height/at(j25,h).height-1)*100:+.2f}%",
          "| Jan25", at(a25, h).height, "Jan26", at(a26, h).height, f"{(at(a26,h).height/at(a25,h).height-1)*100:+.2f}%")
for nm, df in [("Jul25", j25), ("Jul26", j26)]:
    x = at(df, "EDDF").filter(pl.col("MARKET_SEGMENT_flt").is_not_null())
    print("EDDF lowcost share", nm, round((x["MARKET_SEGMENT_flt"] == "Lowcost").mean() * 100, 2))
    t = df.with_columns(t=pl.coalesce("AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt"))
    neo = t.filter(pl.col("t").is_in(["A19N", "A20N", "A21N"])).height; ceo = t.filter(pl.col("t").is_in(["A318", "A319", "A320", "A321"])).height
    print("neo share of A320 family, all hubs", nm, round(neo / (neo + ceo) * 100, 2))

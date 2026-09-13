"""Extra data-quality checks referenced in REPORT 'Ловушки данных'."""
import polars as pl, datetime as dt
from common import load_all, HUBS
df = load_all().select("hub", "month", "date_utc", "PHASE_mvt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", "ADES_flt", "ADES_FILED_flt",
                       "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt", "ADEP_mvt", "ADES_mvt", "ADEP_flt", "other", "other_nm", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt").collect()
print("Rows labelled Military segment:", df.filter(pl.col("MARKET_SEGMENT_flt") == "Military").height, " FLIGHT_TYPE M:", df.filter(pl.col("FLIGHT_TYPE_flt") == "M").height)
print(df.filter(pl.col("MARKET_SEGMENT_flt") == "Military").group_by("hub").len().sort("len", descending=True).to_dicts())
jan26 = df.filter((pl.col("month") == "2026-01") & pl.col("ADES_flt").is_not_null() & (pl.col("ADES_flt") != pl.col("ADES_FILED_flt")))
print("\nJan 2026 diversion rows:", jan26.height)
print(" of which on 2026-01-01..10 (any hub):", jan26.filter(pl.col("date_utc") <= dt.date(2026, 1, 10)).height)
print(" by hub:", jan26.group_by("hub").len().sort("len", descending=True).to_dicts())
print(" top days:", jan26.group_by("date_utc").len().sort("len", descending=True).head(6).to_dicts())
dec25 = df.filter((pl.col("month") == "2025-12") & pl.col("ADES_flt").is_not_null() & (pl.col("ADES_flt") != pl.col("ADES_FILED_flt")))
print("Dec 2025 diversion rows by hub:", dec25.group_by("hub").len().sort("len", descending=True).to_dicts())
print("\nEGLL: DEP rows whose NM ADES_flt == EGLL (returns) :", df.filter((pl.col("hub") == "EGLL") & (pl.col("PHASE_mvt") == "DEP") & (pl.col("ADES_flt") == "EGLL")).height,
      " ARR rows with ADEP_flt == EGLL:", df.filter((pl.col("hub") == "EGLL") & (pl.col("PHASE_mvt") == "ARR") & (pl.col("ADEP_flt") == "EGLL")).height)
for h in HUBS:
    x = df.filter(pl.col("hub") == h)
    print(f"{h}: ARR rows where ADEP_flt == hub (return to origin seen as arrival): {x.filter((pl.col('PHASE_mvt')=='ARR') & (pl.col('ADEP_flt')==h)).height};  DEP rows with ADES_flt == hub: {x.filter((pl.col('PHASE_mvt')=='DEP') & (pl.col('ADES_flt')==h)).height}")
print("\nAirport-reported counterpart != NM counterpart:", df.filter(pl.col("other") != pl.col("other_nm")).height, "of", df.filter(pl.col("other_nm").is_not_null()).height)
print("by hub:", df.filter(pl.col("other") != pl.col("other_nm")).group_by("hub").len().sort("len", descending=True).to_dicts())
print("\nAIRCRAFT_TYPE_mvt != AIRCRAFT_TYPE_flt (both present):", df.filter(pl.col("AIRCRAFT_TYPE_mvt").is_not_null() & pl.col("AIRCRAFT_TYPE_flt").is_not_null() & (pl.col("AIRCRAFT_TYPE_mvt") != pl.col("AIRCRAFT_TYPE_flt"))).height)
print("top mismatched pairs:", df.filter(pl.col("AIRCRAFT_TYPE_mvt") != pl.col("AIRCRAFT_TYPE_flt")).group_by("AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt").len().sort("len", descending=True).head(8).to_dicts())
# segment null (unmatched) by hub and period
u = df.group_by("hub", "month").agg(unmatched_pct=pl.col("MARKET_SEGMENT_flt").is_null().mean() * 100).filter(pl.col("month").is_in(["2025-01", "2025-07", "2026-01", "2026-07"]))
with pl.Config(float_precision=2, tbl_cols=-1):
    print("\nUnmatched (no NM flight) % by hub/month:"); print(u.pivot(on="month", index="hub", values="unmatched_pct").sort("hub"))
# month boundary: UTC month vs local month difference
print("\nMovements whose UTC month differs from local month (all data):", df.filter(pl.col("MVT_TIME_UTC_mvt").dt.month() != pl.col("date_utc").dt.month()).height, "(sanity=0)")
# day-level delay between SCHED and MVT: rows with MVT on a different UTC date than SCHED (month-edge leakage)
print("Rows in Jan/Jul 2026 file whose SCHED time is outside the month:", df.filter(pl.col("month").is_in(["2026-01","2026-07"]) & (pl.col("SCHED_TIME_UTC_mvt").dt.strftime("%Y-%m") != pl.col("month"))).height)

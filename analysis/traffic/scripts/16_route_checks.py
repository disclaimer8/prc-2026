"""Robustness of headline route drops: count ALL movements (both directions, any segment, NM or airport codes) on selected city pairs."""
import polars as pl
from common import load_all
df = load_all().filter(pl.col("month").is_in(["2025-07", "2026-07", "2025-01", "2026-01"])).select("hub", "other", "other_nm", "month", "MARKET_SEGMENT_flt").collect()
pairs = [("EDDF", "EDDS"), ("EDDF", "EDDW"), ("EGLL", "EDDK"), ("EDDM", "EPGD"), ("LEMD", "SVMI"), ("LEMD", "ZGGG"), ("LEMD", "SVVA"), ("LSZH", "KDFW"), ("EDDF", "EGKK"), ("LTFM", "UZTT"), ("LTFM", "UTTT")]
for h, o in pairs:
    x = df.filter((pl.col("hub") == h) & ((pl.col("other") == o) | (pl.col("other_nm") == o)))
    c = x.group_by("month").len().sort("month")
    print(h, o, "all movements (arr+dep, any segment):", {r["month"]: r["len"] for r in c.iter_rows(named=True)},
          "segments:", x.group_by("MARKET_SEGMENT_flt").len().sort("len", descending=True).head(3).to_dicts())
# EDDF & EDDM: domestic (DE) passenger departures Jul 2025 vs Jul 2026
from common import airports
ap = airports().select(pl.col("icao").alias("other_nm"), "iso_country")
d = load_all().filter(pl.col("month").is_in(["2025-07", "2026-07", "2025-01", "2026-01"]) & (pl.col("PHASE_mvt") == "DEP") & pl.col("hub").is_in(["EDDF", "EDDM"])).select("hub", "other_nm", "month", "MARKET_SEGMENT_flt").collect().join(ap, on="other_nm", how="left")
print("\nDomestic (DE) departures, any segment, EDDF/EDDM:")
print(d.filter(pl.col("iso_country") == "DE").group_by("hub", "month").len().sort("hub", "month"))
print("Regional-segment departures EDDF/EDDM:")
print(d.filter(pl.col("MARKET_SEGMENT_flt") == "Regional").group_by("hub", "month").len().sort("hub", "month"))
# A380 share of all A388 movements by hub in 2025
a = load_all().filter(pl.col("src") == "training").select("hub", t=pl.coalesce("AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt")).filter(pl.col("t") == "A388").collect()
print("\nA388 movements 2025 by hub, % of all A388 movements at the 10 hubs:")
print(a.group_by("hub").len().with_columns(pct=pl.col("len") / a.height * 100).sort("len", descending=True))

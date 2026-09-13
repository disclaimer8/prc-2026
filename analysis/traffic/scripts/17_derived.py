"""Derived percentages quoted in REPORT that are not printed elsewhere."""
import polars as pl
from common import load_all, airports
ap = airports().select(pl.col("icao").alias("other_nm"), "iso_country")
d = load_all().filter(pl.col("month").is_in(["2025-01", "2025-07", "2026-01", "2026-07"])).select("hub", "PHASE_mvt", "other_nm", "month", "MARKET_SEGMENT_flt").collect().join(ap, on="other_nm", how="left")
def chg(a, b): return f"{a} -> {b} ({(b/a-1)*100:+.1f}%)"
for h in ["EDDF", "EDDM"]:
    x = d.filter((pl.col("hub") == h) & (pl.col("PHASE_mvt") == "DEP"))
    for mon in ["01", "07"]:
        dom = [x.filter((pl.col("month") == f"{y}-{mon}") & (pl.col("iso_country") == "DE")).height for y in (2025, 2026)]
        reg = [x.filter((pl.col("month") == f"{y}-{mon}") & (pl.col("MARKET_SEGMENT_flt") == "Regional")).height for y in (2025, 2026)]
        print(h, mon, "domestic DE departures", chg(*dom), "| Regional-segment departures", chg(*reg))
# pair totals quoted
for h, o in [("EDDF", "EDDS"), ("EDDF", "EDDW"), ("EDDM", "EPGD"), ("LEMD", "SVMI")]:
    x = load_all().filter((pl.col("hub") == h) & (pl.col("other_nm") == o) & pl.col("month").is_in(["2025-07", "2026-07"])).group_by("month").len().collect().sort("month")
    print(h, o, x.to_dicts())
# heathrow share of widebody in its movements vs rank; share of hub movements in top-10 destinations
y = load_all().filter((pl.col("src") == "training") & (pl.col("PHASE_mvt") == "DEP") & pl.col("MARKET_SEGMENT_flt").is_in(["Mainline", "Lowcost", "Regional"])).group_by("hub", "other_nm").len().collect()
t = y.sort("len", descending=True).group_by("hub").agg(top10=pl.col("len").head(10).sum(), tot=pl.col("len").sum()).with_columns(top10_pct=pl.col("top10") / pl.col("tot") * 100).sort("top10_pct")
print("\nShare of 2025 pax departures going to the hub's top-10 destinations:"); print(t)

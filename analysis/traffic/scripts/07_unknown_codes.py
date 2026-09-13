"""Destination/origin codes not in OurAirports: where, when, which types. Aggregates only."""
import polars as pl
from common import load_all, airports
ap = airports()
codes = ap["icao"].to_list()
df = load_all().select("hub","other","PHASE_mvt","month","MARKET_SEGMENT_flt","AIRCRAFT_TYPE_mvt","ADEP_flt","ADES_flt","FLIGHT_TYPE_flt").collect()
unk = df.filter(~pl.col("other").is_in(codes))
print("movements with counterpart code not in OurAirports:", unk.height, "of", df.height)
top = unk.group_by("other").len().sort("len", descending=True).head(15)
print(top)
for c in top["other"].head(10):
    x = unk.filter(pl.col("other")==c)
    print(f"\n{c}: n={x.height}")
    print(" by hub:", x.group_by("hub").len().sort("len",descending=True).head(5).to_dicts())
    print(" by month:", x.group_by("month").len().sort("month").to_dicts())
    print(" by segment:", x.group_by("MARKET_SEGMENT_flt").len().sort("len",descending=True).head(4).to_dicts())
    print(" by type:", x.group_by("AIRCRAFT_TYPE_mvt").len().sort("len",descending=True).head(4).to_dicts())
    print(" flt ADEP/ADES pairs:", x.group_by("ADEP_flt","ADES_flt").len().sort("len",descending=True).head(4).to_dicts())

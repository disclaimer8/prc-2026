"""Completeness: rows per hub per UTC day; gaps; duplicate IDs; hub derivation sanity."""
import polars as pl
from common import load_all, HUBS, OUT

lf = load_all()
df = lf.select("MVT_ID_mvt", "hub", "PHASE_mvt", "date_utc", "MVT_TIME_UTC_mvt", "src", "month",
               "MARKET_SEGMENT_flt", "ADEP_flt", "ADES_flt").collect()
print("rows total:", df.height, " training:", df.filter(pl.col("src")=="training").height,
      " ranking:", df.filter(pl.col("src")=="ranking").height)
print("hub null:", df["hub"].null_count(), " hub not in HUBS:", df.filter(~pl.col("hub").is_in(HUBS)).height)
print("duplicate MVT_ID:", df.height - df["MVT_ID_mvt"].n_unique())
print("UTC range per src:")
print(df.group_by("src").agg(pl.col("MVT_TIME_UTC_mvt").min().alias("min"), pl.col("MVT_TIME_UTC_mvt").max().alias("max")))
print("rows per month (UTC):")
with pl.Config(tbl_rows=40):
    print(df.group_by("month").len().sort("month"))

daily = df.filter(pl.col("hub").is_in(HUBS)).group_by("hub", "date_utc").len()
# full calendar
import datetime as dt
cal = []
for y, m, nd in [(2025, mm, None) for mm in range(1, 13)] + [(2026, 1, None), (2026, 7, None)]:
    d = dt.date(y, m, 1)
    while d.month == m:
        cal.append(d); d += dt.timedelta(days=1)
grid = pl.DataFrame({"date_utc": cal}).join(pl.DataFrame({"hub": HUBS}), how="cross")
full = grid.join(daily, on=["hub", "date_utc"], how="left").with_columns(pl.col("len").fill_null(0))
full = full.with_columns(month=pl.col("date_utc").dt.strftime("%Y-%m"))
full = full.with_columns(med=pl.col("len").median().over("hub", "month"))
full = full.with_columns(ratio=pl.col("len") / pl.col("med"))
print("\nDays per hub with 0 movements:")
print(full.filter(pl.col("len") == 0).group_by("hub", "month").len().sort("hub", "month"))
print("\nHub-days with count < 70% of hub-month median (possible gaps):")
with pl.Config(tbl_rows=200):
    print(full.filter(pl.col("ratio") < 0.7).sort("hub", "date_utc"))
print("\nDays with data per hub per month (count>0):")
cov = full.filter(pl.col("len") > 0).group_by("hub", "month").len().pivot(on="hub", index="month", values="len").sort("month")
with pl.Config(tbl_rows=40, tbl_cols=-1):
    print(cov)
full.write_csv(OUT + "/scripts/01_daily_counts.csv")

# hourly gaps: hours with 0 movements between 07-21 UTC per hub-day
hourly = df.filter(pl.col("hub").is_in(HUBS)).with_columns(h=pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h")).group_by("hub", "h").len()
hgrid = []
for d in cal:
    for hh in range(6, 20):
        hgrid.append(dt.datetime(d.year, d.month, d.day, hh, tzinfo=dt.timezone.utc))
hg = pl.DataFrame({"h": hgrid}).with_columns(pl.col("h").dt.cast_time_unit("us").dt.convert_time_zone("UTC")).join(pl.DataFrame({"hub": HUBS}), how="cross")
hz = hg.join(hourly, on=["hub", "h"], how="left").filter(pl.col("len").is_null())
print("\nDaytime (06-19 UTC) hours with ZERO movements, per hub:")
with pl.Config(tbl_rows=100):
    print(hz.group_by("hub").len().sort("hub"))
    print(hz.sort("hub", "h").head(60))
print("\nMarket segment null share by src:")
print(df.group_by("src").agg(pl.col("MARKET_SEGMENT_flt").is_null().mean()))

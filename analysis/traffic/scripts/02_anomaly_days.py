"""Hourly structure of low-count days: data gap (whole hours missing) vs reduced ops (spread)."""
import polars as pl, datetime as dt
from common import load_all

days = [("EDDF","2025-03-10"),("EDDM","2025-03-10"),("EDDM","2025-02-27"),("EDDM","2025-02-28"),
        ("EGLL","2025-03-21"),("EHAM","2026-01-04"),("EHAM","2026-01-05"),("EHAM","2026-01-06"),
        ("EHAM","2026-01-07"),("LEMD","2025-09-04"),("LTFM","2025-02-21"),("LTFM","2025-02-22"),
        ("EDDF","2025-12-25")]
lf = load_all()
for hub, d in days:
    dd = dt.date.fromisoformat(d)
    x = lf.filter((pl.col("hub")==hub) & (pl.col("date_utc")==dd)).select("MVT_TIME_UTC_mvt","PHASE_mvt","MARKET_SEGMENT_flt","ADES_FILED_flt","ADES_flt").collect()
    h = x.with_columns(hr=pl.col("MVT_TIME_UTC_mvt").dt.hour()).group_by("hr").len().sort("hr")
    ref = lf.filter((pl.col("hub")==hub) & (pl.col("date_utc")==dd - dt.timedelta(days=7))).select("MVT_TIME_UTC_mvt").collect()
    rh = ref.with_columns(hr=pl.col("MVT_TIME_UTC_mvt").dt.hour()).group_by("hr").len().sort("hr")
    m = pl.DataFrame({"hr": list(range(24))}, schema={"hr": pl.Int8}).join(h.with_columns(pl.col("hr").cast(pl.Int8)), on="hr", how="left").join(rh.with_columns(pl.col("hr").cast(pl.Int8)), on="hr", how="left", suffix="_prevweek").fill_null(0)
    print(f"\n== {hub} {d}: total {x.height} (same weekday -7d: {ref.height}); DEP {x.filter(pl.col('PHASE_mvt')=='DEP').height} ARR {x.filter(pl.col('PHASE_mvt')=='ARR').height}")
    print("hour UTC: " + " ".join(f"{r['hr']:02d}:{r['len']}/{r['len_prevweek']}" for r in m.iter_rows(named=True)))

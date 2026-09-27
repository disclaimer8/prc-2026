import glob, sys
import numpy as np
import polars as pl
sys.argv = ["x"]
exec(open("src/lirf_dayshift_probe_2026_09_27.py").read().split("tot = None")[0])
u = tr.filter(pl.col("utc_shift"))
u = u.with_columns(when=pl.when(pl.col("mvt_h") < 0.42).then(pl.lit("a 00:00-00:25")).when(pl.col("mvt_h") < 3).then(pl.lit("b 00:25-03:00")).otherwise(pl.lit("c >=03:00")))
print(f"2025 LIRF DEP with a UTC date change: {u.height}; July: {u.filter(pl.col('MVT_TIME_UTC_mvt').dt.month() == 7).height}")
for per, x in (("all 2025", u), ("July 2025", u.filter(pl.col("MVT_TIME_UTC_mvt").dt.month() == 7))):
    print(f"\n{per}: regime by takeoff time x NM")
    print(x.group_by("when", "nm").agg(n=pl.len(), normal=(pl.col("regime") == "normal").sum(), SCHED=(pl.col("regime") == "SCHED").sum(),
          plus_day=(pl.col("regime") == "+day").sum(), other=(pl.col("regime") == "other").sum(),
          y_min=pl.col("y").min(), y_med=pl.col("y").median(), y_max=pl.col("y").max(), pred_med=pl.col("pred").median().round(0)).sort("when", "nm"))
print("\nall 2025, takeoff >=03:00 UTC with NM (the RYRR6986 case), row by row:")
print(u.filter(pl.col("nm") & (pl.col("when") == "c >=03:00")).select("FLIGHT_mvt", "CALLSIGN_flt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "D", "mvt_aobt", "y", "regime", "pred").sort("MVT_TIME_UTC_mvt").to_pandas().to_string())

import glob
import numpy as np
import polars as pl

COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt"]
def prep(df):
    tz = "Europe/Rome"
    return df.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF")).with_columns(
        D=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
        utc_shift=(pl.col("MVT_TIME_UTC_mvt").dt.date() > pl.col("SCHED_TIME_UTC_mvt").dt.date()),
        loc_shift=(pl.col("MVT_TIME_UTC_mvt").dt.convert_time_zone(tz).dt.date() > pl.col("SCHED_TIME_UTC_mvt").dt.convert_time_zone(tz).dt.date()),
        mvt_h=pl.col("MVT_TIME_UTC_mvt").dt.hour() + pl.col("MVT_TIME_UTC_mvt").dt.minute() / 60,
        nm=pl.col("FLIGHT_ID_mvt").is_not_null(),
        pfx=pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]+)", 1),
        odd=(pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]+)", 1).str.len_chars() >= 4) |
            (pl.col("CALLSIGN_flt").is_not_null() & (pl.col("FLIGHT_mvt") != pl.col("CALLSIGN_flt"))),
        mvt_aobt=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64))

tr = prep(pl.concat([pl.read_parquet(f, columns=COLS) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]))
oof = pl.concat([pl.DataFrame({"MVT_ID_mvt": (z := np.load(f))["mvt_id"], "pred": z["pred"]}) for f in sorted(glob.glob("data/cv_preds_e106_air/fold_*.npz"))])
tr = tr.join(oof.with_columns(pl.col("MVT_ID_mvt").cast(tr["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt", how="left").with_columns(
    y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
    bs=((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 6))
tr = tr.with_columns(regime=pl.when(pl.col("bs")).then(pl.lit("SCHED"))
                     .when(pl.col("y") >= 80000).then(pl.lit("+day"))
                     .when(pl.col("y") <= 3600).then(pl.lit("normal")).otherwise(pl.lit("other")),
                     err=pl.col("pred") - pl.col("y"))
tot = None
s = tr.filter(pl.col("loc_shift") | pl.col("utc_shift"))
print(f"2025 LIRF DEP with a date change (local or UTC): {s.height} (local {int(s['loc_shift'].sum())}, UTC {int(s['utc_shift'].sum())})")
s = s.with_columns(when=pl.when(pl.col("mvt_h") < 0.42).then(pl.lit("00:00-00:25")).when(pl.col("mvt_h") < 3).then(pl.lit("00:25-03:00")).otherwise(pl.lit(">03:00")))
print("\nregime counts by takeoff time x NM x odd callsign (all months):")
print(s.group_by("when", "nm", "odd").agg(n=pl.len(), normal=(pl.col("regime") == "normal").sum(), SCHED=(pl.col("regime") == "SCHED").sum(),
      plus_day=(pl.col("regime") == "+day").sum(), other=(pl.col("regime") == "other").sum(),
      med_y=pl.col("y").median(), med_pred=pl.col("pred").median().round(0)).sort("when", "nm", "odd"))
j = s.filter(pl.col("MVT_TIME_UTC_mvt").dt.month() == 7)
print(f"\nJuly only: {j.height} rows; >03:00 without NM: {j.filter((pl.col('when') == '>03:00') & ~pl.col('nm')).height}")
print(j.filter(pl.col("when") == ">03:00").group_by("nm", "odd").agg(n=pl.len(), regimes=pl.col("regime").value_counts()).to_dicts())
print("\nthe RYRR6986 analogue: NM present, takeoff >03:00, date change, D 4-12 h:")
a = s.filter(pl.col("nm") & (pl.col("when") == ">03:00") & pl.col("D").is_between(4 * 3600, 12 * 3600))
print(a.select("FLIGHT_mvt", "CALLSIGN_flt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "D", "mvt_aobt", "y", "regime", "pred").sort("MVT_TIME_UTC_mvt").to_pandas().to_string(max_rows=60))
print(a.group_by("odd").agg(n=pl.len(), normal=(pl.col("regime") == "normal").sum(), SCHED=(pl.col("regime") == "SCHED").sum(),
                            plus_day=(pl.col("regime") == "+day").sum(), other=(pl.col("regime") == "other").sum()))

rk = prep(pl.read_parquet("data/prc-2026-datasets/ranking.parquet", columns=[c for c in COLS if c not in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt")]))
sub = pl.read_parquet("submissions/gentle-igloo_v366.parquet").rename({"TAXITIME_SEC_mvt": "v366"})
rk = rk.join(sub.with_columns(pl.col("MVT_ID_mvt").cast(rk["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt", how="left")
r = rk.filter(pl.col("loc_shift") | pl.col("utc_shift")).with_columns(
    when=pl.when(pl.col("mvt_h") < 0.42).then(pl.lit("00:00-00:25")).when(pl.col("mvt_h") < 3).then(pl.lit("00:25-03:00")).otherwise(pl.lit(">03:00")))
print(f"\nranking LIRF DEP with a date change: {r.height} (UTC {int(r['utc_shift'].sum())}, local {int(r['loc_shift'].sum())})")
print(r.group_by("when", "nm", "odd").agg(n=pl.len(), med_D=pl.col("D").median(), med_v366=pl.col("v366").median(), max_v366=pl.col("v366").max()).sort("when", "nm", "odd"))
print(r.filter(pl.col("nm") & (pl.col("when") == ">03:00")).select("MVT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "D", "mvt_aobt", "v366").sort("MVT_TIME_UTC_mvt").to_pandas().to_string())

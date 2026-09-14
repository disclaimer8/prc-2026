"""Coverage and accuracy of OPDI ground events against PRC block times (July 2025 sample)."""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_width_chars(240)
APS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
events_file, list_file, start, end = sys.argv[1:5]


def as_utc(col):
    dt = pl.col(col)
    return pl.when(True).then(dt)


ev = pl.read_parquet(events_file).filter(
    pl.col("type").is_in(["exit-parking_position", "exit-apron", "entry-taxiway", "entry-runway", "take-off"])
)
if ev.schema["event_time"] == pl.String:
    ev = ev.with_columns(t=pl.col("event_time").str.to_datetime("%Y-%m-%d %H:%M:%S", time_zone="UTC"))
else:
    ev = ev.with_columns(t=pl.col("event_time").dt.replace_time_zone("UTC") if ev.schema["event_time"].time_zone is None else pl.col("event_time"))
fl = pl.read_parquet(list_file, columns=["id", "flt_id", "adep"])
ev = ev.join(fl, left_on="flight_id", right_on="id", how="inner").filter(pl.col("adep").is_in(APS))
first = ev.group_by("flight_id", "type").agg(t=pl.col("t").min(), flt_id=pl.col("flt_id").first(), adep=pl.col("adep").first())
wide = first.pivot(on="type", index=["flight_id", "flt_id", "adep"], values="t")

tr = pl.read_parquet("data/prc-2026-datasets/training_*.parquet").filter(pl.col("PHASE_mvt") == "DEP")
s, e = pl.lit(start).str.to_datetime("%Y%m%d", time_zone="UTC"), pl.lit(end).str.to_datetime("%Y%m%d", time_zone="UTC")
tr = tr.filter((pl.col("MVT_TIME_UTC_mvt") >= s) & (pl.col("MVT_TIME_UTC_mvt") < e)).with_columns(
    cs=pl.col("CALLSIGN_flt").fill_null(pl.col("FLIGHT_mvt")).str.replace_all(" ", "")
)
j = tr.join(wide, left_on=["cs", "ADEP_mvt"], right_on=["flt_id", "adep"], how="inner")
if "take-off" in j.columns:
    j = j.filter(((pl.col("take-off") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds().abs() <= 900) | pl.col("take-off").is_null())
j = j.sort("flight_id").unique(["MVT_ID_mvt"], keep="first")
print("PRC departures in window:", tr.height, " matched to an OPDI flight:", j.height)
rows = []
for ap in APS:
    a = j.filter(pl.col("ADEP_mvt") == ap)
    n = tr.filter(pl.col("ADEP_mvt") == ap).height
    row = {"ap": ap, "deps": n, "matched": round(a.height / max(n, 1), 3)}
    for c, tag in [("exit-parking_position", "stand"), ("entry-taxiway", "twy")]:
        if c not in a.columns:
            continue
        x = a.filter(pl.col(c).is_not_null()).with_columns(d=(pl.col(c) - pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds())
        row[f"{tag}_cov"] = round(x.height / max(n, 1), 3)
        if x.height:
            row[f"{tag}_med_s"] = x["d"].median()
            row[f"{tag}_in60"] = round((x["d"].abs() <= 60).mean(), 3)
            row[f"{tag}_in180"] = round((x["d"].abs() <= 180).mean(), 3)
            row[f"{tag}_corr_y"] = round(x.select(pl.corr((pl.col("MVT_TIME_UTC_mvt") - pl.col(c)).dt.total_seconds(), pl.col("TAXITIME_SEC_mvt"))).item(), 3)
    rows.append(row)
print(pl.DataFrame(rows))

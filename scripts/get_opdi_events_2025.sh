#!/bin/bash
# Journal 168: OPDI v002 flight events for Jan/Jul 2025 and 2026 (organiser 06.09: "you can eventually use it"), streamed:
# download one 10-day file, keep departures from our 10 airports (via the monthly flight list) and the ground event types,
# write data/opdi/ev_<window>.parquet, delete the raw file.
cd ~/prc-2026
B=https://www.eurocontrol.int/performance/data/download/OPDI/v002
mkdir -p data/opdi/raw
for m in 202501 202502 202503 202504 202505 202506 202507 202508 202509 202510 202511 202512 202601 202607; do
  [[ -s data/opdi/flight_list_$m.parquet ]] || curl -s -A 'Mozilla/5.0' -o data/opdi/flight_list_$m.parquet $B/flight_list/flight_list_$m.parquet
done
for w in $(cat ${WINDOWS:-data/opdi/windows.txt}); do
  [[ -s data/opdi/ev_$w.parquet ]] && continue
  [[ -s data/opdi/raw/fe_$w.parquet ]] || curl -s --retry 3 -A 'Mozilla/5.0' -o data/opdi/raw/fe_$w.parquet $B/flight_events/flight_events_$w.parquet || { echo "[$(date +%H:%M)] $w download failed"; continue; }
  .venv/bin/python - "$w" <<'PY'
import sys, glob, polars as pl
w = sys.argv[1]
APS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
fl = pl.concat([pl.read_parquet(p, columns=["id", "icao24", "flt_id", "adep", "ades"]).with_columns(pl.col("id").cast(pl.String)) for p in sorted(glob.glob("data/opdi/flight_list_*.parquet"))]).unique("id")
fl = fl.filter(pl.col("adep").is_in(APS)).with_columns(pl.col("flt_id").str.strip_chars())
ev = pl.read_parquet(f"data/opdi/raw/fe_{w}.parquet").with_columns(pl.col("flight_id").cast(pl.String))
types = ev["type"].unique().to_list()
ev = ev.join(fl, left_on="flight_id", right_on="id", how="inner")
t = pl.col("event_time")
ev = ev.with_columns(t=(t.str.to_datetime("%Y-%m-%d %H:%M:%S", time_zone="UTC", strict=False) if ev.schema["event_time"] == pl.String
                        else (t.dt.replace_time_zone("UTC") if ev.schema["event_time"].time_zone is None else t)))
keep = [c for c in ["flight_id", "icao24", "flt_id", "adep", "ades", "type", "t", "longitude", "latitude", "altitude", "source", "version"] if c in ev.columns]
ev.select(keep).write_parquet(f"data/opdi/ev_{w}.parquet")
print(f"{w}: kept {ev.height} events of our departures; types in file: {sorted(types)}")
PY
  [[ -s data/opdi/ev_$w.parquet ]] && rm -f data/opdi/raw/fe_$w.parquet || echo "[$(date +%H:%M)] $w filter failed, raw kept"
done
echo "[$(date +%H:%M)] done"

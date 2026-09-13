#!/usr/bin/env bash
# OPDI v0.0.2 flight_events: удалённое чтение (duckdb httpfs, range requests) без сохранения целых файлов (270–550 МБ).
# Сохраняет сводку типов событий по 10 аэропортам и срез одной даты (<~10 МБ).
set -euo pipefail
O=~/prc-2026/analysis/sources/samples/eurocontrol_opdi; mkdir -p $O
B=https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events
APTS="'EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM'"
for F in flight_events_20250704_20250714 flight_events_20260110_20260120 flight_events_20260709_20260719; do
D=$(date -j -v+1d -f %Y%m%d "${F:14:8}" +%Y-%m-%d)
duckdb -c "INSTALL httpfs; LOAD httpfs; SET http_timeout=300000;
CREATE TEMP TABLE e AS SELECT type, CAST(event_time AS VARCHAR) event_time, longitude, latitude, info, flight_id,
  CASE WHEN json_valid(info) THEN json_extract_string(info,'\$.osm_airport') END apt FROM '$B/$F.parquet';
COPY (SELECT apt, type, count(*) n, count(DISTINCT flight_id) flights, min(event_time) tmin, max(event_time) tmax
  FROM e WHERE apt IN ($APTS) GROUP BY ALL ORDER BY apt, n DESC) TO '$O/${F}_summary_10apt.csv' (HEADER);
COPY (SELECT * FROM e WHERE apt IN ($APTS) AND event_time LIKE '$D%') TO '$O/${F}_${D}_10apt.parquet';"
echo "$F done"
done

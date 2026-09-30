#!/bin/zsh
# External open data used by the model. Run from the repository root.
#   METAR: Iowa Environmental Mesonet ASOS archive (free use, attribution requested)
#   OpenStreetMap aeroway features via Overpass (ODbL 1.0, (c) OpenStreetMap contributors)
#   OurAirports airports.csv (public domain), only to locate the airports for the Overpass bounding boxes
set -euo pipefail
APS=(EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF LSZH LTFM)
# METAR_END (YYYY-MM-DD, default: tomorrow) - the archive end was hard-coded to 2026-08-02, so any later period would get null
# wx_* silently (journal 193). The start stays 2024-12-25 (2025 training plus a margin).
END=${METAR_END:-$(date -u -v+1d +%F 2>/dev/null || date -u -d tomorrow +%F)}
Y2=${END:0:4}; M2=$((10#${END:5:2})); D2=$((10#${END:8:2}))
RANGE="year1=2024&month1=12&day1=25&year2=$Y2&month2=$M2&day2=$D2&tz=Etc/UTC&format=onlycomma&latlon=no&elev=no&missing=null&trace=T&direct=no&report_type=3&report_type=4"
mkdir -p data/metar data/metar_wind data/ext
for ap in $APS; do
  curl -sf -m 300 -o data/metar/$ap.csv "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=$ap&data=tmpc&data=dwpc&data=sknt&data=gust&data=vsby&data=skyl1&data=wxcodes&data=metar&$RANGE"
  curl -sf -m 300 -o data/metar_wind/$ap.csv "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=$ap&data=drct&data=sknt&$RANGE"
done
curl -sf -o data/ext/airports.csv https://davidmegginson.github.io/ourairports-data/airports.csv
# the model was built on the OSM snapshot fetched in September 2026; pin a date for exact reproduction
OSM_DATE="${OSM_DATE:-}" uv run python scripts/osm_overpass_download.py
echo external data ready

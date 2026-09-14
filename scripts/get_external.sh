#!/bin/zsh
# External open data used by the model. Run from the repository root.
#   METAR: Iowa Environmental Mesonet ASOS archive (free use, attribution requested)
#   OpenStreetMap aeroway features via Overpass (ODbL 1.0, (c) OpenStreetMap contributors)
#   OurAirports airports.csv (public domain), only to locate the airports for the Overpass bounding boxes
set -euo pipefail
APS=(EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF LSZH LTFM)
RANGE="year1=2024&month1=12&day1=25&year2=2026&month2=8&day2=2&tz=Etc/UTC&format=onlycomma&latlon=no&elev=no&missing=null&trace=T&direct=no&report_type=3&report_type=4"
mkdir -p data/metar data/metar_wind data/ext
for ap in $APS; do
  curl -sf -m 300 -o data/metar/$ap.csv "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=$ap&data=tmpc&data=dwpc&data=sknt&data=gust&data=vsby&data=skyl1&data=wxcodes&data=metar&$RANGE"
  curl -sf -m 300 -o data/metar_wind/$ap.csv "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=$ap&data=drct&data=sknt&$RANGE"
done
curl -sf -o data/ext/airports.csv https://davidmegginson.github.io/ourairports-data/airports.csv
# the model was built on the OSM snapshot fetched in September 2026; pin a date for exact reproduction
OSM_DATE="${OSM_DATE:-}" uv run python scripts/osm_overpass_download.py
echo external data ready

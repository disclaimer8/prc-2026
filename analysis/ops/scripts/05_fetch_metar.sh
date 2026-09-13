#!/bin/bash
# Source: Iowa Environmental Mesonet (IEM), Iowa State University, ASOS/METAR archive.
# https://mesonet.agron.iastate.edu/request/download.phtml  (free, no key)
set -e
cd "$(dirname "$0")/../data_ext"
for s in EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF LSZH LTFM; do
  curl -s -m 300 -o metar_$s.csv "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=$s&data=tmpc&data=dwpc&data=vsby&data=wxcodes&year1=2024&month1=12&day1=31&year2=2026&month2=1&day2=2&tz=Etc/UTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=3&report_type=4"
  echo "$s $(wc -l < metar_$s.csv) lines"
  sleep 2
done

#!/bin/bash
# Open-Meteo Historical Weather API (ERA5 / ERA5-Land reanalysis), no key. CC BY 4.0.
# Usage: open_meteo_era5.sh START END  -> samples/open_meteo_era5/<ICAO>_<START>_<END>.json
OUT=~/prc-2026/analysis/sources/samples/open_meteo_era5
S=$1; E=$2
while IFS=, read icao lat lon; do
  curl -s -o $OUT/${icao}_${S}_${E}.json -w "$icao %{http_code} %{size_download}\n" \
   "https://archive-api.open-meteo.com/v1/archive?latitude=$lat&longitude=$lon&start_date=$S&end_date=$E&hourly=temperature_2m,dew_point_2m,precipitation,rain,snowfall,snow_depth,weather_code,wind_speed_10m,wind_gusts_10m,cloud_cover_low,freezing_level_height&timezone=GMT&models=era5_seamless"
  sleep 1
done <<'L'
EDDF,50.0333,8.5706
EDDM,48.3538,11.7861
EGLL,51.4706,-0.4619
EHAM,52.3086,4.7639
LEBL,41.2971,2.0785
LEMD,40.4719,-3.5626
LFPG,49.0097,2.5479
LIRF,41.8045,12.2508
LSZH,47.4647,8.5492
LTFM,41.2753,28.7519
L

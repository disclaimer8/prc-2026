#!/bin/bash
# DWD CDC (CC BY 4.0): Frankfurt 01420, München-Flughafen 01262 — суточные KL (снежный покров SHK_TAG) и часовые явления WW.
# KNMI (CC0): Schiphol 240 часовые данные. Meteostat (CC BY 4.0) суточные по WMO.
S=~/prc-2026/analysis/sources/samples; B=https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate
mkdir -p $S/dwd_cdc $S/knmi $S/meteostat
for s in 01420 01262; do
  curl -s -o $S/dwd_cdc/tageswerte_KL_${s}_akt.zip $B/daily/kl/recent/tageswerte_KL_${s}_akt.zip
  curl -s -o $S/dwd_cdc/stundenwerte_WW_${s}_akt.zip $B/hourly/weather_phenomena/recent/stundenwerte_WW_${s}_akt.zip
done
# для янв–мар 2025: $B/daily/kl/historical/tageswerte_KL_01262_19920517_20251231_hist.zip (и 01420_19350701_20251231)
curl -s -o $S/knmi/uurgeg_240_2021-2030.zip https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/uurgegevens/uurgeg_240_2021-2030.zip
for y in 2025 2026; do for s in 10637 10866 03772 06240 08181 08221 07157 16242 06670 17062; do
  curl -s -o $S/meteostat/d${y}_$s.csv.gz https://data.meteostat.net/daily/$y/$s.csv.gz; done; done

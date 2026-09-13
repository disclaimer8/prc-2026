#!/bin/bash
# Праздники (Nager.Date, без ключа) и школьные каникулы (OpenHolidays API, ODbL, без ключа).
S=~/prc-2026/analysis/sources/samples
mkdir -p $S/nager_date $S/openholidays
for c in DE GB NL ES FR IT CH TR; do for y in 2025 2026; do
  curl -s -o $S/nager_date/${c}_${y}.json https://date.nager.at/api/v3/PublicHolidays/$y/$c; done; done
# GB и TR в OpenHolidays отсутствуют (список /Countries проверен 13.09.2026)
for c in DE NL ES FR IT CH; do
  curl -s -H 'accept: application/json' -o $S/openholidays/school_${c}_2025-2026.json \
   "https://openholidaysapi.org/SchoolHolidays?countryIsoCode=$c&languageIsoCode=EN&validFrom=2025-01-01&validTo=2026-12-31"
  curl -s -H 'accept: application/json' -o $S/openholidays/public_${c}_2025-2026.json \
   "https://openholidaysapi.org/PublicHolidays?countryIsoCode=$c&languageIsoCode=EN&validFrom=2025-01-01&validTo=2026-12-31"
done

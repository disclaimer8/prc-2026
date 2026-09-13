#!/bin/bash
# FAA Aircraft Characteristics Database (Oct 2024), xlsx, без ключа.
mkdir -p ~/prc-2026/analysis/sources/samples/faa_acd
curl -sL -o ~/prc-2026/analysis/sources/samples/faa_acd/faa_acd_2024-10.xlsx https://www.faa.gov/airports/engineering/aircraft_char_database/aircraft_data

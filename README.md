# PRC Data Challenge 2026: taxi-out time (team gentle-igloo)

Open solution for the [EUROCONTROL PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/):
predict the taxi-out time (take-off minus off-block, seconds) of departures from ten European
airports. Ranking metric: RMSE on January and July 2026.

Code licence: GNU GPL v3 (see `LICENSE`). External data keep their own licences (below).

## Key observation

Most of the squared error comes from a few hundred departures whose recorded block time is not an
actual off-block time. For long delays several airports report the **scheduled time** (or another
planning time: IOBT, LOBT, EOBT), and a handful of records carry a block time dated one day early.
For those rows the recorded taxi-out equals `MVT_TIME - <planning time>` to the second.

The model therefore predicts a mixture:

* a multiclass LightGBM model estimates the probability of each regime
  (`regular`, `sched`, `iobt`, `lobt`, `eobt`, `dayshift`);
* regressors (LightGBM, CatBoost) predict ordinary taxi-out on regular-regime rows;
* the prediction is `sum_k p_k * candidate_k`, which is the squared-error optimal point estimate.

Validation on January and July 2025 (trained on the other 2025 months) shows the classifier is
well calibrated (predicted probability matches the observed share in every bin).

## Features (`src/features.py`)

Every feature is computable for the ranking set, where departure `BLOCK_TIME` and `TAXITIME` are
blank; departure block times are never used as inputs.

* Time differences between take-off and NM / airport planning times (AOBT_3, EOBT_1, LOBT, IOBT, schedule).
* Airport congestion from the movement table: departures and arrivals in windows around take-off and
  around the estimated off-block, aircraft taxiing at off-block, runway-level take-off counts, recent
  arrival taxi-in times (arrival block times are available in the ranking set).
* Runway configuration per airport-hour inferred from `RUNWAY_mvt` of all movements.
* Aerodrome geometry from OpenStreetMap: stand position matched on the normalised stand reference,
  straight distance and bearing offset to the departure runway start (`src/geometry.py`).
* Weather from METAR (temperature, dew-point spread, wind, visibility, snow/freezing/icing flags).
* Smoothed out-of-fold target encodings (stand x runway, runway x hour, stand, operator, type, destination)
  (`src/target_encoding.py`).

After the mixture is assembled, two more stages run: an empirical tail table with TabPFN-2 probabilities
for LIRF departures without an NM record and a delay of 4 h or more (`src/lirf_table.py`,
`src/lirf_tabpfn.py`), a logistic correction for the same rows below 4 h (`src/e14_apply.py`), and a
LightGBM stage over the ADS-B surface features (`src/e36_adsb_stand_geometry.py`).

## Reproduce

```bash
uv sync
uv run python src/download.py                          # challenge data, needs PRC_ACCESS_KEY/PRC_SECRET_KEY in .env
# METAR: see "External data" below, saved as data/metar/<ICAO>.csv
# OSM extracts: analysis/sources/scripts/osm_overpass_download.py
FEATURES_OUT=data/features_v4.parquet uv run python src/features.py
FEATURES=data/features_v4.parquet uv run python src/train.py validate
FEATURES=data/features_v4.parquet uv run python src/train.py submit <N>
```

## External data

| Source | Use | Licence |
|---|---|---|
| [adsb.lol globe_history](https://github.com/adsblol/globe_history_2026) (feeds of adsb.lol, FlyItalyADSB and TheAirTraffic) | surface tracks: when the aircraft was seen standing, when it first moved, path and stops before take-off — a second-stage correction of the prediction | ODbL 1.0 |
| Iowa Environmental Mesonet ASOS/METAR archive, Iowa State University | weather features | freely usable, attribution requested |
| OpenStreetMap via Overpass API, © OpenStreetMap contributors | stand and runway geometry | ODbL 1.0 |

Surface coverage is uneven and decides how much the ADS-B stage can do: the aircraft is seen standing
at its stand before push-back on 70% of departures at EDDM but on 0.0% at LTFM and 21-25% at LFPG,
because no receiver near those aerodromes hears ground-level transmissions. The other two replicas of
the same archive (`staging-0`, `mlatonly-0`) were checked and add nothing there (`src/e49_replica_probe.py`).

Exploratory probes that did not enter the model: ERA5 snow variables, OPDI stand events (no licence in
the repository), ADSB IQ (no surface points at our aerodromes), OpenSky (terms of use are
non-commercial research only, which is not an open licence).

## Analyses

`analysis/traffic`, `analysis/ops` contain reproducible descriptive analyses of the challenge data
(scripts with their outputs). `analysis/sources` documents the external-data survey.

## Citation

Data: EUROCONTROL Performance Review Commission Data Challenge 2026, with the support of the
OpenSky Network.

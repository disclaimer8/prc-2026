# Model card: gentle-igloo taxi-out reconstruction

A model card in the usual sections (Mitchell et al., 2019). Every number is
copied from the page it cites; nothing here is new measurement. Written 8 October
2026 for the v12075 chain and the uniform final recipe derived from it
([final submission](final-submission.md)).

## Model details

| | |
|---|---|
| Name | gentle-igloo taxi-out chain, version v12075 (leaderboard) / `gentle-igloo_final.parquet` recipe (blind final) |
| Developer | Team gentle-igloo: one developer, Denys Kolomiiets, with AI coding assistants ([README](../README.md#team)) |
| Date | Developed 13 September – 8 October 2026; v12075 scored 7 October 2026 |
| Task | Predict `TAXITIME_SEC_mvt`, the recorded taxi-out time of a departure at ten European airports (EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM), for the EUROCONTROL PRC Data Challenge 2026; metric RMSE in seconds |
| Type | Probability mixture over six recording regimes (LightGBM multiclass classifier) whose "regular" candidate is a blend of gradient-boosted residual models (CatBoost 15 seeds, LightGBM 5 seeds), five anchored LightGBM members and an ordered-trajectory neural network; followed by LightGBM surface-correction stages on the aircraft's own ADS-B track, a TabPFN-2 classifier for a sparse Rome tail, a confidence gate, LOBT-window clipping and an OPDI correction at Zurich ([model design](model-design.md)) |
| Size | Tree models: classifier 1,080 trees, regressor 2,319 trees, 170 input columns; neural member 1,908,437 parameters per seed, 189 tabular columns + 2 streams × 64 tokens × 12 channels ([model design](model-design.md)) |
| Licence | Code GNU GPLv3 ([LICENSE](../LICENSE)); data and pretrained weights keep their own terms ([SOURCES.md](../SOURCES.md)) |
| Where documented | [README](../README.md), [model design](model-design.md), [validation](validation.md), [development history](development-history.md), [negative results](negative-results.md), [glossary](glossary.md) |

## Intended use

**Primary use: post-operations reconstruction of the recorded off-block time.**
In the challenge data only the departure block time and the target are hidden; the
take-off time, the Network Manager timestamps of the flight (scheduled, initial,
latest and estimated off-block, and the minute-rounded actual off-block `AOBT_3`),
the stand, the runway, all arrivals and the aircraft's own post-pushback ADS-B
track remain visible ([README](../README.md#the-problem-as-the-data-presents-it)).
The organiser confirmed on 19 September 2026 that "the model is for post-ops, not
for tactical use" ([organiser clarifications](organiser-clarifications.md)). The
model therefore answers "when did this record say the aircraft left the stand,
given everything recorded afterwards", at the ten airports, for the recording
process of 2025–2026.

**Secondary uses demonstrated on 2025 data:**

- a detector of block times copied from planning timestamps, as a data-quality
  tool for the PRU taxi-out indicator: the regime probabilities are calibrated
  (mean 9.46 % against 9.58 % labelled copies across the seven labelled airports),
  and copied block times inflate the LIRF additional-time indicator by about 42 s
  per departure ([data quality for PRU](data-quality-regimes.md));
- the prediction expressed as unimpeded reference plus additional taxi-out time;
  the emulated monthly indicator matches the published 2025 values from April to
  October with MAE 3.3 s and correlation 0.999 ([PRU view](pru-view.md));
- a per-flight predictive distribution from the same mixture ([uncertainty](uncertainty.md)).

**Not intended for:**

- pre-departure or tactical forecasting (it uses the take-off time and the
  post-pushback track);
- airports, years or recording processes other than those it was trained and
  scored on, without retraining and re-checking the regime labelling rules, which
  are dataset-specific heuristics ([model design](model-design.md#recording-regime-mixture));
- real-time operation or a changed NM schema ([model design](model-design.md#what-this-design-does-not-establish));
- any use that treats a predicted value as a measured physical taxi duration: the
  target is a recorded interval, which at LIRF is often a planning time.

## Inputs

| Family | Source | Columns (trees / neural) |
|---|---|---|
| Planning-time intervals, airport/runway/aircraft/flight identity, local time, airport and runway traffic counts, neighbour NM proxies, stand turnaround, forward and filed-time neighbours, planned-arrival residuals, wake separation | Challenge data (movements and NM records of the same and other flights) | 142 / 145 |
| Weather | Iowa Environmental Mesonet METAR archive | 15 / 15 |
| Stand-to-runway geometry and taxi-route length | OpenStreetMap via Overpass | 8 / 8 |
| Destination daily arrival ATFM delay | EUROCONTROL Aviation Intelligence Portal (ansperformance.eu) | 5 / 5 |
| Target encodings (month-out) | 2025 labels | 0 / 16 |
| Ordered ADS-B observations of the current and previous visit | adsb.lol `globe_history` | 0 / 2 × 64 × 12 |
| Surface-stage features (movement onset, stops, stand distances, previous visit) and OPDI exit-parking events at LSZH | adsb.lol; OPDI v0.0.2 | stage inputs, not base columns |

Full table with gain shares: [model design](model-design.md#feature-families).
The tree models drop stand identity, runway configuration, `month`, day of year,
absolute timestamps and the 2025 target encodings; the neural member keeps stand,
configuration and month-out encodings. No ADS-B feature enters the base model: ADS-B
enters only through coverage-gated correction stages ([data insights §4](data-insights.md)).

## Training data

- All 2025 departures at the ten airports: 2,085,047 rows ([negative results](negative-results.md#how-the-evidence-was-produced)).
- Labels: `TAXITIME_SEC_mvt` from the challenge files. 6.8 % of 2025 labels
  coincide within 6 s with take-off minus a planning timestamp; at LIRF 18.1 % of
  block times fall in the scheduled minute; 14 labels are shifted by one day; 122
  labels exceed four hours; 369 are negative ([README](../README.md), [data insights](data-insights.md#data-caveats)).
- All challenge timestamps carry a provider dithering of up to ±6 s on each of the
  two times, so labels carry up to ±12 s of noise ([jitter analysis](../analysis/final_jitter_sensitivity_20261008.md)).
- Regime labels for the classifier are assigned by explicit comparison with the
  target at a 6 s tolerance; EDDF, EHAM and LSZH are excluded from planned-regime
  labelling ([model design](model-design.md#recording-regime-mixture)).
- External data: adsb.lol 2025 and 2026 archives (ODbL 1.0), OSM (ODbL 1.0), IEM
  METAR (public domain), ansperformance.eu daily aggregates (portal terms; accepted
  for the challenge by the organiser on 17 September 2026), OPDI v0.0.2 events at
  LSZH (accepted on 4 October 2026) ([SOURCES.md](../SOURCES.md)).
- Pretrained weights: TabPFN-2 classifier by Prior Labs (Prior Labs License 1.1,
  Apache 2.0 plus attribution; "Built with PriorLabs-TabPFN"); used for the LIRF
  tail only, weights not redistributed.
- The final recipe refits the tabular members on the same 2025 rows with the
  feature table rebuilt against the four-month final file, because two feature
  builders take medians over training and ranking rows together
  ([pipeline readiness](../analysis/final_pipeline_readiness_20261008.md)).

## Evaluation

**Cross-validation.** Six temporal folds of 2025: month pairs 1+7, 2+8, 3+9,
4+10, 5+11, 6+12 held out together, training on the other ten; inner validation
month for round selection is the month after each held-out month. Gates were
written before each run: lower paired squared error in at least 4 of 6 folds and
trimmed/body legs below zero ([validation](validation.md#temporal-splits-and-integrity-checks)).
There is no fully nested cross-validation of the exact final chain: later stages
were trained on historical out-of-fold predictions of earlier ones, and the E81
donors have no out-of-fold counterpart ([validation](validation.md#evidence-for-the-current-members)).

**2025 out-of-fold error of the chain before E81** (E195 `joint` arm; 2,084,921
rows; the 126 TabPFN tail rows excluded; the E203 member is not in these numbers):

| Airport | RMSE, all rows (s) | RMSE, labels ≤ 7,200 s (s) | Share of squared error |
|---|---:|---:|---:|
| EDDF | 103.9 | 103.9 | 2.5 % |
| EDDM | 92.7 | 92.7 | 1.4 % |
| EGLL | 196.0 | 188.6 | 9.3 % |
| EHAM | 101.7 | 98.5 | 2.6 % |
| LEBL | 149.0 | 149.0 | 4.0 % |
| LEMD | 138.3 | 138.3 | 4.1 % |
| LFPG | 298.9 | 211.9 | 21.6 % |
| LIRF | 436.4 | 363.5 | 30.8 % |
| LSZH | 263.8 | 102.1 | 9.5 % |
| LTFM | 228.2 | 224.2 | 14.3 % |
| All | 218.26 | 183.17 | 100 % |

Source: [validation](validation.md#2025-out-of-fold-rmse-by-airport). The
E203 replacement improved the same comparison by −0.614 s (body) and −0.546 s
(decided), 6 of 6 pairs.

**Public leaderboard (January + July 2026, 344,841 rows).** Best file v12075:
230.9387 s, processed 7 October 2026; 16th of 238 teams on 8 October 2026 at
13:47 UTC (public leaderboard).
The leaderboard was used to choose between complete pre-built variants during
development, so this score carries selection effects; the eight such decisions are
listed in the [leaderboard use disclosure](development-history.md#6-leaderboard-use-disclosure).
The uniform final recipe (no E81, fresh tail ensemble) scored 232.1933 s on the
same rows as a canary ([final submission §4](final-submission.md#4-measured-cost-on-januaryjuly)).

**Generalisation across months (2025 out-of-fold).** February + June: 247.3 s all
rows, 196.3 s on labels ≤ 7,200 s; January + July: 279.2 s and 211.2 s. The gain
over the NM-proxy baseline is 181–227 s in every month, and in no airport-month
(0 of 120) is the model worse than the proxy ([generalisation](generalisation.md)).
Body RMSE by month lies between 159 and 206 s in eleven months and 236 s in July.

**Transfer from CV to the board.** Across 38 single-change pairs the ratio of
leaderboard change to decided-row CV change was about 1.0 for ADS-B stages, 0.7
for blend members, 0.2–0.3 for new base features and 0 or negative for rules and
calibrations fitted on 2025; the sign matched in only 60–70 % of pairs
([negative results, lesson 1](negative-results.md#methodological-lessons)).

**Uncertainty.** The mixture gives a predictive distribution per flight. Central
50/80/90 % intervals cover 51.0/80.1/89.7 % of 2025 out-of-fold departures (chain
variant); by month group Feb + Jun 51.0/80.3/89.9 %, Jan + Jul 47.4/76.6/87.1 %;
CRPS 76.5 s against point MAE 101.8 s ([uncertainty](uncertainty.md)).

**Blind final (January, February, June, July 2026; 670,790 rows).**
Result: `[[to be filled after the final assembly]]` ([final submission §10](final-submission.md#10-results)).

## Limitations and failure modes

- **Departures without an NM record.** 1.07 % of 2025 rows hold 36.6 % of the
  out-of-fold squared error; their RMSE is 500–2,800 s against 154–212 s for NM
  rows, and the predictive distribution fails on them (coverage 30.3/56.6/70.7 %,
  CRPS 224 s) ([data insights §5](data-insights.md), [uncertainty](uncertainty.md)).
  The share is 1.53 % in the January/July ranking set and reached 4.96 % at EHAM in
  January 2026.
- **The Rome tail.** LIRF departures without NM and with take-off minus schedule
  ≥ 4 h have labels of minutes or hours (schedule copies, one-day shifts); a single
  row can be worth more than a model improvement (the worst 2025 row was worth
  7.47 s on the leaderboard scale; the ten worst rows held 24.3 % of the squared
  error) ([negative results, lesson 5](negative-results.md#methodological-lessons)).
  The TabPFN-2 ensemble is bitwise repeatable on the same sealed input and
  machine, but TabPFN inference is not bit-identical across runs and machines in
  general ([SOURCES.md](../SOURCES.md)).
- **ADS-B coverage collapse.** The surface stages were trained on 2025 reception;
  in 2026 LIRF has an own track for 78.5 % of January departures and 6.2 % in July,
  0.7 % in June; EDDF February 32.2 % against 96.3 % in 2025; LTFM has none in
  either year ([final submission §5](final-submission.md#5-ads-b-coverage-in-the-final-months)).
  Where the track is absent the stages are inert and the base model decides; the
  neural member sees empty trajectory tokens in combinations absent from training
  (LIRF in summer without tokens). An audit on 2025 out-of-fold data found the
  member without tokens still better than no member in every fold, so it is kept;
  a scenario estimate (not a measurement) puts the cost of the lost coverage at about +0.18 s on the final
  ([coverage audit](../analysis/coverage_collapse_audit_20261008.md)). Lower coverage
  goes with higher error (correlation −0.45 over the twelve 2025 months, descriptive).
- **Extreme labels dominate RMSE.** In January and July 2025, 45 of 344,419 labels
  exceed four hours and hold 52.4 % of the label's squared deviation from its mean
  ([data insights](data-insights.md#data-caveats)). Small changes in how such rows
  are hedged move the score by more than most design changes.
- **Selection effects.** 109 leaderboard uploads informed development; blend
  weights were never fitted to the board, but eight design choices were kept or
  dropped on a board result ([disclosure](development-history.md#6-leaderboard-use-disclosure)).
  Out-of-fold errors are optimistic for the same reason.
- **Learned 2025 capacity does not transfer reliably.** Features and rules that
  encode 2025 aggregates (per-airport models, regime rates by group, PRU
  calibration, flight-number habits) improved CV and worsened the board
  ([negative results, lessons 1–2](negative-results.md#methodological-lessons)).
- **Not a physical model.** A trajectory observes movement, the target is a
  recorded interval; neither is declared correct when they disagree
  ([validation](validation.md#remaining-scientific-limits)). The chain has no proven
  irreducible-error floor.
- **Calibration of intervals is airport-level only.** EDDM and LEBL intervals are
  slightly too wide, LIRF too narrow in June–July ([uncertainty](uncertainty.md#limitations)).

## Ethical and data considerations

- **Data subjects.** Inputs describe aircraft movements (flight numbers,
  registrations through ADS-B `icao24`, operators, stands, times). No passenger or
  crew data are used. Outputs are per-departure reconstructions of a recorded time.
- **Licences and permissions.** Each external source is listed with its licence
  evidence and the organiser's dated permission in [SOURCES.md](../SOURCES.md) and
  [organiser clarifications](organiser-clarifications.md). Challenge data, OPDI
  extracts and tables that embed challenge labels are not redistributed; the
  reproduction archive is available to the organisers on request
  ([reproduction](reproduction.md#what-you-need)). If the two EUROCONTROL-portal
  inputs were set aside, the affected parts are the destination ATFM delay features
  (board effect −0.17 s) and the half-weight LSZH correction (−0.25 s over two
  steps) ([README](../README.md#data-openness-at-a-glance)).
- **Use of the leaderboard.** No submission was tuned row by row and no weight was
  optimised on the board; leaderboard-optimal weights (−1.004 s for one blend) were
  rejected ([development history §5](development-history.md#5-a-note-on-the-late-phase)).
  For the blind final the two leaderboard-dependent parts were removed at a
  measured cost ([final submission](final-submission.md)).
- **Misuse.** The regime detector flags records whose block time equals a planning
  time; it is a statistical signal (calibrated probability), not proof about any
  individual record, airport or operator ([data quality for PRU §6](data-quality-regimes.md)).
- **AI assistance.** Code, logs and documents were drafted with AI coding
  assistants; every modelling decision, gate and upload is recorded with evidence
  in `HYPOTHESES_LOG.md` ([README](../README.md#team)).

## Maintenance

- **Reproduction.** The best leaderboard file was replayed byte-for-byte from
  archived components on 7 October 2026 (101 s, no refits); a clean end-to-end
  retrain has not been completed and several step times are not measured
  ([reproduction](reproduction.md)). The final file is produced by the dated scripts
  in [final submission §6](final-submission.md#6-pipeline-scripts-inputs-and-receipts),
  which save every fitted model and write receipts.
- **Environment.** CPU stack locked in `uv.lock` (Python ≥ 3.12, Polars, LightGBM,
  CatBoost, TabPFN); GPU stack for the neural member pinned in `requirements-gpu.txt`
  (PyTorch 2.13.0 + ROCm, `rtdl_num_embeddings` 0.0.12). Cached stage models record
  the full interpreter and package versions and refuse others.
- **Freeze.** The pipeline code is frozen by tag before the final upload and the
  repository deadline (14 October 2026, 10:00 UTC); later commits carry logs and
  manifests only (`analysis/win_program_20261008.md` §2).
- **Known gaps to close before reuse elsewhere.** Nested validation of the full
  chain; regime labelling rules for other recording processes; ADS-B stage
  behaviour under coverage changes; an airport × month residual table for the
  intervals.
- **Contact.** Denys Kolomiiets, [@disclaimer8](https://github.com/disclaimer8), through the repository.

# Model card: gentle-igloo taxi-out reconstruction

A model card in the usual sections (Mitchell et al., 2019). Every number is
copied from the page it cites; nothing here is new measurement. Written 8 October
2026 for the v12075 chain and the uniform final recipe derived from it
([final submission](final-submission.md)); updated 9 October 2026 after the final
file was assembled and checked, and 11 October 2026 after the E218 PRU level stage
was added to the best leaderboard file (v12081) and to the final file.

## Model details

| | |
|---|---|
| Name | gentle-igloo taxi-out chain, version v12081 (leaderboard: v12077, the v12075 chain with the neural member as a ten-seed mean at weight 0.35, plus the E218 PRU level stage) / `gentle-igloo_final.parquet` recipe (blind final, with the same E218 stage) |
| Developer | Team gentle-igloo: one developer, Denys Kolomiiets, with AI coding assistants ([README](../README.md#team)) |
| Date | Developed 13 September – 11 October 2026; v12075 scored 7 October 2026, v12077 8 October 2026, v12081 11 October 2026; final file assembled 9 October 2026 and rebuilt with the E218 stage 11 October 2026 |
| Task | Predict `TAXITIME_SEC_mvt`, the recorded taxi-out time of a departure at ten European airports (EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM), for the EUROCONTROL PRC Data Challenge 2026; metric RMSE in seconds |
| Type | Probability mixture over six recording regimes (LightGBM multiclass classifier) whose "regular" candidate is a blend of gradient-boosted residual models (CatBoost 15 seeds, LightGBM 5 seeds), five anchored LightGBM members and an ordered-trajectory neural network (mean of ten seeds at weight 0.35 in v12077 and the final; three seeds at 0.25 in v12075); followed by LightGBM surface-correction stages on the aircraft's own ADS-B track, a TabPFN-2 classifier for a sparse Rome tail, a confidence gate, LOBT-window clipping, an OPDI correction at Zurich ([model design](model-design.md)) and, in v12081 and the final, a post-mixture level shift per airport × month towards EUROCONTROL's published additional taxi-out time (E218, [final submission §2.5](final-submission.md#25-e218-pru-level-stage)) |
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
| Published monthly additional taxi-out time per airport, 2025 and the scored months of 2026 | EUROCONTROL (`taxi_out_additional_time_{2025,2026}.csv`) | input of the E218 stage only (v12081, final file), one value per airport × month |

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
  LSZH (accepted on 4 October 2026), and for the E218 stage EUROCONTROL's monthly
  additional taxi-out time files for 2025 and 2026 (no open licence stated; use
  answered "yes" by the organiser on 27 September 2026 and confirmed for the 2026
  file on 8 October 2026) ([SOURCES.md](../SOURCES.md)).
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

**Public leaderboard (January + July 2026, 344,841 rows).** Best file v12081:
230.4478 s, processed 11 October 2026; 18th of 242 teams on 11 October 2026 at
08:52 UTC (public leaderboard). It is v12077 (230.4812 s, 8 October; 16th of 239
at the time) with the E218 PRU level stage added, −0.0334 s; the stage's rule was
fixed and passed a gate on 2025 out-of-fold data before the upload, and its input
is EUROCONTROL's published monthly additional taxi-out time for the scored months
([final submission §2.5](final-submission.md#25-e218-pru-level-stage)). v12077 is
the v12075 chain (230.9387 s, 7 October)
with the neural member averaged over ten seeds (v12076, −0.1204 s) and its weight
raised from 0.25 to 0.35 with the anchored members scaled by 0.65/0.75 (v12077,
−0.3371 s); the weight was chosen on 2025 out-of-fold data before the upload,
where 0.35 is lower than 0.25 in all twelve months
([final submission §2.3](final-submission.md#23-the-ten-seed-neural-member)).
The leaderboard was used to choose between complete pre-built variants during
development, so this score carries selection effects; the eight such decisions are
listed in the [leaderboard use disclosure](development-history.md#6-leaderboard-use-disclosure).
The uniform final recipe (no E81, fresh tail ensemble) scored 232.1933 s on the
same rows as a canary run with the v12075 members and weights. On the v12077 chain
its two removals were then measured one at a time: without E81 231.0488 s
(v12078, +0.5676 s) and with the fresh tail ensemble instead of the 42 frozen rows
231.1770 s (v12079, +0.6958 s); together +1.2634 s against the canary's +1.2546 s
([final submission §4](final-submission.md#4-measured-cost-on-januaryjuly)).

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

**Blind final (January, February, June, July 2026; 670,790 rows).** The file
`gentle-igloo_final.parquet` was first assembled on 9 October 2026 (SHA-256
`76c32426…d554`) with no fits, passed every hard check of the validator and was
rebuilt from scratch with the same SHA-256. Its January/July rows, uploaded as the
control v12080, scored 231.1805 s, 0.02 s below the corridor written beforehand
([231.2; 232.3]) and 0.564 s better than the measured expectation of 231.745 s;
the deviation was explained without labels, mainly as noise in the regime
probabilities of the refitted classifier on about 350 mixture rows, and the recipe was not
changed ([final submission §7](final-submission.md#7-validation-of-the-file)).
On 11 October the file was rebuilt with the E218 stage (SHA-256 `bbae2cc5…34a1`;
stages up to the last as on 9 October, 415,697 rows moved by RMS 4.687 s, validator
passed, same SHA on a second build). Its January/July rows scored 231.1445 s as the
control v12082 (−0.0360 s against v12080, inside the corridor), and under the rule
written beforehand this file is the final. A third upload, v12083 (231.1749 s),
removed the stage on the LIRF rows only and split the gain exactly: −0.0304 s from
the two LIRF cells, −0.0056 s from the other ten, so the 2025 relation transferred
only partly and mostly at LIRF ([final submission §2.5](final-submission.md#25-e218-pru-level-stage)).
Blind result: `[[to be filled after 14 October]]`
([final submission §10](final-submission.md#10-results)).

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
- **Known infrastructure change in the final period.** Schiphol announced maintenance of
  taxiways A and B from 9 March to 12 April and from 4 May to 26 June 2026, with fewer
  aircraft taxiing at the same time. 18,194 of the 20,881 EHAM departures of June 2026 in the
  final file fall in that window. The model has no input describing it: the announcement
  carries no open licence, so under the organiser's rule it cannot be a model input, and 2025
  holds no comparable episode to learn from ([checks](../analysis/breakthrough_review_checks_20261008.md)).
- **Extreme labels dominate RMSE.** In January and July 2025, 45 of 344,419 labels
  exceed four hours and hold 52.4 % of the label's squared deviation from its mean
  ([data insights](data-insights.md#data-caveats)). Small changes in how such rows
  are hedged move the score by more than most design changes.
- **Score noise of a refitted chain.** Refitting the members and the regime
  classifier on the four-month view moves the January/July score with a noise of
  about ±0.43–0.55 s (1σ, estimated without labels from 2025 out-of-fold data),
  concentrated on a few hundred rows whose prediction mixes regime candidates with
  a median spread of 2,818 s; the control v12080 deviated from its measured
  expectation by −0.564 s, a 1.0–1.3σ event
  ([final submission §7](final-submission.md#7-validation-of-the-file)). One score
  cannot separate refitted variants that differ by less than about 0.5 s.
- **Selection effects.** 117 leaderboard uploads informed development; blend
  weights were never fitted to the board, but eight design choices of v12075 were
  kept or dropped on a board result, and the E218 stage entered v12081 and the final
  under rules that depended on board results ([disclosure](development-history.md#6-leaderboard-use-disclosure)).
  Out-of-fold errors are optimistic for the same reason.
- **A published aggregate of the scored months as an input (E218).** The stage reads
  EUROCONTROL's monthly additional taxi-out time for January, February, June and
  July 2026, computed by PRU from the same kind of record as the labels. Its form
  was chosen after a diagnosis on 2025 and gated on 2025, but on January/July 2026
  only part of the 2025 relation transferred (LIRF about 0.69 of its gap, the other
  airports 0.37–0.48, against 0.93 on 2025). The largest shift of the blind months,
  LIRF June +21 s, has no board measurement. The stage needs the published value of
  the month, so it is a post-operations correction, not available in real time.
- **Learned 2025 capacity does not transfer reliably.** Features and rules that
  encode 2025 aggregates (per-airport models, regime rates by group, the first PRU
  calibration E145, flight-number habits) improved CV and worsened the board
  ([negative results, lessons 1–2](negative-results.md#methodological-lessons)). The
  narrower PRU level stage E218 transferred, but at about half of its 2025 size.
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
  ([reproduction](reproduction.md#what-you-need)). If the EUROCONTROL inputs were
  set aside, the affected parts are the destination ATFM delay features (board
  effect −0.17 s), the half-weight LSZH correction (−0.25 s over two steps) and the
  E218 stage (−0.0334 s on v12077, −0.0360 s on the final's January/July rows)
  ([README](../README.md#data-openness-at-a-glance)). The monthly additional
  taxi-out time files read by E218 carry no open licence; their use rests on the
  organiser's answers of 27 September and 8 October 2026
  ([organiser clarifications](organiser-clarifications.md)).
- **Use of the leaderboard.** No submission was tuned row by row and no weight was
  optimised on the board; leaderboard-optimal weights (−1.004 s for one blend) were
  rejected ([development history §5](development-history.md#5-a-note-on-the-late-phase)).
  For the blind final the two leaderboard-dependent parts were removed at a
  measured cost ([final submission](final-submission.md)). The one stage added
  afterwards, E218, was gated on 2025 data, and its inclusion was then decided by
  board uploads with rules written beforehand (v12081, v12082, v12083); its input is
  a published aggregate of the scored months, which we state plainly rather than
  treat as ordinary external data.
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
  which save every fitted model and write receipts. On 9 October 2026 the final file
  was rebuilt from scratch from the same contract (on the same machine) with a
  byte-identical result, and so was the E218 version on 11 October 2026
  ([final submission §7](final-submission.md#7-validation-of-the-file)). Byte
  identity holds on the build platform: a replay on Linux ARM64 matched the first
  three stages by value, gave 55 of the 62 TabPFN tail values up to 35 s apart
  (different numerical backend) and stopped at the cached E141 stage on the Python
  build string, as designed ([reproduction §2c](reproduction.md#2c-rebuild-the-final-file)).
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

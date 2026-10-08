# Model design: v12075

This document describes v12075, scored at 230.9387 s RMSE on 7 October 2026.
The E203 recipe (production run of the ordered-trajectory neural member) was
registered at `6e95e3a65936a3401e41499bb340f2d54a1a2121`.
Its `branch` and `build` functions in
[`scripts/e203_trajectory_production.py`](../scripts/e203_trajectory_production.py)
replace the neural member of the v12074 control (the previous best, 231.4438 s)
while retaining the fixed weights and complete tail. The historical control is
defined by `build_e190`, `build_joint` and `tails` in
[`scripts/today_submissions_20261006.py`](../scripts/today_submissions_20261006.py).
Both scripts need the archived artifacts listed in [reproduction](reproduction.md).
Experiment IDs (E-numbers) and submission versions (v-numbers) are explained in the
[glossary](glossary.md).

## Target and available information

We predict the recorded departure interval `TAXITIME_SEC_mvt`, indexed by
`MVT_ID_mvt`. Training is the full year 2025; ranking is January and July 2026.
The model uses the take-off timestamp and other retrospective observations.
It does not consume the ranking departure block time or target. Arrivals and
their available block times can contribute movement-context features.

The distinction between an actual physical taxi duration and its recorded label
matters. In 2025, some labels closely coincide with take-off minus scheduled,
initial, latest or estimated off-block timestamps. There are also observations
consistent with a one-day displacement. This motivates an explicit recording
regime model without assuming that every extreme value is an operational delay.

## Base features

The FIN1b feature table combines the E163 table (the base features plus the
destination airport's daily arrival ATFM delay) with taxi-route geometry (E51),
wake-separation features (E157) and planned-arrival residuals (E165). It excludes
the rejected E170 en-route ATFM extension. The construction is recorded in
[`hz_queue/20_fin1b_cv.sh`](../hz_queue/20_fin1b_cv.sh) and uses
[`src/build_final_table.py`](../src/build_final_table.py); most base columns are
built in [`src/features.py`](../src/features.py).

The E163 destination join is by airport and UTC take-off date, including published
data for the ranking year. This is another reason the model is a retrospective
estimator.

### Feature families

The saved regime classifier `submissions/clf_v870.txt` and LightGBM regressor
`submissions/reg_v870.txt` list their input columns. Both use the same 170 FIN1b
columns. The CatBoost member and the five anchored LightGBM members use the same
column filter (`DROP_PREFIX` in
[`hz_queue/40_fin1b_base.sh`](../hz_queue/40_fin1b_base.sh) and
[`hz_queue/hp_e182_rank_20261005.sh`](../hz_queue/hp_e182_rank_20261005.sh)).
The neural member reads 189 tabular columns, listed in its prepared-input manifest
`data/e190_rank_inputs_20261006/manifest.json`, plus ordered ADS-B observations.

| Family | Example columns | Columns: trees / neural | Source | Gain share: classifier / regressor |
|---|---|---:|---|---:|
| Planning-time intervals | `d_mvt_sched`, `d_aobt_eobt`, `aobt_infeasible` | 11 / 11 | Challenge data: movement and NM flight times | 40.4 % / 36.7 % |
| Airport, runway, aircraft and flight identity | `ADEP_mvt`, `ap_runway`, `airline_prefix` | 12 / 15 | Challenge data | 45.5 % / 23.2 % |
| Local time | `local_hour`, `local_dow` | 2 / 2 | Challenge data | 0.2 % / 0.4 % |
| Airport traffic counts | `dep_before_30`, `taxiing_at_ob`, `taxi_in_mean_60` | 28 / 28 | Challenge data: all movements at the airport | 1.2 % / 3.3 % |
| Runway traffic and configuration | `rwy_dep_before_10`, `rwy_since_last_arr`, `n_dep_rwys` | 12 / 12 | Challenge data: movements on the runway | 0.4 % / 1.5 % |
| Neighbour NM taxi proxies | `nb_ap_30_30_mean`, `nb_own_minus_ap`, `nb_stand_last` | 23 / 23 | Challenge data: other departures' take-off minus `AOBT_3` | 3.6 % / 11.7 % |
| Stand turnaround | `stand_prev_arr_age`, `stand_next_arr_gap` | 4 / 4 | Challenge data: arrivals on the same stand | 1.0 % / 3.4 % |
| Forward neighbours (E28) | `fw_ap_15_mean`, `fw_rwy_next_gap` | 14 / 14 | Challenge data: departures in the next 15–60 min | 1.0 % / 2.9 % |
| Filed-time neighbours (E77) | `nbf_rwy_30_30_mean`, `nbs_own_minus_ap` | 27 / 27 | Challenge data: other departures' take-off minus LOBT or schedule | 3.9 % / 8.7 % |
| Planned-arrival residuals (E165) | `plan_taxi_res`, `plan_nm_taxi` | 2 / 2 | Challenge data: NM planned arrival | 0.3 % / 0.7 % |
| Wake separation (E157) | `e157_gap`, `e157_ladder_q`, `e157_heavy_10` | 7 / 7 | Challenge data: runway sequence and wake category | 0.5 % / 1.2 % |
| Weather (METAR) | `wx_temp`, `wx_vis_mi`, `metar_age_min` | 15 / 15 | Iowa Environmental Mesonet METAR archive | 0.6 % / 1.9 % |
| Stand-to-runway geometry | `geo_dist_m`, `geo_bearing_offset` | 5 / 5 | OpenStreetMap | 0.6 % / 1.8 % |
| Taxi-route length (E51) | `route_m`, `route_detour` | 3 / 3 | OpenStreetMap taxiway graph | 0.6 % / 2.2 % |
| Destination arrival ATFM delay (E163) | `dst_dly_per_arr`, `dst_known` | 5 / 5 | EUROCONTROL Aviation Intelligence Portal | 0.2 % / 0.4 % |
| Target encodings | `te_stand_rwy`, `ter_op_ap` | 0 / 16 | 2025 labels, encoded month-out | — |
| Ordered ADS-B observations (E197/E203) | time to take-off, position, altitude, speed, heading, on-ground flag | 0 / 2 streams × 64 tokens × 12 channels | adsb.lol | — |
| **Total tabular columns** | | **170 / 189** | | 100 % / 100 % |

The classifier, the tree regressors and the neural member all drop the airport-day
context (`day_*`), runway head- and crosswind (`wxr_*`), `month`, day of year and
absolute timestamps. The tree models also drop stand identity (`STAND_mvt`,
`ap_stand`), runway configuration (`rwy_config`, E116) and target encodings (E99).
The neural member keeps these three groups. Its target encodings exclude the row's
own calendar month, so a 2025 row never sees its own month's labels. Data sources
and licences are listed in [SOURCES](../SOURCES.md).

Gain share is a family's share of the total split gain in the saved booster. The
same table, with a one-line description per family, is in
[`docs/tables/feature_families.csv`](tables/feature_families.csv).

### Most important features

[`scripts/export_feature_importance.py`](../scripts/export_feature_importance.py)
loads both saved boosters and writes gain and split counts for all 170 features to
[`importance_classifier_v870.csv`](tables/importance_classifier_v870.csv) and
[`importance_regressor_v870.csv`](tables/importance_regressor_v870.csv). Both models
are the ones v12075 loads: the E203 build calls `src/blend.py 870`, which reads
`clf_v870.txt` and `reg_v870.txt`, and their SHA256 hashes match the pinned inputs
of `data/today_chain_20261006/setup.json`.

The classifier is used as saved: 180 boosting rounds × 6 classes = 1,080 trees,
with no seed average. The regressor file is the seed-42 fit of the regressor
recipe (2,319 trees). The chain does not use its own predictions:
[`src/blend.py`](../src/blend.py) replaces them with the five-seed average in
`lgb_reg_seeds_v870.npz` (seeds 1000–1004, same features, parameters and rounds),
whose boosters were not saved. Its importance therefore describes the recipe, not
one of the averaged models.

Top 15 features of the regime classifier by gain (84.8 % of its total gain):

| # | Feature | Gain share | Plain meaning |
|---:|---|---:|---|
| 1 | `ADEP_mvt` | 17.1 % | Departure airport; planned-time regimes are not labelled at EDDF, EHAM and LSZH |
| 2 | `ap_runway` | 13.6 % | Airport and departure runway combined |
| 3 | `d_aobt_sched` | 9.1 % | NM actual off-block (`AOBT_3`) minus scheduled time |
| 4 | `d_eobt_iobt` | 8.7 % | Estimated off-block (`EOBT_1`) minus initial off-block (IOBT): how far the plan moved |
| 5 | `d_mvt_sched` | 8.5 % | Take-off minus scheduled time: the schedule-copy candidate |
| 6 | `d_aobt_eobt` | 6.7 % | `AOBT_3` minus `EOBT_1`: off-block delay against the last estimate |
| 7 | `airline_prefix` | 4.3 % | First three letters of the callsign (airline code) |
| 8 | `AIRCRAFT_OPERATOR_flt` | 3.9 % | Aircraft operator in the NM flight record |
| 9 | `ades_region` | 3.7 % | First two letters of the destination ICAO code (country or region) |
| 10 | `d_mvt_iobt` | 3.1 % | Take-off minus IOBT: the IOBT-copy candidate |
| 11 | `d_mvt_lobt` | 1.7 % | Take-off minus LOBT: the LOBT-copy candidate |
| 12 | `d_mvt_eobt` | 1.2 % | Take-off minus `EOBT_1`: the EOBT-copy candidate |
| 13 | `nb_ap_last_age` | 1.1 % | Seconds since the previous departure at the airport with a usable NM taxi proxy |
| 14 | `RUNWAY_mvt` | 1.1 % | Departure runway |
| 15 | `AIRCRAFT_TYPE_mvt` | 1.0 % | Aircraft type |

Top 15 features of the LightGBM regressor by gain (66.6 % of its total gain):

| # | Feature | Gain share | Plain meaning |
|---:|---|---:|---|
| 1 | `d_aobt_eobt` | 18.1 % | `AOBT_3` minus `EOBT_1`: off-block delay against the last estimate |
| 2 | `d_mvt_aobt` | 9.3 % | Take-off minus `AOBT_3`: the NM taxi proxy whose residual the regressor learns |
| 3 | `ap_runway` | 6.1 % | Airport and departure runway combined |
| 4 | `nb_own_minus_ap` | 4.2 % | This flight's NM taxi proxy minus the mean of other departures at the airport within 30 min of its take-off |
| 5 | `airline_prefix` | 3.6 % | First three letters of the callsign (airline code) |
| 6 | `AIRCRAFT_OPERATOR_flt` | 3.6 % | Aircraft operator in the NM flight record |
| 7 | `nb_own_minus_rwy` | 3.4 % | The same comparison with departures from the same runway |
| 8 | `d_aobt_sched` | 3.1 % | `AOBT_3` minus scheduled time |
| 9 | `RUNWAY_mvt` | 2.9 % | Departure runway |
| 10 | `ades_region` | 2.8 % | First two letters of the destination ICAO code |
| 11 | `stand_prev_arr_age` | 2.4 % | Time from the previous arrival's in-block on the same stand to this flight's off-block estimate |
| 12 | `aobt_infeasible` | 2.0 % | Flag: `AOBT_3` is more than 3,606 s from LOBT, so it cannot be the off-block |
| 13 | `AIRCRAFT_TYPE_mvt` | 1.8 % | Aircraft type |
| 14 | `d_mvt_eobt` | 1.7 % | Take-off minus `EOBT_1` |
| 15 | `nbf_rwy_30_30_mean` | 1.7 % | Mean take-off minus LOBT of other departures on the same runway within 30 min |

The classifier relies on context and on the planning-time intervals that define
its candidate durations: 14 of its top 15 features belong to these two families.
The regressor relies more on the NM delay against plan, the NM taxi proxy and the
comparison with neighbouring departures. Gain measures training-loss reduction
inside the boosters. High-cardinality categoricals such as `ap_runway`,
`airline_prefix` and `ades_region` offer many split points, and correlated columns
share credit. Gain is therefore not the RMSE change from removing a feature; the
feature-removal experiments are in the [experiment register](../HYPOTHESES_LOG.md).

## Recording-regime mixture

[`src/train.py`](../src/train.py) defines six classes: regular, schedule, IOBT,
LOBT, EOBT and day shift. Training labels are assigned by explicit comparisons
with the target. The ordinary default tolerance is 6 seconds; EDDF, EHAM and LSZH
are excluded from planned-regime labelling under the registered base settings.
These choices are dataset-specific heuristics and must be reassessed for another
source of movement records.

Let `r(x)` be the regular-duration prediction. The candidate vector contains
`r(x)`, the four take-off-minus-planning-time intervals, and `r(x) + 86400`.
A multiclass LightGBM estimates regime probabilities. Missing/infeasible
candidates are masked and the remaining probabilities renormalised. Before later
corrections, the prediction is:

```text
y_hat(x) = sum_k p_k(x) * candidate_k(x)
```

The probability-weighted mean is motivated by squared-error loss. Its optimality
requires the probabilities and candidate conditional means to be correct; we do
not claim that the fitted mixture is perfectly calibrated. The registered
classifier uses cost weighting when feasible candidate durations span more than
3,600 seconds: weights rise with squared spread and are capped at 400. This
emphasises expensive regime errors while leaving the smaller-spread rows at
weight one. It remains an approximation to minimising the final mixture's RMSE.

The ordinary LightGBM/CatBoost models learn a residual to a calibrated NM proxy:
first take-off minus AOBT_3, then LOBT, then EOBT_1 when eligible. The submitted
`PROXY_SCALE=1` setting (E46) learns a least-squares scale for each airport/source,
shrunk towards a global coefficient; an airport median supplies the no-proxy
fallback. Other tree members use the raw target or a particular time anchor to
provide different residual errors.

The submitted regular candidate has these fixed weights:

| Component | Weight within the regular candidate |
|---|---:|
| Residual-model blend `0.65 * CatBoost15 + 0.35 * LightGBM5` | 19/60 |
| Raw-target LightGBM member | 1/30 |
| LOBT-anchored LightGBM member | 0.10 |
| Schedule-anchored LightGBM member | 0.10 |
| EOBT-anchored LightGBM member | 0.10 |
| AOBT-anchored LightGBM member | 0.10 |
| E197/E203 ordered-trajectory neural member, mean of three seeds | 0.25 |

The weights sum to one. They apply to the regular candidate, not directly to the
final prediction after regime routing and corrections. The five additional tree
members each use their saved five-seed ranking artifact.

The E190 backbone (the earlier neural member) uses the standard piecewise-linear
embeddings implemented by
[`rtdl_num_embeddings`](https://github.com/yandex-research/rtdl-num-embeddings),
with 48-bin training quantiles, 12 embedding dimensions and a width-512 network.
The package is version 0.0.12, with its `version='B'`, `activation=False` preset.
The full-training recipe fixes
eight epochs, AdamW, learning rate 0.002, weight decay 0.00001, OneCycleLR and
seeds 0, 1 and 2. Numerical knots are learned from the fit data. See
[`scripts/e190_rank_20261006.py`](../scripts/e190_rank_20261006.py) and
[`scripts/e190_numeric_embeddings.py`](../scripts/e190_numeric_embeddings.py).
E197 extends that backbone with ordered observations from the current and previous
aircraft visit: up to 64 actual observation tokens per stream, twelve channels
with missingness masks, and message blocks using previous, current and next tokens.
This adds trajectory information to the ordinary-duration member before the
existing surface stages. It is not a graph of interacting flights.
See [`scripts/e197_trajectory_encoder.py`](../scripts/e197_trajectory_encoder.py).

The E203 ranking receipt records three seeds, eight epochs and 1,908,437 parameters
per seed on the AMD Radeon RX 7900 XTX. Each fit took approximately 269 seconds;
these are component timings, not end-to-end training times.
Evidence: `data/e203_rank_20261007/complete.json`. The fixed 0.25 slot replaces E190;
it does not add a second neural weight or tune weights against the ranking score.

For rows where LOBT is missing outside LIRF, E185 further replaces 7/30 of the
regular candidate with a movement-only predictor, then recomputes the mixture.
E167 applies the existing NM matching-window rules. These are restricted
transformations; the stored mixture parts retain their ID alignment.

## Surface observations

Ground observations can constrain when an aircraft first moved and how much of
its taxi has already occurred. The ADS-B path builds movement onset, stops,
stand distances, previous visits, reception-gap repairs and initial track-motion
summaries. E141 (the own-track surface stage) uses this information for a
correction to the base; E64 adds the correction from the same aircraft's previous
visit to the airport. The recipe averages seeds 42, 1, 2, 3 and 4.

Coverage is a modelling variable and a limitation: observations may start after
pushback, disappear within a taxi or be absent. A first received message is not
automatically an off-block event. Seven reception-density features are dropped
from E141 and three from E64 in the registered recipe because their historical
behaviour did not transfer reliably. Removing those features does not remove all
possible sensitivity to changes in the receiver network.

E188, a previous-visit stage for LIRF rows with an NM record retrained without the
ground-coverage mask, replaces the E64-stage result on LIRF rows with an NM record,
no own surface track and a previous visit, using separately registered models. It
uses each branch's own pre-E64 prediction, including the ordered-trajectory branch.
Its incremental evidence is small and mixed; see [validation](validation.md).

E152 restores the pre-surface prediction when the maximum probability of the
four planning-time classes is at least 0.95. This is a gate based on model
confidence, not proof that a row's true regime is known.

## Sparse tail and final transformations

LIRF without an NM match receives special treatment because few examples support
very different possible recorded durations. The historical approach combines an
empirical regime table with TabPFN predictions. In v12075, **42 predictions are
copied from the fixed v361 donor**. Their IDs are the rows where v361 differs from
v291, two earlier submissions that differ in exactly these 42 TabPFN rows. The
final protection stage restores them after intervening corrections.
This preserves the submitted output across builds, but makes the donor artifacts
explicit dependencies. It is not a rule for choosing arbitrary new-period IDs.

After E152, the E81 stage blends the whole file with two older complete
submissions. Both are full-chain files scored on all 344,841 ranking rows
([leaderboard table](tables/leaderboard_gentle-igloo.csv)):

- **v44** (256.3401 s, processed 16 September 2026) is the per-airport variant of
  the chain at that time: a separate regime classifier and regressor for each
  airport (E38), the LIRF TabPFN tail, E14 (flight-number history for LIRF
  without NM) and a per-airport surface stage with stand geometry (E42).
- **v60** (252.1044 s, processed 17 September 2026) is the global chain with ADS-B
  features inside the base model (E32) and the per-airport/source proxy scale
  (E46), followed by the TabPFN tail, E14 and the E36 stand-geometry surface stage.

Each is worse than the current chain on its own. A small weight still helps
because their errors differ from the main chain's. On 20 September their
cross-validation residuals correlated with those of the then-accepted chain at
0.939 (v44) and 0.964 (v60) ([`src/blend_submissions.py`](../src/blend_submissions.py)).
Under squared error, moving a small weight to a partner lowers the error when
ρ·σ_partner < σ_main, where ρ is the residual correlation and σ the RMSE. The
weights were fitted by non-negative least squares on 2025 out-of-fold predictions,
against the staged chain of 20 September, and have been fixed since; they were not
tuned on the leaderboard:

```text
p_E81 = 0.8516 * p_current + 0.1214 * p_v44 + 0.0270 * p_v60
```

Complete out-of-fold counterparts of v44 and v60, through their own tail and
surface stages, were not kept. Both donors therefore enter the chain only as
archived ranking prediction files (`gentle-igloo_v44.parquet`, `v60.parquet`).
This is the main obstacle to a fully nested validation: the 2025 comparisons in
[validation](validation.md) stop before E81. A rebuild of the donor out-of-fold
predictions from their historical recipes (E201) was registered on 7 October and
stopped the same day in favour of documentation and reproduction work; it has no
result.

LOBT edge adjustment and feasibility clipping follow. Exact reproduction must
retain the stage order and integer rounding, as well as the underlying floating
point predictions.

The last stage uses OPDI exit-parking/taxiway events **at LSZH only**, with no
cross-airport time shift. It fits a small LightGBM residual model on the 2025 LSZH
data matched to historical staged out-of-fold predictions. Inputs are the OPDI
duration, its difference from the existing prediction, the prediction itself and
the exit-to-taxiway interval. The output is a 50% blend of the corrected and
uncorrected file. The production sources are
[`src/e159b_apply.py`](../src/e159b_apply.py) and
[`src/e159_fixed_lineage.json`](../src/e159_fixed_lineage.json).

The broader OPDI ingestion/research pipeline is separate. Downloading additional
OPDI tables does not imply that they entered v12075. The frozen LSZH input
inventory, including its source gaps, defines this submitted component.

## What this design does not establish

The chain predicts the supplied recording process as well as physical taxi
duration. It has not been validated for online operational use, unseen airports
or a changed NM schema. Historical stages and donor predictions complicate clean
cross-validation and retraining. A large residual associated with missing data
does not prove that the error is irreducible.

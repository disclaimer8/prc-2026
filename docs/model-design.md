# Model design: v12075

This document describes v12075, scored at 230.9387 s RMSE on 7 October 2026.
The E203 recipe was registered at `6e95e3a65936a3401e41499bb340f2d54a1a2121`.
Its `branch` and `build` functions in
[`scripts/e203_trajectory_production.py`](../scripts/e203_trajectory_production.py)
replace the neural member of the v12074 control while retaining the fixed weights
and complete tail. The historical control is defined by `build_e190`,
`build_joint` and `tails` in
[`scripts/today_submissions_20261006.py`](../scripts/today_submissions_20261006.py).
Neither file should be invoked casually: they require pinned local artifacts and
some stages fit models.

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

The FIN1b table combines the E163 table with route geometry (E51), wake-separation
features (E157) and planning residual features (E165). It excludes the rejected
E170 en-route ATFM extension. The construction is recorded in
[`hz_queue/20_fin1b_cv.sh`](../hz_queue/20_fin1b_cv.sh) and uses
[`src/build_final_table.py`](../src/build_final_table.py).

The feature families include differences between take-off and NM/planning times;
airport, runway, aircraft and flight context; surrounding arrivals/departures;
METAR observations; OSM stand/runway geometry; and the destination airport's daily
arrival ATFM delay. The E163 destination join is by airport and UTC take-off date,
including published data for the ranking year. This is another reason the model
is a retrospective estimator.

Feature availability in the table does not imply use by every learner. The
production tree configuration drops several month, target-encoding, stand-identity
and runway-configuration prefixes. The neural path has its own preprocessing and
month-out encodings. Exact filters and category mappings belong to each saved
component's source/runtime manifest, not to a shared informal feature list.

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
`PROXY_SCALE=1` setting learns a least-squares scale for each airport/source,
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

The E190 backbone uses the standard piecewise-linear embeddings implemented by
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
summaries. E141 uses this information for a correction to the base; E64 adds the
visit-based correction. The recipe averages seeds 42, 1, 2, 3 and 4.

Coverage is a modelling variable and a limitation: observations may start after
pushback, disappear within a taxi or be absent. A first received message is not
automatically an off-block event. Seven reception-density features are dropped
from E141 and three from E64 in the registered recipe because their historical
behaviour did not transfer reliably. Removing those features does not remove all
possible sensitivity to changes in the receiver network.

E188 replaces the E64-stage result for its natural LIRF/NM population, using
separately registered models. It uses each branch's own pre-E64 prediction,
including the ordered-trajectory branch. Its incremental evidence is small and
mixed; see [validation](validation.md).

E152 restores the pre-surface prediction when the maximum probability of the
four planning-time classes is at least 0.95. This is a gate based on model
confidence, not proof that a row's true regime is known.

## Sparse tail and final transformations

LIRF without an NM match receives special treatment because few examples support
very different possible recorded durations. The historical approach combines an
empirical regime table with TabPFN predictions. In v12075, **42 predictions are
copied from the fixed v361 donor**; their IDs are the difference between v361 and
v291. The final protection stage restores them after intervening corrections.
This preserves the submitted output across builds, but makes the donor artifacts
explicit dependencies. It is not a rule for choosing arbitrary new-period IDs.

After E152, the E81 stage forms a whole-file mixture with historical predictions:

```text
p_E81 = 0.8516 * p_current + 0.1214 * p_v44 + 0.0270 * p_v60
```

LOBT edge adjustment and feasibility clipping follow. Exact reproduction must
retain the stage order and integer rounding, as well as the underlying floating
point predictions. The donor models' exact historical out-of-fold counterparts
are not available for the complete validation chain.

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
does not prove that the error is irreducible. These limitations are part of the
model description, not assumptions hidden behind the final RMSE.

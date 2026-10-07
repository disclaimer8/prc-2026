# PRC Data Challenge 2026 — gentle-igloo

A model for the recorded taxi-out time of departures from ten European airports,
developed for the [EUROCONTROL PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/).
The best verified submission documented here is **v12075: 230.9387 seconds RMSE**
on **344,841 departures in January and July 2026**, processed on 7 October 2026.
The [result record](analysis/e203_upload_20261007/RESULTS.md) links the stored scoring response.
This is a result on the challenge ranking set, not an estimate for an unseen year.

The code is licensed under [GNU GPLv3](LICENSE). Data and pretrained models retain
their own terms; the [source register](SOURCES.md) distinguishes published licences
from permissions recorded for this challenge.

## What the model estimates

The target is `TAXITIME_SEC_mvt`, the recorded interval between departure block time
and take-off. Training uses 2025 departures at EDDF, EDDM, EGLL, EHAM, LEBL, LEMD,
LFPG, LIRF, LTFM and LSZH. The FIN1b feature table contains 2,085,047 training
departures. Departure block time and the target are absent from ranking inputs.

This is a **retrospective reconstruction task**. The model uses the known take-off
time, planning and Network Manager timestamps, surrounding movements, and, where
available, the flight's own surface trajectory. It is not validated as a forecast
made before pushback. That information boundary is central to interpreting its result.

## Why a mixture, surface observations and a separate tail

Recorded taxi-out is not described well by one ordinary-duration regressor. Some
training labels closely match take-off minus a planning timestamp; a small number
are consistent with a one-day displacement. The model treats these as competing
recording regimes, with probabilities rather than a hard correction of the label.
These are inferred statistical regimes, not independently verified causes of every record.

The regular-duration candidate combines LightGBM, CatBoost and a neural network
with numerical embeddings. ADS-B corrections then use observed motion and stand
geometry to improve the estimate where surface observations exist. A gate preserves
the earlier prediction when the classifier strongly favours a planning-time regime:
physical movement need not explain a recorded planning-time label.

LIRF departures without a Network Manager match have a sparse, high-impact tail.
The submitted recipe retains an empirical/TabPFN tail treatment and freezes 42
historical TabPFN predictions. Their exact values are part of the reproduction
requirements; rerunning a stochastic model is not evidence of exact reproduction.

The complete v12075 order is:

1. **FIN1b features and regime mixture:** LightGBM classifier; regular candidate
   from CatBoost/LightGBM, five raw or timestamp-anchored tree members, and the
   E197/E203 ordered-trajectory neural member at weight 0.25, replacing E190.
2. **Restricted adjustments:** movement-only member for rows with missing LOBT outside
   LIRF (E185), NM matching-window rules (E167), and the 42 frozen tail predictions.
3. **Surface corrections:** E141 and E64, with E188 replacing the applicable
   LIRF/NM part of E64; all use the registered coverage and feature rules.
4. **Final controls:** planning-regime gate E152, historical prediction blend E81,
   LOBT edge adjustment/clipping, restoration of the 42 tail predictions, then
   a 50% blend of the LSZH-only OPDI correction.

The [model-design document](docs/model-design.md) gives the equations, weights,
stage dependencies and reasons for these choices. Experiment names such as E190
are stable identifiers in [HYPOTHESES_LOG.md](HYPOTHESES_LOG.md).

## Validation and limitations

Our main cross-validation scheme holds out six pairs of 2025 months:
January/July, February/August, through June/December. Selected base components use
an additional inner month split for round selection. We inspect paired squared-error
changes, month consistency, the ordinary-duration population and the extreme tail.
An artifact audit checks IDs, targets, months, finite predictions and provenance.

**There is no complete, fully nested cross-validation result for the exact v12075
chain.** E203 improves the E190+E188 control by 0.545761 s on the defined
`decided` subset and 0.614049 s on the `body` subset, with all six pairs improving.
This comparison reaches E152, before E81 and the remaining tail, and inherits
historical upstream out-of-fold predictions. It is selection-informed validation,
not a full-chain or all-row RMSE estimate. The separate ranking improvement over
v12074 is 0.5051 s.
See [validation definitions, results and negative experiments](docs/validation.md).

The small leaderboard difference between v12070 and v12074 is **0.0011 s**.
It does not establish a transferable benefit for E188. More broadly, repeated
development against the same ranking period creates selection risk. Surface
reception varies by airport and year, and a few extreme labels can dominate RMSE.
We do not claim that a diagnostic residual decomposition proves an irreducible
error floor.

## Reproduction status

The submitted artifact is `submissions/gentle-igloo_v12075.parquet`:

```text
SHA256 90d3cd0b5a68cc8008f2142f1801857752574b525b2ff046ad6bf4089a70aee2
344841 unique MVT_ID_mvt values; output TAXITIME_SEC_mvt in seconds
recipe registration 6e95e3a65936a3401e41499bb340f2d54a1a2121
submission registration 3bcb89d7c62fa86b233d24755006b14965a7a522
```

Large data, fitted models and submission files are excluded from Git. Paths in
the following table describe the development artifact layout; they are not a
claim that a fresh clone contains those artifacts.

| Reproduction level | Current evidence | Remaining requirement |
|---|---|---|
| Verify the submitted file | Local SHA, ID count and receipt agree; scoring response records 230.9387 s | Obtain the artifact and receipt through an appropriately licensed release |
| Reassemble v12075 from saved inputs | Independent replay on 7 October: all 344,841 predictions and Parquet bytes exact; zero new fits | Package the required inputs, frozen sources, Git objects and runtime records for another checkout |
| Retrain the full recipe from raw sources | Component trainers, extractors and stage builders are present | A clean end-to-end run of the current recipe, complete dependency specification, data manifests and documented parity criteria |
| Run on another period | `src/run_final_any_period.sh` implements an earlier v808-family recipe | Port and validate the current recipe; historic ranking donors cannot simply be reused on new IDs |

The [current v12075 replay](docs/reproduction.md) completed on 7 October at
22:33 UTC. It reproduced every prediction and the complete Parquet bytes,
including all 42 protected donors, in 101.19 seconds with no new model fits.
Its saved-model and input checks cover 3,711 dependencies. Peak combined RSS
was 8.49 GB after removing unused data copies; the first attempt exceeded its
memory limit and was preserved as a failed run. Raw-data rebuilding and a
distributable artifact package remain separate requirements. Start with the
read-only archive check:

```bash
.venv/bin/python scripts/reproduce_v12075_v2.py check
```

The linked guide distinguishes saved-component replay from fresh model training.
The older `run_final_any_period.sh` reproduces the v808 family, not the current best.
The earlier v12074 replay also passed; its historical runner remains available.

For the existing development environment, use `.venv/bin/python`. Do not recreate
an environment underneath running jobs. `pyproject.toml` and `uv.lock` describe
the main Python stack; the E197/E203 neural component additionally uses the recorded
PyTorch/ROCm and numerical-embedding environment. A lockfile alone does not cover
that separate runtime.

## Data and review guide

The submitted chain reads challenge movements/NM records, adsb.lol trajectories,
OpenStreetMap geometry, IEM METAR weather, EUROCONTROL destination ATFM aggregates,
and OPDI events for the Zurich correction. [SOURCES.md](SOURCES.md) gives each
source's role, primary URL, attribution and licensing evidence.

Start with [model design](docs/model-design.md) and [validation](docs/validation.md).
The [experiment register](HYPOTHESES_LOG.md) preserves tested and rejected ideas;
dated reports under `analysis/` provide the associated evidence. Operational notes
in `HANDOFF.md` and old runner comments contain historical states and should not
override this version-specific recipe description.

The repository is being prepared for public release on **9 October 2026**. The
team's final completion cutoff is **11 October 2026, 21:59 UTC**. In the organiser
messages recorded in the user-verified 7 October Discord extract, model design
and documentation contribute to prize review, the last pre-deadline repository
commit is used, and no final round is planned. These are recorded organiser
clarifications, not a guarantee of review eligibility or a statement of current rank.
The public [eligibility page](https://prc-data-challenge-2026.netlify.app/eligibility.html)
requires open code, documented external data and sufficient reproduction documentation.

Data acknowledgement: EUROCONTROL Performance Review Commission Data Challenge
2026, supported by the OpenSky Network; adsb.lol and its contributors;
© OpenStreetMap contributors; Iowa Environmental Mesonet, Iowa State University;
EUROCONTROL Performance Review Unit and the Open Performance Data Initiative.

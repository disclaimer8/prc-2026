# Glossary

Experiment IDs (E-numbers) and submission versions (v-numbers) are stable
identifiers. IDs up to about E99 (13–22 September) are documented in
[REVIEW_BRIEF.md](../REVIEW_BRIEF.md) §2–§21; later IDs each have a numbered entry
in [HYPOTHESES_LOG.md](../HYPOTHESES_LOG.md). Most also have a dated report under
`analysis/`. This page gives a plain-language
name for the IDs that appear in the English documentation.

## Terms

| Term | Meaning |
|---|---|
| Recording regime | How the recorded off-block time relates to reality: a regular pushback time, or a copy of a planning timestamp (schedule, IOBT, LOBT, EOBT), or a one-day shift |
| NM proxy | Take-off time minus a Network Manager off-block timestamp: `AOBT_3` (actual, minute-rounded), else LOBT, else EOBT; scaled per airport and source |
| Regular candidate | The model's estimate of the taxi-out time if the recorded off-block is a real pushback |
| Surface stage | A model applied after the mixture that corrects the estimate using the flight's own ADS-B ground observations |
| LIRF without NM | Rome departures with no matching Network Manager flight record; the sparse tail with the largest per-row errors |
| Month-pair fold | Cross-validation fold holding out two 2025 months six apart (1+7, 2+8, …, 6+12) and training on the other ten |
| OOF | Out-of-fold prediction: a prediction for a 2025 row made by a model that did not see that row's fold |
| `decided` / `body` subsets | Validation subsets: all rows except a small LIRF tail with no NM match and schedule gap ≥ 4 h / rows with target ≤ 7,200 s (defined in [validation](validation.md#subset-definitions)) |
| FIN1b | The final feature table: base features plus route geometry (E51), wake-separation features (E157), planning residuals (E165) and destination arrival ATFM delay (E163) |
| PLE | Piecewise-linear encoding of numerical features (Gorishniy et al., NeurIPS 2022), used by the neural member |
| Gram identity | RMSE of a blend of two submitted files is predictable from their scores and their RMS difference: `sqrt(w·s1² + (1−w)·s2² − w(1−w)·d²)` |

## Experiment IDs used in the documentation

| ID | Plain-language name | Status in the best submission |
|---|---|---|
| E11 | TabPFN-2 classifier for regime probabilities of LIRF departures without NM | Used (tail) |
| E14 | Flight-number history for LIRF without NM, delay under 4 h | Removed later; the base absorbed it |
| E25 | First own-aircraft ADS-B surface stage (adsb.lol ground tracks) | Superseded by later stages |
| E26a | Local Rome date as a TabPFN tail feature | Used (tail) |
| E36 | Surface stage with stand geometry | Superseded by E70/E141 |
| E46 | Per airport × source scale of the NM proxy, fitted by least squares and shrunk towards a global value (`PROXY_SCALE=1`) | Used |
| E51 | Taxi-route geometry on the OSM taxiway graph | Used (FIN1b) |
| E64 | Previous-visit surface stage: the same aircraft's earlier observations at the airport | Used |
| E70 | Surface-stage family built on a 1-knot movement extraction | Used through E141 |
| E76 / E81 | Whole-file blend with two earlier complete pipelines (v44 per-airport models, v60 earlier global chain), weights 0.1214 / 0.0270 from cross-validation | Used |
| E77 | Neighbour features ordered by filed time | Used (features) |
| E89 | Regime-classifier sample weights: rows whose regime candidates differ by more than a threshold get weight (spread / threshold)², capped at 400; other rows keep weight 1 | Used |
| E99 | Removal of 2025 target encodings from the regressor | Used |
| E100 / E176 / E181a | Neural (MLP) member of the regular candidate on the E77 table / on FIN1b | Superseded by E190, then E203 |
| E106 | Airline prefix parsed from the flight number for rows without NM | Used |
| E116 | Removal of runway-configuration features | Used |
| E118 | Airline prefix as a TabPFN tail feature | Used (tail) |
| E130, E171–E173 | Anchored members: the same LightGBM target relative to raw, LOBT, schedule, EOBT and `AOBT_3` anchors | Used |
| E141 | Kinematic estimate of the unseen start of the taxi (distance at first fix / initial speed) inside the own-track surface stage | Used |
| E147 | Removal of reception-density features that drifted between 2025 and 2026 | Used |
| E150 | Surface stage trained on all rows, including those where the record contradicts the track | Used |
| E152 | Gate: keep the pre-surface prediction when a planning regime has probability ≥ 0.95 | Used |
| E157 | Runway wake-separation ladder and heavy aircraft ahead | Used (FIN1b) |
| E159 / E159b | OPDI exit-parking / taxiway events as a second off-block source, applied at LSZH only | Used at 50% |
| E163 | Destination airport daily arrival ATFM delay (ansperformance.eu aggregates) | Used (FIN1b) |
| E165 | Planned-arrival residual features | Used (FIN1b) |
| E167 | Rules that use the LOBT window to rule out impossible regimes (for example, a schedule-copy label more than an hour away from LOBT) | Used |
| E182 | Five LightGBM anchored members on FIN1b | Used |
| E185 | Movement-only member for rows without LOBT outside LIRF | Used |
| E188 | Previous-visit stage for LIRF rows with NM, retrained without the ground-coverage mask | Used |
| E190 | Neural member with piecewise-linear numerical embeddings | Replaced by E203 in v12075 |
| E195 | Retrospective four-arm comparison of the chain before E81 | Validation only |
| E197 / E202 / E203 | Ordered-trajectory neural member: screen, confirmation, production in the current chain | Used (E203) |

## Submission versions

`vN` refers to `submissions/gentle-igloo_vN.parquet`. Version numbers are not
consecutive in time: blocks of numbers were reserved for build pipelines.
Scores of all 108 submissions are in
[`docs/tables/leaderboard_gentle-igloo.csv`](tables/leaderboard_gentle-igloo.csv).

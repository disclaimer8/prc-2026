# Validation and evidence boundaries

This record accompanies the [v12075 model description](model-design.md). Negative
changes in RMSE mean improvement. Results below identify their population and
pipeline boundary; a component result is not a full-chain result.

## Submitted result

| Submission | Registered change | Ranking RMSE, seconds | Processed, UTC |
|---|---|---:|---|
| v12070 | E190 in the existing complete chain | 231.4449 | 2026-10-07 07:01:39 |
| v12071 | Fixed 50:50 whole-file blend of v12061 and v12070 | 231.6095 | 2026-10-07 07:02:00 |
| v12074 | E190 + E188, complete chain | 231.4438 | 2026-10-07 07:02:34 |
| **v12075** | **E203 ordered-trajectory replacement, complete chain** | **230.9387** | **2026-10-07 21:53:59** |

All four scored 344,841 rows. The original recipes were registered before these
scores. The [delivery/result report](../analysis/submission_results_20261007.md)
links the stored responses. v12074 has SHA256
`e5c78ab061492b69fe72d4a46d2e286f7a2f6b3129ebf2ae4a34731bb662374b`.
A local documentation check on 7 October confirmed this hash, 344,841 unique IDs
and no null predictions against its saved manifest.

v12075 improves on v12074 by **0.5051 s**. Its SHA256 is
`90d3cd0b5a68cc8008f2142f1801857752574b525b2ff046ad6bf4089a70aee2`.
All IDs, finite predictions and the 42 fixed tail rows passed the upload checks.
The [E203 delivery report](../analysis/e203_upload_20261007/RESULTS.md) records the
scoring response, prior forecast and exact registered recipe. The result is a
ranking-set measurement, not an independent future-period estimate.

The 0.0011-second difference between v12074 and v12070 is a property of these
files on this ranking set. It is not evidence that E188 will improve another
period. We do not infer a current rank, prize-review threshold or remaining
submission allowance from a historical scoring response.

## Temporal splits and integrity checks

The main 2025 cross-validation scheme has six outer folds: months 1+7, 2+8, 3+9,
4+10, 5+11 and 6+12. Each fold holds out both months and trains on the other ten.
For components using nested round selection, the next month after each outer
month supplies the inner validation pair; the selected iteration count is then
used for a refit on the outer training months. An earlier January/July-only
holdout and targeted January-to-July transfer tests also appear in historical
experiments; they are not interchangeable with six-pair evidence.

[`src/strict_oof_gate.py`](../src/strict_oof_gate.py) checks exact fold inventory,
unique IDs within/across folds, complete training-ID coverage, targets against
the feature table, held-out months, shape and finite values. The more recent
experiment-specific evaluators also bind source/input hashes, seed receipts,
mean predictions and stage outputs. These checks establish artifact consistency;
they cannot establish that every upstream model was independently cross-fitted.

The historical strict gate reports a scaled SSE change using the fixed denominator
`6 * 2 * 252.43 * 344841`. Its printed “seconds” are a first-order
leaderboard-scale approximation, not an exact RMSE difference on the evaluated
subset. The tables below use actual RMSE differences from their named reports.
These two conventions must not be mixed in a pooled results table.

## Subset definitions

| Name | Definition and purpose |
|---|---|
| Ranking RMSE | All scored January/July 2026 departures; the challenge metric |
| E195 `included` | 2,084,921 of 2,085,047 training IDs; excludes 126 fixed historical TabPFN rows |
| E195 `decided` | Included rows excluding LIRF with no `FLIGHT_ID_mvt` and take-off-minus-schedule ≥14,400 s; 2,084,920 rows |
| E195 `body` | Included rows with observed target ≤7,200 s; 2,084,448 rows |
| Paired SSE change | Sum of candidate squared error minus control squared error on identical IDs |
| Within-fold trimming | Remove the three or twenty largest and smallest paired squared-error differences in each fold, then sum; sensitivity analysis, not the challenge metric |

The `decided` label is an internal name for an exclusion rule. It does not mean
that the true recording regime is known. The body subset uses training labels
and is a diagnostic, not a ranking-time routing condition. Trimming by pooled
rows differs from trimming within each fold; experiment reports preserve both
where calculated. A bootstrap interval on a selected screen is descriptive
uncertainty, not a correction for the earlier model search.

## Evidence for the current members

E195 retrospectively compared four fixed variants on all six outer pairs, through
E152 but **before E81, LOBT tail adjustments, final protection and half-LSZH**.

| Variant | Decided RMSE | Change | Body RMSE | Change | Decided pairs improved |
|---|---:|---:|---:|---:|---:|
| Control | 218.609760 | — | 183.540525 | — | — |
| E190 | 218.270114 | −0.339646 | 183.174325 | −0.366200 | 6/6 |
| E188 | 218.602766 | −0.006994 | 183.535612 | −0.004913 | 3/6 |
| E190 + E188 | 218.262731 | −0.347029 | 183.168965 | −0.371560 | 6/6 |

Source: [E195 report](../analysis/e195_pre_e81_results_20261007.md), with fold
results and registered input/model receipts. The comparison includes the saved
Cat15/LGB5/neural members, fixed weights, E185/E167, fold-specific surface stages
and E152. No feasibility-mask exclusions were newly introduced; the 126
historical TabPFN rows were excluded from every arm.

This is **selection-informed retrospective validation**. Historical upstream
out-of-fold predictions were not fully nested through the complete downstream
chain. The stored classifier probabilities are already feasibility-masked;
the original unmasked mass is unavailable. Exact historical OOF donors for v44
and v60 are unavailable, and the legacy tail cache does not reproduce the 42-row
production treatment. These limitations prevent presenting E195 as an unbiased
full-v12074 estimate. The original v12074 manifest explicitly records
`quality_validated_on_full_chain=false`.

## Negative experiments and later work

| Experiment | Measured result | Decision and scope |
|---|---|---|
| E193: six-hour METAR history | Against the equal-capacity current-weather control, body RMSE changed by +0.01538 / +0.00640 s on pairs 1+7 / 4+10 | Failed the fixed continuation gate after 12 fits. This tests the selected history encoding; it does not imply that weather is useless. [Report](../analysis/e193_weather_results_20261007.md) |
| E199: spatial surface-state summaries | Spatial-minus-counts RMSE changed by −0.022049 / +0.062688 s on the two screen pairs; the registered body/decided/trim requirements did not all pass | Rejected; no four-pair extension or ranking model. This is a test of the fixed summaries and residual learner, not all possible spatial models. [Report](../analysis/e199_results_20261007.md) |
| E198 → E200: density-model replacement | E200 improved body RMSE by 0.200152 s but worsened decided RMSE by 0.054472 s; only 3/6 pairs improved, below the required four | Rejected before ranking fit. The earlier component-level MDN success did not survive the current pre-E81 comparison. [Report](../analysis/e200_mdn_results_20261007.md) |
| E197: ordered trajectories | Before the ADS-B stages, ordered-minus-PLE RMSE was −4.465932 / −7.113443 s; ordered-minus-equal-capacity-flat was only −0.249545 / −0.126464 s | Positive two-pair component screen. The large PLE difference includes changed inputs and capacity; it is not solely an ordering effect or a measured full-chain gain. [Report](../analysis/e197_verified_results_20261007.md) |

E202 completed the four remaining month-pair comparisons (24 fits); both
registered component gates passed. E203 then compared the ordered replacement
against the E190+E188 control through E152, before E81 and the remaining tail:

| Subset | Control RMSE | E203 RMSE | Change |
|---|---:|---:|---:|
| Body | 183.168965 | 182.554916 | −0.614049 |
| Decided | 218.262731 | 217.716970 | −0.545761 |

All six pairs improved and the three-/twenty-row trimming checks passed. The
four confirmation pairs had body change −0.735814 s. This remains
selection-informed validation using historical upstream predictions, with the
same full-chain limitations as E195. The complete ranking candidate used the
fixed three-seed member and unchanged downstream recipe; its control exactly
matched v12074. [Completion evidence](../analysis/e203_reproduction_status_20261007.md).

E197's 4.5/7.1-second component improvements are not the achieved leaderboard
gain. The measured complete-file improvement is 0.5051 s, as recorded above.

The full [experiment register](../HYPOTHESES_LOG.md) also preserves older rejected
features and models. This summary is deliberately selective; it is not a claim
that all conceivable alternatives have been excluded.

## Remaining scientific limits

- **Period and airport transfer:** the observed ranking result covers two 2026
  months at ten airports. Receiver coverage, operating patterns and the recording
  process can drift. A validation win does not determine its ranking magnitude.
- **Selection:** many experiments and repeated leaderboard submissions informed
  development. Fixed protocols for later comparisons reduce local flexibility
  but do not make the whole research process confirmatory.
- **Rare labels:** a few one-day or long-planning-time records can dominate squared
  error. A smooth probability mixture is still uncertain when examples are scarce.
- **Sensor and label disagreement:** a trajectory observes movement, while the
  target is a recorded interval. Neither must be declared correct solely because
  the two disagree.
- **No proven irreducible floor:** proxy residual variance, correlations with block
  time, and mixture uncertainty are useful diagnostics. They are not independent
  error components by assumption; adding them can double-count error and does not
  establish an information-theoretic lower bound.
- **Reproduction versus generalisation:** matching a saved submission verifies an
  implementation path. It does not supply an independent quality estimate or a
  valid recipe for an arbitrary future period.

## Review and rules provenance

The public [eligibility page](https://prc-data-challenge-2026.netlify.app/eligibility.html),
read on 7 October 2026, requires GPLv3 source code, documented/open external data
and sufficient information to understand and reproduce the result.

The user-verified local extract
`analysis/review_fable_2026-10-07/discord_2026-09-26_10-05_extract.md`
records organiser statements dated 1 October (last repository commit before the
deadline), 2 October (model design/documentation contribute to prize review) and
3 October (no final round). We read this supplied extract for documentation; we
did not independently authenticate those Discord messages in this pass. No
number of teams eligible for code review is established by those statements.

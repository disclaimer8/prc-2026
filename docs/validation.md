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

E190 is the neural member with piecewise-linear numerical embeddings; E188
retrains the previous-visit surface stage for LIRF rows with an NM record; E203
replaces E190 with the ordered-trajectory neural member (see the
[glossary](glossary.md)). All four scored 344,841 rows. The original recipes were
registered before these scores. The [delivery/result report](../analysis/submission_results_20261007.md)
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
period.

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

These subsets were defined for E195, the retrospective four-arm comparison of the
chain before E81 described in the next section.

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
Cat15/LGB5/neural members, fixed weights, the E185 movement-only member, the E167
NM-window rules, fold-specific surface stages and the E152 confidence gate. No
feasibility-mask exclusions were newly introduced; the 126 historical TabPFN rows
were excluded from every arm.

This is **selection-informed retrospective validation**. Historical upstream
out-of-fold predictions were not fully nested through the complete downstream
chain. The stored classifier probabilities are already feasibility-masked;
the original unmasked mass is unavailable. Exact historical OOF donors for v44
and v60 are unavailable
([model design](model-design.md#sparse-tail-and-final-transformations)), and the
legacy tail cache does not reproduce the 42-row production treatment. These
limitations prevent presenting E195 as an unbiased full-v12074 estimate. The
original v12074 manifest explicitly records `quality_validated_on_full_chain=false`.

## 2025 out-of-fold RMSE by airport

The table uses the E195 `joint` arm: the E190 + E188 chain of v12074 through E152,
before E81 and the later tail steps. The E203 member of v12075 is not in these
predictions. Rows are the 2,084,921 evaluated 2025 departures; the 126 LIRF rows of
the TabPFN tail segment are excluded. The numbers are printed by
`scripts/make_doc_figures.py --only 5` (output lines `[F5] airport ...`), which reads
`data/e195_pre_e81_20261007_v2/predictions.parquet`.

**2025 out-of-fold RMSE by airport (E195 `joint`, current chain before E81)**

| Airport | Evaluated rows | Share of squared error | RMSE, all rows (s) | RMSE, labels ≤ 7,200 s (s) |
|---|---:|---:|---:|---:|
| EDDF | 230,141 | 2.5 % | 103.9 | 103.9 |
| EDDM | 167,334 | 1.4 % | 92.7 | 92.7 |
| EGLL | 239,546 | 9.3 % | 196.0 | 188.6 |
| EHAM | 247,951 | 2.6 % | 101.7 | 98.5 |
| LEBL | 179,705 | 4.0 % | 149.0 | 149.0 |
| LEMD | 212,242 | 4.1 % | 138.3 | 138.3 |
| LFPG | 239,552 | 21.6 % | 298.9 | 211.9 |
| LIRF | 160,578 | 30.8 % | 436.4 | 363.5 |
| LSZH | 134,907 | 9.5 % | 263.8 | 102.1 |
| LTFM | 272,965 | 14.3 % | 228.2 | 224.2 |
| **All** | **2,084,921** | **100 %** | **218.26** | **183.17** |

The totals agree with the E190 + E188 row of the E195 table above to two
decimals; that table's `decided` subset excludes one further row. At LSZH, six labels above 7,200 s hold
85 % of the airport's squared error (recomputed from the same file), which explains
the gap between 263.8 s and 102.1 s. LIRF, LFPG and LTFM have the largest errors on
labels up to 7,200 s.

**Why 218.3 s out-of-fold differs from 230.9 s on the leaderboard (explanation,
not measurement).** The two numbers cover different rows: the out-of-fold figure
spans all twelve 2025 months, excludes the 126 TabPFN tail rows (which hold 111 of
the 122 labels above four hours) and stops before E81 and the LSZH stage, while the
leaderboard scores every row of the complete file for January and July 2026; on
January and July 2025 alone the out-of-fold RMSE is 279.2 s, and four labels above
four hours hold 38.2 % of that squared error. The inputs also shift between years:
own-aircraft ADS-B coverage changes strongly by airport and month
([data insights §4](data-insights.md#4-ads-b-surface-coverage-drift)), and
departures without an NM record, which hold 36.6 % of the out-of-fold squared
error, are 1.53 % of the ranking set against 1.08 % of 2025. Finally, the chain was
developed by selecting changes on the same 2025 data, so its out-of-fold error is
optimistic; we have not measured how much of the gap each factor explains.

## Negative experiments and later work

| Experiment | Measured result | Decision and scope |
|---|---|---|
| E193: six-hour METAR history | Against the equal-capacity current-weather control, body RMSE changed by +0.01538 / +0.00640 s on pairs 1+7 / 4+10 | Failed the fixed continuation gate after 12 fits. This tests the selected history encoding; it does not imply that weather is useless. [Report](../analysis/e193_weather_results_20261007.md) |
| E199: spatial surface-state summaries | Spatial-minus-counts RMSE changed by −0.022049 / +0.062688 s on the two screen pairs; the registered body/decided/trim requirements did not all pass | Rejected; no four-pair extension or ranking model. This is a test of the fixed summaries and residual learner, not all possible spatial models. [Report](../analysis/e199_results_20261007.md) |
| E198 → E200: density-model replacement | E200 improved body RMSE by 0.200152 s but worsened decided RMSE by 0.054472 s; only 3/6 pairs improved, below the required four | Rejected before ranking fit. The earlier component-level MDN success did not survive the current pre-E81 comparison. [Report](../analysis/e200_mdn_results_20261007.md) |
| E197: ordered trajectories | Before the ADS-B stages, ordered-minus-PLE RMSE was −4.465932 / −7.113443 s; ordered-minus-equal-capacity-flat was only −0.249545 / −0.126464 s | Positive two-pair component screen. The large PLE difference includes changed inputs and capacity; it is not solely an ordering effect or a measured full-chain gain. [Report](../analysis/e197_verified_results_20261007.md) |
| E204: a wider second trajectory-message scale | Body/decided changes were +0.092642/+0.096240 s on pair 1+7 and −0.203661/−0.203193 s on pair 4+10 | Failed the preregistered requirement to improve both pairs after six fits. Small pooled improvements do not override that gate; no confirmation or ranking fit followed. [Report](../analysis/night_gpu_20261007/RESULTS.md) |
| E205: missed cloud-CB flags | Against existing weather inputs, decided RMSE improved by 0.009903 s; against the current E203 chain, the complete correction worsened decided RMSE by 0.087097 s, with only 3/6 improved pairs and worse trims | Rejected after 12 fits. The additional input signal did not make this correction useful. Independent ASUS reconstruction matched all 2,429,888 flight IDs and four CB columns. [Report](../analysis/night_cb_20261007/RESULTS.md) |

E202, the confirmation run of the ordered-trajectory member, completed the four
remaining month-pair comparisons (24 fits); both registered component gates
passed. E203 then compared the ordered replacement
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
features and models; this page summarises only recent experiments.

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

Organiser clarifications on data use, the final repository commit, the role of
model design and documentation in prize review, and the absence of a final round
are quoted with their dates in [organiser clarifications](organiser-clarifications.md).

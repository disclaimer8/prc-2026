# Final blind submission: `gentle-igloo_final.parquet`

This page describes the one file that team gentle-igloo submits for the blind
final ranking: the rules it answers, the recipe that produces it, why that recipe
differs from the best leaderboard file (v12075), what the difference was measured
to cost, what is known about the new months' inputs, and how the file is built,
validated and uploaded. Numbers are taken from the dated analysis reports they
cite; items marked `[[to be filled after the final upload]]` or
`[[to be filled after 14 October]]` do not exist yet. Written on 8 October 2026
while the final model fits were still running; updated on 9 October 2026 after the
final file was assembled, validated and checked on the public board (log 430–447);
updated on 11 October 2026 after one post-mixture stage, E218 (PRU level), was
added to the recipe and the final file was rebuilt with it (log 449–461,
[section 2.5](#25-e218-pru-level-stage)). The file of 9 October (SHA-256
`76c32426…d554`) is the predecessor: the new final is that file plus the E218
stage, and every stage up to `half` is byte-identical to it.

Related pages: [model design](model-design.md) (the v12075 chain),
[final submission runbook](final-submission-runbook.md) (upload procedure),
[model card](model-card.md), [generalisation](generalisation.md),
[leaderboard use disclosure](development-history.md#6-leaderboard-use-disclosure).

## 1. Rules of the final

Quoted with timestamps in [organiser clarifications](organiser-clarifications.md).

- On 8 October 2026 at 07:34 UTC the organiser announced "a final dataset for a
  single blind ranking" with "two additional months", to "rank against predictions
  for the following combinations: initial two months, final two months, combined
  four months". Teams ranked 1–30 on the initial two months are reviewed, "BUT we
  will consider any submission with the final that will then rank in the first 30
  positions". The stated reason: "the signs of 'optimizing for the ranking' are too
  strong". Review is "human and AI-supported … probably in adversarial mode".
- 07:46 UTC: "the design of the model shouldn't change"; the deadline for the
  final single submission is 2026-10-14 12:00 CET, confirmed at 11:16 UTC as
  **2026-10-14T10:00Z**, which also applies to the repository and documentation.
- 11:16 and 11:55 UTC: the final file is computed from `final_ranking.parquet`
  only; its timestamps carry deliberate shifts of up to ±12 s ("the 'randomness'
  I had to introduce").
- The input published at 11:04 UTC has SHA-256 `4b0692a7…bc69f`
  ([ADS-B derivatives report](../analysis/final_adsb_derivatives_20261008/RESULTS.md)).
  It holds 1,341,197 rows; the departures to predict are 670,790:
  January 152,719, February 144,158, June 181,791, July 192,122
  ([runbook §1](final-submission-runbook.md)). The template
  `final_submitting.parquet` has the same 670,790 `MVT_ID_mvt` values
  (`Float64`) and an `Int32` target column.
- One object `gentle-igloo_final.parquet` in the team bucket; the site rejects any
  mismatched, missing or extra `MVT_ID_mvt` ([runbook §1](final-submission-runbook.md)).

## 2. The recipe

One procedure on all 670,790 rows. It is the v12075 chain
([model design](model-design.md)) with two parts removed, one part widened and,
since 11 October, one post-mixture stage added (E218, section 2.5).
Protocol name: `final-uniform-tail-E156fresh-noE81-fourmonths-20261008-v1`
([tail-assembly protocol](../analysis/final_tail_assembly_20261008/PROTOCOL.md),
log entry 389); the E218 stage is `E218-PRU-level-halfgap-nodeice-additive-v1`
(log 451), and the rebuild with it was registered as log 457.

| | v12075 (leaderboard best) | Final file |
|---|---|---|
| Feature table | FIN1b built on 2025 + the January/July ranking file | FIN1b rebuilt on 2025 + the four-month `final_ranking.parquet`: 2,755,837 rows = 2,085,047 training + 670,790 final (log 385) |
| Base classifier and regressor, CatBoost 15 seeds, LightGBM 5 seeds, five anchored members, movement-only member | Fitted on the January/July view | Refitted with the same rounds, seeds and weights on the new view: 52 fits (log 388), all completed (log 430, 436). Fitted models are saved this time |
| Neural member (ordered trajectories, E203) | Weight 0.25, mean of 3 seeds | Weight **0.35**, mean of **10 seeds** (0–9), same recipe and 8 epochs; the 3-seed mean is kept for comparison only (log 394, 400). The five anchored members are scaled by 0.65/0.75 (raw 0.028889, LOBT/schedule/EOBT/AOBT 0.086667 each) and the tree pair keeps the remainder 0.27444 (`WEIGHTS['final']`, log 425; section 2.3) |
| Whole-file blend with v44 and v60 (E81, weights 0.1214 / 0.0270) | Present | **Absent.** The donors are archived January/July leaderboard files; nothing equivalent exists for February and June |
| LIRF tail without NM record | 42 predictions frozen from v361 (a single TabPFN draw) and protected | **Fresh 10-seed TabPFN-2 ensemble (E156, airline prefix and local date) on every row of the tail segment in all four months**; every served row is protected from later stages |
| Surface stages E141, E64, E188; gate E152; LOBT lift and clip; LSZH OPDI stage at 50 % | Present | Present and unchanged (cached models replayed only on an exact training signature) |
| Input timestamps | Old ranking file | `final_ranking.parquet` as published, including its ±12 s shifts |
| Post-mixture PRU level stage (E218) | Absent (v12075 and v12077; the leaderboard file v12081 is v12077 plus this stage only) | **Present** since 11 October: stage 12 `pru_level` after `half`, enabled by `E218_PRU_LEVEL=1`; reads EUROCONTROL's published monthly additional taxi-out time (section 2.5) |

### 2.1 Stage order

From the literal `stage_plan()` of the assembly adapter
([protocol](../analysis/final_tail_assembly_20261008/PROTOCOL.md)) with
`E218_PRU_LEVEL=1`; without the flag the plan ends at stage 11 and is
byte-identical to the plan of 9 October (log 457). Weights as in
[model design](model-design.md).

| # | Stage | What it does in the final |
|---|---|---|
| 0 | premix | Raw regime probabilities recomputed from the saved classifier; regular candidate = 0.27444 × (0.65 CatBoost15 + 0.35 LightGBM5) + 0.028889 raw-target member + 0.086667 each of the LOBT-, schedule-, EOBT- and AOBT-anchored members + 0.35 neural member (`WEIGHTS['final']`; the canary used the v12075 weights 19/60, 1/30, 0.10 and 0.25, `WEIGHTS['janjul']`); probability-weighted mixture over regime candidates |
| 1 | E185 | Movement-only member at 0.35 × 2/3 on rows with no LOBT outside LIRF |
| 2 | E167 | NM matching-window rules against the arrivals of the final file |
| 3 | E156 | TabPFN-2 classifier, `TABPFN_EXTRA=pfx`, seeds 0–9, CPU, pinned checkpoint (SHA `cf8c519c…`), offline; served-ID manifest written |
| 4 | E141 | Own-track surface stage, seeds 42, 1, 2, 3, 4 (cached models) |
| 5 | E64 | Previous-visit stage, same seeds (cached models) |
| 6 | E188 | LIRF previous-visit stage for rows with an NM record (5 saved models) |
| 7 | E152 | Keep the pre-surface value where a planning regime has probability ≥ 0.95 |
| — | E81 | **not applied** |
| 8 | LOBT | Edge lift and window clip |
| 9 | protect | Every served E156 row restored to its stage-3 value |
| 10 | LSZH | OPDI exit-parking correction at LSZH only; a row with no matched event is unchanged |
| 11 | half | 0.5 × (stage 9) + 0.5 × (stage 10), rounded |
| 12 | pru_level | E218: per airport × month, half the gap between the indicator implied by the stage-11 predictions and EUROCONTROL's published monthly additional taxi-out time, net of the airport's 2025 emulation offset, added to the rows of the PRU population; airport-months with de-icing and the served E156 rows unchanged (section 2.5) |

### 2.2 The tail segment

In the January/July dry run the segment had 42 rows
([dry-run results](../analysis/final_tail_assembly_20261008/DRYRUN_RESULTS.md)).
Counted on `final_ranking.parquet`, LIRF departures without an NM record and with
take-off minus schedule ≥ 4 h number 15 in January, 8 in February, 12 in June and
28 in July ([final strategy §0](../analysis/final_strategy_20261008.md)); the
authoritative list is the served-ID manifest written by stage 3. On the same
sealed input the ten-seed ensemble was bitwise repeatable: two runs gave identical
probabilities for all seeds and 0 of 42 rounded values differed
([dry-run results](../analysis/final_tail_assembly_20261008/DRYRUN_RESULTS.md)).
In the final run stage 3 served 62 rows over the four months, again bitwise
repeatable ([stage receipt](../analysis/final_tail_20261009/receipts/03_e156.json));
in January/July they are the same 42 IDs as in the dry run
([control explanation §4](../analysis/v12080_control_explanation_20261009.md)).

### 2.3 The ten-seed neural member

Seeds 3–9 were trained after seeds 0–2 with the unchanged recipe (width 512,
PLE 48 bins, 2 × 64 tokens, batch 4096, AdamW, 8 epochs), 270–274 s per seed on
the AMD Radeon RX 7900 XTX, each checkpoint replayed exactly (log 393, 400). On the
670,790 final rows the 10-seed mean differs from the 3-seed mean by RMS 23.98 s
with a mean shift of −0.28 s; the pairwise RMS between single seeds is 70.6 s
(0.20–0.24 in units of the residual scale)
([statistics](../analysis/final_ordered_trajectory_20261008/execution_10seed/statistics.json)).
This is a variance reduction with no added capacity. The decision to use the
10-seed mean was written before the extra seeds were trained (log 394).

**The member's weight is 0.35, not the 0.25 of v12075.** The 0.25 had been
inherited from the earlier MLP member and never chosen at the level of the full
chain. E212 (log 407a, 410) recomputed the exact 2025 out-of-fold chain up to E81
at twelve weights (0.05–0.60 in steps of 0.05) with the three-seed and a ten-seed
member, taking the weight from the other members in proportion (`regular(w) =
((1 − w)/0.75) × [tree pair and anchors at the v12075 weights] + w × NN`; at
w = 0.25 this reproduces the production column byte for byte on all 2,085,047
rows). Results on the `decided` rows, pooled over the six month pairs
([E212 report](../analysis/e212_nn_weight_10seed_20261008.md)):

| | 0.25 (v12075) | 0.35 (final) | 0.40 | 0.45 |
|---|---:|---:|---:|---:|
| RMSE, 3 seeds (s) | 217.717 | 217.549 | 217.523 | 217.546 |
| RMSE, 10 seeds (s) | 217.723 | 217.535 | 217.496 | 217.504 |
| Months of 2025 better than 0.25 (10 seeds) | — | 12 of 12 | 11 of 12 | 9 of 12 |

The registered leave-one-pair-out rule selected 0.40 in all six pairs (−0.228 s
pooled against 0.25, 6 of 6 pairs, gate passed). Two checks outside the gate
argued for less: 0.40 is worse than 0.25 in August (+0.016 s), and a weight chosen
on January–June (0.50) is worse than 0.25 on July–December (+0.017 s); the
per-month optimum ranges from 0.30 to 0.60. The coordinator therefore fixed 0.35,
the only grid value with lower squared error than 0.25 in every month of 2025 at
three and at ten seeds (log 425; the per-month table is in
[generalisation](generalisation.md#is-the-neural-weight-tuned-to-januaryjuly)).
The ten seeds alone, at the old weight, change nothing in cross-validation
(+0.0065 s); the gain comes from the weight.

Both changes were then scored on the January/July board on the v12075 chain, with
E81 and the frozen rows kept, as two single-change uploads with forecasts written
beforehand (log 416, 417, 426, 427): v12076 (ten-seed mean at 0.25) 230.8183 s,
−0.1204 s against v12075 (forecast −0.03, corridor [230.74; 230.99], inside);
v12077 (weight 0.35, anchors scaled) 230.4812 s, −0.3371 s against v12076
(forecast −0.08 from the January + July 2025 estimate of −0.11 s times a 0.7
transfer, corridor [230.62; 230.87]; the result was 0.14 s below the corridor on
the favourable side). The canary v12090 in section 4 ran with the v12075 members
and weights, before either change.

### 2.4 The ±12 s shifts

The shifts are not new noise: every challenge file, including all of 2025, carries
the same dithering of the movement time (`T − η/2`) and block time (`T + η/2`), so
the label carries `η ~ U(−12, 12)`; `final_ranking` is a fresh independent draw
([jitter analysis](../analysis/final_jitter_sensitivity_20261008.md)). The regime
tolerance of 6 s and the 3,606 s LOBT window are the half-width of that dithering.
Expected effect on RMSE: 0; the realised deviation on January/July is ±0.08 s
(1σ) for the production chain. No augmentation, tolerance change or de-noising was
added; the input is used as published. The ±0.08 s holds for a fixed chain only:
the control upload of section 7, item 4, showed that the refitted base classifier's
response to the re-dithered inputs alone carries about 0.29 s of noise on
January/July ([control explanation §3, §6](../analysis/v12080_control_explanation_20261009.md)).

### 2.5 E218: PRU level stage

**What it does.** For every airport × month of the final, stage 12 computes the
additional taxi-out indicator implied by the stage-11 predictions, with the
emulation of [PRU view](pru-view.md) and [data quality for PRU](data-quality-regimes.md):
the reference is the 10th percentile of the 2025 label per airport × stand × runway
(valid with at least 10 flights at or below it), the population P is every
departure with a stand and a runway, not a helicopter, a prediction between 0 and
7,200 s and a valid reference, and the indicator `ind` is the mean of prediction
minus reference over P. It compares `ind` with EUROCONTROL's published monthly
additional taxi-out time of the same airport and month (`TOTAL_ADD_TIME_MIN` × 60 /
`TOTAL_REF_NB_FL`) and subtracts `emu_hat`, the airport's 2025 emulation offset (the
mean of "indicator computed from the 2025 labels minus the published value" over
the airport's 2025 months without de-icing). Every row of P in the cell moves by

c = −0.5 × (ind − published − emu_hat), rounded to whole seconds.

The factor 0.5 was fixed in the registration and not tuned. A cell is eligible only
if the published value exists and PRU's excluded share is less than 0.5 percentage
points above the airport's April–October 2025 level, that is, if PRU did not
exclude de-iced flights in that month; cells with de-icing are unchanged, and so are
rows outside P and the 62 served E156 rows. Code: `pru_level_shift` in
`scripts/final_tail_assembly_20261008.py` (the patch
`analysis/e218_pru_level_recheck_20261010/final_tail_e218_pru_level_stage.patch`,
SHA-256 `0613fdd9…`, applied unchanged; log 451, 452, 457).

**Why.** An earlier calibration to the published PRU values (E145) had improved
cross-validation by 0.045 s and worsened the board by 0.114 s on 28 September
([negative results](negative-results.md)). The re-analysis of 10 October (log 451,
452; [report](../analysis/e218_pru_level_recheck_20261010.md)) found that E145 set
the mean taxi time of all rows to the published reference plus additional time plus
a same-month 2025 offset, so it absorbed PRU's recomputation of its rolling
reference (EGLL July: −20.8 s for PRU, −11.9 s for ours) and the change in the
difference between rows with a valid reference and all rows (LIRF July: −24.1 s).
It moved against the gap in the additional-time indicator in 3 of 10 cells, and its
largest shift (LEMD January, −25 s) fell on a cell where PRU excluded de-iced
flights. On the production chain's 2025 out-of-fold predictions the gap between our
indicator and the published one is unrelated to the mean residual over all 120
airport-months (r = 0.16), but in the 95 airport-months without de-icing it follows
it closely (r = 0.91, slope 0.85). The rule above was registered before its gate was
run (log 451). The gate, on the exact 2025 out-of-fold chain (2,084,920 `decided`
rows, reference and `emu_hat` from the training months of each split), passed every
condition: lower in 6 of 6 month pairs; pooled `decided` −0.137 s, body (labels
≤ 7,200 s) −0.162 s, trimmed < 0; forward in time, first half-year → second
−0.126 s and expanding window −0.131 s (9 of 9 months lower); not worse in 12 of 12
months (log 452). The leave-one-airport-out bias model gives β = 0.93 (σ 0.07
across airports) for the share of the gap that is real bias. The form of the rule
(cells without de-icing, the additional-time indicator, a constant shift) was chosen
after the diagnosis on the same 2025 data, so the gate checks the implementation and
its stability, not the transfer to 2026; that was left to the board (below).

**Data use, disclosed.** The stage reads two files that EUROCONTROL publishes:
`taxi_out_additional_time_2025.csv` (SHA-256 `4aedf195…`, for the reference year)
and `taxi_out_additional_time_2026.csv` (SHA-256 `b72040e0…`), whose January,
February, June and July values are **the scored months of the final**. These are
monthly aggregates of the same kind of record as the labels: PRU computes them from
the airport operator data flow behind the challenge's block times, its monthly
flight counts agree with our departure counts within 132 flights per airport-month
in 2026 ([PRU check §1](../analysis/final_pru_and_regime_check_20261009.md)), and our
emulation from the 2025 labels reproduces the published April–October 2025 values to
a mean absolute difference of 3.3 s ([data quality for PRU](data-quality-regimes.md)).
The stage therefore brings published information about the label level of the
scored months into the file, one number per airport and month; it uses no row-level
or daily value of these files. The organiser answered "I already answered above: yes"
(27 September 2026, 15:14 UTC) to our question whether the published monthly taxi-out
figures for January and July 2026 may be used as a model input, and on 8 October 2026
confirmed in the challenge channel that `taxi_out_additional_time_2026.csv` may still
be used, in answer to another team that reported using it (recorded in log 452 and in
[the E218 report §4.4](../analysis/e218_pru_level_recheck_20261010.md); the message text
is not in our export, so it is not quoted). The files carry no open licence; their
terms are in [SOURCES.md](../SOURCES.md) and the statements in
[organiser clarifications](organiser-clarifications.md). Both files are pinned by
SHA-256 in the final contract and the stage refuses to run on any other copy.

**Board evidence.** Three uploads, each with a forecast and a decision rule written
beforehand:

| Upload | What it is | RMSE (s) | Against | Forecast, corridor | Rule and outcome |
|---|---|---:|---|---|---|
| v12081 | v12077 plus the E218 stage, the only change (215,682 rows changed, RMS 3.53 s) | **230.4478** | −0.0334 vs v12077 | 230.408, [230.394; 230.422] | ≤ 230.4512 ⇒ rebuild the final with E218. Outside the corridor on the unfavourable side by 0.025 s; rule met with a margin of 0.0034 s (log 455, 456) |
| v12082 | the January/July rows of the final rebuilt with E218 (control) | **231.1445** | −0.0360 vs v12080 | 231.144, [230.994; 231.294] | ≤ 231.1805 ⇒ the E218 file becomes the final. Inside the corridor, 0.0005 s from the point (log 458, 459) |
| v12083 | v12082 with the E218 shift removed on the LIRF rows only (25,017 rows; LIRF rows equal v12080) | **231.1749** | +0.0304 vs v12082 | 231.153, [231.130; 231.174] | ≤ 231.1475 ⇒ rebuild the final without the LIRF cells. Outside the corridor by 0.0009 s; rule not met, the final stays `bbae2cc5…` (log 460, 461) |

v12080, v12083 and v12082 differ only by the E218 shifts on the same labels: v12083
adds the ten non-LIRF cells of January/July to v12080, and v12082 adds the two LIRF
cells to v12083. The gain of the stage on the final's own January/July rows,
−0.0360 s, therefore splits exactly: **LIRF −0.0304 s (84 %), the other ten cells
−0.0056 s** ([v12083 record](../analysis/v12083_upload_20261011/RESULTS.md),
[verdict.json](../analysis/v12083_upload_20261011/verdict.json)). In the bias model
of log 452, the LIRF cells transfer about 0.69 of their gap to the published value
and the non-LIRF cells 0.37–0.48, against β = 0.93 on 2025. The reading suggested by
v12081 and v12082 alone, "the non-LIRF cells transfer fully and the LIRF shift is
about neutral", was one of two readings that a single number could not separate;
v12083 separates them and refutes it (log 461). The 2025 relation thus transfers only
partly, and most of what transfers is at LIRF.

**Risk.** LIRF carries the largest shifts: +9, +8, +21 and +11 s in January,
February, June and July. On January/July the LIRF cells are now the best-measured
part of the stage; whether February and June follow is not measured. If they follow
the January/July relation, removing the LIRF cells would raise the final file's RMSE
by about 0.045 s through the February/June rows (model, not measured; break-even at a
LIRF transfer of 0.25) on top of 0.0156 s through the January/July rows (measured;
log 461, [verdict.json](../analysis/v12083_upload_20261011/verdict.json)). The worst
case of the model of log 452, a gap that is pure noise in every cell, would cost
+0.066 s on February/June
([board.json](../analysis/e218_pru_level_recheck_20261010/board.json)); on
January/July both parts of the stage gained, so that case did not occur there. LIRF
June (+21 s on 14,362 rows) remains the largest single bet of the final.

**The new file.** Rebuilt on the Mac on 11 October 2026 with `E218_PRU_LEVEL=1` on
`make-contract`, `prepare` and `assemble --repeat-e156` (231 s, peak 10.8 GB, 0 fits;
log 457, 458). SHA-256
`bbae2cc544989a0faf68abb85dbb4a9aab5ea38eec98d01fa9282c332b4234a1`, 2,145,576 bytes,
670,790 rows; byte-identical to the E218 dry run of log 452. Contract `7a1cde30…`
(4,027 files: the 4,025 of the 9 October contract `6af6c4ca…` plus the two PRU files,
and a `pru_level` block with k 0.5 and the 0.005 de-icing threshold); builder
SHA-256 `21f46756…`. Without the flag the plan is byte-identical to that of
9 October and a fresh contract equals `6af6c4ca…` except its creation time; the
adapter's 22 tests pass with and without the flag. The receipts of stages 0–11 equal
those of 9 October field by field except three explained fields (the row-group split
of the premix Parquet with identical values, the timing of the E156 repeat, and the
absolute workspace paths in the E188 receipt); the outputs of stages 1–11 are
byte-identical, and the stage-11 output is the 9 October file `76c32426…`
([receipt parity](../analysis/final_tail_20261011/receipt_parity.json)). Only stage 12
changes rows: 415,697 of 670,790, RMS 4.687 s, mean −0.456 s, maximum 21 s; 29 of
40 airport-months eligible; January/July 201,598 rows (RMS 3.80 s), February/June
214,099 (RMS 5.47 s). Shift per row, seconds ("de-icing": not eligible; 0: eligible,
|c| < 0.5 s; [cells](../analysis/final_tail_20261011/pru_level_cells.csv),
[rows](../analysis/final_tail_20261011/rows_changed_by_cell.csv)):

| Airport | January | February | June | July |
|---|---:|---:|---:|---:|
| EDDF | de-icing | de-icing | +3 | +1 |
| EDDM | de-icing | de-icing | −3 | −3 |
| EGLL | −1 | 0 | −1 | −1 |
| EHAM | 0 | −2 | −3 | −3 |
| LEBL | 0 | +5 | −4 | −4 |
| LEMD | de-icing | −2 | −5 | −7 |
| LFPG | de-icing | de-icing | −8 | −5 |
| LIRF | +9 | +8 | +21 | +11 |
| LSZH | de-icing | de-icing | −3 | −3 |
| LTFM | de-icing | de-icing | −2 | −4 |

The validator against v12081 passed every hard check (section 7, item 1), and a
second assembly in a fresh workspace gave the same SHA-256 (section 7, item 5).

## 3. Why this recipe

- **Generalisation over the two months no one has scored.** The chain's 2025
  out-of-fold error on February + June (247.3 s all rows, 196.3 s on labels
  ≤ 7,200 s) is not worse than on January + July (279.2 s, 211.2 s), and the gain
  over the NM-proxy baseline is 181–227 s in every month
  ([generalisation](generalisation.md)). The design is therefore not changed for the
  new months.
- **No leaderboard-specific parts.** Two items of the v12075 chain depend on
  January/July leaderboard files and were decided on the board
  ([disclosure, items 1 and 2](development-history.md#6-leaderboard-use-disclosure)):
  the 42 frozen TabPFN rows are one favourable draw kept after a seed ensemble
  scored worse on the board, and the E81 donors v44/v60 are archived submissions
  without out-of-fold counterparts. Copying v12075's January/July rows into the
  final and predicting only February/June with a model was rejected
  ([final strategy §1, variant (c)](../analysis/final_strategy_20261008.md)): it is
  two procedures in one file and exactly what the organiser named as the reason for
  the blind final.
- **One code path for four months.** The same adapter, cached stage models and
  weights serve all rows; nothing is chosen per month. The E218 stage applies one
  rule to all four months; which airport-months it moves follows from PRU's
  published de-icing exclusions, not from a choice.
- **One stage added after the 9 October file, decided on 2025 data and checked on
  the board.** E218 passed a gate registered before it ran on the 2025 out-of-fold
  chain, then had to beat v12077 by at least 0.03 s in one upload (v12081) under a
  rule written before that upload, its January/July rows were checked once more
  in the control v12082, and its LIRF part was measured separately by v12083
  (section 2.5). Its input is a published aggregate of the
  scored months; this is disclosed in section 2.5 and in the
  [leaderboard use disclosure](development-history.md#6-leaderboard-use-disclosure).

## 4. Measured cost on January/July

The uniform recipe was run on the v12075-era January/July artifacts (old members,
the three-seed neural member at weight 0.25, E81 removed, fresh E156) and uploaded
once as a technical canary, `v12090`
([canary report](../analysis/v12090_upload_20261008/RESULTS.md), log 390–391):

| | RMSE on January/July 2026 (s) |
|---|---:|
| v12075 (best at the time, with E81 and frozen rows) | 230.9387 |
| v12090 (same chain, no E81, fresh E156 on the 42 rows) | **232.1933** |
| Difference | +1.2546 |

The forecast written before the upload was 231.6 s with a corridor of
[231.1; 232.3] s; the result is inside the corridor near its upper edge. Against
v12075 the file changed 313,616 of 344,841 rows with RMS 18.209 s. The canary
removed E81 and replaced the 42 donor rows at the same time, so its +1.2546 s does
not identify either effect alone.

Later the same day the two member changes of the final recipe were scored on the
v12075 chain with E81 and the frozen rows still in place (section 2.3): v12076
(ten seeds) 230.8183 s and v12077 (weight 0.35) 230.4812 s, −0.4575 s together.
On 9 October the two removals were then **measured one at a time on the v12077
chain**, each as the only change against v12077 and each with a forecast written
before the upload (log 431–434, 444):

| File | Only change against v12077 | RMSE (s) | Against v12077 (s) | Forecast, corridor |
|---|---|---:|---:|---|
| v12077 | — (E81 and the 42 frozen rows present) | 230.4812 | — | — |
| v12078 | E81 whole-file blend removed | 231.0488 | **+0.5676** | 230.60, [230.40; 231.20] |
| v12079 | 42 frozen rows replaced by the fresh ten-seed E156 values | 231.1770 | **+0.6958** | 231.17, [230.83; 231.48] |
| Sum of the two parts | | | **+1.2634** | |

The two measured parts add up to +1.2634 s against the canary's +1.2546 s, which
was measured with the older weights and the three-seed member; the interaction is
about −0.009 s (log 444). The cost of the uniform recipe on January/July is
therefore measured, not inferred: +0.5676 s for dropping E81 and +0.6958 s for
replacing the single favourable TabPFN draw by the fresh ensemble. Both
results fell inside their corridors. The central forecast for v12078 came from the
Gram identity on the current files and missed by +0.45 s: the identity is exact for
the unrounded blend of fixed files, but v12078 also differs by the tail stages
applied after E81 (LOBT lift, protect, half LSZH), and that interaction is
correlated with the residual; the historical price of E81 (−0.6564 s on v98) was
the better guide ([v12078 record](../analysis/v12078_upload_20261009/RESULTS.md)).
v12079 was uploaded during an outage of the organiser's scorer (no file of any team
was processed between 08:54:51 and 16:28 UTC) and was scored after it was
repaired, 0.007 s from its central forecast
([v12079 record](../analysis/v12079_upload_20261009/RESULTS.md), log 434, 444).

On this decomposition the January/July rows of the final file were expected at
230.4812 + 0.5676 + 0.6958 = 231.745 s plus the noise of the refits. They scored
231.1805 s as the control v12080; why that is 0.564 s better is explained in
section 7, item 4. The canary and the three uploads of 9 October are technical
checks of the code path on real labels, not candidates for best. The January/July
labels do not transfer one-to-one to the blind months, but the sign and size of the
cost are known and accepted. The E218 stage added on 11 October lowered the
January/July rows of the final file from 231.1805 s (v12080) to 231.1445 s (v12082),
−0.0360 s (section 2.5).

## 5. ADS-B coverage in the final months

Seven daily surface products were rebuilt for all 120 final days from the adsb.lol
archives with the final movement times (log 387, 392;
[report](../analysis/final_adsb_derivatives_20261008/RESULTS.md)). `seg` is the
share of departures with an own surface segment (the input of stage E141);
`seg25` is the same calendar month of 2025, on which the stage models were trained.

| Month | Departures | seg 2026 | seg 2025 | visit 2026 |
|---|---:|---:|---:|---:|
| 2026-01 | 152,719 | 67.6 % | 50.3 % | 81.7 % |
| 2026-02 | 144,158 | 63.8 % | 51.8 % | 83.5 % |
| 2026-06 | 181,791 | 64.0 % | 64.0 % | 80.4 % |
| 2026-07 | 192,122 | 62.9 % | 55.9 % | 83.0 % |

Three airport-months collapse relative to 2025:

| Airport, month | seg 2026 | seg 2025 | Note |
|---|---:|---:|---|
| EDDF 2026-02 | 32.2 % | 96.3 % | Every day at 15–46 %; aircraft are seen airborne (`visit` 92–98 %) but the ground segment is not received |
| LIRF 2026-06 | 0.7 % | 80.3 % | 0.0–2.8 % every day; `visit` 51.5 % against 98.3 % in 2025 |
| LIRF 2026-07 | 6.2 % | 93.9 % | Known since the public ranking; unchanged by the final movement times |

Other shifts: LTFM has no ground reception in any month (0.0–0.3 %, as in 2025);
LFPG 15–31 %; LEMD January/February 96–97 % against 0–2 % in 2025 (reception
appeared at Madrid); EGLL June/July 93–96 % against 34–36 %.

**EHAM taxiway works in June 2026.** Schiphol's announced maintenance of taxiways A and B
(4 May – 26 June 2026) covers 18,194 of the 20,881 EHAM final departures in June. The model has
no input for it (the announcement has no open licence and 2025 has no comparable episode), so
EHAM June is a known risk rather than a modelled effect.

**Tabular inputs against 2025 coverage.** A label-free audit of the sealed four-month feature table against the
2025 training rows ([input audit](../analysis/final_months_input_audit_20261009.md)) finds no new departure runway at
any airport and at most 0.1 % of an airport-month on runways with fewer than 100 training departures. Runway
configurations are novel only at Schiphol (5.8–11.9 % of EHAM rows in configurations with fewer than 100 training
departures, 3.7–7.9 % never seen in 2025) and marginally at Istanbul in January/February (2.4–3.5 %). Destination ATFM
delay and hourly traffic features stay inside the 2025 1st–99th percentile band of their airport (at most 1.3 % and
3.3 % outside). Two inputs leave it: the daily movement count (June/July 2026 above every 2025 day at LEMD, +7 %, 51–69 %
of rows, and at LEBL, LIRF, LSZH, +5 %; January 2026 below 2025 at EHAM, −12 % departures per day, and EDDM) and
temperature (June 2026 above the 2025 maximum at EDDF, P99 40 °C against 33, and at LFPG, EDDM, LSZH, EGLL, EHAM). Tree
models extrapolate flat beyond their training range, so these rows get the prediction of the most extreme 2025 days.
The no-NM share is 0.9–1.6 % per month against 0.8–2.1 % in 2025; only EHAM January (5.0 %) and LSZH January (3.0 %)
exceed every 2025 month of their airport. No tabular input separates the EHAM works window from the rest of June; the
EDDF collapse is visible only in ADS-B.

**2026-06-11 is defective in the archive.** The day's surface file holds 91,473
points against a median of 1.24 million; LIRF, LEBL, LEMD and LTFM are absent. Both
replicas (`staging-0`, 42,736 traces; `prod-0`, 81,883 points) are truncated at the
source. The day is processed as it is: 6,039 final departures, 2 own-track matches,
300 visit records; no stage input for that day is recoverable.

**How the chain behaves without coverage.** E141 is applied only to rows with
`coverage ≥ 2` and E64 only to rows with `coverage < 2` and a previous visit, so
where the track is missing the stages leave the base prediction unchanged; the base
model has no ADS-B features by design ([data insights §4](data-insights.md)). The
neural member does see the own aircraft's trajectory tokens: in LIRF June 2026
58.6 % of rows (8,773) have no tokens against 2.7–3.2 % in June–August 2025, LIRF
July 25.9 % (4,101), EDDF February 10.2 % (1,496) (log 397). A protection rule
(neural weight 0 on rows with no tokens and no visit record, weight redistributed)
was pre-registered with a 2025 out-of-fold gate (log 397). It failed in 0 of 6 folds:
on the 417,194 out-of-fold rows it would touch, RMSE rose from 269.83 s to 271.87 s
(+2.03 s). The neural member without trajectory tokens still beats no neural member in
every fold and in the collapsed months, so the rule is not applied (log 401;
[coverage audit](../analysis/coverage_collapse_audit_20261008.md)). The audit's scenario estimate
of the cost of the lost coverage is about +0.18 s (range +0.1 to +0.3 s) on the final: it
transfers 2025 premix penalties to the 2026 masks and is not a measured full-chain loss.
No rule recovers it.

## 6. Pipeline: scripts, inputs and receipts

Every step is a dated script registered in `HYPOTHESES_LOG.md` before it ran, with
tests that use synthetic rows only, and a receipt directory under `analysis/`.
Interpreter on the Mac: `.venv/bin/python`.

| # | Step | Script | Status on 9 Oct, receipts |
|---|---|---|---|
| 1 | Sealed four-month feature workspace: FIN1b, E106/E113 prefix tables, motion view | `scripts/final_feature_workspace_20261008.py`, `scripts/final_feature_mac_20261008.py` | Done on the Mac: 11 stages, 64.89 s, peak 12.53 GiB; WORK `data/final_feature_work_mac_20261008`, FIN1b SHA `2bb90b76…` (log 385; [protocol](../analysis/final_feature_workspace_20261008/PROTOCOL.md)) |
| 2 | 52 tabular fits with saved models: classifier (180 rounds) + regressor (2,319), CatBoost 15 × 5,600, LightGBM 5 × 2,319, anchors raw 2,565 / LOBT 3,065 / schedule 2,958 / EOBT 2,736 / AOBT 2,738 × 5 seeds, motion 5 × 2,481 | `scripts/final_tabular_models_20261008.py` | **Done: all 52 fits**, on prc-cloud (ARM) and hetzner; cross-node parity on a 20-round probe: max abs diff 7.96e-13 over 670,790 predictions ([execution journal](../analysis/final_tabular_models_20261008/EXECUTION.md), [contract](../analysis/final_tabular_models_20261008/CONTRACT.json), [receipt](../analysis/final_tabular_models_20261008/receipt_42a6f28.json), log 388). The two CatBoost queues hit their 12 h unit runtime limit (01:00/01:03 UTC on 9 October) about 12 minutes before seeds 5 and 13 finished; seeds 0–4 and 8–12 were complete and untouched, the two empty seed directories were set aside (the fitter refuses to resume a partial seed), and the remaining five seeds (5, 6, 7, 13, 14) were relaunched with the same receipt `42a6f28`, the same resources and a 10 h limit per queue; recipe, seeds and parameters unchanged (log 430). All nine families were merged on the Mac (`models/final_tabular/<kind>/merged/`; CatBoost 15 seeds, LightGBM 5, anchors raw/LOBT/schedule/EOBT/AOBT 5 each, motion 5, base). On the January/July rows each family differs from the v12075-era member by RMS 19–22 s with a mean shift of +0.3 to +0.6 s (input dithering and refit; log 436) |
| 3 | Neural member: 2025 token reuse (364 days) + 120 final days re-encoded on final times; seeds 0–2, then 3–9 | `scripts/final_ordered_trajectory_20261008.py`, `scripts/final_ordered_trajectory_seeds_20261008.py` | Done on the Home GPU; 3-seed `prediction.npz` SHA `6d50f9cb…`, 10-seed `prediction_10seed.npz`; exact checkpoint replay ([receipts](../analysis/final_ordered_trajectory_20261008/), log 387, 393, 394, 400) |
| 4 | Seven daily ADS-B products for 120 days on final movement times | `scripts/final_adsb_derivatives_20261008.py` | Done on the Mac: 203.9 s, 120/120 days, 840 files, 114 MB, `MANIFEST.sha256` SHA `a3240876…` ([report](../analysis/final_adsb_derivatives_20261008/RESULTS.md), log 392) |
| 5 | OPDI events for February and June 2026 (LSZH stage input) | `scripts/opdi_final_extract_20261008.py` | **Done** on prc-cloud: February and June windows plus the two boundary windows, 7 files, 4.8 MB, `data/opdi_final_20261008/`, pinned in the final contract (`make-contract --opdi-extra`, 51 files by SHA; log 398, 404, 405). LSZH departures with an exit-parking event: January 4,151 of 9,995 (41.5 %, a gap in the source), **February 6,886 of 9,374 (73.5 %)**, **June 9,517 of 12,384 (76.8 %)**, July 10,131 of 13,155 (77.0 %) ([report §B.3](../analysis/tail_e156_opdi_20261008.md)). Rows without a matched event stay unchanged. The January/July set of IDs with an event is unchanged, so the cached LSZH model stays valid |
| 6 | Tail assembly: premix, stages 1–11, served-ID manifest, `complete.json` | `scripts/final_tail_assembly_20261008.py` (`make-contract --mode final`, `prepare`, `assemble --repeat-e156`) | Adapter dry-run on January/July (141 s, 0 fits, premix byte-identical to the archived v12075 premix; [dry run](../analysis/final_tail_assembly_20261008/DRYRUN_RESULTS.md), log 389). **Final run done** on the Mac on 9 October: 157.6 s including the E156 repeat, `fits_performed = 0`, E81 not applied, 0 donor rows copied; stage 3 served 62 rows and its repeat was bitwise identical (all ten probability arrays, 0 rounded rows differing). Rows changed per stage: E185 8,224; E167 306; E156 62; E141 429,789; E64 95,421; E188 18,128; E152 287; LOBT 169; protect 11; LSZH 30,407 (now also in February and June); half 30,099 ([complete.json](../analysis/final_tail_20261009/complete.json), [stage receipts](../analysis/final_tail_20261009/receipts/), [contract](../analysis/final_tail_20261009/contract.json), log 436, 438). **Rebuilt on 11 October** with `E218_PRU_LEVEL=1` (builder `21f46756…`, contract `7a1cde30…`): 231 s, peak 10.8 GB, `fits_performed = 0`; stages 0–11 as on 9 October, stage 12 `pru_level` 415,697 rows ([complete.json](../analysis/final_tail_20261011/complete.json), [stage receipts](../analysis/final_tail_20261011/receipts/), [contract](../analysis/final_tail_20261011/contract.json), log 457, 458; section 2.5) |
| 7 | Validator | `scripts/validate_final_submission.py --file … --reference submissions/gentle-igloo_v12081.parquet --json …` (on 9 October with v12077 as the reference) | **Done**: all hard checks pass; against v12081 on the 344,841 January/July IDs 329,836 rows changed (95.65 %), RMS 26.485 s, max 2,926 s ([validator report](../analysis/final_tail_20261011/validate_vs_v12081.json)). Independent rebuild from scratch (a new `prepare` and `assemble` from the same contract into a separate work directory on the Mac) gave the same SHA-256 `bbae2cc5…34a1` ([replay log](../analysis/final_tail_20261011/assemble_replay.log), [MANIFEST](../analysis/final_tail_20261011/MANIFEST.json), log 458). The 9 October file was checked the same way (against v12077: 95.64 %, RMS 26.5 s; replay 112.7 s, SHA `76c32426…d554`; [report](../analysis/final_tail_20261009/validate_vs_v12077.json), log 438) |
| 8 | Single PUT to the bucket | Mode `final` of the transport registered in log 414 (`scripts/final_transport_20261009.py`, built on `OnePutHTTP` and `single_attempt`, per [runbook §3](final-submission-runbook.md)) | Planned for **12 October** from the Hetzner host, after the last regular upload (log 447, 459). The manifest of 9 October (`manifest_final.json`, file `76c32426…`) is superseded by [`analysis/final_transport_20261009/manifest_final_v2.json`](../analysis/final_transport_20261009/manifest_final_v2.json) (SHA-256 `3648f566…`, file `bbae2cc5…`, day 2026-10-12, builder `21f46756…`), sealed after v12082; registration commit `6b130265…`, transport source SHA-256 `831712bf…`; the new file and its manifest were copied to Hetzner and verified by SHA-256, and the preflight passed on the Mac and on Hetzner, and again on Hetzner after v12083. The final PUT has not been made ([v12082 record](../analysis/v12082_control_20261011/RESULTS.md), [v12083 record](../analysis/v12083_upload_20261011/RESULTS.md), log 459, 461). Receipt: `[[to be filled after the final upload]]` |

Cached stage models (E141 5, E64 5, LSZH 1, E188 5) are loaded only when the
reconstructed training call matches the stored signature (parameters, rounds,
data/label/weight hashes, feature order, package and interpreter versions);
otherwise the run stops before any fit, and a refit is a separate explicit action
(`--refit`). The TabPFN-2 checkpoint is pinned by SHA in the contract after a macOS
cache purge deleted the cached copy on 8 October
([protocol](../analysis/final_tail_assembly_20261008/PROTOCOL.md)).

Measured in the final run: the four-month assembly took 157.6 s on the Mac with the
E156 repeat and 112.7 s for the rebuild without it (the January/July dry run took
141 s); the rebuild of 11 October with the E218 stage took 231 s with the E156
repeat. Fit times of step 2 per family were not collected into one table; the
bound that mattered was CatBoost, whose queues of 5,600-iteration fits finished five
seeds each within 12 h and were stopped by the unit limit during the sixth (log 430).

## 7. Validation of the file

1. **Hard checks** (exit code 0 required): object name, column names and order,
   dtypes (reported, never cast), 670,790 rows, ID set equal to the template, no
   duplicates, no null/NaN/inf, 0 ≤ value ≤ 172,800, integer seconds
   ([runbook §2](final-submission-runbook.md)).
   **Result: all hard checks pass**: columns `MVT_ID_mvt` (`Float64`) and
   `TAXITIME_SEC_mvt` (`Int32`), 670,790 rows, ID set identical to the template,
   0 duplicates, 0 null/NaN/inf, values 0…110,578, all integer
   ([validator report](../analysis/final_tail_20261009/validate_vs_v12077.json)).
   The file with the E218 stage passes the same checks with the same range of values
   (0…110,578, mean 1,011.6 s, median 939 s;
   [validator report](../analysis/final_tail_20261011/validate_vs_v12081.txt)).
2. **Distribution checks**: per month × airport medians and 95th percentiles,
   share of rows equal to take-off minus a planning time ±15 s (on v12075 about
   2.4–3.5 % for schedule/IOBT/LOBT/EOBT and about 6 % for `AOBT_3` per month);
   February/June values far from January/July indicate a changed regime mixture.
   **Result**: mean/median per month 1,086/971 s (January), 985/922 s (February),
   984/927 s (June), 1,001/940 s (July). Shares equal to take-off minus a planning
   time: schedule 2.39–2.80 %, IOBT 2.89–3.21 %, LOBT 2.87–3.20 %, EOBT 3.33–3.89 %,
   `AOBT_3` 5.67–6.14 % per month. February is the highest month in every column
   (schedule 2.80 %, EOBT 3.89 %, `AOBT_3` 6.14 %); June lies within the
   January/July spread except EOBT (3.33 % against 3.35–3.59 %) and `AOBT_3`
   (6.04 % against 5.67–5.96 %). The full month × airport table (n, mean, median,
   95th percentile, maximum) is in the validator report. After the E218 stage the
   mean/median per month are 1,087/972 s (January), 985/922 s (February), 983/926 s
   (June) and 999/939 s (July), and the shares equal to take-off minus a planning time
   stay within 0.1 percentage point of the values above
   ([validator report](../analysis/final_tail_20261011/validate_vs_v12081.txt)).
3. **Reference check on the 344,841 January/July IDs**: RMS and maximum
   difference against the best file at assembly time. An RMS of exactly 0 would
   mean the rows were copied and is not acceptable.
   **Result** against v12077: 329,804 rows changed (95.64 %), RMS 26.5 s, maximum
   2,926 s. The expectation of RMS 16–25 s written beforehand
   (`analysis/win_program_20261008.md` §9) was stated against the canary, which has
   no E81 and no frozen rows; v12077 has both, so 26.5 s does not test it. Against
   H, the file v12077 without E81 and with the fresh 42 tail rows (section 7,
   item 4), the difference is RMS 20.36 s on 325,612 changed rows
   ([control explanation](../analysis/v12080_control_explanation_20261009.md)).
   The validator's Gram-identity line assumes that the reference scored
   230.9387 s, the v12075 score, while the reference file was v12077 (230.4812 s);
   that line is therefore not used. The board check in item 4 replaces it.
4. **Integration check on the public board: the control v12080.** The January/July
   rows of the final file (344,841 IDs, values unchanged, SHA-256 `ae1680cb…`)
   were uploaded once as the regular submission v12080. Before the upload the
   forecast was written into the journal and its line pinned by SHA-256 in the
   sealed control manifest (10:40 UTC; [manifest](../analysis/final_transport_20261009/manifest_control.json)):
   **231.7 s, corridor [231.2; 232.3] s**, from two routes, v12077 plus the
   measured E81 price plus the estimated fresh-tail effect (231.65) and the canary
   minus the measured member changes (231.74), with ±0.3 s for the noise of the
   refits and the new input dithering (log 439). The rule written with it: a result
   outside the corridor stops the final upload until a stage-by-stage comparison
   explains the difference. (The older forecast in this section, 232.2 s
   [231.8; 232.7], was written for the old members and weights and was replaced by
   this one.) Once v12079 was scored, the measured decomposition of section 4 gave
   231.745 s, inside the corridor (log 444).

   **Result: 231.1805 s**, processed at 16:32:54 UTC on 9 October (one PUT
   16:16:28–16:16:40 UTC from the Hetzner host). That is **0.02 s below the
   corridor**, on the favourable side, and 0.564 s better than the measured
   expectation. Under the pre-registered rule the final upload was paused (log 445).
   Label leakage was excluded first: in `final_ranking.parquet` the departure
   `BLOCK` and `TAXITIME` fields are null in all four months, as in
   `ranking.parquet`.

   **Explanation, without labels and without fits** (log 446;
   [report](../analysis/v12080_control_explanation_20261009.md)). The reference
   point is a real file, H = v12078 with the 42 fresh rows of v12079; its score
   follows exactly from the Gram identity, √(231.0488² + 231.1770² − 230.4812²) =
   231.7429 s, because v12079 and v12077 differ on exactly those 42 rows. The final
   rows F differ from H by RMS 20.36 s on 325,612 rows, ΔMSE −260.3 s². Each group
   of inputs was swapped back to its v12077-era version one at a time and run
   through the frozen final chain; the identity run reproduces all twelve stages
   byte for byte. Share of the difference (projection on F − H):

   | Source | Share |
   |---|---:|
   | Regime probabilities of the base classifier (refit on the four-month view of the `plan_*` medians, and the re-dithered January/July inputs) | 55 % |
   | Neural member (fresh fits on the final view) | 23 % |
   | Tabular members (CatBoost, LightGBM, anchors) | 16 % |
   | Daily ADS-B products rebuilt on the final movement times | 2 % |
   | Candidate dithering in the premix; movement-only member | 0.4 % each |
   | 42 tail rows (final E156 against the canary's E156) | 0.4 % |
   | Remainder (dithering inside stage features, E167, LSZH) | 4 % |

   Single swaps do not add exactly because the mixture is non-linear; the joint
   swaps close the decomposition at 99.9 %. The difference is concentrated: 345
   rows (0.1 %) carry 61.7 % of its squared size, and LIRF in July alone 53.5 %.
   On these rows (185 of them LIRF in July) the prediction is a mixture of regime
   candidates whose median spread is 2,818 s, and 87 % of the change comes from
   the probabilities.
   Nothing points to an error: the members show the same seed spread as before
   (CatBoost 32.6 s against 32.6 s), the classifier has the same parameters, trees,
   features and training rows, and the stages serve almost the same rows (E141
   222,740 against 222,660). The noise of such a control, calibrated label-free on
   the 2025 January + July out-of-fold predictions, is σ ≈ 200–250 s² in MSE, that is
   **±0.43–0.55 s in RMSE (1σ)**; the observed deviation is a 1.0–1.3σ event. v12079
   and v12080 differ by RMS 25.3 s and scored 231.1770 and 231.1805 s.

   **Lesson.** The ±0.3 s corridor was about half as wide as it should have been.
   The control of a chain whose members and classifier were refitted carries about
   ±0.5 s of noise on January/July, and the ±0.08 s dithering estimate of
   section 2.4 does not cover the refitted classifier's response to new inputs. The
   rule did what it was written for: it stopped the upload until the deviation was
   explained. The same draw is not guaranteed to fall with the same sign on the
   blind months.

   **Decision** (log 447): explained, no leakage, no substituted member; the recipe
   is unchanged and the file `76c32426…` remains the only final candidate.

   **Second control, after the E218 stage (v12082, 11 October).** The January/July
   rows of the rebuilt file (SHA-256 `e87b7978…`) were uploaded once with a forecast
   of 231.144 s, corridor [230.994; 231.294] (±0.15 s, the coordinator's minimum after
   the lesson above), and the rule written with it: worse than v12080 (231.1805 s) ⇒
   the final stays `76c32426…`; otherwise the E218 file becomes the final. **Result:
   231.1445 s**, inside the corridor, −0.0360 s against v12080 (log 458, 459;
   [record](../analysis/v12082_control_20261011/RESULTS.md)). Because the two files
   differ only by the E218 shift on the same labels, this is an exact measurement of
   the stage on the final's January/July rows, not a test of the refits. Under the rule
   the final is now the E218 file `bbae2cc5…` (section 2.5). A third upload,
   v12083 (231.1749 s), removed the stage on the LIRF rows only; under its rule the
   LIRF cells stay in the final (log 460, 461).
5. **Determinism**: `assemble --repeat-e156` must reproduce all ten TabPFN
   probability arrays bitwise; cached-model replay reports `fits_performed = 0`
   unless `--refit` was requested.
   **Result**: the repeat gave identical probabilities for all ten seeds (maximum
   absolute difference 0.0) and 0 differing rounded rows; all cached stages report
   `fits_performed = 0` ([E156 receipt](../analysis/final_tail_20261009/receipts/03_e156.json)).
   An independent rebuild from scratch, a new `prepare` and `assemble` from the same
   contract into a separate work directory, gave the byte-identical file, SHA-256
   `76c32426…d554` ([replay log](../analysis/final_tail_20261009/assemble_replay.log),
   log 438). That rebuild ran on the same Mac. A replay on a second machine
   (prc-cloud, ARM64 Linux, the same 4,025 pinned files verified by SHA-256) did **not**
   reproduce the file byte for byte, for two reasons that are limits of portability
   rather than errors ([replay report](../analysis/final_replay_prc_cloud_20261010.md),
   log 453–454): premix, E185 and E167 matched on all 670,790 rows (classifier
   probabilities differ by about one unit in the last place and round to the same
   values), but the TabPFN-2 tail ensemble gives probabilities up to 0.010 apart with
   identical inputs and package versions, because the Mac uses Apple Accelerate and
   the ARM server OpenBLAS/NVPL (55 of the 62 rounded tail values differ, by at most
   35 s; each machine is bitwise repeatable with itself); and the cached E141 stage
   refused to run because the Python build string differs (build date only), which is
   the signature check working as designed. The final file is therefore reproducible
   byte for byte on the build machine, and to within the TabPFN numerical-backend
   difference elsewhere.
   **With the E218 stage (11 October)**: the E156 repeat was again bitwise identical
   with the same 62 served rows, and a second assembly in a fresh workspace gave the
   same SHA-256 `bbae2cc5…34a1`, receipts 00–12 equal except timing and the E188
   workspace paths ([receipt parity](../analysis/final_tail_20261011/receipt_parity.json),
   log 458). The second-machine replay was run on the 9 October file, whose stages 0–11
   are those of the current final; the E218 stage was not replayed on a second
   platform.
6. **Label-free diagnostics** of the new months. Planned beforehand: rows near the
   3,606 s, 4 h and LOBT-window thresholds, the no-NM floor audit
   (`analysis/nonm_disruption_floor_20261008/audit_2026.py`) and regime-probability
   means per airport × month against 2025. No separate report of these three was
   produced for the final file. The label-free evidence that exists is the
   [input audit](../analysis/final_months_input_audit_20261009.md) of the sealed
   four-month feature table against 2025 (section 5), the validator's per
   month × airport table and fallback-regime shares (items 1–2), and the
   stage-by-stage control explanation (item 4).
7. **Manifest** ([`MANIFEST.json`](../analysis/final_tail_20261011/MANIFEST.json)):
   SHA-256 of the file (`bbae2cc5…`), the template (`b3445d80…`), the ranking file
   (`4b0692a7…`), the builder (`21f46756…`), the contract (`7a1cde30…`) and
   `complete.json` (`b708f42d…`); `complete.json` pins the thirteen stage receipts,
   which pin the cached stage models, and the contract pins 4,027 input files by
   SHA-256, including the saved base classifier, the merged predictions of every
   member and the two PRU files. An `e218` block records the flag, the patch
   (`0613fdd9…`), both PRU files, the parent file `76c32426…`, the rows changed
   (415,697), the RMS change (4.687 s) and the dry-run SHA-256. It also records the
   validator report, the replay SHA-256, the git commit of the frozen pipeline
   (`f9694de3…`), the journal entries (457–459), the machine and the UTC time
   (09:07:57 on 11 October 2026). The transport manifest for the upload is
   [`analysis/final_transport_20261009/manifest_final_v2.json`](../analysis/final_transport_20261009/manifest_final_v2.json).
   The predecessor's [`MANIFEST.json`](../analysis/final_tail_20261009/MANIFEST.json)
   (file `76c32426…`, contract of 4,025 files, git commit `2d85c07…`, 9 October
   09:12:29 UTC) is kept unchanged.

## 8. Fallbacks written down in advance

| Situation | Behaviour |
|---|---|
| No OPDI event for an LSZH row (all of February and June if the extraction fails) | Row unchanged by stage 10; the 50 % blend then returns the stage-9 value. The stage is applied identically in all months; its coverage per month is reported |
| Defective archive day (2026-06-11) | Processed as produced; stages inert where no track exists; no interpolation |
| Coverage collapse (EDDF Feb, LIRF Jun/Jul) | Stages E141/E64/E188 inert by their masks (verified: 0 rows changed outside the masks); the neural member keeps its full weight (0.35) on every row because the protection rule failed its gate (section 5; the gate was evaluated at 0.25) |
| Cached stage model does not match its training signature | Run stops; explicit `--refit e141 e64 lszh` with fresh signatures recorded, no silent fit |
| Partial seed set in any member family | Refused by the contract (`members` must cover every ID exactly once) |
| E218 flag and contract disagree, or a PRU file differs from its pinned SHA-256 | `admit` and the stage refuse to run (log 457). The predecessor `76c32426…` and its manifest `manifest_final.json` are kept; returning to it needs the old builder `9ceab78e…` back on the upload host (kept in git at `185df32` and in a backup on Hetzner) and a new preflight, as written in the rule of log 458 |
| Pipeline not assembled by 11 October 12:00 UTC | Pre-registered backup: `src/run_final_any_period.sh` with `E81=drop TABPFN_SEEDS=0..9` (the older v808 line, 233.5877 s on the board, period-independent) (`analysis/win_program_20261008.md` §2; [strategy §1](../analysis/final_strategy_20261008.md)). Not triggered: the pipeline was assembled on 9 October (section 6) |
| Upload failure | [Runbook §6](final-submission-runbook.md): reconcile the ledger before any second PUT; never a second final name |

## 9. What is deliberately not in the final

- **E81 and the 42 frozen rows** (sections 2–4).
- **E207 v3** (full-token trajectory encoding): rejected on 8 October after 18
  fits, pooled body +0.007 s / decided +0.016 s ([negative results](negative-results.md)).
- **E209** (floor for no-NM rows at the neighbours' live level): rejected,
  1 of 6 folds, +0.018 s ([report](../analysis/nonm_disruption_floor_20261008.md)).
- **E211** (level correction by our own chain's residuals at neighbouring departures,
  with exponential time decay and airport × day centring; the method of a CityLearn 2023
  forecasting solution and of team smart-jigsaw's public notes): it passed the
  registered month-pair gate (decided −0.102 s, 5 of 6 folds), but the whole gain came
  from LTFM in February 2025, and every forward-in-time check made it worse
  (+0.08 to +0.16 s). The final applies 2025 coefficients to later months, which is the
  forward case, so the stage stays off (log 399;
  [report](../analysis/neighbour_residual_level_20261008.md)).
- **Neural-member protection at coverage collapse**: failed its gate, 0 of 6 folds
  (section 5).
- **E210** (20 TabPFN seeds, or averaging the tail ensemble over ±6 s shifts of the
  inputs): leave-one-month-out on the 2025 tail, all three variants failed the gate
  registered before the run (squared error +0.005 % to +0.13 % against 10 seeds, June
  worse in every variant); stage 3 stays at 10 seeds (log 397b, 403).
- **Arrival-based day state** (E213): the share of arrivals whose on-block equals the
  schedule, and the day's NM-versus-APDF clock offset from arrivals, add nothing beyond
  what the regime classifier and the chain already use (partial correlations ≤ 0.013
  at LIRF, LFPG, LTFM and EGLL), so no rule was built (log 407c, 411).
- **Aircraft identified by stand occupancy** for departures without an own track: on
  three 2025 days 2 departures without an NM record received a candidate, and 0 on
  three 2026 days; ground reception at the stands is missing, so the idea was closed
  before any cross-validation (log 405b–407b).
- **E215** (12 training epochs for the neural member instead of 8; 18 fits, exact
  2025 out-of-fold chain at the final weight 0.35): pooled −0.107 s on `decided`
  rows and 11 of 12 months better, but the registered gate failed on the month pair
  2+8 (+0.610 s) and on January–June (+0.055 s). The whole loss is February
  (+1.61 s on the chain; all three 12-epoch seeds worse there), the regime month in
  which the chain leans on the neural member most. The member stays at 8 epochs and
  no retrain was registered (log 428, 429;
  [report](../analysis/e215_nn_epochs_20261008.md)).
- **E216** (the remaining global scalars of the chain re-checked one at a time from
  the final configuration: CatBoost share in the tree pair, the block of four
  anchored members, the raw-target member, the movement-only member weight and the
  E152 gate threshold; 15 values, exact 2025 out-of-fold chain): none of the 15
  values is lower than the current one in all twelve months, both half-years and
  pooled, which the rule registered beforehand required (0 of 15). The best pooled
  gain was −0.013 s (CatBoost share 0.75, better in 6 of 12 months); one grid step
  moves the pooled RMSE by 0.004–0.016 s against 0.18 s for the neural weight in
  E212, and February again pulls in the opposite direction to most other months.
  No chain scalar besides the neural weight was mis-set; the final configuration is
  confirmed unchanged (log 442, 443;
  [report](../analysis/e216_chain_scalars_20261009.md)).
- **E217** (a "winter" weight of the neural member: 0.55 instead of 0.35 in December
  to March, chosen on January–March 2025 and checked forward on December 2025):
  December improved (−0.083 s) but January 2025 got worse (+0.124 s), so the rule
  registered beforehand failed; the expected effect on the final came entirely from
  February. The weight stays a constant 0.35 (log 449, 450;
  [report](../analysis/e217_winter_nn_weight_20261010.md)).
- **E188** stays although its measured contribution is small (CV −0.007 s, 3/6;
  board −0.0011 s): removing it would be a design change ([validation](validation.md)).
- No weights, seeds or rows are chosen on the public board for the final: the
  ten-seed mean was decided before the seeds were trained and the 0.35 weight on
  2025 out-of-fold data before v12077 was built (section 2.3). The two uploads of
  8 October (v12076, v12077), the two measurement uploads of 9 October (v12078,
  v12079, section 4) and the control v12080 (section 7, item 4) are checks, not
  selections: the control fell outside its corridor on the favourable side and the
  recipe was not changed (log 447). The one exception is the **E218 stage**: its rule
  and its parameters (k = 0.5, the 0.5-point de-icing threshold) were fixed and gated
  on 2025 data, but whether it entered the final was decided by one board upload
  (v12081) under a rule written before that upload, confirmed by the control
  v12082 under a second rule, and its LIRF cells were kept after v12083 under a
  third (section 2.5). Its input is EUROCONTROL's published
  monthly aggregate for the scored months; nothing was tuned on the board.

## 10. Results

- **Final file**: `gentle-igloo_final.parquet`, SHA-256
  `bbae2cc544989a0faf68abb85dbb4a9aab5ea38eec98d01fa9282c332b4234a1`,
  2,145,576 bytes, 670,790 rows; rebuilt on the Mac on 11 October 2026 in 231 s
  with 0 fits from git commit `f9694de3…`, builder `21f46756…`, with
  `E218_PRU_LEVEL=1` (log 457, 458). It is the 9 October file plus stage 12
  (section 2.5): stages up to `half` are byte-identical, and stage 12 changes
  415,697 rows by RMS 4.687 s.
- **Predecessor**: SHA-256
  `76c32426d750040690ff68bacd6ecbf3a7aa1e63340df5b9271a324b7064d554`, 2,145,126
  bytes, assembled on 9 October 2026 in 157.6 s with 0 fits from git commit
  `2d85c07…` (log 436, 438); final candidate until the control v12082 (log 459).
- **Validator**: all hard checks pass
  ([`analysis/final_tail_20261011/validate_vs_v12081.json`](../analysis/final_tail_20261011/validate_vs_v12081.json));
  **manifest**: [`analysis/final_tail_20261011/MANIFEST.json`](../analysis/final_tail_20261011/MANIFEST.json);
  receipts, contract and `complete.json` in the same directory; **transport
  manifest**: [`analysis/final_transport_20261009/manifest_final_v2.json`](../analysis/final_transport_20261009/manifest_final_v2.json)
  (SHA-256 `3648f566…`, day 2026-10-12; preflight passed on the Mac and on Hetzner).
- **Replay**: an independent rebuild from scratch from the same contract gave the
  byte-identical file, SHA-256 `bbae2cc5…34a1`
  ([replay log](../analysis/final_tail_20261011/assemble_replay.log)); the same held
  for the predecessor on 9 October. A replay of the predecessor on a second platform
  (prc-cloud, Linux ARM64) matched stages 0–2 by value, differed in 55 of the 62
  TabPFN tail values of stage 3 (numerical backend) and stopped at stage 4 on the
  Python build string, by design (section 7, item 5).
- **January/July rows on the public board**: control v12080 (predecessor)
  231.1805 s, 0.02 s below the corridor written beforehand, explained stage by stage
  (section 7, item 4; log 445–447); control v12082 (current file) 231.1445 s, inside
  its corridor, −0.0360 s against v12080 (log 459); v12083 (the current file
  without the E218 shift on LIRF rows) 231.1749 s, +0.0304 s against v12082 (log 461).
- **Best regular submission**: v12081 (v12077 plus the E218 stage), 230.4478 s, 18th of
  242 teams at 08:52 UTC on 11 October 2026 and still at 09:30 UTC
  ([record](../analysis/v12081_upload_20261011/RESULTS.md), log 456).
- **Per month × airport distribution and label-free diagnostics**: the validator
  report (month × airport table, fallback-regime shares) and the
  [input audit](../analysis/final_months_input_audit_20261009.md) (sections 5, 7).
- Upload receipt (ledger lines, bucket listing, SHA-256 of the object):
  `[[to be filled after the final upload]]`.
- Blind-ranking result (January/July, February/June, four months), published by
  the organiser after 14 October 10:00 UTC: `[[to be filled after 14 October]]`.

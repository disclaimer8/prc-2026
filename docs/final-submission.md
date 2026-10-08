# Final blind submission: `gentle-igloo_final.parquet`

This page describes the one file that team gentle-igloo submits for the blind
final ranking: the rules it answers, the recipe that produces it, why that recipe
differs from the best leaderboard file (v12075), what the difference was measured
to cost, what is known about the new months' inputs, and how the file is built,
validated and uploaded. Numbers are taken from the dated analysis reports they
cite; items marked `[[to be filled after the final assembly]]` do not exist yet.
Written on 8 October 2026 while the final model fits were still running.

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
([model design](model-design.md)) with two parts removed and one part widened.
Protocol name: `final-uniform-tail-E156fresh-noE81-fourmonths-20261008-v1`
([tail-assembly protocol](../analysis/final_tail_assembly_20261008/PROTOCOL.md),
log entry 389).

| | v12075 (leaderboard best) | Final file |
|---|---|---|
| Feature table | FIN1b built on 2025 + the January/July ranking file | FIN1b rebuilt on 2025 + the four-month `final_ranking.parquet`: 2,755,837 rows = 2,085,047 training + 670,790 final (log 385) |
| Base classifier and regressor, CatBoost 15 seeds, LightGBM 5 seeds, five anchored members, movement-only member | Fitted on the January/July view | Refitted with the same rounds, seeds and weights on the new view: 52 fits (log 388). Fitted models are saved this time |
| Neural member (ordered trajectories, E203), weight 0.25 | Mean of 3 seeds | Mean of **10 seeds** (0–9), same recipe and 8 epochs; the 3-seed mean is kept for comparison only (log 394, 400) |
| Whole-file blend with v44 and v60 (E81, weights 0.1214 / 0.0270) | Present | **Absent.** The donors are archived January/July leaderboard files; nothing equivalent exists for February and June |
| LIRF tail without NM record | 42 predictions frozen from v361 (a single TabPFN draw) and protected | **Fresh 10-seed TabPFN-2 ensemble (E156, airline prefix and local date) on every row of the tail segment in all four months**; every served row is protected from later stages |
| Surface stages E141, E64, E188; gate E152; LOBT lift and clip; LSZH OPDI stage at 50 % | Present | Present and unchanged (cached models replayed only on an exact training signature) |
| Input timestamps | Old ranking file | `final_ranking.parquet` as published, including its ±12 s shifts |

### 2.1 Stage order

From the literal `stage_plan()` of the assembly adapter
([protocol](../analysis/final_tail_assembly_20261008/PROTOCOL.md)); weights as in
[model design](model-design.md).

| # | Stage | What it does in the final |
|---|---|---|
| 0 | premix | Raw regime probabilities recomputed from the saved classifier; regular candidate = 19/60 × (0.65 CatBoost15 + 0.35 LightGBM5) + 1/30 raw-target member + 0.10 each of the LOBT-, schedule-, EOBT- and AOBT-anchored members + 0.25 neural member; probability-weighted mixture over regime candidates |
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

### 2.3 The ten-seed neural member

Seeds 3–9 were trained after seeds 0–2 with the unchanged recipe (width 512,
PLE 48 bins, 2 × 64 tokens, batch 4096, AdamW, 8 epochs), 270–274 s per seed on
the AMD Radeon RX 7900 XTX, each checkpoint replayed exactly (log 393, 400). On the
670,790 final rows the 10-seed mean differs from the 3-seed mean by RMS 23.98 s
with a mean shift of −0.28 s; the pairwise RMS between single seeds is 70.6 s
(0.20–0.24 in units of the residual scale)
([statistics](../analysis/final_ordered_trajectory_20261008/execution_10seed/statistics.json)).
This is a variance reduction with no added capacity; the member weight stays 0.25.
The decision to use the 10-seed mean was written before the extra seeds were
trained (log 394).

### 2.4 The ±12 s shifts

The shifts are not new noise: every challenge file, including all of 2025, carries
the same dithering of the movement time (`T − η/2`) and block time (`T + η/2`), so
the label carries `η ~ U(−12, 12)`; `final_ranking` is a fresh independent draw
([jitter analysis](../analysis/final_jitter_sensitivity_20261008.md)). The regime
tolerance of 6 s and the 3,606 s LOBT window are the half-width of that dithering.
Expected effect on RMSE: 0; the realised deviation on January/July is ±0.08 s
(1σ) for the production chain. No augmentation, tolerance change or de-noising was
added; the input is used as published.

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
  weights serve all rows; nothing is chosen per month.

## 4. Measured cost on January/July

The uniform recipe was run on the v12075-era January/July artifacts (old members,
E81 removed, fresh E156) and uploaded once as a technical canary, `v12090`
([canary report](../analysis/v12090_upload_20261008/RESULTS.md), log 390–391):

| | RMSE on January/July 2026 (s) |
|---|---:|
| v12075 (best, with E81 and frozen rows) | 230.9387 |
| v12090 (same chain, no E81, fresh E156 on the 42 rows) | **232.1933** |
| Difference | +1.2546 |

The forecast written before the upload was 231.6 s with a corridor of
[231.1; 232.3] s; the result is inside the corridor near its upper edge. The
historical price of E81 alone (about 0.66 s) explains roughly half of the change;
the remainder is the fresh ensemble against the frozen donors on the 42 rows (RMS
713.5 s between them) and/or a stronger E81 effect on this chain. Against v12075
the file changed 313,616 of 344,841 rows with RMS 18.209 s. The canary is a
technical check of the code path on real labels, not a candidate for best; no
further probing uploads were made. The two Jan/Jul labels do not transfer
one-to-one to the blind months, but the sign of the cost is now known and
accepted.

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
[coverage audit](../analysis/coverage_collapse_audit_20261008.md)). The audit estimates
the cost of the lost coverage at about +0.18 s (range +0.1 to +0.3 s) on the final; it is
a cost of missing data that no rule recovers.

## 6. Pipeline: scripts, inputs and receipts

Every step is a dated script registered in `HYPOTHESES_LOG.md` before it ran, with
tests that use synthetic rows only, and a receipt directory under `analysis/`.
Interpreter on the Mac: `.venv/bin/python`.

| # | Step | Script | Status on 8 Oct, receipts |
|---|---|---|---|
| 1 | Sealed four-month feature workspace: FIN1b, E106/E113 prefix tables, motion view | `scripts/final_feature_workspace_20261008.py`, `scripts/final_feature_mac_20261008.py` | Done on the Mac: 11 stages, 64.89 s, peak 12.53 GiB; WORK `data/final_feature_work_mac_20261008`, FIN1b SHA `2bb90b76…` (log 385; [protocol](../analysis/final_feature_workspace_20261008/PROTOCOL.md)) |
| 2 | 52 tabular fits with saved models: classifier (180 rounds) + regressor (2,319), CatBoost 15 × 5,600, LightGBM 5 × 2,319, anchors raw 2,565 / LOBT 3,065 / schedule 2,958 / EOBT 2,736 / AOBT 2,738 × 5 seeds, motion 5 × 2,481 | `scripts/final_tabular_models_20261008.py` | Running on prc-cloud (ARM, 15 CatBoost + 10 LightGBM) and hetzner (26 LightGBM); cross-node parity on a 20-round probe: max abs diff 7.96e-13 over 670,790 predictions ([execution journal](../analysis/final_tabular_models_20261008/EXECUTION.md), [contract](../analysis/final_tabular_models_20261008/CONTRACT.json), log 388). Completion: `[[to be filled after the final assembly]]` |
| 3 | Neural member: 2025 token reuse (364 days) + 120 final days re-encoded on final times; seeds 0–2, then 3–9 | `scripts/final_ordered_trajectory_20261008.py`, `scripts/final_ordered_trajectory_seeds_20261008.py` | Done on the Home GPU; 3-seed `prediction.npz` SHA `6d50f9cb…`, 10-seed `prediction_10seed.npz`; exact checkpoint replay ([receipts](../analysis/final_ordered_trajectory_20261008/), log 387, 393, 394, 400) |
| 4 | Seven daily ADS-B products for 120 days on final movement times | `scripts/final_adsb_derivatives_20261008.py` | Done on the Mac: 203.9 s, 120/120 days, 840 files, 114 MB, `MANIFEST.sha256` SHA `a3240876…` ([report](../analysis/final_adsb_derivatives_20261008/RESULTS.md), log 392) |
| 5 | OPDI events for February and June 2026 (LSZH stage input) | `scripts/opdi_final_extract_20261008.py` | Running on prc-cloud; local `data/opdi` covers 31 Dec–9 Feb and 29 Jun–8 Aug only (log 398). Rows of LSZH without a matched event stay unchanged. Count of LSZH rows with an event per month: `[[to be filled after the final assembly]]` |
| 6 | Tail assembly: premix, stages 1–11, served-ID manifest, `complete.json` | `scripts/final_tail_assembly_20261008.py` (`make-contract --mode final`, `prepare`, `assemble --repeat-e156`) | Adapter done and dry-run on January/July (141 s, 0 fits, premix byte-identical to the archived v12075 premix; [dry run](../analysis/final_tail_assembly_20261008/DRYRUN_RESULTS.md), log 389). Final run: `[[to be filled after the final assembly]]` |
| 7 | Validator | `scripts/validate_final_submission.py --file … --reference submissions/gentle-igloo_v12075.parquet --json …` | Written; used on the canary (all hard checks passed). Final: `[[to be filled after the final assembly]]` |
| 8 | Single PUT to the bucket | New dated wrapper built on `OnePutHTTP` and `single_attempt`, per [runbook §3](final-submission-runbook.md) | Planned for 13 October ≤ 20:00 UTC from the Hetzner host. Receipt: `[[to be filled after the final assembly]]` |

Cached stage models (E141 5, E64 5, LSZH 1, E188 5) are loaded only when the
reconstructed training call matches the stored signature (parameters, rounds,
data/label/weight hashes, feature order, package and interpreter versions);
otherwise the run stops before any fit, and a refit is a separate explicit action
(`--refit`). The TabPFN-2 checkpoint is pinned by SHA in the contract after a macOS
cache purge deleted the cached copy on 8 October
([protocol](../analysis/final_tail_assembly_20261008/PROTOCOL.md)).

Unmeasured until the final run: wall-clock of the four-month assembly on the Mac
(the January/July dry run took 141 s; the v12075 replay of the same stages peaked at
7.9 GiB), and the fit times of step 2 per family: `[[to be filled after the final assembly]]`.

## 7. Validation of the file

1. **Hard checks** (exit code 0 required): object name, column names and order,
   dtypes (reported, never cast), 670,790 rows, ID set equal to the template, no
   duplicates, no null/NaN/inf, 0 ≤ value ≤ 172,800, integer seconds
   ([runbook §2](final-submission-runbook.md)).
2. **Distribution checks**: per month × airport medians and 95th percentiles,
   share of rows equal to take-off minus a planning time ±15 s (on v12075 about
   2.4–3.5 % for schedule/IOBT/LOBT/EOBT and about 6 % for `AOBT_3` per month);
   February/June values far from January/July indicate a changed regime mixture.
3. **Reference check against v12075 on the 344,841 January/July IDs**: RMS and
   maximum difference, Gram-identity range of the January/July score. The canary
   had RMS 18.209 s; the final file should differ from the canary by the dithering
   and the refits, expected RMS 16–25 s (`analysis/win_program_20261008.md` §9);
   an RMS of exactly 0 would mean the rows were copied and is not acceptable.
   Result: `[[to be filled after the final assembly]]`.
4. **Integration check on the public board** (planned, one regular slot): the
   January/July rows of the assembled four-month file uploaded as a regular
   submission. Forecast written in advance: 232.2 s, corridor [231.8; 232.7] s;
   a result outside the corridor stops the process and triggers a stage-by-stage
   comparison with v12090 (`analysis/win_program_20261008.md` §1, item 2).
   Result: `[[to be filled after the final assembly]]`.
5. **Determinism**: `assemble --repeat-e156` must reproduce all ten TabPFN
   probability arrays bitwise; cached-model replay reports `fits_performed = 0`
   unless `--refit` was requested. Planned: an independent replay of the assembly
   on a second node. Result: `[[to be filled after the final assembly]]`.
6. **Label-free diagnostics** on February/June: rows within 12 s of the 3,606 s,
   4 h and LOBT-window thresholds (January/July: 0, and 79 → 10 after the shifts);
   the no-NM floor audit `analysis/nonm_disruption_floor_20261008/audit_2026.py`
   (share of no-NM rows below the neighbours' level; diagnostic only, the rule E209
   was rejected); regime-probability means per airport × month against 2025
   (`scripts/regime_quality_report.py export`). Results: `[[to be filled after the final assembly]]`.
7. **Manifest** (`MANIFEST.json`): SHA-256 of the file, the template, the ranking
   file, every saved model and receipt, the git commit of the frozen pipeline, the
   journal entry, machine and UTC time ([runbook §2](final-submission-runbook.md)).

## 8. Fallbacks written down in advance

| Situation | Behaviour |
|---|---|
| No OPDI event for an LSZH row (all of February and June if the extraction fails) | Row unchanged by stage 10; the 50 % blend then returns the stage-9 value. The stage is applied identically in all months; its coverage per month is reported |
| Defective archive day (2026-06-11) | Processed as produced; stages inert where no track exists; no interpolation |
| Coverage collapse (EDDF Feb, LIRF Jun/Jul) | Stages E141/E64/E188 inert by their masks (verified: 0 rows changed outside the masks); the neural member stays at weight 0.25 because the protection rule failed its gate (section 5) |
| Cached stage model does not match its training signature | Run stops; explicit `--refit e141 e64 lszh` with fresh signatures recorded, no silent fit |
| Partial seed set in any member family | Refused by the contract (`members` must cover every ID exactly once) |
| Pipeline not assembled by 11 October 12:00 UTC | Pre-registered backup: `src/run_final_any_period.sh` with `E81=drop TABPFN_SEEDS=0..9` (the older v808 line, 233.5877 s on the board, period-independent) (`analysis/win_program_20261008.md` §2; [strategy §1](../analysis/final_strategy_20261008.md)) |
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
- **E188** stays although its measured contribution is small (CV −0.007 s, 3/6;
  board −0.0011 s): removing it would be a design change ([validation](validation.md)).
- No weights, seeds or rows are chosen on the public board for the final; the
  Jan/Jul slot in section 7 is a check, not a selection.

## 10. Results

- Final file: SHA-256, size, validator report, manifest: `[[to be filled after the final assembly]]`.
- Per month × airport distribution and the three label-free diagnostics: `[[to be filled after the final assembly]]`.
- Upload receipt (ledger lines, bucket listing, SHA-256 of the object): `[[to be filled after the final assembly]]`.
- Blind-ranking result (January/July, February/June, four months), published by
  the organiser after 14 October 10:00 UTC: `[[to be filled after the final assembly]]`.

# Planning timestamp alignment and stand-evidence results

**Registered retrospective analysis #464.** Registration commit: `bf08d014000bd51ad65ae045be644ab0629f6c8f`. Exactly one outcome run completed; frozen source, tests, protocol and preflight were unchanged. Inputs and all ten original output hashes verify. The pre-run direct synthetic suite passed 21/21.

The known LIRF alignment concentration and LEBL July–September episode replicate under fixed minute-preserving controls. Derived geometry supplies directionally consistent, selected-subset evidence. The stricter raw stationary-prefix audit is **inconclusive because no episode passed its admission gates**. These findings do not establish physical AOBT, copying causality or detector precision.

## Population and timestamp contract

From 2,085,047 departures in 2025, the primary population contains **2,084,663** rows; 384 durations are nonfinite or outside 0–86,400 seconds. All primary rows have at least one planning timestamp. Implied BLOCK and supplied BLOCK agree within one second on all 2,085,047 comparable rows. This verifies representation, not physical truth.

At ±6 seconds, 148,452 rows (7.1212%) align with at least one of SCHED, IOBT, LOBT or EOBT_1. Of those, 109,555 (73.80%) align with multiple fields, precluding attribution to a unique copied field. The all-airport mean shifted rate is 4.9747%, giving an excess of 2.1465 percentage points (pp). Controls shift all available timestamps on the same row by −180, −120, −60, +60, +120 and +180 seconds; denominators remain identical.

## All primary airports

Union of four timestamps, tolerance ±6 s. “All controls” means the zero-offset rate exceeds each of the six fixed controls; it is not a significance test.

| Airport | N | Matches | Zero % | Mean control % | Excess pp | Above all controls |
|---|---:|---:|---:|---:|---:|---|
| EDDF | 230,141 | 2,815 | 1.2232 | 1.2056 | +0.0176 | no |
| EDDM | 167,334 | 13,118 | 7.8394 | 7.5588 | +0.2806 | no |
| EGLL | 239,546 | 20,321 | 8.4831 | 6.9200 | +1.5632 | yes |
| EHAM | 247,948 | 3,148 | 1.2696 | 1.1637 | +0.1059 | no |
| LEBL | 179,700 | 16,642 | 9.2610 | 6.6541 | +2.6069 | yes |
| LEMD | 212,242 | 16,326 | 7.6922 | 6.2210 | +1.4712 | no |
| LFPG | 239,491 | 16,875 | 7.0462 | 5.8459 | +1.2003 | yes |
| LIRF | 160,690 | 38,785 | 24.1365 | 6.4986 | +17.6379 | yes |
| LSZH | 134,608 | 1,513 | 1.1240 | 1.5031 | -0.3791 | no |
| LTFM | 272,963 | 18,909 | 6.9273 | 6.2988 | +0.6285 | no |

LIRF exceeds every control in all 12 months (monthly excess 14.899–20.208 pp). The SCHED-only rate is 18.1019% versus 2.5269% in controls (+15.5750 pp); the union gives 24.1365% versus 6.4986%. LIRF remains positively enriched across recorded delay and coverage strata and with or without NM input, but the no-NM stratum is small (1,475 rows). Full per-anchor, airport-month, delay, NM and coverage results are retained in `alignment.csv` and `alignment_contrasts.csv`.

## Predetermined LEBL windows

| Window | N | Matches | Zero % | Mean control % | Excess pp |
|---|---:|---:|---:|---:|---:|
| January–June | 86,742 | 7,376 | 8.5034 | 7.2762 | +1.2272 |
| July–September | 49,491 | 5,474 | 11.0606 | 5.4855 | +5.5751 |
| October–December | 43,467 | 3,792 | 8.7239 | 6.7430 | +1.9808 |

For the fixed input-delay >30-minute subset, the same windows have excesses +0.5774, +4.2812 and +1.1667 pp, respectively. These are the preselected previously observed episode and comparison windows, not a new event search.

## Descriptive tolerance sensitivity

| Airport | Tolerance s | Zero % | Mean control % | Excess pp |
|---|---:|---:|---:|---:|
| LIRF | 1 | 5.9842 | 1.6332 | +4.3510 |
| LIRF | 3 | 14.0363 | 3.7909 | +10.2454 |
| LIRF | 6 | 24.1365 | 6.4986 | +17.6379 |
| LIRF | 12 | 24.1365 | 6.4986 | +17.6379 |
| LIRF | 30 | 24.1365 | 6.4986 | +17.6379 |
| LEBL | 1 | 2.3472 | 1.6694 | +0.6779 |
| LEBL | 3 | 5.4302 | 3.9028 | +1.5274 |
| LEBL | 6 | 9.2610 | 6.6541 | +2.6069 |
| LEBL | 12 | 9.2615 | 6.6553 | +2.6063 |
| LEBL | 30 | 9.2643 | 6.6582 | +2.6061 |

LIRF counts plateau exactly from 6 to 30 seconds (38,785 matches); this is compatible with timestamp construction/quantisation and cannot validate physical copying. At 30 seconds adjacent control windows touch and are not independent.

## Derived stand geometry

Geometry admission is label-blind. Of 2,085,047 departures, 2,080,332 have each derived product (31 December is absent), 1,210,091 have a known stand, and 596,461 were seen near it. The fixed strict geometry filter retains 160,249 rows, including **160,240 primary rows (7.69%)**: 6,152 aligned and 154,088 unaligned. These are selected observation-rich visits, not a representative airport sample.

Outcome: first observed near-stand time ≥300 s after recorded BLOCK. Risk differences compare aligned minus unaligned visits, standardized within airport × month × input-delay × NM cells with ≥10 per arm. The reporting gate additionally requires ≥50 retained per arm, ≥3 cells and ≥3 months. No confidence intervals or population-level inference were registered.

| Scope | Retained aligned / unaligned | Cells | Months | Discarded | Difference pp | Status |
|---|---:|---:|---:|---:|---:|---|
| ALL | 5,606 / 114,612 | 97 | 12 | 40,022 | +1.0086 | descriptive positive |
| EDDF | 11 / 883 | 1 | 1 | 8,263 | +0.0000 | inconclusive |
| EDDM | 3,354 / 34,599 | 28 | 12 | 1,846 | +0.0074 | descriptive positive |
| EGLL | 90 / 679 | 5 | 5 | 657 | +4.3780 | descriptive positive |
| EHAM | 871 / 58,655 | 16 | 12 | 18,489 | +0.9328 | descriptive positive |
| LEBL | 540 / 4,390 | 15 | 12 | 1,525 | +7.0308 | descriptive positive |
| LEMD | 342 / 4,008 | 13 | 10 | 1,800 | +1.7785 | descriptive positive |
| LFPG | 0 / 0 | 0 | 0 | 14 | null | inconclusive |
| LIRF | 245 / 795 | 9 | 7 | 438 | +8.9712 | descriptive positive |
| LTFM | 0 / 0 | 0 | 0 | 0 | null | inconclusive |
| LSZH | 153 / 10,603 | 10 | 10 | 6,990 | +0.9620 | descriptive positive |
| LEBL_JanJun | 225 / 1,963 | 7 | 6 | 583 | +2.8636 | descriptive positive |
| LEBL_JulSep | 164 / 1,225 | 5 | 3 | 387 | +15.8320 | descriptive positive |
| LEBL_OctDec | 151 / 1,202 | 3 | 3 | 555 | +4.7344 | descriptive positive |

EDDF’s computed zero contrast fails the sufficiency gate and is not a null-effect finding. LFPG/LTFM have no common support. EDDM’s positive contrast is only +0.0074 pp. LIRF support spans June–December and retains 245 aligned versus 795 unaligned visits; its +8.9712 pp contrast cannot be extrapolated to the full airport. LEBL July–September has +15.8320 pp on 164 versus 1,225 retained visits, compared with +2.8636 pp in January–June and +4.7344 pp in October–December.

The geometric observations are generated without the taxi label, but both the alignment flag and discrepancy outcome contain recorded BLOCK; their contrast is not an independent error-rate validation. First-near observation, first movement and exit from a 150 m stand neighbourhood are not AOBT. Aggregated products do not certify continuous fresh stationary positions. Misassociation, stand mapping, returns/waits near a stand and residual confounding remain possible.

## Fixed-date raw supplement: inconclusive

All 36 preregistered dates (day 5/15/25 of each month) were locally available and pinned: **15,758 derived-eligible visits, zero validated stationary prefixes**. Missing `stale` schema excludes 7,888 visits on 16 dates; 7,870 visits on the remaining 20 dates fail the initial stationary-prefix requirement. Every raw common-support contrast, for all ten airports and all LEBL windows, is **JSON null / inconclusive**, not zero effect.

A separately authorized post-hoc technical check preserved the frozen result and inspected only identity, geometry, speed, stale status and time. Of the 7,870 inspected first points, 7,851 were marked stale; all 19 fresh first points were moving at 73.0–122.5 kt and >40 m from the stand. It reproduced zero-length prefixes without a dtype, time-unit or Boolean-negation mismatch. The local extractor encodes the archived stale flag from trace bit 0; external flag semantics were not independently verified. See `POSTHOC_ADMISSION_DIAGNOSTIC.md` and its pinned script/JSON/row evidence.

The strict initial-prefix rule has no admissible support in these archives. The raw audit therefore adds no independent physical validation; it neither refutes nor confirms the planning-copy mechanism. No later episode was searched, no threshold relaxed, and no frozen scientific run repeated.

## Novelty and inference boundary

LIRF concentration, adjacent-minute comparisons and the LEBL July–September episode were already recorded in `docs/data-quality-regimes.md`. Incremental evidence is the common-denominator six-offset/four-anchor analysis, tolerance and missingness accounting, fixed-stratum derived comparison and a transparent failed raw-admission audit. This remains a selection-informed retrospective replication, even though this follow-up’s code, sample and decision gates were committed before its outcomes.

Alignment enrichment may be constructional. The former 99.2% classifier precision measures agreement with a constructed label rule, not physical correctness. No estimate here supplies AOBT truth or the fraction of erroneous records. Prior PRU deletion sensitivities, including 42.1 s, remain deletion sensitivities rather than corrected ground truth. No new model, prediction, submission or public publication follows from this analysis.

## Execution and artifact integrity

Command:

```sh
/usr/bin/time -l .venv/bin/python scripts/planning_copy_triangulation_20261012.py run --registration bf08d014000bd51ad65ae045be644ab0629f6c8f
```

Exit 0. Measured wall 20.20 s; user 26.01 s; system 5.23 s; script runtime 20.0361 s. Peak RSS **1,804,697,600 bytes (1.681 GiB)**, peak footprint 1,766,590,672 bytes, zero swaps. Bound: 2 Polars threads, 900 s, 6 GiB. Execution log is outside the immutable outcome directory: `analysis/scientific_followup_20261012/triangulation_run.log`.

Input manifest SHA-256: `debd46591a8b4f7ee16965dfd2bb6a2f11522aada1b4390aba70656a4181a7fa`. `results.json` contains the original ten output hashes; all were independently recomputed and matched. Supplemental report/diagnostic hashes are in `FINAL_REPORT_HASHES.json`; they do not replace or mutate the frozen outcome manifest.

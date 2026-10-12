# Planning timestamp alignment and independent stand-geometry triangulation

Protocol `planning-copy-triangulation-20261012-v1`; proposed journal #464.
Status: **pre-outcome protocol**, local 2025 data only. No model fitting,
inference, external data access or submissions. This is an extension of a
selection-informed retrospective case study, not prospective ground-truth
validation. LIRF concentration, adjacent-minute contrasts and the LEBL
July–September episode were previously observed in `docs/data-quality-regimes.md`.
They are replication targets, not discoveries from this analysis.

## Question and fixed populations

Does the recorded block implied by `MVT_TIME_UTC_mvt - TAXITIME_SEC_mvt`
concentrate near planning timestamps more than nearby minute-preserving controls?
Where existing independent geometric evidence is available, do aligned records
also precede later observations near the assigned stand?

Use every 2025 departure at the fixed ten airports EDDF, EDDM, EGLL, EHAM, LEBL,
LEMD, LFPG, LIRF, LTFM and LSZH. The primary population has finite recorded taxi
duration `0 <= y < 86400` and at least one finite planning timestamp. Report all
excluded durations and all missing planning timestamps. Day-shift records are
excluded from the primary planning-alignment analysis; this does not reclassify
them as valid. Production's three-airport exception is deliberately not used.

The four planning timestamps are SCHED, IOBT, LOBT and EOBT_1. For each separately
the denominator is primary-duration rows with that timestamp available. The union
denominator requires any timestamp and counts each row once. Report multiple
matching anchors, which cannot identify which timestamp was copied. No row is
selected on a classifier score, prediction residual, or closeness to ADS-B.

## Timestamp controls, denominators and estimands

Let B = MVT minus recorded taxi duration, residual = B minus planning timestamp.
Primary tolerance is 6 seconds; descriptive sensitivities are 1, 3, 12 and 30
seconds. Preserve microsecond timestamps when converting to epoch seconds.

Controls shift **every available planning timestamp on the same row** by
`-180, -120, -60, +60, +120, +180` seconds. Whole minutes preserve minute-grid
rounding. No half-minute controls are used. For a timestamp and stratum, the
denominator is identical across every offset and tolerance; shifted timestamps
are not discarded because they cross a date boundary or pass takeoff. The
control is a local alignment benchmark, not a counterfactual departure time.
At tolerance 30 seconds adjacent windows touch; their counts must not be treated
as independent. The fixed six controls are reported individually, never selected
by the most favourable result. Coinciding planning fields can make shifted union
matches overlap another anchor; the per-anchor results expose this ambiguity.

Estimands: zero-offset match rate; each shifted rate; zero minus the equal-weight
mean of six controls; zero greater than all six controls. No p-values, threshold
search or estimated physical-copy precision. A zero excess or an excess reproduced
by controls weakens the narrow alignment interpretation. Positive excess only
establishes concentration at the published timestamp, which can be constructional.

Report all airports and airport-months. Fixed input-delay strata use MVT minus
SCHED: missing, negative, 0–1800, >1800–3600, >3600–7200 and >7200 seconds.
Also report NM presence (`ADEP_flt` non-null), coverage categories including
missing, and airport-month-delay. LIRF and LEBL July–September, January–June and
October–December are named before outcomes. This is not an event-search exercise.
Compare B with the supplied raw BLOCK timestamp before the analysis. Any
comparable row differing by >1 second stops outcome analysis: the representation
contract requires investigation, not silently choosing one interpretation.
Missing raw BLOCK is counted, not used to exclude otherwise eligible rows.
Agreement is a data-contract check, not independent proof that B is true.

## Independent ADS-B triangulation

Use already frozen `offblock5_2025-*.parquet` and `adsblol_kt1/offblock2_2025-*.parquet`,
joined one-to-one by integral MVT ID. The v5 source associates aircraft by
callsign/airport and computes geometry without taxi labels or model predictions.
The earlier `stand_occupancy_probe` is **not** used because its candidate gate
depends on an existing taxi prediction. No learned pipeline is rerun.

Before looking at B or the alignment flag, an eligible geometric record must have:

- known stand and observed point within 150 m (`gs_seen_at_stand=1`);
- first point and first movement point each within 40 m of assigned stand;
- v5 stand-neighbourhood dwell >=90 s;
- unique candidate (`n_cand=1`), >=10 surface points and coverage >=2;
- finite first-near = `gs_t_leave - gs_dwell_stand`, first-near >= takeoff−7200,
  and finite leave <= takeoff.

The 150 m neighbourhood exit is **not AOBT**. The first movement is used only for
a geometry-quality filter, never as block-time truth. Aggregated v5 products do
not certify continuous stationary dwell, speed quality or no inter-point gap.
Callsigned tracks can still be mismatched; an aircraft can wait or return near
its stand after off-block. Therefore geometric discordance is supporting or
weakening evidence, not a sensitivity/specificity reference standard.

Primary geometric quantity: the proportion whose first observed near-stand time
is >=300 s later than B, alongside its complementary diagnostic B >= leave+300 s.
Also report median/P10/P90 first-near minus B and median exit minus B.
The 300 s threshold is a fixed five-minute discrepancy screen, deliberately
larger than the six-second alignment tolerance; it is not calibrated to outcomes.

Compare aligned (any timestamp, tolerance6, offset0) versus unaligned eligible
rows. Standardise the late-presence risk difference within fixed airport x month
x input-delay-bin x NM-presence strata. Each stratum needs >=10 in both arms;
weights are combined eligible population in that common-support stratum. Publish
discarded rows and number of retained strata. No-overlap is null/inconclusive,
not zero effect. List airport and month availability, product-missingness, known
stand, observed stand and stricter geometry counts before reporting differences.

## Supplementary raw-episode audit, fixed before outcomes

Metadata inventory found raw surface observations on the mounted external disk,
in `e194_inputs_20261007_s0/surface` and `e194_inputs_20261007_s1/surface` under
`/Volumes/PRCEXT/prc-2026-data`. The local `data/adsblol` directory had no raw
2025 files; raw observations were not physically unavailable. Use exactly
January–December 2025 day **5, 15 and 25** (36 dates). Missing dates are reported,
not replaced. Resolve identical duplicate dates by s0 then s1; differing hashes
abort preflight. Select all derived-eligible visits on these dates, before using
labels, planning matches or outcomes. Whole-year primary analysis is unchanged.

Pin the 36 available raw files, all ten OSM extracts, `src/geometry.py`, and the
training columns providing callsign/flight fallback and assigned stand. Use the
v5 association criterion: callsign and airport, exactly one ICAO whose last
surface point at or before takeoff+300 s is no more than 900 s before takeoff.
The surface predicate is the existing v2 predicate (on-ground OR speed<=40 kt
and altitude absent or <=airport elevation+800 ft). Resolve stand by exact or
suffix-stripped OSM name, no prefix centroid.

Walk back from that last surface point through the terminal surface segment
while each preceding point is surface and its gap <=900 s, matching the default
v5 segment. Test only its **initial prefix**, never a later stop chosen to pass:

- every retained point has a finite speed between 0 and 1 kt, is not marked
  stale (missing stale status fails), and is <=40 m from assigned stand;
- every point remains <=25 m from the first point, all times <=takeoff;
- every inter-point gap is strictly positive and <=30 s;
- prefix contains >=5 points and spans >=90 s.

The prefix terminates at the first failed point. Later movement does not
invalidate an already sufficient early stationary prefix; a failed short prefix
is not rescued by a later favourable stop. These conditions apply to **all**
points, not just endpoints. The independent timestamp is the first point of the
validated prefix. Compare it with B using the unchanged >=300 s discrepancy and
same common-support strata/weights/sufficiency gate as the derived analysis.
Report each fixed date, every rejection category, retained IDs and per-scope
attrition. Missing files, no qualified episodes or inadequate overlap give an
explicit inconclusive result. No p-values or new threshold search.

Raw data validate observable episode continuity and stand position, **not AOBT**,
nor the mechanism that generated an aligned recorded timestamp. Aircraft identity,
stand geocoding, returns to a stand and coverage selection remain limitations.
Calendar dates are a fixed convenience sample, not a probability sample. Numerical
contrasts and month-by-month replication are descriptive; no general-population
confidence claim follows.

Schema-only admission found that some older raw files lack the `stale` field.
Do not replace those dates or assume positions are fresh: the manifest records
field availability, and every such visit receives `missing_stale_status` with no
accepted prefix. Missing stale values within a present column also fail the gate.
This is missingness handling, not a change in the sample or quality thresholds.

## Fixed interpretation gates and forecast

**Forecast, deliberately qualitative:** the already observed LIRF central
alignment peak is expected to reproduce; other airports may have much smaller
excess despite a large six-second rule prevalence. Long-delay LEBL July–September
is expected to differ from the first half of 2025. Independent geometry may be
too sparse at LIRF to resolve a physical interpretation. No numerical effect was
inferred from the new data before registration.

Report three separate statuses, never one binary claim of "true copies":

1. Alignment enrichment present when primary zero-offset rate exceeds every
   minute-preserving control; otherwise absent/mixed. Always retain signed excess.
2. Geometric comparison is informative only with >=50 flagged and >=50 unflagged
   rows **retained in common support**, retained evidence in >=3 months, and >=3 common-support strata;
   otherwise explicitly inconclusive. These are reporting sufficiency gates,
   not statistical significance criteria.
3. Where informative, positive standardised late-presence risk difference is
   directionally consistent with an early recorded timestamp; non-positive is
   conflicting/neutral. Neither establishes AOBT truth, causal copying, or the
   fraction of physically wrong records. No final recipe changes follow.

The prior 99.2% detector precision is agreement with the constructed rule, not
physical precision. PRU deletion effects, including the previously reported
42.1 s, remain deletion sensitivities; this protocol does not correct PRU truth.

## Files, verification, execution and resource bound

`preflight.json` lists each selected input (including external-disk raw files) and provenance-source SHA-256, byte
size, schema, missing daily files, and source/test/protocol SHA. Preflight reads
no outcomes; inventory inspection found 12 training months and 364 daily files
in each derived product. Inputs and code must match the committed manifest at
runtime. Runtime also verifies this protocol, source, tests, manifest and #464
marker in the supplied preregistration commit. No outcomes run before root review
and journal registration. Outputs remain under this analysis directory.

Commands:

```sh
.venv/bin/python tests/test_planning_copy_triangulation_20261012.py
.venv/bin/python scripts/planning_copy_triangulation_20261012.py preflight
.venv/bin/python scripts/planning_copy_triangulation_20261012.py run --registration FULL_COMMIT
```

Estimated runtime 3–12 minutes, <=2 Polars threads, estimated peak RAM <6 GiB.
The process has a 900 s alarm; raw-day checkpoints stop if peak RSS exceeds6GiB.
Preflight rejects a selected raw archive total >1GB compressed. These are execution
bounds, not sample selection rules; exhaustion reports an incomplete analysis.
Only existing `.venv`; no installs. Output: alignment and contrast CSVs,
availability CSV, stand-geometry summary CSV, standardised contrasts JSON and
results JSON with hashes and limitations. The supplement adds raw-day inventory,
episode audit and attrition CSVs, common-support cells and raw-supplement JSON.
No output is a prediction file.

Synthetic checks cover identical denominators, null timestamps, multiple anchors,
minute-grid controls, injected matches, closed tolerance boundaries, timestamp
units, one-to-one IDs, label-blind geometry, weak geometry rejection, common
support standardisation, and the registration guard. The test harness uses
stdlib unittest because pytest is absent in the existing environment.
Raw synthetic checks cover continuous stationary dwell, speed missingness,
early motion, gaps, displacement, stale points, terminal rather than inbound
segment, and refusal to search later favourable episodes.

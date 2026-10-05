# Protocol290: restore exact original visit sorting before audit rerun

Operational correction to protocol289; mechanism,124 fixed days, label-free
inputs, resource limits and every feasibility gate remain unchanged. New
protocol marker: `visit-airborne-label-free-124days-v2-original-sort`. Commit
this amendment and code with journal290 before a fresh complete124-day run.
Keep the failed289 bundle, partial output, logs and failure JSON unchanged.

## Observed failure and independent diagnosis

Attempt289 stopped on2026-01-02 after63 completed days. Against the saved
control, vi_alt_first differed in1 cell and vi_gs_first in2 cells; every other
vi field matched. No model or target was used and no feasibility pass occurred.

Read-only diagnosis on the immutable2026-01-02 bundle (1,169,426 observations,
5,386 departures) captured the actual unchanged `adsb_visit.main`, including
its actual `load_surface`, with only file I/O replaced by in-memory data.
That original matched the saved control exactly in all8 fields. The audit's
extractor on unsorted observations reproduced1 altitude/2 speed mismatches.
Passing observations first through `sort('icao','t')` removed every mismatch.

The original `load_surface` globally sorts by ICAO and timestamp, then main
groups by ICAO/airport and sorts each group by timestamp again. Attempt289
omitted the first sort. Equal timestamps can have different altitude or speed,
so the omitted ordering changed which tied observation becomes the first point.
Example: EDDF departure MVT202499662, THA921, ICAO885177, timestamp1767360176.72
has two ground observations with missing altitude and speeds10.8/11.2. Original
double sorting selected11.2; the audit's single sorting selected10.8.

## Minimal correction and validation

Restore only the original global `points.sort('icao','t')` before grouping;
keep identity-map construction from the same raw same-day point set, as in the
original. Apply the same extraction to the mask-filtered candidate. Do not
deduplicate points, change sorting options, round timestamps, weaken equality,
drop a compared field, or alter any visit/mask threshold.

Add an adversarial duplicate-timestamp regression against the actual original
main, with conflicting altitude/speed and unsorted points. All existing12
synthetic contracts remain required. The live diagnosis is source-parity
evidence only; complete124-day parity must be measured by the newly registered
run. Fresh bundle/source checksums and output directory are mandatory; no
in-place update, resumption or rewriting of failed289 evidence.

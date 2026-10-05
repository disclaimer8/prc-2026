# Protocol289: label-free E64 ground-mask removal audit

Fixed dates: all124 days in January/July2025 and January/July2026. No model fit,
target read, prediction scoring, quality selection, submission or existing-file
replacement. Code and this protocol must be committed with journal289 before
packing/running real data. This measures feasibility, not an expected RMSE gain.

## Mechanism and distinction from closed work

The current E64 extractor uses all near-airport points, including ground. A
ground reception collapse can change every visit boundary and vi feature.
Masking only the coverage field cannot simulate that observation process.
For each day we delete observations selected by the exact canonical
`adsb_offblock_v2.surface_mask`: on_ground OR (speed<=40kt, missing speed filled
with0, AND altitude missing or <=airport elevation+800ft). Recompute callsign
identity and all8 original vi features from the remaining same-day observations.
Polars drops rows whose mask is unknown. Missing altitude with speed>40kt and
on_ground=False is retained. Therefore this is canonical-ground-mask removal,
not a guarantee that every retained point is physically airborne. Report both
unknown masks removed and unknown-altitude observations retained.

The control literally reproduces `adsb_visit.py`: unique same-day callsign and
airport identity, 12h lookback, observations through MVT+600s, strict gap>1800s,
last visit and previous visit, identical null semantics. No E75 previous-day
history. Control must match existing saved visit outputs exactly for every ID
and all8 fields; a mismatch is a failed feasibility gate, never silently fixed.

This differs from rejected E39/E134b, which cut a ground-track prefix after
first movement and recomputed offblock/shape fields while retaining ground
observations. E150c already keeps contradictory training labels but does not
change reception. E147b already drops vi_n_pts/vi_prev_pts/vi_alt_first; changed
active fields are therefore assessed after those drops. The exact D3 config
and current E182+E185 E64 stage invocation are source-hash pinned as evidence.

## Inputs and provenance

All124 surface files are present on Mac and Hetzner:1,710,942,516 bytes,
125,049,648 observations, largest compressed day20,365,667 bytes. Read only
icao/airport/t/alt/gs/callsign/on_ground. Matched saved visit and production
1kt offblock2 coverage sidecars add21,933,956 bytes. Do not use old3kt coverage.

Departure source is the exact January2025 and July2025 training parquets and
ranking.parquet, projected at read to MVT_ID_mvt, PHASE_mvt, ADEP_mvt,
CALLSIGN_flt, FLIGHT_mvt, MVT_TIME_UTC_mvt and FLIGHT_ID_mvt. No target column is
loaded or copied. Keep DEP rows in the four fixed months. Input projection,
observations, control visits, coverage, four source/config snapshots and audit
script are immutable SHA-pinned. Original packing sources are checked before
and after; the complete bundle is checked before and after extraction. One
audit sidecar lock and atomic final directory publication prevent partial
results from masquerading as complete. Preserve partial directories on error.

## Outputs and locked feasibility gate

Daily label-free outputs preserve IDs, airport, NM availability, original
coverage, original/masked identity and all8 original/masked vi fields. Report
by airport/month/NM status: identity retained/lost/gained/switched, previous
visit retained/lost, current and masked E64 reach, all-field and active-field
changes, paired delta quantiles. Unknown-altitude counts are daily evidence.

PASS requires all of:

1. All124 inputs and complete departure/coverage/control ID populations verify;
   zero unfiltered-control mismatched cells across all8 vi fields.
2. No already unique identity switches to another airframe after deletion.
3. July2026 LIRF with NM retains a nonzero masked previous-visit population at
   original coverage<2.
4. Both January and July2025 LIRF with NM have nonzero synthetic rows with
   original coverage>=2, a masked previous visit, the same unique airframe,
   and at least one changed active vi field.
5. July2025 LIRF-NM has changed active vi values (explicit redundant audit).

These are mechanical/availability rejection gates, with no arbitrary quality
threshold. A pass licenses preparation of a separately registered experiment,
not model acceptance. Existing production E64 already reaches11359 of15562
LIRF-NM-July2026 flights in diagnostic281; this audit is about training-support
shift, and must not call that existing coverage new. Added signal/headroom
beyond the current assembled chain is unknown until later valid OOF testing.
No numeric gain forecast is made here. Removing low/slow points may also remove
true low airborne observations; artificial and natural missingness can differ.

## Bounded laptop run

Laptop a1@192.168.1.171:14GiB RAM,204GiB free disk. One day at a time,6CPU,
Polars<=6, MemoryMax10G, MemorySwapMax0, nice10; no training environment changes.
Expected per-day working set is well below10GiB from20.4MB compressed maximum,
but the real first day is the resource measurement. Stop on input/parity/shape
errors; preserve evidence. All raw inputs already transferred may be hardlinked
into the bundle, then verified against the Mac-generated manifest; only
projected metadata, source/config files and manifest need copying again.

`pack --root <Mac-root> --input <fresh-bundle> --registration-commit <289>`
creates the immutable label-free bundle. `verify --input <bundle>
--registration-commit <289>` must pass on laptop before `run --input <bundle>
--output <fresh-output> --registration-commit <289>`. Run the committed script
in the fixed resource unit and copy/report the verified result afterwards.

# Fresh final-view original ordered trajectory member

Protocol: `final-original-ordered-E203-fresh2025-three-seed-eight-epoch-v1`.

This is the final-period implementation of the already selected E203 ordered
member. It is not a new model experiment. The final feature union contains
2,085,047 training departures and 670,790 final departures, January, February,
June and July 2026. Every final row follows the same procedure. Old January/July
predictions and old 42-row tail outputs are not inputs.

The parent E207 full-token experiment and its separately registered current-chain
child remain independent. This adapter always calls the original, unfilled
`e194_trajectory_sequence.encode_day` and the original
`e197_trajectory_encoder.fit(..., arm='ordered')`. E207 admission cannot silently
change this source or its sampler.

## Registration and source/data admission

Root registers this helper, its tests, this protocol and the journal entry before
any real preparation, raw extraction, GPU smoke or fit. `attest` reads that full
Git commit and compares each live file with it. Its small portable receipt binds
the exact source closure and the separately sealed feature WORK manifest SHA.
All execution actions require that receipt, compare every active helper SHA, and
check that loaded original helper modules come from the attested tree. The four
original trainer files must additionally equal `mlp_gpu_component_oof.SOURCES`.
The shared `final_tabular_models_20261008.py` proxy helper is a read-only dependency
and must be reviewed and frozen before registration.

Feature WORK is the separate
`final-fresh-FIN1b-E113-motion-20261008-v1` build. Its manifest binds canonical
training files, the new final ranking copied as `ranking.parquet`, and the new
template copied as `submitting.parquet`. ID hashes are contiguous little-endian
float64 bytes. `queries` requires the feature completion receipt, feature SHA,
template SHA, exact split counts, ordered ID hashes, all 12 training months,
unique disjoint IDs, null final Y and BLOCK, and a final ID set equal to the
official template. No target or model feature arrays have been prepared during
implementation/review.

## Fixed scientific recipe and persisted state

1. Fresh FIN1b training/final features come from the new union-dependent feature
   build. Training labels and the original regime labelling rule remain unchanged.
2. Call the original month-out target encoding helper with ordinary-2025 rows.
   The actual E203 recipe uses 16 TE/TER mean/count columns. Persist all eight
   group-statistic plans and airport priors for full training and each month-out
   scope, smoothing 20. Verify exact replay against the original helper. `tq_`
   columns are dropped and regime rates are off; unused statistics are not saved.
3. Fit the shared serialized E46 proxy (scale ridge 1000, AOBT window 3606,
   unseen source scale 1.0). Require exact equality with original `T.proxy_base`
   on ordinary training and final rows. Residual is the original clipped Y minus
   proxy, clipped to ±7200; save its ordinary-training standard deviation.
4. Call original `mlp_lowmem_prepare.prepare` once. Save numeric quantile knots,
   missing-indicator decisions and the actual category codes produced by that
   call, including the frequency-20 boundary, learned nulls and unknown code0.
   Chunkwise saved-state replay must reproduce both numeric and categorical
   tensors exactly. No second unordered vocabulary enumeration is substituted.
5. Trajectory query population is 2,755,837 rows. Normalization uses only ordinary
   2025 indices; original `norm_stats`, `normalization` and `normalize` are reused.
   Persist count/mean/std and ordered fitted-ID hash. Channels0–5 are clipped to8;
   float16 values, uint8 masks and zero unknown/PAD values are unchanged.
6. Three fresh fits, seeds0/1/2, eight epochs, width512, PLE48/d12/versionB,
   two64-token streams, two local message blocks, batch4096, AdamW lr.002,
   weight_decay.00001, OneCycle pct_start.1. Use unchanged `S.fit`; no early
   selection, no fitting controls, no weight search, no extra model architecture.
   PLE bins come only from ordinary training numeric tensors and are saved.
7. Each checkpoint is immediately reloaded and must reproduce all670,790 seed
   predictions exactly. Retain all three checkpoints, seed arrays, bins, query
   specification, transforms and normalization. The raw member is exactly the
   float32 seed mean times saved sd plus proxy; output keys are `mvt_id/reg/reg2`.
   The .25 blend, E185 routing, tail/network stages and final submission assembly
   belong to other registered adapters. This helper never uploads.

## Original raw trajectory reuse and fresh final extraction

`raw` projects exactly original E194 `DEP_COLS`; Y and BLOCK are not read. It
derives UTC day shards from the actual movement rows, not a fixed427-day list.
The original model observes same-day histories; previous-day concatenation would
change its design and is not introduced. Token timestamps remain ≤ movement
time, with the original visit boundaries, identity resolution and raw duplicate
timestamp handling. This is not a claim to fix sampling density bias.

The caller supplies existing local `--surface-dir` directories. There is no
download, network operation, SSH runner or extraction of source ADS-B archives.
Conflicting copies of a day are rejected. A missing surface file is allowed only
for the historical original missing day2025-12-31. Other missing days fail.

Optional repeated `--reuse ORIGINAL_INPUT ORIGINAL_OUTPUT` pairs admit only
original E194 registration `fc7107de145a87a5a1ff93314f318b440b1de825`, original
source SHA, bound input/output/day receipts, identical hashes of all12 canonical
training files, and exact projected daily movement rows and surface SHA. Only
2025 products can be reused. A failed proof aborts; it does not silently reuse
or retry. Without reuse pairs, original extraction is performed for every day.
Every2026 day, including January/July, is encoded from the new final movement
times. Old2026 derived tokens are never copied by ID. Compression members are
materialized once per day before validation, avoiding the E207 original repeated
decompression error.

Raw outputs retain departure projections, the bound source plan, and per-day
NPZ arrays. IDs, movement times, airport, finite values, masks, source-row padding,
timestamp padding and future-point boundaries are checked before admission.

## Operations and resource bounds

No execution was authorized by preparing this patch. Root owns registration,
transport, admission, service launch and cleanup; this file adds no queue. CLI
actions are `attest`, `raw`, `queries`, `context`, `smoke`, `run`, `replay`.
Fresh output namespaces are mandatory; failed/partial outputs are retained with
failure markers and are never automatically retried. No existing output is
overwritten. SIGTERM/SIGINT/6h alarm stop the worker and record failure. Root's
durable bounded service is still required: detached durable execution is not
provided by this CLI.

All real data/model actions are Linux-only, with swap0 and finite cgroup limits.
`raw`: ≤3CPU/10GiB, ≥8GiB free disk. `queries`: ≤6CPU/40GiB,
≥40GiB free disk. `context`: ≤3CPU/10GiB, ≥20GiB free disk.
`smoke/run/replay`: ≤2CPU/12GiB, ≥25GiB free disk, existing qualified ROCm
RX7900XTX environment. These are caps/guards, not measured new-run memory/ETA.
No Mac bulk, no new environment, no concurrent owner/unknown job interruption.
Home GPU admission is after E207 releases the device. Claude's raw February/June
prefetch on175 is a separate source producer and must not be duplicated/touched.

`smoke` is separately invoked after preregistration and query preparation. It
uses synthetic4096-row inputs and actual numeric/category dimensions, reserves
the full dynamic trajectory-table bytes, performs six finite forward/backward
steps, checks unknown/PAD invariance and checkpoint reload, and binds runtime
and query receipt. Synthetic columns are all continuous, making its query
capacity conservative; this smoke is not a target fit or an accuracy estimate.
`run` requires that exact smoke/runtime and dynamic context receipt before fits.

`replay` is a separate zero-fit check: reapply saved full-training TE/TER,
numeric/category transforms and proxy to the admitted final features, require
exact equality with prepared final tensors, reload all three checkpoints using
saved bins and context, and require exact equality with saved seed/mean outputs.
No new statistics or models are fitted in replay. Full source/data staging and
host lifecycle remain root's responsibility.

## Verification and limits of this preregistration

Targeted tests cover exact original preparer and JSON replay, null/rare category
boundaries, month-out/full TE/TER, dynamic670,790 IDs, seed completeness/finiteness,
original duplicate-time sampler behavior, raw future/PAD rejection, training-only
normalization, original raw reuse lineage, artifact hashes/inventory, failure
markers, and PLE-bin save/load. No full suite, challenge target read, actual raw
encode, full GPU smoke, model fit or final prediction has run during development.
New runtime memory, source transport and saved-checkpoint equality on real final
inputs remain to be established by the registered execution; future RMSE/LB
quality is not inferred from these tests.

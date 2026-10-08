# Final tail and assembly implementation plan

> **For agentic workers:** use `superpowers:executing-plans` to implement these
> tasks in the assigned namespace. Root owns registration, actual execution and
> source commits. No source changes or real execution have begun for this plan.

**Goal:** assemble one uniformly processed four-month final candidate from the
fresh feature/model outputs, preserving the selected v12075-style arithmetic
with fresh E156 and no E81/donor copy.

**Architecture:** one small adapter with explicit actions for admission, native
classifier inference/premix, E156, cached stage execution and final assembly.
Reuse unchanged stage code in fresh child processes, capture/replay LightGBM
calls using the existing E203 signature contract, and change only declared
population/protection interfaces. No network, extraction scheduler or trainer
framework is added.

**Tech stack:** existing Python/NumPy/Polars/LightGBM; existing TabPFN8.5.0 and
cached V2 weights for the separately qualified E156 process.

**Spec:** `analysis/final_pipeline_readiness_20261008.md`, root's explicit
8 October final recipe, and `scripts/e203_trajectory_production.py::branch`.
The new final output is deliberately not byte-identical to old v12075.

## Global constraints

- Own only `scripts/final_tail_assembly_20261008.py`, its same-named test file,
  and this analysis directory. No shared/frozen source edits.
- No real model/TabPFN call, target-conditioned cache reconstruction, assembly,
  transport or ranking output before root's code/protocol/journal registration.
- Fresh immutable WORK; exact input/source hashes; no overwrite, automatic
  retry, old Jan/Jul prediction splice, 42-row donor inference or submission.
- Expected train/final DEP counts2085047/670790; exact template IDs/order/schema;
  final target and departure BLOCK are null. All four months share the recipe.
- Cached E141/E64/LSZH/E188 models require exact training/source admission.
  A mismatch stops. Explicit fixed refit is a separate registered action.

## Exact pipeline and interfaces

1. Read completed base classifier/state, complete Cat15/LGB5/anchor5/motion5
   families, and complete ordered E203 seeds0..2 from their new receipts.
   Recompute **raw classifier probabilities** from the saved classifier using
   its saved columns/categories. Base `predictions.npz.proba` has already been
   masked using its original regular/day candidate and must not be the input.
2. Execute the frozen blend arithmetic: `.65*Cat + .35*LGB`; then one member
   sum in order raw,lobt,sched,eobt,aobt,NN, with weights
   `0.0333333333333333,.10,.10,.10,.10,.25`. The raw literal is the existing
   branch representation of1/30. Rebuild candidates, apply unchanged
   `T.mask_infeasible`/normalization/LOBT clip and template rounding. Persist
   `mvt_id,reg,proba,values,pred` parts and their dtypes.
3. Run unchanged E185 application at `.35*2/3`, no-NM/outside-LIRF mask.
4. Run unchanged E167 A+B against the **final provided ARR and DEP rows**.
5. Run fresh E156: `TABPFN_EXTRA=pfx`, seeds0..9, V2 CPU classifier, existing
   default estimator count, existing bins/candidate/median rules; no E136
   expanded bins and no extra day-probability threshold. Record all served IDs,
   per-seed probabilities/class order, averaged probabilities, prefix mapping,
   exact training/query feature arrays and package/checkpoint identity.
6. Run E141 seeds42,1,2,3,4,800 rounds, keep-all,D7; then E64 same seeds,
   400 rounds,keep-all,D3. Both consume E167 parts. E141's base is E156 output.
7. E188 replaces only its natural LIRF/NM/coverage<2/previous-visit mask, using
   E141 pre-E64 values as its base and E64 post values outside that mask.
8. E152 `.95`: preserve E156 pre-stage values on confident filed-time rows.
9. No E81 stage. Apply unchanged LOBT lift1074/165 and window clip3606.
10. Protect **every served E156 ID**, including a row whose rounded E156 value
    happened to equal its input. Take values from the saved E156 output.
11. LSZH E159b,all2025 months,shift0,300 rounds; use current final OPDI matches.
    No matched event means unchanged prediction. Round its whole output using
    the original rule, then apply the existing rounded `.5*pre+.5*LSZH` blend.
    LIRF protection must still hold after this LSZH-only step.
12. Publish final template-ordered candidate plus stage hashes, all model and
    receipt identities, exact served-ID list and validator result. Upload is
    outside this adapter.

## Data and derivative gaps

| Input | Existing evidence | Required final admission / owner |
|---|---|---|
| Feature WORK | Feature helper33479b6d independently READY; no real completed WORK yet | manifest.json+feature_complete.json, E106/E113/FIN1b/motion hashes, final raw/template hashes and ordered split IDs; feature owner/root |
| Base and regular member outputs | Root's tabular adapter READY; neural adapter separate | All fixed seeds complete, model/state/prediction hashes and common feature/ID lineage; no old component predictions; root/neural owner |
| 2025 stage inputs | All18 OOF files exist (six each E116prod,nomonth,E150c/E64); all archived input paths exist locally | Reuse exact captured per-file inventories, including labels/OOF parents. Do not regenerate stage OOF on the new feature view |
| E141/E64/LSZH cache | All11 model file hashes verified against receipts | Reconstruct the original training calls and compare x/y/weights/order/features/params/rounds/runtime exactly before load |
| E188 cache | All5 model hashes verified; all368 bound fit inputs exist locally | Exact original fit-source/input manifest, seeds/features/weights/400 rounds and source hashes; no new masked extraction simply because ranking changed |
| **Final daily ADS-B derivatives** | Old 2026 derived rows are invalid for changed final MVT; E194 tokens are not substitutes | Fresh final source SHA + day receipt +7 daily products listed below for every requested day. No receipt proving this complete is available to this adapter yet. Extraction ownership must be assigned by root |
| OPDI | Existing44 event files total1,532,598,644bytes locally; previously reported Feb/Jun gaps | Explicit sealed event inventory and no-match fallback; no hidden download or month-specific correction; root/external-data owner |
| TabPFN checkpoint/runtime | Local29,009,539byte cached V2 weight, SHA cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49 | Offline load/configuration qualification before registered10-seed execution; exact checkpoint path passed, no download |

Minimum fresh daily product set consumed by this tail:

| Output | Existing producer/configuration |
|---|---|
| `adsblol_kt1/offblock2,3,5_<day>.parquet` | corresponding `adsb_offblock_v{2,3,5}.py`,MOVING_KT=1,STAND_WALK unset |
| `adsblol_sw_kt1/offblock2_<day>.parquet` | offblock_v2,MOVING_KT=1,STAND_WALK=1 |
| `adsblol/visit_<day>.parquet` | adsb_visit.py,original same-day VISIT_PREV_DAY unset/default0 |
| `adsblol/visit_m_<day>.parquet` | e69_visit_merge.py,original same-day/default thresholds |
| `adsblol/head_<day>.parquet` | e134_head_features.py,original default MOVING_KT=3 |

All read final `ranking.parquet` and the corresponding raw surface day. Preserve
existing raw identity/segment semantics; do not introduce cross-midnight E75 or
new onset thresholds. E194 has its own separate raw/source requirements.
Unconsumed default3kt offblock2/3/5 and standwalk offblock3/5 outputs are not
needed merely because the older any-period shell produced them.

Local 2025 cache inventories have364 files per each of these seven groups,
missing2025-12-31. That is the captured historical inventory, not a reason to
silently complete a new historical day and invalidate stage signatures. The
new final daily receipt must distinguish a processed no-match flight from a
missing/unprocessed whole day. Detailed metadata: source_cache_inventory.json.

## Cache/runtime boundary and smallest implementation changes

E141 cache:1260599×66float64,5×800rounds,6threads. E64:318697×11,5×400,
6threads. LSZH:84758×4,1×300,4threads. Their signatures bind Python's **entire**
`3.12.14 (main, Sep 1 2026, 14:09:38) [Clang22.1.3]` string as stored,
NumPy2.5.3/Polars1.44.2/LightGBM4.7.0. The unmodified Mac runtime is the direct
reuse candidate. Linux/ARM is not admitted by matching package names alone.
Do not weaken the full signature; choose explicit fixed refit registration if
root selects a runtime that cannot satisfy it.

The archived stage dependency source differs from live source only in known
resource/import corrections: E36/e64 parameter threads8→6 and removal of
unused import-time CV from e64_visit_stage. E141/E64/E159b themselves match
their live source. Preserve the archived safe dependency closure, without
copying the entire221-file source archive. E159b's fixed344841 assertion needs
one hash-bound AST transformation to compare the sealed final row count; no
expression or training-call transformation. E188 has a dynamic `ROWS` interface
adapter and a registration-receipt bridge, while its apply arithmetic remains
unchanged. The bridge verifies the original source/fit/input hashes itself and
never treats disabled validation as proof. Both transformations must be tested
for exact single-match scope and stored with before/after source hashes.

E156 needs a bounded capture wrapper around the original classifier factory and
existing script output: collect per-seed classes/probabilities and fit-array
fingerprints, enforce the exact seed sequence, save the served-ID predicate.
Keep estimator objects/arrays only as long as needed; no new classifier math.
The final protection step is the original ID-join/coalesce operation, with
explicit served IDs replacing historical/changed-row ID discovery.

## Implementation tasks and tests

### Task1 — admission and immutable stage plan

- [ ] Add tests rejecting partial/mixed seed families, reordered or duplicate
  IDs, final labels, absent derivative receipts, wrong raw-final SHA, incomplete
  day products and pre-existing output. Fixture files are synthetic.
- [ ] Implement `admit(root, receipt)` returning pinned contract, and
  `stage_plan()` returning the literal stage list above. Fail if E81/donors,
  a second NN application, alternative weights or a missing model family enter.
- [ ] Bind source closure/cache receipts/derivative schemas and actual final
  template counts; no actual heavy reads until an authorized CLI action.

### Task2 — original arithmetic and explicit E156 protection

- [ ] Run original blend expressions on synthetic classifier/candidate/member
  arrays; compare all5parts and rounded output exactly to the adapter. Include
  an r+DAY window boundary that proves masked base proba cannot be reused.
- [ ] Test E185 off-mask identity, E167 final-ARR linkage, all-month routing,
  and no-match OPDI identity using synthetic rows and unchanged source bodies.
- [ ] Test `protect(final, tab, served_ids)` where tab equals its input but a
  later stage changed the row; it must restore tab. Reject missing/duplicate IDs.
- [ ] Implement only the narrow original-source wrappers and explicit capture.
  A fake classifier exercises ten ordered seeds/classes/probabilities without
  importing or fitting TabPFN; a real cached-weight smoke awaits preregistration.

### Task3 — stage replay/refit refusal and final receipts

- [ ] Test `cached_train()` with exact x/y/weight/feature/order signatures and
  a stub native model: exact reuse succeeds; any mismatch fails before fit.
  Default mode must never call a real train function. Capture mode requires a
  separately bound explicit fixed-refit action and fixed expected call count.
- [ ] Test scoped row-count adaptation, E188 off-mask identity, premature
  completion rejection, final schema/ID/order/nonnegative checks and protected
  values surviving LSZH half-blend. No output is accepted from partial stages.
- [ ] Implement sequential existing subprocess calls with failure propagation,
  immutable stage receipts and one final completion. Root supplies existing
  resource supervisor/units; no new scheduler or network setup here.
- [ ] Run targeted synthetic tests; independent review; freeze source/test/
  protocol hashes. Root registers and commits before any real execution.

## Resource proposal and execution limits

Cache signature reconstruction needs full2025 stage inputs, but not raw ADS-B.
Observed old complete saved-component replay peak7.90GiB does not prove the
new670790-row pipeline cap; initial proposal is Mac≤16GiB process-group RSS,
6threads, diskfloor4GiB,2h bounded assembly excluding separately bounded E156.
Use root's qualified supervisor, never an unbounded Mac child. No large training
matrices or full new feature WORK are copied to Mac automatically.

The alternative is explicit registered fixed-stage refits on a qualified Linux
host. That is at most11existing-recipe stage models, plus5E188 only if its input/
source signature actually fails. It is not an implicit fallback and does not
authorize fresh OOF or scientific model selection. E156's resource/time envelope
must be set after offline runtime inspection; no fit-time/RMSE forecast is
invented from checkpoint presence.

## Review focus

1. Raw versus already-masked classifier probability semantics — Task2 boundary.
2. Stale same-ID Jan/Jul derivatives after MVT changes — Task1 source receipts.
3. Silent cached-model runtime/training drift or import-time fits — Task3 guard.
4. E156 served row with unchanged rounded value — Task2 explicit-ID protection.
5. No OPDI event, empty/partial stage results and final order — Tasks2/3 identity
   and completion checks. No final-byte equality to historicalv12075 is claimed.

Current status: plan/source/metadata only. Adapter/test code is not written;
real training, extraction, TabPFN, inference and final assembly counts are zero.

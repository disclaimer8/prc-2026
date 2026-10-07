# E192 — observed event graph against a matched event-pooling model

Implementation: `scripts/e192_event_context.py`. Root owns the single code+journal
registration commit, launch authorization and all submission decisions. No real
E192 target fit has run while this protocol was written. Synthetic checks use
random arrays only. E190 outputs, prior helpers and package environments remain
unchanged. This document is the implementation plan and fixed experiment design
for the separately authorized overnight task.

## Question and existing negative results

Does relation-aware exchange among individual observed airport movements improve
on an equally sized model seeing exactly the same movement tokens? The fixed
flat arm distinguishes gains from new inputs/capacity from gains from relational
computation. Existing saved E190 PLE predictions are a third, descriptive query-only
reference; they are not the parameter-matched graph control.

E151 reduced observed flows to interval counts; its production CV was +0.034s and
leaderboard +0.041s (REVIEW_BRIEF closure). E183 averaged neighbors' ADS-B/NM proxy
residuals and failed its probe (journal253, HANDOFF). E192 uses individual raw,
label-free events and nonlinear event-to-event interactions, no neighbor residuals
or target encodings. These distinctions justify testing; they do not show gain.
This is a one-layer local event-relation model inspired by temporal graph ideas,
not an implementation of TGN memory/TGAT, a GPS trajectory model, or a physical
collision graph. No leaderboard improvement is promised.

## Fixed population and observations

Outer month pairs1+7 and4+10, seeds0/1/2 in BOTH arms, eight epochs, no early
stopping or hyperparameter search. All original regular training rows are used:
1,622,159 and1,607,392 respectively. Validation populations344,419/360,962. Query
arrays are referenced from immutable `data/mlp_split_20261005/inputs/fold_*`
registered at a926f3848baaae7415bd3cb773cbd39728ce1851, verified by their complete
SHA manifests. Arrays are not copied or re-encoded; train IDs are reconstructed
with the original source+FIN1b SHA and exact original row order/regime function.
Pack additionally reconstructs the first eight numeric query columns over ALL
training and validation rows and compares them exactly against immutable xnf/xnp,
so raw event/query alignment is checked independently of assumed join order.
Targets remain `(clipped_y - calibrated_proxy)/train_sd` from those arrays.

Raw inputs: all12 `data/prc-2026-datasets/training_*.parquet`; ranking only for a
coverage audit. Local airport is DEP.ADEP / ARR.ADES. Events sorted by local
airport, observed MVT time and ID. Each departure gets32 previous and32 following
events in that order within±3600s; self excluded by its unique sorted position.
Equal-time events use deterministic ID order. Missing slots PAD0. All event IDs,
query IDs, window limits and airport boundaries are checked. Subsequent MVTs are
observed challenge inputs, as in existing `dep_after` features; this is retrospective
ranking, not a causal real-time prediction claim. No hidden departure block time
is used to define an event, window or relation.

The ONLY raw allowlist: MVT_ID_mvt, PHASE_mvt, ADEP_mvt, ADES_mvt,
MVT_TIME_UTC_mvt, RUNWAY_mvt, STAND_mvt, AIRCRAFT_TYPE_mvt, WK_TBL_CAT_flt,
AIRCRAFT_OPERATOR_flt, AOBT_3_flt, LOBT_flt, SCHED_TIME_UTC_mvt. BLOCK_TIME_UTC_mvt
and TAXITIME_SEC_mvt are never read into the event table, including ARR events.
No label-derived regime, residual, prediction, counts or teacher signals enter
neighbor tokens. Own training labels only select the unchanged query population;
held-out labels are used solely for evaluation.

Event categorical fields: local airport, phase, runway, full stand, aircraft type,
wake category, operator. Vocabulary sorted from outer-fit-month events only,
minimum count20; PAD0, unseen/null1, known categories2+. Unknown string sentinels
(empty/UNKNOWN/UNK/NA/N/A/NONE/NULL/-/?/??/???) map to null before encoding.
Held-out vocabularies never expand embeddings. Label-free global identity codes
for runway/stand/operator are used only for exact observed equality, never as a
learned category or ordered number; unknown identity0 never creates equality.

Event numbers: MVT minus each of AOBT3/LOBT/SCHED, clipped±7200 and divided7200,
plus each field's presence indicator; no fitted statistics. Query-relative time
ΔMVT/3600 and fixed sine/cosine periods60/300/900/3600s; three equality flags for
query runway, stand and operator. No absolute timestamp, ID or month is a model
input. Arrival AOBT/LOBT observations remain explicitly flight-level observations,
not an inferred local off-block event. Both arms receive identical numbers and
categorical fields; observed relation keys also come from those fields.

## Exactly matched learned modules

Both arms instantiate the same modules under the same seed and minibatch order:

1. Original E190 PLE48/d12/B (activation=False), categorical embeddings and
   width512 query network through its256-dimensional penultimate representation.
2. Seven event embeddings, each8-dimensional; concatenate56 categorical dimensions
   and18 numeric/time/equality dimensions. Token MLP74→64→64 with SiLU.
3. One shared learned node update128→64→64 with SiLU, applied to concatenated
   current token and aggregated neighbor message, then residual addition.
4. Masked mean pooling into64dimensions, concatenate with query256dimensions,
   fusion MLP320→256→1 with SiLU.

The ONLY arm switch is the fixed event-to-event aggregation adjacency:

- **flat:** uniform weights over every other valid event, independently of runway,
  stand and time; each row normalizes by its valid neighbor count.
- **graph:** average of three independently row-normalized binary adjacencies:
  same known runway; same known full stand; absolute event-event time gap≤120s.
  Each excludes self and PAD; missing-degree relation contributes zero. Relations
  have fixed equal1/3 coefficients, with no learned attention weights.

Both feed the same learned node-update/fusion/pooling modules. The nonlinear node
update permits different relational neighborhoods to matter after pooling. Flat
uses all the same event observations and has exactly the same trainable parameter
count, optimizer and query network. An unseen category can still have an observed
exact equality relation; this is an explicit property of relational input, not a
new measured coordinate or evidence of physical conflict. No learned TGN state or
cross-flight label memory is used. Entirely empty context pools to exact zero.

Original AdamW lr0.002/weight_decay0.00001; original OneCycleLR pct_start0.1;
batch4096, eight epochs, floor(n/batch) batches per epoch; NumPy seed permutation
and torch seed0/1/2. Prediction batch4096. No seed reduction, schedule retuning or
post-result arm changes. All seed residual predictions and model state dicts saved.

## Packing, capacity, runtime and validation

Pack global event tables once and one int32[N_departures,64] neighbor index table;
never materialize context for all queries. Store fold-specific train-only event
codes and query-index mappings. Reuse original query tensor paths+SHA references.
GPU holds compact event lookup tables; context expands only for the current batch.
Host RAM cap12GiB/swap0/CPUQuota200%/nice10 on isolated `.venv-numemb`; existing Cat
keeps running. Mac preparation may use its existing `.venv/bin/python`, never uv.

Observed ranking support:689,534 movements=344,841DEP+344,693ARR;344,835 departures
have another event±1h,327,402 have≥64. LTFM47,581 all have context,46,985≥64 (research
`event_support.json`). Those≥64 counts are total-window support; balanced32+32 can
still contain padding when one side is sparse. No GPS coverage assumption.

After sentinel normalization, ranking missing runway DEP4/344841=0.001160%,
ARR31/344693=0.008994%; missing stand DEP253=0.073367%, ARR274=0.079491%.
LTFM has zero unknown runway/stand in both phases. DEP AOBT3/LOBT/operator missing
5290=1.53404%; wake5304=1.53810%; aircraft type388=0.112516%; schedule0.
ARR AOBT3/LOBT/operator2552=0.740369%; wake2564=0.743850%; type21=0.006092%; schedule0.
Full train/rank phase×airport availability and held-out unknown-category fractions
are preserved in the pack manifest/availability report.

Real full-shape synthetic probe on the intended Radeon environment, CPU2/RAM12G,
passed finite forward/backward gradients, padded-context handling, event permutation
invariance and equal parameter count2,408,917. Batch4096×64events×64latent; after
2warmups,12timed batches per arm with query CPU→GPU transfer: flat0.0489454s/batch,
graph0.0520862s/batch. GPUpeak1.6467/1.8646GB. Runtime torch2.13.0+rocm10.0.0,
rtdl-num-embeddings0.0.12; exact package/runtime hashes retained in
`data/e192_smoke_20261006/result_v3.json`. These are synthetic throughput estimates,
not actual full dataset I/O or model quality. The first4batch probe included cold
kernel startup; the warm benchmark supersedes its timing only, both evidence files
retained remotely.

12fits require37,824 training batches (18,912 per arm), approximately31.9min pure
synthetic compute. Budget1.2–2.5h including packing/transfer/context gather and CPU
contention, with a conservative4× compute margin. No parameter reduction required
by this measured budget. Report actual first-epoch time without selecting on loss.

Selftests cover exact window boundaries/ties, self exclusion, airport isolation,
empty context, train-only categorical vocab including held-out-only values, missing
and duplicate ID rejection and explicit raw target/block exclusion. Synthetic
permutation check includes both empty and real context. Basic source/input/runtime
and whole-output SHA checks must pass before any artifact is accepted. Final
experiment directory publishes atomically only after all12fits. Interrupted hidden
staging is retained; no automatic resume/retry or partial mean.

## Evaluation and fixed continuation rule

Each three-seed member replaces only the existing MLP slot of the original E190
matched LGB5/pre-ADS-B screen: coefficient0.25, or23/120 in the original motion
subset. Existing planned candidates, probabilities, LOBT mapping, day endpoint and
base correction remain as in that screen. Do not call this complete production-chain
CV; Cat15 and subsequent ADS-B/E152/E81/LSZH stages are absent.

Verify exact three-seed reconstruction, all validation IDs/targets/months and
mixture arithmetic. Compare graph−flat (primary architectural comparison),
graph−savedPLE and flat−savedPLE. Per fixed outer pair report all, decided,
body(y≤7200), symmetric trim3 and trim20 SSE differences, all-row RMSEs, month summaries,
NM-LTFM body separately. Report a descriptive95% paired airport-day bootstrap CI,
1000resamples with seed192; do not use a confidence interval as an adaptive gate.

Advance the **relation** mechanism only if graph beats both flat and savedPLE in
body, decided, trim3 and trim20 SSE in BOTH fixed pairs. Advance **observed input**
as a separate finding if flat beats savedPLE by the same criteria in BOTH pairs.
This screen is not a six-fold acceptance or automatic ranking/submission trigger.
Any six-pair/ranking continuation gets separate root registration, retaining the
requested3seeds and all fixed architecture choices. A negative result is reported
without search over windows, architectures, weights, endpoints or seeds.

## Execution plan / commands

- [x] Read E151/E183 closures and measured raw-event coverage.
- [x] Implement isolated pack/network/train/evaluation and synthetic contract checks.
- [x] Measure real full-batch forward/backward on the intended GPU before real targets.
- [ ] Root register code+this protocol+journal in one commit; supply full SHA.
- [ ] Mac: `.venv/bin/python scripts/e192_event_context.py pack --root . --output data/e192_event_inputs_20261006 --registration-commit SHA`.
- [ ] Transfer immutable pack only to home and verify there using its included scripts;
  existing original query bundles stay in place, no duplication.
- [ ] Home bounded service: `.venv-numemb/bin/python data/e192_event_inputs_20261006/scripts/e192_event_context.py run --root . --input data/e192_event_inputs_20261006 --output data/e192_event_20261006 --smoke data/e192_smoke_20261006/result_v3.json --registration-commit SHA`.
- [ ] Copy complete outputs to Mac, independently verify/evaluate with the same
  script and registration; retain hashes, metrics and honest continuation verdict.

## Pre-launch independent review

A separate read-only reviewer checked the worker/protocol and underlying tensor
producer. No launch-blocking correctness/resource/leakage bug found. For both
fixed folds, upstream control hashes match the original E190 input receipts; PLE
prediction hashes match fold receipts; fold receipts occur in E190 complete; PLE
input hashes match the controls; control and PLE IDs/y/month exactly match original
MLP metadata (344419/360962 rows). Frozen production sources and final synthetic
worker SHA427b3ab0e063f421f8cc3b150c06a4030d5870d0417e63d0b868a36ba4d0dff5
match. This external receipt verification supplements pack's immutable copies.

Preregistration expert prior, uncalibrated: graph−flat body RMSE center0seconds,
range[−1,+1]seconds for the two-pair screen. This is not a leaderboard forecast,
probability interval or acceptance threshold. Selection criteria above stay fixed.

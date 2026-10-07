# E200: MDN4 through the current pre-E81 recipe and conditional full ranking

Protocol `E200-MDN4-fixed-joint-preE81-gated-ranking-fulltail-20261007-v1`.
Workers: `scripts/e200_mdn_production.py` and the one-shot durable controller
`scripts/e200_execution_queue_20261007.py`. Register both workers, this protocol,
the sealed input receipt and journal together before any new real comparison or
fit. No upload belongs to this experiment.

## Fixed comparison and inference boundary

E196 selected MDN4 on two pairs; E198 measured the remaining four, with all six
body improvements and the registered aggregate gate passing. E200 uses these
same six pairs, so **all six E200 comparisons are selection-informed**. They
are not a new independent confirmation sample.

Control is exactly E195's joint E190+E188 arm, current FIN1b Cat15/LGB5 with the
fixed five E130 anchor weights and neural slot0.25. Candidate changes only the
PLE neural regular member to the MDN4 analytic conditional mean. MDN OOF files
come from E196 for1+7/4+10 and E198 for2+8/3+9/5+11/6+12; no density refit,
weight search, extra seed, Gaussian or alternative mixture size.

Reuse E195's unchanged `upstream` and `stage_predictions` helpers and the96
registered held-out-pair stage files. Recompute control and require exact
integer equality to every original included E195 joint prediction before
scoring. Candidate gets its own regular/day candidates, base and stage inputs.
Use E185/E167/E141/E64+E188/E152 with the same ordering and rounding.

Before errors, union E195's existing exclusion mask with any candidate-induced
change in the original cached FIN1b feasibility/missing mask. This uses only
observables and candidate predictions, not labels/errors. Keep all2085047 IDs
but mark excluded rows explicitly and give both arms the same finite historical
fallback. Report original/additional exclusions per pair and globally.

Primary populations are the existing E195 decided definition and body y≤7200,
inside the common supported mask. Fixed operational continuation gate:

- decided SSE improves in≥4/6 pairs;
- summed body SSE<0;
- **sum of within-pair DECIDED trim3 SSE<0 and DECIDED trim20 SSE<0**.

Trimming sorts candidate-minus-control squared-error differences within each
pair/population and removes k extremes on both sides. This is not global trim.
Report included/body/decided, all months and airports, pooled RMSE, per-pair SSE,
and both global and sum-of-fold trims. No pooled-decided sign condition or gain
threshold is added. Expert prior for pre-E81 body change: center−0.2s,
uncalibrated interval[−1,+0.5]s; this is not a gate.

E195's historical upstream label dependence remains. Raw classifier probability
mass is unavailable,126 legacy TabPFN rows remain excluded, and exact held-out
E81 donor chains are unavailable. The comparison ends after E152/before E81.
Passing authorizes the registered production work below, **not a claim that
full-chain OOF or leaderboard improvement has been demonstrated**.

## Conditional production model

Only on the fixed gate PASS: reuse the already immutable full-training E190
ranking tensor bundle `data/e190_rank_inputs_20261006`, registration
`f40a465633cc32bff959c598d27d59f9bdc11cc3`, spec
`9c9e67f37c16db24d1e4a40245c28f56fdb206f8a5a85fca5ff47415cb2c9399`.
No array duplication or new encodings are needed. Verify its original source,
array, ID, no-ranking-target and admitted runtime receipts.

Call unchanged E196 `fit` with armMDN4, seeds0/1/2, eight epochs each, using the
full regular population and original train-only PLE48/d12/B bins, width512,
batch4096, AdamW .002/1e−5 and OneCycle pct_start.1. Preserve clipped standardized
residual target, analytic softmax-weighted mean and original proxy restore.
Exactly three fits, no retraining/search or seed replacement after failure.
Publish344841 finite unique ranking IDs with exact arithmetic three-seed mean
as `reg`/`reg2`, models, runtime and input/source receipts atomically.

GPU is Home `.venv-numemb`, CPUQuota200%, MemoryMax12GiB, swap0, nice10. No environment
changes. The durable GPU worker waits for E197 exact invocation
`0b48f00de65a4f1daada3b2fdcd094cc`, unit
`prc-e197-trajectory-20261007.service`. It must observe that live invocation,
then actual process exit plus successful12-fit artifacts/evaluation with
registration `bfe793c62f7cecf633c7cc31c747f48eb0676f2d` and input spec
`93d90131cdfa32c04b8f8c3513f3e7f7cc19c9695d8f47d3fe98b4cc8c0a407d`.
E197 immutable receipts encode epochs as eight finite losses, not an `epochs`
field. Verify four arms, each seeds0/1/2 and all model/prediction hashes. E197's
quality gate does not select E200. If the exact running invocation was never
observed, stop rather than infer success from unit absence. Arm this queue while
E197 is active; no simultaneous GPU fitting. Queue maximum12h, no automatic retry.
The Home unit has RuntimeMaxSec45000 (12.5h), independently bounding its lifetime
if the Mac controller disappears.

## Concurrent CPU production control, then exact model replay

After gate PASS, Mac prepares a private copy of the frozen
`data/today_chain_20261006` workspace. Preserve its source hashes and links;
new output version prefixes910100/920100 cannot overwrite old hardlinks.
Original global sources and helpers are untouched. Mac stage subprocesses use
the original clean environment/6-thread cap and **never import Torch**.

Run the entire original PLE+E188 production chain once while GPU waits/runs.
In dedicated stage subprocesses, wrap only `lightgbm.train` to capture the
original E141 five800-round, E64 five400-round, and LSZH one300-round models.
Record ordered feature names, full numeric matrix shape/dtype/bytes, labels,
weights, categorical semantics, all parameters, rounds, source hash and
LightGBM/NumPy/Polars/Python runtime. Existing E188 five saved models are reused.
Require exact final344841-row schema/ID/value equality to frozen v12074,
SHA `e5c78ab061492b69fe72d4a46d2e286f7a2f6b3129ebf2ae4a34731bb662374b`.
Failure preserves artifacts and stops candidate assembly; no tolerance is fitted.

For MDN assembly rerun the same scripts; their unchanged historical training
calls must match every captured fingerprint. Return the saved model for these
calls, performing **zero additional stage fits**. Model hash or feature/label/
parameter/runtime mismatch fails closed. This is reuse of unchanged training,
not a substitute implementation of stage prediction.

Full candidate chain: FIN1b classifier and cached Cat15/LGB5; unchanged E130
weights with MDN0.25; E185; E167; original42 TabPFN values; E141; E64 with E188
only on its registered natural LIRF/NM target at candidate's own pre-E64 base;
E152 .95; exact original E81 donors44:0.1214 and60:0.0270; LOBT lift/clip;
42-row protection; LSZH all2025/SHIFT0; original half-LSZH whole-file blend.
No half blend against v12074 or changed coefficient is introduced.

Ranking donor/source/input hashes are sealed separately from the limited CV
boundary. Validate exact344841 IDs, finite nonnegative integer predictions,
42 original protected values and candidate difference against v12074. Publish
only `data/e200_candidate_20261007/candidate.parquet` plus receipt. A public
submission number and upload are outside this task. Ranking expert forecast
center0s, interval[−1,+1]s is uncalibrated, not a leaderboard promise.

## One-shot durable execution

Before registration, `seal` hashes inputs/provenance only and `selftest` uses
synthetic arrays, a tiny synthetic LightGBM fit/save/reload parity check, changed
label/feature-order rejection and an E197 eight-loss fixture. No new real target
fit or error comparison occurs. Final registration binds both scripts, protocol
and input inventory. Existing files are never overwritten on a retry.

After root registration REG, start the controller on Mac under `caffeinate`:

```sh
caffeinate -i .venv/bin/python -u scripts/e200_execution_queue_20261007.py \
  --root /Users/denyskolomiiets/prc-2026 --registration-commit REG
```

It automatically runs CPU CV at≤8 threads; stops on negative gate; otherwise
packs/transfers the small GPU bundle, arms Home E200 unit behind E197, and starts
the Mac CPU control capture concurrently. It waits for actual GPU exit+complete,
collects only results/logs, independently verifies exact seed means and runtime,
waits for exact CPU control, then builds the full candidate by model replay.
Persistent state: `data/e200_execution_20261007.json`; stage/worker logs remain.
Any failure is recorded without automatic fitting retry, recipe adjustment,
silent fallback, removal of failed staging or stopping unrelated jobs.

CPU pre-E81 apply estimate2–4min; production control/model capture several
minutes to tens of minutes; MDN ranking about3–7min after E197. These estimates
never change the fixed seed/epoch/round counts. Root owns journal and upload.

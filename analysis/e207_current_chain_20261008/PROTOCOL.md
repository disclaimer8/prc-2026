# E207 current E203 chain: CV only, zero new fits

Protocol: `E207-fullK64-fixed025-current-E203-preE81-sixpair-zero-fit-v1`.

Status: implementation and synthetic verification only. Root owns independent review, journal registration and all real execution. Before registration only `seal` (metadata and opaque hashes) and synthetic tests are allowed. No actual parent errors, current-chain comparison, model fitting or ranking was evaluated during implementation.

Purpose: conditional on the fixed E207 v3 six-pair PASS, replace the saved ordered trajectory member in the current E203 chain and determine whether the already specified improvement survives through E152/pre-E81. Prior mean body improvement is 0 seconds; expert range is [-1,+1] seconds. These are assumptions, not observed effects. This is a retrospective selected-upstream check, not independently nested CV.

## Fixed lineage and gate

Parent `E207-full-K64-retain-original-raw-index-fill-sixpair-3seed8epoch-v3`, registration `f43405e7c6836230862837b66fad081ae4d69777`. Packed specification `598b6ff1398e85166665ed11cab066f402131cc883d5707ce25be14963c06eae`; materialized specification `56e1417bb71aa917847a7536ffbf8f0a9110c768a66bbaa6bf32e851fbffc4ca`. Both hashes and the two complete raw-shard receipts are checked; the materialized spec is reconstructed from the packed spec and raw completion metadata.

Current E203 registration `6e95e3a65936a3401e41499bb340f2d54a1a2121`; result SHA `455c6d0f6242e649817f743d90262f67d5422ffe81e537b3324c1e1be7027cab`; prediction SHA `381563def761db5e201efbb6349ad6668ff4dc8c2ae00b39eed19e8d3694d4b7`. The original E203 input receipt SHA is `c137929b58c498195d010e24326ea11a6de038bd3d07daebedbf276aac330ba4`. Required control/data/model hashes come from that receipt; the four E202 ordered controls are bound to the unchanged ordered files in the E207 packed manifest.

The child gate is fixed before reading its comparison: **at least 4/6 decided folds improve**, and all four quantities are strictly negative: pooled body SSE delta, pooled decided SSE delta, sum of within-fold decided symmetric trim3 SSE deltas, and sum of within-fold decided symmetric trim20 SSE deltas. Trims remove the 3/20 largest and smallest row deltas **inside each fold** before summation. Pooled sorted trimming is only descriptive and is not the gate. Exactly six distinct month pairs (1,7)..(6,12), finite metrics, no additional body-win threshold, no E204 selected-pair subgroup. No alternate weights, tests or adaptive fallback.

## Parent admission and small transport

Parent `complete.json` alone cannot admit the child: E207 writes it before evaluation. Require matching parent complete/evaluation, reconstructed six-pair metrics/gate, six ordered manifests, 18 distinct per-fold seeds0/1/2 with eight finite losses, unchanged parameter counts/runtime and bounded GPU memory. Require the terminal controller events receipt with correct registration, exactly one owned GPU start/exit, matching InvocationID, successful exit0/MainPID0, final `complete_no_ranking`, gate true and exact evaluation SHA. Failure or incomplete events reject admission.

After child preregistration, `attest` on Mac verifies its exact Git source/test/protocol/inventory blobs plus protocol presence in the registered journal. It creates `registration.json` once. Root transfers this small Git-admission receipt and frozen files to Home; no Git history/bootstrap or new remote runner is introduced. The Home audit validates the receipt and every active helper hash. This assumes root-controlled transport of the same reviewed artifacts; the receipt is not a cryptographic signature from an external authority.

`audit-parent` runs directly on Home after parent terminal PASS. It reuses frozen `Q.checked_arm` to verify every saved artifact SHA, seed arithmetic and component-mixture reconstruction. The parent metric helper is reused exactly and all six metric reports are reconstructed; this imports the frozen `night_gpu_20261007` metric helper through E207 but never selects/calls its E204 model, fitter, screen, queue or output. No prepared trajectory tensor table is read by the audit. Query bundles, seed arrays and model files already on Home are checked locally. Exactly2,085,047 unique parent IDs are required. A new `parent_audit.json` binds all checked model/seed hashes and enumerates a portable `files` map.

Root collects only `parent_audit.json` and the files in its `files` map: complete/evaluation/controller events, packed/materialized and two raw-shard metadata receipts, and six evaluation/ordered-control NPZs plus six candidate predictions/manifests. The 55GiB prepared tables, raw sequence arrays and model/seed files stay on Home. After transfer, `cv` checks every portable SHA, repeats parent metadata/gate/metric reconstruction, and verifies the audit's exact source/inventory/registration identities. No candidate comparison begins if any proof fails. Parent artifact hashes are bound at runtime after their completion, not falsely listed as known before preregistration.

## Chain arithmetic and population

Use unchanged E203/A helpers for ID/y/month alignment, rules, feasibility, masks, saved E195 stage prediction and paired metrics. For both saved ordered control and new E207 raw `member` separately:

`r = (1-sum(anchor weights + .25))*(.65*Cat90+.35*LGB5) + weighted anchors + .25*member`.

Use `member`, never the parent `pred` or `reg` (which already contains a component mixture). Apply the neural slot once. Existing E185 motion blending subsequently reduces its coefficient to23/120 on that subset; do not apply that attenuation twice or use a linear correction to rounded/nonlinear finished predictions. Keep FIN1b probabilities, rules, anchors, Cat90/LGB5/motion and all saved E141/E64/E188 stage models unchanged. Stage code loads/predicts with 15 cached models per fold; **zero new control or candidate CPU/GPU fits**.

Exactly2,085,047 unique IDs, the saved E203 evaluation mask, exactly126 historical TabPFN exclusions and identical placeholders. Changed scored-row feasibility rejects the experiment; it never expands exclusions. Each reconstructed control must equal saved E203 `ordered` on every row before that fold's errors are evaluated. Saved output reports are reloaded to reconstruct metrics/gate and verify the original control/population before a final completion marker is written.

The boundary is E152/pre-E81. It does not test full-tail OOF, the42 old-ranking donor values, ranking readiness or future final-dataset months. It neither modifies nor materializes those objects. The newly announced final Jan/Feb/Jun/Jul dataset does not enter this 2025 six-pair CV check. Ranking/final architecture requires separate period-independent work and authorization.

## Execution and limits

No scheduler, automatic SSH, retry, ranking command, model-training command or submission capability is present. Root runs the standalone CLI directly. The source/test/protocol/inventory must be committed with the exact protocol string in the journal before `attest`, `audit-parent`, `cv` or `check`. Existing qualified environments are unchanged; no installs. Prediction threads are restricted to1..4, BLAS defaults1 and Polars2. Run one instance; Root bounds the direct job with the available machine/disk budget and keeps at least4GiB Mac disk free. Outputs are only in this namespace; output/audit/registration files use exclusive creation, no resume or overwrite. Failure leaves partials and no successful completion marker; retry requires a separately reviewed recovery.

Commands after root registration and parent PASS:

```sh
# Mac: Git admission for the reviewed child, once.
.venv/bin/python scripts/e207_current_chain_20261008.py attest --registration-commit FULL_CHILD_COMMIT
# Root transfers frozen source/test/protocol/inventory/registration and terminal parent events to Home.
# Home: existing CPU Python; no GPU calculation or fit.
.venv/bin/python scripts/e207_current_chain_20261008.py audit-parent --registration-commit FULL_CHILD_COMMIT
# Root collects parent_audit.json and exactly its portable files map to Mac.
.venv/bin/python scripts/e207_current_chain_20261008.py cv --registration-commit FULL_CHILD_COMMIT --threads 4
.venv/bin/python scripts/e207_current_chain_20261008.py check --registration-commit FULL_CHILD_COMMIT
```

`seal` precedes preregistration and does not decode predictions, targets or errors. `attest` creates only metadata. Parent audit/verification does read completed parent predictions and target metadata, but only after prerequisites; the child comparison reads targets only for the fixed registered check. `result.json` and `complete.json` always retain `accepted_for_submission=false`, `ranking_authorized=false`, `new_cpu_fits=0`, `new_gpu_fits=0`, `full_tail_oof_validated=false`, `selection_informed=true`. PASS produces a report, not an automatic next action.

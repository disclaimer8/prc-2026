E207 potential PASS: focused current-chain readiness review, 2026-10-08.

**Verdict: reuse the E206 draft's zero-fit CV arithmetic as a template, not its executable protocol or queue.** The smallest next stage is a new, separately registered **CV-only** child of completed E207 v3. It must compare the full-token member against saved E203 ordered predictions through E152/pre-E81, with zero new control or candidate CPU fits. E204 stays closed; the existing E206 draft must not launch. No E207 quality result was read for this preparation, and no fit, comparison, remote action, queue or source mutation was performed. Only this note was written.

Read in full: `scripts/night_trajectory_chain_20261007.py` (411 lines), its queue (166 lines), its protocol and four contract tests. Also inspected the relevant E203 CV/ranking preparation/control code, E195 alignment/upstream/stage application, E202 checked-arm reconstruction, E207 v3 output/provenance code and the qualified full-tail replay receipts. This is a bounded code/readiness review, not end-to-end execution qualification or a fresh audit of all 4,044 inherited dependencies.

**Important incompatibilities if the draft is used for E207 unchanged.**

1. Wrong parent and schema. Worker lines26–35,57–63,142–149 and queue20–22,88–107 hardcode E204 protocol/registration/spec/unit/invocation, require E204 `run/result.json` plus `audit.json`, `phase=six`, `status=six_pair_pass`, and load `run/fold_*/prediction.npz`. E207 has `complete.json`, `evaluation.json`, and `fold_*/ordered/prediction.npz`; it has no E204-shaped audit. E204's screen failed. Replacing just a path or label would neither authorize nor correctly validate E207.

2. Wrong conditional ranking mechanism. Worker263–274 verifies/reuses **E203's old prepared table** and temporarily installs E204 `P.network`, fitting arm `multiscale`. E207 requires the unchanged E197 ordered model on newly filled F tokens. E207 v3 extracts 365 training days only; ranking F tokens for Jan/Jul2026 and a new all-training normalization/table do not yet follow from a six-fold PASS. The old E203 query tensors, IDs, targets and index mappings may be reused; the old trajectory table and its normalization moments may not.

3. Wrong lifecycle for retained E207 units. Queue30–40,89–96 treats every `ActiveState=active` as running and does not read SubState; an E207 v3 `active/exited/MainPID0` parent would be waited on until timeout. Its terminal checks also allow missing status fields, and its ranking launch lacks retained state/unique Description fallback for lost ACK (120–125,144–148). Do not copy this controller. A CV-only child can start from verified terminal artifacts; if it waits on a unit, use the reviewed E207 v3 state model with exact ownership, SubState and strict success checks. This note does not authorize any ranking unit.

4. Accidental gate inheritance. Draft48–54 demands ≥4/6 body wins AND ≥4/6 decided wins, four negative pooled SSE/trim sums over all six pairs **and** over pairs2/3/5/6 alone. That subgroup came from E204's selected1/4 screen. E207 has one fixed arm and all six pairs; the four pairs must not be called independent confirmation of E207. Keep any additional restriction only if explicitly preregistered as such, not inherited silently.

**Exact minimal CV-only delta.**

Create new source/test/protocol/inventory/output namespaces; preserve E206, E204 and E207 sources/results. Proposed descriptive protocol identifier, not yet registered: `E207-fullK64-fixed025-current-E203-preE81-sixpair-zero-fit-v1`. Do not allocate a journal number or include automatic ranking/full-tail work in this minimal child.

- Replace E204 parent/import/gate/path handling with an E207 v3 artifact adapter; remove E204 source dependencies from the new inventory. Parent protocol is `E207-full-K64-retain-original-raw-index-fill-sixpair-3seed8epoch-v3`, registration `f43405e7c6836230862837b66fad081ae4d69777`. The original packed input spec is `598b6ff1398e85166665ed11cab066f402131cc883d5707ce25be14963c06eae`. **This is not the later materialized spec digest**: parent `complete.json.input_spec_sha256` must match `data/e207_v3_materialized_20261008.json.spec_sha256`, whose parent_spec_sha256 must match the packed spec. Bind both manifests and both raw-shard completion receipts.
- Require completed parent output `data/e207_v3_trajectory_20261008/{complete,evaluation}.json`, all six ordered arm manifests/predictions, and a terminal controller receipt proving successful completion. A printed fit or `complete.json` alone is insufficient: evaluate runs after writing complete. Require exact six distinct folds, three distinct seeds0/1/2 per fold, eight finite losses per fit, parameter/runtime checks and artifact SHA. Require `evaluation.six_pair_gate_pass` and recompute the fixed E207 gate from all six reports.
- Independently validate each parent arm via the frozen `Q.checked_arm`: exact metadata, saved seed arithmetic, member and component-mixture reconstruction; recompute its metrics against packed `ordered.npz` and `evaluation.npz` using `G.measured`, then compare with the saved parent evaluation. This can be audited on Home where seeds/model files exist, with a digest-bound small receipt. Do not copy the six large prepared tables to Mac or call the full parent `evaluate` on Mac merely to audit member predictions; it additionally expects prepared support tables. Collect small prediction/proof files only, and pin the audited parent file hashes in the child inventory.
- Keep draft CV152–193 arithmetic, but load candidate from `data/e207_v3_trajectory_20261008/fold_f_f+6/ordered/prediction.npz['member']`; rename candidate output columns honestly (e.g. `full_tokens`, not `multiscale`). Keep ordered control from `W.trajectory_path(root,fold)` and require exact equality with saved E203 `ordered` for every row, including excluded placeholders. Keep `A.aligned` ID/y/month checks and all six month pairs.
- Keep identical E203 evaluation_mask, exactly126 exclusions, exact2,085,047 unique IDs and no added exclusions. `fixed_feasibility` must reject a changed scored-row feasibility mask before evaluating errors. Keep the saved FIN1b probabilities/rules, Cat90/LGB5/anchors/motion, and existing E195 stage models. `A.stage_predictions` loads models and calls predict, not fit; at this boundary it uses E141/E64/E188 (15 models per fold), while the six cached LSZH models are not applied pre-E81.
- Reuse `W.masks` and `W.paired`; trim decided per fold before summing, not across the pooled sorted delta vector. Store fold reports, pooled descriptive metrics, exact control proof, input/output SHAs, `new_cpu_fits=0`, `full_tail_oof_validated=false`, `selection_informed=true`, `accepted_for_submission=false`. `checked_cv` should reconstruct saved metrics/gate from output files as the draft already does.
- Before comparisons, root must fix the child gate. Smallest E203-equivalent gate is ≥4/6 decided wins, negative pooled body SSE and negative sums of within-fold decided trims3/20 (`e203_trajectory_production.py:125`). For consistency with E207's parent gate, a reasonable explicitly preregistered child gate adds negative pooled decided SSE; then it is ≥4/6 decided and all four pooled body/decided/decided-trim3/20 sums<0. This is a proposal, not an already authorized gate. Do not silently inherit the draft's separate four-pair requirement or select a gate after seeing the chain comparison.

**The slot must be replaced once.**

Draft170–180 correctly builds, separately for ordered and candidate:

`r=(1-sum(A.WEIGHTS))*(.65*cat+.35*lgb)+sum(anchor_weight*anchor)+.25*member`

Then `A.upstream` applies the existing E185 motion blend on its subset. Therefore the member's effective coefficient there is `.25*(1-.35*2/3)=23/120`; elsewhere it is .25. Do not pre-scale the member, apply23/120 manually in this loop, append both old and new neural members, or treat E207 `pred`/`reg` as a neural member. E207 `pred` is already assembled with the component control/mixture (E207339–340), so using it in the .25 slot double-counts the upstream mixture. Reconstruct both branches from the original components; do not apply a linear delta to already rounded/nonlinear pre-E81 E203 predictions.

**42 donors and full tail are a later boundary.**

Pre-E81 OOF preserves the **126 excluded historical TabPFN rows** and their placeholders; it does not test preservation of the42 ranking donor values or full-tail OOF. Keep this distinction explicit.

After a separately authorized ranking preparation/fit stage, the draft's `replay_ready`, `full_branch`, and `fulltail` provide useful reviewed arithmetic: qualified low-memory stage0, exact v12075 control first, then candidate substitution, cached11 E141/E64/LSZH models and5 E188 models, E81/LOBT/half-LSZH unchanged, and final42 donor equality check. Reuse source by a new frozen adapter; do not resume the draft controller or recapture/refit controls. Full training query reuse still requires a new F-token table for2,429,888 train+rank rows, train-only normalization and full-size GPU smoke; candidate ranking would need exactly3 new ordered fits, not control fits.

Existing replay qualification was read, not rerun: registration `70aff7199731c381945c45754ee70c7e3a964aa5`,344841 exact predictions/bytes,0fits. The live best file SHA matches `90d3cd0b5a68cc8008f2142f1801857752574b525b2ff046ad6bf4089a70aee2`. Qualification receipt SHA `f620d88d4a8cf67d3fa37df32760923bc327ad0a9d16d7e1b7ba2be25cf6adb3`; supervisor SHA `7cc362164ef2b68beeaed9dda6dafbe72f29ff84b6d24b4421af7d4455d2b822` points to it. Lowmem helper SHA `c30655b5eca6236928ab72fa58e5d404abdbf20b24cb5c80310c9fffdd72ec53` matches both small qualification receipts. This establishes an available prerequisite, not execution qualification of a new candidate branch.

**Pinned control and source snapshot.**

Current E203 protocol: `E203-ordered-trajectory-fixed025-joint-preE81-sixpair-gated-fullrank-fulltail-v1`; registration `6e95e3a65936a3401e41499bb340f2d54a1a2121`. Its result records gate_pass/control_exact,2,085,047 rows and126 excluded. Saved result SHA in the draft inventory: `455c6d0f6242e649817f743d90262f67d5422ffe81e537b3324c1e1be7027cab`; predictions SHA `381563def761db5e201efbb6349ad6668ff4dc8c2ae00b39eed19e8d3694d4b7`. The child must verify these actual files before comparison, not merely trust the JSON flags.

The draft inventory's19 active source hashes all currently match; it is internally consistent **for E204**, not repurposed E207. Snapshot:

| File | SHA256 |
|---|---|
| `scripts/night_trajectory_chain_20261007.py` | `23983e7e4437551713e14a2247e6efc501e61310c13622c7d94d6c3540148a2d` |
| `scripts/night_trajectory_chain_20261007_queue.py` | `571d928084e867c6939dac97041c2d57378ed6bdb262031e8c6049aef61cd2f9` |
| `analysis/night_trajectory_chain_20261007/PROTOCOL.md` | `500d0599cc674b9e37b2965bd2feb5a300536536a7191c3d3cc410a95ea12dec` |
| `analysis/night_trajectory_chain_20261007/inventory.json` | `33a23115e532e7e5e819ab66e76f864d12cfb179a8c5b171b80af72847bf872d` |
| `scripts/e203_trajectory_production.py` | `874fe8e2855d814d60ce47375ba03fade036a5e9f071d1cd852f5b756f9d8174` |
| `scripts/e195_pre_e81_comparison.py` | `a9439cf5ab51ceb52dd1d94ed7524631f8e13901f173f4e9af924a4be7f2ed59` |

Declined to judge: E207 outcome, prospective chain gain, new GPU ranking feasibility/runtime, end-to-end adapted controller behavior, and future LB. No child implementation, preregistration, model fitting or upload is authorized by this note.

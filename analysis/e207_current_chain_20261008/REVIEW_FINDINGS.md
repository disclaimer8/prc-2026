# Independent review — E207 current E203 chain adapter

**READY for root preregistration; 0 Critical, 0 Important, 0 Minor open findings in the reviewed scope.** Review completed 2026-10-08 11:22:35 UTC. This verdict permits review/registration of the fixed CV-only adapter; it does not attest that the parent passed, authorize execution before its prerequisites, or approve ranking/submission.

Reviewed the full new worker, tests, protocol and inventory, plus the active called paths in `e203_trajectory_production`, `e195_pre_e81_comparison`, `e207_full_token_sampler_v3`, `e202_trajectory_extension` and `night_gpu_20261007`. No source edits. No actual parent errors, predictions, targets or current-chain comparisons were opened; no `attest`, `audit-parent`, `cv`, `check`, model fit, queue or remote action was run for this review. Only this findings file was written.

## Verified contracts

- Git registration checks the exact worker/test/protocol/inventory and journal protocol. The small root-produced registration receipt explicitly uses trusted root transport; it is not represented as an external signature. Runtime source checks cover the 16 active frozen source pins.
- Parent admission reconstructs packed/materialized lineage and complete raw-shard metadata; checks fixed six-fold/three-seed/eight-loss evidence, capacity/runtime/GPU bound, parent metrics and the fixed parent gate. A successful complete marker alone is insufficient: terminal controller/GPU InvocationID, successful exit and evaluation SHA are required.
- Home audit calls the frozen `Q.checked_arm` to validate artifact hashes, seed arithmetic and mixture reconstruction. Mac receives a bound portable file map and audit receipt, repeats lineage/metrics/gate checks, and does not require the large prepared tensors or copied model/seed files.
- Candidate arithmetic substitutes raw `member` once at coefficient .25. Existing motion blending then gives 23/120 on its original subset. Control reconstruction uses unchanged A/W rules and saved stage predictors, and must exactly equal the frozen E203 ordered prediction before fold metrics are written. Candidate feasibility cannot broaden the fixed exclusions.
- Exactly 2,085,047 unique IDs, six distinct month pairs, 126 frozen exclusions and unchanged placeholders are enforced. The child gate uses at least four decided-fold wins and strict negative sums for body, decided, within-fold trim3 and within-fold trim20; no E204 screen-pair requirement or tuned alternative is introduced.
- Output checking reloads each fold and pooled predictions, verifies source/control/population/hash identities, recalculates fold and pooled metrics, and precedes the exclusive completion marker. CLI exposes no ranking, submission, fitting, scheduler, automatic retry or automatic next stage.

## Independent validation

`python -m unittest discover -s tests -p test_e207_current_chain_20261008.py`: **11/11 PASS**, 0.247 seconds in the existing project environment. All four readiness SHA pins and all 16 inventory source pins independently matched.

Additional synthetic checks, separate from the supplied tests, passed:

1. A raw member increase of 120 with zero other components produces final upstream increases **23** on the EDDF motion subset and **30** on the LIRF non-motion row, confirming 23/120 and .25 are applied once.
2. Six synthetic folds passed the actual `W.paired` symmetric trim3/20 arithmetic against explicit within-fold sorted row deltas; `chain_gate` accepted the valid six-fold reports.

## Frozen reviewed files

| File | SHA-256 |
|---|---|
| `scripts/e207_current_chain_20261008.py` | `2eaff5d0640a6d9d7f10d01208239bd692b95b0f1081e6e6edd42386ec9770f7` |
| `tests/test_e207_current_chain_20261008.py` | `31aaf9eca88f728bc3daa30966f3d8c810c5f8d2944fd17d8ebc0d3339af3fde` |
| `analysis/e207_current_chain_20261008/PROTOCOL.md` | `21b8b2b9615c359e34d922b286d63bf2ecd39d72621f31938c8edd234b602024` |
| `analysis/e207_current_chain_20261008/inventory.json` | `41b9a0158b7220a28f127ee49ad52b7bcdb7c1cb36650110b38168e16c4e3e03` |

The scope remains retrospective, selection-informed 2025 six-pair CV through E152/pre-E81, conditional on the exact saved upstream. This review does not establish real parent completion/PASS, actual exact-control replay, prediction quality, resource usage of the later real job, full-tail OOF, ranking readiness or transfer to the newly announced final months. Those remain runtime evidence and separately authorized work.

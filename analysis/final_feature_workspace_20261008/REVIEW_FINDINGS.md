# Independent review — fresh final feature workspace

Status: **READY for root preregistration; 0 Critical, 0 Important, 0 Minor open findings in the reviewed scope.** Final delta-review completed **2026-10-08 11:35:20 UTC** on worker `33479b6d8e8e9648b0c5c6410cba157f940ce6ca0389cd031f8164326f56b1be`. Two initial Important admission defects were independently reproduced, fixed by the author, and independently retested. No real seal, stage, feature build, fit or remote action was run. Source files were not edited by the reviewer. This review does not substitute for root registration or actual resource/feature completion receipts.

## Closed findings

1. **Fixed — incomplete completion receipt passed `verify`.** Initial worker `8577f11c…` accepted `outputs={}` without any final output. The final worker requires exactly four output paths, rechecks every actual output, reconstructs completion identities and consumer aliases, checks the exact completion field set and validates elapsed time. Independently rerunning the original minimal CLI counterexample now rejects with `completion requires exact four output paths`. Tests also reject partial inventory and changed protocol/registration/path/hash/rows/ordered IDs/template/source map/fit count, while a valid synthetic receipt passes.

2. **Fixed — code-only Git commit satisfied registration.** The final worker requests `HYPOTHESES_LOG.md` first and requires the exact protocol in that commit before source/input metadata. The independent mocked-Git counterexample now rejects with `protocol absent from preregistration journal`; the only Git call before rejection is the journal. A synthetic seal regression confirms no Parquet metadata read occurs first.

## Completed scope checks

The full adapter and tests were read. Existing producer order and actual input paths were traced through the eleven pinned producers. E106 coalesce matches `run_final_any_period.sh`; E113 remains separate. E163/route/wake/plan residual merge order is explicit. All **11 source SHA** match the source contract; **35 external paths** are unique and cover the actual producer reads.

Read-only schema footers independently match the contract: E106 193 columns, E113 193, FIN1b 210, motion 37, final raw 30. No historical feature values were read. Final DEP TAXITIME/BLOCK rejection, exact train/final/template IDs and splits, unchanged ARR context boundary, training-label fingerprints and input-only E165/E157 union semantics were inspected. Fresh copies, read-only destination modes, path containment, no symlink/hardlink staging, no resume/overwrite and subprocess stop-on-failure are implemented. Mac metadata limits are explicitly soft phase RSS checks plus CPU/wall limits; Linux cgroup admission and external root-owned lifecycle/resource scheduling remain separate.

**12/12 final synthetic tests PASS** in 0.961 seconds; the two independent original counterexamples are rejected. All **five final readiness pins and eleven producer pins** independently match. The initial suite had 10 passing tests before the two uncovered admission regressions were added.

| Frozen file | SHA-256 |
|---|---|
| `scripts/final_feature_workspace_20261008.py` | `33479b6d8e8e9648b0c5c6410cba157f940ce6ca0389cd031f8164326f56b1be` |
| `tests/test_final_feature_workspace_20261008.py` | `c14ac43f56d4b33efbee027772340d80a37523cb63f4f0837dbfee5a5bf67feb` |
| `PROTOCOL.md` | `25323d69615df342b45a50c8e9f7dbc2dd6cd2929e867f2d61be36cb9a2ccec9` |
| `source_contract.json` | `0e0b16de32c76529f490afd59da601cc51e15d14d305e860347442f9261082da` |
| `external_availability.json` | `bae0395d96c48a2e544e6a78da06e569a4c1d881a6c4395b46c1691ad3e121c6` |

Remote stage/build/verify need the trusted root manifest SHA and exact executing helper, not `.git`. Home execution still requires root's qualified environment, exact owned lifecycle/resource admission and E207 resource release. Mac's 3GiB check remains an explicitly soft phase high-water RSS rejection with 900-second CPU/wall bounds; Linux enforces the documented cgroup caps. The existing Feb/Jun external-availability receipt establishes calendar presence, not per-flight match quality; the actual post-registration seal checks all required dates.

This review makes no claim about real feature-build RAM/time, final-period weather match rates, final prediction quality, or readiness of downstream training. It authorizes no download, transport, model fit or submission by itself.

# E208 independent review, 2026-10-08

Reviewer: `/root/fresh_competitors`. **READY for root preregistration of this label-free coverage pilot**, at the exact frozen hashes below. Open findings: **Critical 0 / Important 0 / Minor 0**. This verdict does not authorize a model, fit, ranking inference, or submission.

Read the complete worker, tests and protocol, inventory/readiness/source summary, and the original two-day scanner's candidate-selection path. Ran only local synthetic tests and read existing receipts. No source/test edits, metadata pack, archive request, remote job, or target evaluation were performed by this reviewer. The only file written is this report.

## Resolved Important finding

`scripts/e208_datis_coverage.py:87` in the previous frozen source `f52a14e1ec1f8ca9497dddbb1a5bba009bb63bedb4a5c55a61e492412ca9da80` recognized `AT AT`, but ordinary `AT` and `BTN` were omitted. The independently reproduced bulletin `EGLL DEP ATIS A 2355Z RWY 09L CLSD AT 0030Z`, received at 23:56Z, incorrectly set current closure and operational coverage. `BTN 0030Z AND 0430Z` did the same. Timed condition codes also required suppression.

The author corrected the bounded timing grammar in `scripts/e208_datis_coverage.py:91` and suppresses both closures and conditions at line114. Tests at `tests/test_e208_datis_coverage.py:84` cover AT/BTN/BETWEEN closures and AT/BTN/FROM conditions, including the distinction between ordinary ATIS coverage and operational coverage. I reran the counterexamples on the final frozen source: all pass. `PROTOCOL.md:23` now explicitly limits the recognized cues, 65-character scope and clock forms, and explains conservative exclusion of mixed current/scheduled text. This is not a comprehensive aviation grammar claim.

## Verification

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s tests -p test_e208_datis_coverage.py` — **17 tests PASS**, including actual local zstd compression/decompression with synthetic payload, malformed JSON rejection, no-request expiry/asset cap, midnight/future/naive timestamps, ARR/DEP separation, expiry boundary, complete-only messages, same-issue conflicts, supersession and unique-flight unions. The real zstd test executed, rather than being skipped. No dependencies were installed.

The initial attempted pytest invocation could not run because pytest is absent; the existing unittest runner above was then used successfully. This is not a claim of a pytest result.

Independently reran `reuse_assets` against the existing receipts. Exact result: 253 assets / 143,913,547 compressed bytes; 58 complete accepted messages, 408 assembly-not-complete, 54 not-A9; 4 accepted messages with recognized operational state. This agrees with the frozen parser-check receipt. These are message counts across two historical days, **not flight coverage**, and the earlier 466 A9/header candidates do not establish usable coverage.

The inventory lists 12 of 16 fixed days, 1,800 assets and 766,417,808 declared bytes. Four January 2025 HTTP404 days remain source-unavailable. After verified reuse, the declared new requests total 622,504,261 bytes. Asset reservation, declared-size reads, compressed digest checks, no retries, and per-asset limits implement the network budget. Partial/unavailable days remain distinct from absence of ATIS; the final artifact cannot become model-approved.

Reviewed the full source-to-output path: canonical projection selects only ID/phase/airport/MVT, keeps DEP and fixed days, and stores only ID/airport/MVT. Whole-file source SHA reads opaque bytes without decoding targets. Reuse is checked against the complete asset URL set, release SHA, individual GitHub compressed digest, size and successful parsing, then passed through the current sanitizer. Candidate selection matches the old uppercase ATIS/target-ICAO superset. The as-of index enforces received and issue timestamps no later than departure, fixed issue+60-minute expiry, role separation, conflicts and deduplication. Retained records start at the ATIS header and omit source receiver/aircraft identifiers.

The network/decode lifecycle has bounded per-read timeouts, an internal 44-minute deadline, decoder timeout/kill and process reap. **CPU, total RSS and the hard 45-minute process-tree deadline depend on the specified root-owned unit**: CPU quota at most 4, MemoryMax=4GiB, swap=0, RuntimeMaxSec=2700, KillMode=control-group, nice10. This review verifies the Python side and protocol; it does not certify an unlaunched systemd unit. Root must retain that existing launch contract. No automatic retry or adaptive date selection was added.

## Exact reviewed frozen files

| File | SHA256 |
|---|---|
| `scripts/e208_datis_coverage.py` | `2759d327e63b2085e857ab6d9a6d3da2dae86f48e74c7d64a066edb43c3ab2c8` |
| `tests/test_e208_datis_coverage.py` | `d02db06584dc626da1193883d39b2c6b033e30d0b7035afc830462268e113443` |
| `analysis/e208_datis_20261008/PROTOCOL.md` | `959b62897c4823ed3d1c649fde1f7ac0df9e9903d96d441bfb9161c078464427` |
| `analysis/e208_datis_20261008/inventory.json` | `4fde6d144587c29c213f99cb009fc09b8fa6b92fc894d748190f84543acf12fa` |
| `analysis/e208_datis_20261008/existing_receipts_parser_check.json` | `5112995b7ffe4f38c0c7bf48c40146d6c5d794cde612aab517a575bfef02c33d` |
| `analysis/e208_datis_20261008/SOURCE_INVENTORY.md` | `aa2670f6cbc5417fdfdb3c01ee0ea1101226a384a52f6ff26c361a3fffacb96c` |
| `analysis/e208_datis_20261008/readiness.json` | `dd9238e3bd8c2e340b28f38e24009b7545502109fc612565a35127f3f7b9e30a` |

All six file hashes in readiness matched independent SHA256 reads after the author's 10:47:40Z re-freeze.

## Declined to judge

No real pack/scan/coverage run was executed by this reviewer. I therefore decline to judge realized source/flight coverage, real-run memory/time use, unlaunched remote cleanup, all possible ATIS wording, operational truth of published bulletins, predictive usefulness, future CV/LB effect, or ranking availability beyond the fixed-date source inventory. The bounded parser pilot and the source's missing days must remain visible in the results.

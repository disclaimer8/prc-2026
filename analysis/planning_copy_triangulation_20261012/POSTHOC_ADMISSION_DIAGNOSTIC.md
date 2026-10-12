# Post-hoc technical admission diagnostic

Status: bounded correctness diagnostic after frozen registration #464; not a new scientific estimand or a corrected outcome run.

The original raw result remains **0 accepted episodes out of 15,758 derived-eligible visits** on the fixed 36 dates. This diagnostic inspected first-point admission only; it did not read taxi labels, BLOCK, planning timestamps, alignment flags or contrasts. No date, threshold, frozen source or scientific output was changed.

Command: `.venv/bin/python analysis/planning_copy_triangulation_20261012/posthoc_admission_diagnostic.py`. Runtime: 6.92 s.

Source SHA-256: `8fcb3854d9771958df620d7f07c72c756c5f33dbc752c0dd9db571462d59822e`.
Input manifest SHA-256: `debd46591a8b4f7ee16965dfd2bb6a2f11522aada1b4390aba70656a4181a7fa`.
Frozen ID-pool file SHA-256: `790d8733ca408e7a72abce537f4c42ade43331e7b04ecbd7363c42754e23f219` (only MVT ID, date and airport columns read).

The frozen run loaded all 36 raw files. This diagnostic verified their pinned manifest, then reread raw points only for the 20 dates containing `stale` (7,870 visits); the other 16 dates / 7,888 visits retain `missing_stale_status` without assuming freshness.

## Result and scope

- First point marked stale: **7,851 / 7,870 (99.76%)**; none were null values filled to true.
- All 19 fresh first points also failed the speed and stand-distance gates: speed 73.0–122.5 kt and distance >40 m. No first point passed every gate.
- Marginal failures overlap: nonfinite speed 8; speed outside 0–1 kt 4,549 (including nonfinite); stale 7,851; stand distance 19; after takeoff 1; surface 0; self-displacement 0.
- The mutually exclusive first-failure ordering in the diagnostic gives speed range 4,541; nonfinite speed 8; freshness 3,321. This ordering is only a reporting decomposition.
- `stale` is NumPy `bool`, speed and time are `float64`. Bitwise inversion agrees with logical negation in all 7,870 cases. Raw and movement timestamps are both Unix seconds. Reproduced prefix count is zero for every inspected visit.

The reviewed source and diagnostic show no dtype, unit or Boolean-negation defect in this admission path. The exact unique callsign/airport association and initial terminal-surface-segment mapping reproduce the frozen zero-prefix result. This is a bounded technical check, not a certification of every archive extraction or physical aircraft identity.

Local provenance: `src/adsb_extract.py:85–87` encodes `stale = bool(int(p[6]) & 1)` when known. Its source comment describes that bit as a stale/repeated position after 20 seconds without reception. Extractor SHA-256: `b2a13491c5b8dcf5c0749a49588f0c9f09a775e2ddb2d0ed8635f3ddc97ae9a5` (read-only local provenance checked after outcomes, not a preregistered source pin). This is the local extractor interpretation; no external decoder verification was performed. It does not independently prove what each archived observation physically means.

**Interpretation:** the preregistered requirement that the terminal segment begin with a fresh stationary prefix has no support in this sample. The raw supplement is inconclusive; zero admissions is not evidence that the aligned timestamps are physically correct, nor evidence against planning-copy mechanisms. No later favourable stationary episode was searched or rescued.

## Fixed dates and first-point marginal failures

Missing-stale dates have no point-level diagnostic. Failure counts below overlap.

| Date | Visits | Stale field | Stale first | Speed fails | Stand fails | Time fails |
|---|---:|---|---:|---:|---:|---:|
| 2025-01-05 | 301 | present | 301 | 165 | 0 | 0 |
| 2025-01-15 | 321 | present | 319 | 193 | 2 | 0 |
| 2025-01-25 | 355 | present | 354 | 218 | 1 | 0 |
| 2025-02-05 | 371 | present | 370 | 215 | 1 | 1 |
| 2025-02-15 | 381 | present | 381 | 228 | 0 | 0 |
| 2025-02-25 | 404 | present | 403 | 233 | 1 | 0 |
| 2025-03-05 | 418 | present | 417 | 250 | 1 | 0 |
| 2025-03-15 | 408 | present | 407 | 239 | 1 | 0 |
| 2025-03-25 | 412 | present | 410 | 227 | 2 | 0 |
| 2025-04-05 | 412 | present | 412 | 249 | 0 | 0 |
| 2025-04-15 | 292 | present | 291 | 152 | 1 | 0 |
| 2025-04-25 | 318 | present | 316 | 200 | 2 | 0 |
| 2025-05-05 | 385 | present | 385 | 238 | 0 | 0 |
| 2025-05-15 | 322 | present | 321 | 217 | 1 | 0 |
| 2025-05-25 | 464 | present | 462 | 253 | 2 | 0 |
| 2025-06-05 | 427 | present | 427 | 241 | 0 | 0 |
| 2025-06-15 | 477 | present | 476 | 261 | 1 | 0 |
| 2025-06-25 | 505 | present | 504 | 268 | 1 | 0 |
| 2025-07-05 | 361 | absent | — | — | — | — |
| 2025-07-15 | 361 | present | 361 | 195 | 0 | 0 |
| 2025-07-25 | 505 | absent | — | — | — | — |
| 2025-08-05 | 475 | absent | — | — | — | — |
| 2025-08-15 | 415 | absent | — | — | — | — |
| 2025-08-25 | 568 | absent | — | — | — | — |
| 2025-09-05 | 523 | absent | — | — | — | — |
| 2025-09-15 | 481 | absent | — | — | — | — |
| 2025-09-25 | 424 | absent | — | — | — | — |
| 2025-10-05 | 480 | absent | — | — | — | — |
| 2025-10-15 | 536 | present | 534 | 307 | 2 | 0 |
| 2025-10-25 | 535 | absent | — | — | — | — |
| 2025-11-05 | 525 | absent | — | — | — | — |
| 2025-11-15 | 463 | absent | — | — | — | — |
| 2025-11-25 | 529 | absent | — | — | — | — |
| 2025-12-05 | 572 | absent | — | — | — | — |
| 2025-12-15 | 539 | absent | — | — | — | — |
| 2025-12-25 | 493 | absent | — | — | — | — |

Machine-readable evidence: `POSTHOC_ADMISSION_DIAGNOSTIC.json` and `posthoc_admission_rows.csv`. Frozen scientific outputs remain unchanged.

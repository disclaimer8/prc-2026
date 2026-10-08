# v12090 canary → 232.1933 s: the uniform final tail costs +1.25 s on Jan/Jul

**v12090: 232.1933 s RMSE**, **+1.2546 s worse** than the best v12075 (230.9387), +0.2051 s worse than
v12061 (231.9882). Processed 08.10.2026 **13:20:58.405033 UTC**, all **344841** pairs.
Submission ID `0f63954a-f553-484a-941e-36003323f9a1`; raw board item in [score_response.json](score_response.json).
Board at 13:21 UTC: team best still v12075, rank **15/237**, leader 213.2541 s.
For 8 October **1 of 5** slots is used; no other automatic uploads are planned by this transport.

File: `submissions/gentle-igloo_v12090.parquet` (hard link of
`data/final_tail_dryrun_janjul_20261008/work/gentle-igloo_v12090.parquet`),
SHA-256 `180a185e232bfb5c13fe4f2dd518558f30005fc716024b2c77498665e9ed5fe3`, 1 096 989 bytes.
This was a technical canary, not a candidate for best: it checks the new final-pipeline code path
(№389 adapter, [DRYRUN_RESULTS](../final_tail_assembly_20261008/DRYRUN_RESULTS.md)) against real Jan/Jul labels
and measures what dropping E81 and the 42 frozen v361 donor rows costs.

## Forecast check

Pre-registered in №390 before any network action: central **231.6 s**, expert corridor **[231.1; 232.3] s**,
Gram range [212.73; 249.15], uncorrelated estimate 231.6554.
Result 232.1933: **✓ inside the corridor** (near its upper edge), ✓ inside the Gram range,
**+0.59 s worse than the central forecast** and +0.54 s worse than the uncorrelated estimate.

Decomposition against v12075 (identity RMSE² = ref² + RMS²(Δ) + 2·cov):
232.1933² − 230.9387² = 581.05 s²; RMS²(Δ) = 18.2087² = 331.56 s²; so 2·cov = +249.5 s² (correlation of the change
with the v12075 residual ≈ +0.03). The change is not neutral noise; it moves predictions away from the labels on
average. The historical E81-alone price (≈ −0.66 s, v98 identity) explains roughly half of +1.25 s; the remainder is
the fresh 10-seed E156 ensemble on the 42 LIRF rows (RMS 713.5 s vs the v361 donors; weight in RMSE² ≈ 62 s² of
the 331.6 s² of change) and/or a stronger E81 effect on this chain. A single upload cannot separate the two parts;
no further uploads were made to probe it (rule 4: no probing rows through the board).

## What this means for the blind final

- The code path works end to end on the real template: schema, 344841 IDs, 42 served E156 IDs, single PUT,
  board accepted the file and scored all pairs.
- The uniform recipe **without E81 and without frozen donor rows is worse by about +1.25 s on Jan/Jul** than the
  historical best chain. Any decision to keep E81 out of the final (it blends whole-file donors v44/v60 that do not
  exist for the four-month file) or to replace the v361 donors with a fresh E156 should weigh this measured cost;
  the Jan/Jul labels do not transfer one-to-one to the blind months, but the sign is now known.
- Nothing in this result was used to pick weights; the final recipe decision stays with the coordinator.

## Delivery

Transport `scripts/submission_transport_20261008.py` (SHA `79808a22…`), copied from the 07.10 pattern:
`inspect` (network-free), `seal` (Git object proofs of both registrations), `preflight`, `upload`
(OnePutHTTP, `single_attempt`, shared ledger + lock, 90 s SIGALRM, Oct 8 UTC window ≥ 120 s before midnight).
Pinned: builder `9ff246b8` (scientific registration `9bd1f2e0`), guard `e2815dbb`, e185_submission `b1031e15`,
single-put `d7cc42d6`, template `0d383408`, v12075 `90d3cd0b`, `complete.json` `da219ef0`,
`e156_served.json` `80f5f261`; difference vs v12075 pinned (changed 313616, RMS 18.208699, max 2058) and
re-verified in preflight on both hosts. Four negative scenarios (wrong diff, wrong candidate SHA, wrong receipt SHA,
wrong protocol) stop on the Mac. Transport registration `2cc86beaee483b7359eb3a2b10bc78682009d694` (№390),
pushed before `seal`. The 1.6 MB registration proof is kept on both hosts (SHA `b1f177df…`), not committed.

Host **Hetzner** `/root/prc-2026`, interpreter `.venv-upload/bin/python` (3.14.4, minio 7.2.20, urllib3 2.8.0).
Candidate, builder, receipts and proof copied with scp; SHA-256 verified on the host before preflight.
First Hetzner preflight stopped on `hash mismatch: scripts/final_tail_assembly_20261008.py` (the pinned builder was
absent on the host); after copying it (SHA exact) preflight passed. Quota before the PUT: 0 team uploads processed on
08.10 by the public API, 0 ledger lines for 08.10, lock free, no object named v12090 anywhere.
Exactly **one PUT**, 13:19:45 → 13:20:11 UTC, no retry. Ledger line:

```
{"at": "2026-10-08T13:20:11.094253+00:00", "event": "intent",   "filename": "gentle-igloo_v12090.parquet", "sha256": "180a185e…", "used_before": 0}
{"at": "2026-10-08T13:20:11.094253+00:00", "event": "uploaded", "filename": "gentle-igloo_v12090.parquet", "sha256": "180a185e…", "used_before": 0}
```

Receipts: [upload_receipt.json](upload_receipt.json), [upload_stdout.log](upload_stdout.log),
[canonical_ledger.jsonl](canonical_ledger.jsonl) (copy of the host ledger after the PUT),
[score_response.json](score_response.json).

# v12078 → 231.0488 s: removing the E81 whole-file blend costs +0.5676 s on the v12077 chain

**v12078: 231.0488 s RMSE**, **+0.5676 s worse** than v12077 (230.4812). Processed 09.10.2026 **08:54:51.901367 UTC**,
all **344841** pairs. Submission ID `aa78ab4e-f3f7-4093-a8c5-66dcb1742c3d`; raw board item in [score_response.json](score_response.json).
Board at 08:55 UTC: team best still v12077, rank **16/239**, leader 213.0252 s, third 213.2898 s. For 9 October **1 of 5** slots is used.
This is a measurement, not a candidate for best.

File: `submissions/gentle-igloo_v12078.parquet`, SHA-256 `b0f735e4344c6c9982f992c961e76f70b1e6ad7e847568d7ec0423e5e519e287`;
build report `submissions/gentle-igloo_v12078.manifest.json` (SHA `9f8d1b58…`).

## The single change

The v12077 build path (`scripts/v12078_v12079_builds_20261009.py --variant noe81`, zero fit, 08:47–08:49 UTC) with the E81 whole-file
blend skipped: `blend_submissions.py` with `44:0.1214,60:0.0270` is not run and `lobt_edge_lift.py` reads the E152 output directly,
exactly as the final assembly does. The premix is byte-identical to the v12077 premix (`premix_parity.json`); the 42 frozen TabPFN rows,
the ten-seed NN member, the E212 weights, every other stage and the 11+5 cached stage models are unchanged.
Difference against v12077: 313 500 rows changed, RMS 16.507 s, mean −0.075 s, max |Δ| 1324 s.

## Forecast check

Pre-registered in №431 before the upload: central **230.60** (+0.12 s, from the Gram identity on the current files), expert corridor
**[230.40; 231.20]** whose upper part covered the historical E81 price (−0.6564 s on v98, i.e. ≈ 231.14).
Result 231.0488: **✓ inside the corridor**, but **+0.45 s worse than the identity-based point**; the historical price was the better guide.
Why the identity failed: it is exact for the unrounded blend of three fixed files, and the exact "deblend" X of v12077 scores 230.54 by it;
v12078 differs from X by the tail stages applied after E81 (LOBT lift, protect, half LSZH): RMS 5.48 s on 62 163 rows. With the actual
score, 231.0488² − 230.5394² = 234.9 s² = RMS²(Δ) 30.0 s² + 2·cov 204.9 s², i.e. the tail-stage interaction is strongly correlated with
the residual (corr ≈ +0.08), not noise. Lesson: the E81 blend interacts with the LSZH half-blend and the LOBT lift; the identity cannot be
applied across those stages.

## Decomposition (first half)

- **Price of E81 on this chain: v12078 − v12077 = +0.5676 s** (removing it costs 0.57 s; historical v98 identity −0.6564 s, №76/192).
- Canary v12090 (older weights 0.25/anchors ×1, three-seed member) was +1.2546 s against v12075 with both E81 removed and the 42 frozen
  rows replaced by fresh E156. If the parts add, the fresh-E156 share is ≈ +0.69 s; v12079 (№433) measures it directly.

## Delivery

Transport `scripts/submission_transport_v12078_20261009.py` (SHA `2a13a46a…`), the v12077 wrapper pattern with new pins; registration
`bf1716724924317edfdd2e9e27afd2412a8892cb` (transport, builder, journal №431 in one commit). Pinned: candidate, builder, build report,
member receipt `e5910d44`, parent v12077 `f06c66bf`, v12075, template, guard/e185/single-put/canary helpers, difference vs v12077
(changed 313500, RMS 16.506542, max 1324), stop rule RMS ∈ [0.5; 30] s; four negative scenarios stop on the Mac. Host **Hetzner**
`/root/prc-2026`, `.venv-upload/bin/python`; SHA of every copied file verified; preflight passed ([preflight_hetzner.json](preflight_hetzner.json)).
Quota before the PUT: 0 slots used on 09.10. Exactly **one PUT**, 08:53:54 → 08:54:06 UTC, no retry. Ledger lines (`used_before` 0) in
[canonical_ledger.jsonl](canonical_ledger.jsonl); stdout in [upload_stdout.log](upload_stdout.log). Registration proof (SHA `abe4913f…`) kept on
both hosts, not committed.

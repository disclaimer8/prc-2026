# v12076 → 230.8183 s: new best, −0.1204 s against v12075 (ten-seed E203 NN member)

**v12076: 230.8183 s RMSE**, **−0.1204 s better** than v12075 (230.9387). Processed 08.10.2026
**20:05:56.697106 UTC**, all **344841** pairs. Submission ID `5be7ef97-8244-4f9a-94cb-5027f50c20d6`;
raw board item in [score_response.json](score_response.json). Board at 20:07 UTC: rank **16/239**, leader 213.2541 s,
third place 213.5212 s. For 8 October **2 of 5** slots are used (v12090 canary, v12076).

File: `submissions/gentle-igloo_v12076.parquet`, SHA-256 `c22de523cd392fb035f9223c1ac2222173de0967ef5a8909ee9641876828e1b0`,
1 097 355 bytes; build report `submissions/gentle-igloo_v12076.manifest.json` (SHA `90bb9eba…`).

## The single change

v12075 replay path (`scripts/reproduce_v12075_v2.py`, journal №372/374, zero fit) with one input substituted:
the 0.25-weight NN member `data/e203_rank_20261007/prediction.npz` (seeds 0..2) → `data/e203_rank_seeds3to9_20261008/models/prediction_10seed.npz`
(seeds 0..9). E81 donors v44/v60, the 42 frozen TabPFN rows (v361), all weights and the 11+5 cached stage models unchanged.
Builder `scripts/v12076_nn10seed_build_20261008.py` (SHA `3fc1e93f…`), output `data/v12076_nn10seed_20261008/`, 1 min 57 s on the Mac, 0 fits.

**Premise correction (journal №416).** The final 10-seed NN member (№387/394/400) is a set of fresh fits on the final-view
features (`model_0.pt` 678bd50c ≠ archived E203 822b9aab; RMS 47 s against the old member, №393) and cannot be applied to the
E203 Jan/Jul queries. Seeds 3..9 were therefore fitted anew on the frozen E203 ranking pack on the Home GPU
(`scripts/e203_rank_seeds3to9_20261008.py`, unit `prc-e203-rank-seeds3to9-20261008`, 17:46–18:18 UTC after E212 left the GPU,
7 × ≈225 s, 1904 s wall). Controls, all exact: pack/prepared/archived-output SHA, runtime = admitted, PLE bins = archived
`bins.npz`, archived 3-seed mean = `prediction.npz`, archived `model_0.pt` reproduces `seed_0.npy` through the same inference
path, every new checkpoint reproduces its seed array. Member statistics: pairwise seed RMS 71.8–79.6 s (0.22–0.24 sd units),
RMS(10-seed − 3-seed) 25.38 s, mean shift −0.31 s (`models/statistics.json`, receipt `models/complete.json` SHA `e5910d44…`).

Difference against v12075: 250 190 rows changed (72.6 %), RMS 4.849 s, mean −0.003 s, max |Δ| 909 s; premix parity: `mvt_id`/`proba`
exact, member-dependent `reg`/`values`/`pred` differ by design. Implied stage pass-through of the 6.3 s premix shift ≈ 0.76.

## Forecast check

Pre-registered in №416 before the fits: central **230.91** (−0.03 s), expert corridor **[230.74; 230.99]**.
Result 230.8183: **✓ inside the corridor**, 0.09 s better than the central forecast. Decomposition (RMSE² = ref² + RMS²(Δ) + 2·cov):
230.8183² − 230.9387² = −55.6 s²; RMS²(Δ) = 23.5 s²; so 2·cov = −79.1 s² — the change is anti-correlated with the v12075 residual
(correlation ≈ −0.035): the three-seed mean carried noise that the ten-seed mean removes, consistent with the seed-noise
argument but larger than the g ≈ 0.45 estimate (the realised pass-through is ≈ 0.76). No weights were chosen from the board.

## Delivery

Transport `scripts/submission_transport_v12076_20261008.py` (SHA `b7acf02d…`), modelled on the 08.10 canary: `inspect`
(network-free), `seal` (Git object proof of registration `d706c9b6e85bfe1e618e518f4c382ac8b0d6bab7`: transport, builder,
journal), `preflight`, `upload` (OnePutHTTP, `single_attempt`, shared ledger + lock, 90 s SIGALRM, Oct 8 UTC window).
Pinned: candidate SHA, builder, build report, member receipt, guard `e2815dbb`, e185_submission `b1031e15`, single-put `d7cc42d6`,
canary helpers `79808a22`, template `0d383408`, v12075 `90d3cd0b`, difference vs v12075 (changed 250190, RMS 4.848684, max 909);
four negative scenarios stop on the Mac. Host **Hetzner** `/root/prc-2026`, `.venv-upload/bin/python`; SHA of every copied
file verified on the host; preflight passed ([preflight_hetzner.json](preflight_hetzner.json)). Quota before the PUT: 1 slot used.
Exactly **one PUT**, 20:03:51 → 20:05:22 UTC, no retry. Ledger lines (`intent`/`uploaded`, `used_before` 1) in
[canonical_ledger.jsonl](canonical_ledger.jsonl); stdout in [upload_stdout.log](upload_stdout.log). The 0.9 MB registration proof is kept
on both hosts (SHA `aa5adf6c…`), not committed.

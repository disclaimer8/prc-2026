# v12077 → 230.4812 s: new best, −0.3371 s against v12076 (E212 final weights: NN 0.35, anchors × 0.65/0.75)

**v12077: 230.4812 s RMSE**, **−0.3371 s better** than v12076 (230.8183) and −0.4575 s better than v12075 (230.9387).
Processed 08.10.2026 **20:14:08.419637 UTC**, all **344841** pairs. Submission ID `ecc78bca-21ed-4ce4-89ff-5b9765d3f74c`;
raw board item in [score_response.json](score_response.json). Board at 20:15 UTC: rank **16/239**, leader 213.2541 s, third 213.5212 s
(gap to third 16.96 s). For 8 October **3 of 5** slots are used (v12090 canary, v12076, v12077).

File: `submissions/gentle-igloo_v12077.parquet`, SHA-256 `f06c66bfa3388b17f5b6ba2362fa45a7da1be8fd5ae2d37de47cfe0e6bf2ba44`;
build report `submissions/gentle-igloo_v12077.manifest.json` (SHA `368798d2…`).

## The single change

The v12076 build (v12075 zero-fit replay with the ten-seed E203 NN member, journal №416/417) with the premix member weights of
the regular candidate changed to the E212 final weights (journal №410/425, `WEIGHTS['final']` of
`scripts/final_tail_assembly_20261008.py`, commit 2650099): raw 0.028888888888888857, lobt/sched/eobt/aobt 0.08666666666666667 each,
NN (ten-seed mean) 0.35; tree pair = remainder 0.27444 (CatBoost 0.65 / LightGBM 0.35 inside it, unchanged; frozen `blend.py`:
`r = (1 − Σw)·r + Σ w·g`). E81 donors v44/v60, the 42 frozen TabPFN rows, all stages and the cached stage models are byte-identical
to the v12076 build. Builder `scripts/v12077_nn035_build_20261008.py` (SHA `d1944cd4…`), output `data/v12077_nn035_20261008/`,
1 min 38 s on the Mac, 0 fits. Difference against v12076: 286 560 rows changed, RMS 7.422 s, mean +0.214 s, max |Δ| 909 s;
against v12075: RMS 8.682 s.

## Forecast check

Pre-registered in №426 before the upload (from №425: Jan+Jul 2025 chain-level OOF −0.11 s × transfer 0.7): central **230.74**
(−0.08 s), expert corridor **[230.62; 230.87]**. Result 230.4812: **✗ outside the corridor — on the favourable side**, 0.26 s better
than the central forecast and 0.14 s below the lower edge. Decomposition (RMSE² = ref² + RMS²(Δ) + 2·cov):
230.4812² − 230.8183² = −155.5 s²; RMS²(Δ) = 55.1 s²; 2·cov = −210.6 s² (correlation of the change with the v12076 residual ≈ −0.06).
The weight shift transferred to the board about three times stronger than the OOF-derived forecast (−0.34 s vs −0.11 s on the
2025 OOF before the 0.7 transfer factor); the sign and the direction agree with E212. Nothing was tuned on the board: both the
weights and the forecast were fixed before the upload.

## Delivery

Transport `scripts/submission_transport_v12077_20261008.py` (SHA `0990d29b…`), the v12076 wrapper with new pins; registration
`58df6b9c8653b3fef3e1c53c62c91359322c34ed` (transport, builder, journal №426 in one commit). Pinned: candidate, builder, build report,
member receipt `e5910d44`, parent v12076 `c22de523`, v12075 `90d3cd0b`, template `0d383408`, guard/e185/single-put/canary helpers,
difference vs v12076 (changed 286560, RMS 7.422038, max 909), stop rule RMS ∈ [0.5; 20] s; four negative scenarios stop on the Mac.
Host **Hetzner** `/root/prc-2026`, `.venv-upload/bin/python`; SHA of every copied file verified; preflight passed
([preflight_hetzner.json](preflight_hetzner.json)). Quota before the PUT: 2 slots used. Exactly **one PUT**, 20:12:26 → 20:13:33 UTC,
no retry. Ledger lines (`intent`/`uploaded`, `used_before` 2) in [canonical_ledger.jsonl](canonical_ledger.jsonl); stdout in
[upload_stdout.log](upload_stdout.log). The registration proof (SHA `f221f88f…`) is kept on both hosts, not committed.

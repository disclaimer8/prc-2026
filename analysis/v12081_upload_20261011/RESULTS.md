# v12081 → 230.4478 s: E218 PRU level stage on v12077, −0.0334 s (new best); ✗ outside the β corridor, = the "LIRF gap is an artefact" scenario

**v12081: 230.4478 s RMSE**, **−0.0334 s** against v12077 (230.4812). Processed 11.10.2026 **08:51:08.916526 UTC**, all **344841**
pairs, 34 s after the PUT. Submission ID `95bd150b-e18c-45ca-9ea1-fdf50624e0eb`; raw board item in [score_response.json](score_response.json).
Board at 08:52 UTC (best per team): rank **18/242**, leader strong-crane 212.8457, third jolly-lobster 213.2898. Slots on 11.10: **1 of 5**.

File: `submissions/gentle-igloo_v12081.parquet`, SHA-256 `719622263e3e66fb39ad67d3b4255dc7317f9a6449385b136de5d8d80dd42f4c`;
build report `submissions/gentle-igloo_v12081.manifest.json` (SHA `4c498885…`).

## The single change

Builder `scripts/v12081_e218_build_20261011.py` (SHA `fa5b2bcb…`), output `data/v12081_e218_20261011/`, 1 min 59 s on the Mac, 0 fits.

1. The v12077 chain (v12075 replay, ten-seed E203 NN member, E212 final weights, E81 v44/v60, 42 frozen TabPFN rows, cached
   E141/E64/LSZH models, E188) was rebuilt unchanged. The rebuilt `half` output (b+15) is **byte-identical to v12077**
   (`f06c66bf…`, [rebuild_v12077.json](../../data/v12081_e218_20261011/rebuild_v12077.json)); the build stops otherwise.
2. One post-mixture stage `pru_level` (b+15 → b+16), placed where the E218 patch places it in the final (after `half`). The functions
   `pru_read/pru_ok/pru_reference/pru_level_shift` and the stage's `dep` reader are executed from the post-image of the SHA-pinned patch
   `analysis/e218_pru_level_recheck_20261010/final_tail_e218_pru_level_stage.patch` (`0613fdd9`), not retyped. k = 0.5, cell eligible if
   PRU's excluded share is < 0.5 pp above the airport's Apr–Oct 2025 level; reference = P10 stand × runway over the 12 training files;
   ranking = `data/prc-2026-datasets/ranking.parquet` (`9db75e3a`); PRU 2025/2026 `4aedf195`/`b72040e0`. Protected = the 42 frozen
   TabPFN rows (their role here = the 62 E156 rows in the final). Row order restored to v12077.
3. Extraction proof: the same code on the final `76c32426` (read only, final workspace ranking and E156 IDs) reproduces the E218 dry-run
   `bbae2cc5` on all 670 790 rows, 415 697 changed ([stage_parity_final_dryrun.json](stage_parity_final_dryrun.json)).

Difference against v12077: **215 682 rows changed, RMS 3.532 s**, mean −0.984 s, max |Δ| 8 s; everything else byte-identical (row order,
dtypes, the 42 TabPFN rows). 14 of 20 Jan/Jul cells eligible (January EDDF/EDDM/LEMD/LFPG/LSZH/LTFM excluded as de-icing). Every row of P in
a cell moves by rint(c): LIRF Jan/Jul +8/+8 (c 8.49/8.07), LEMD Jul −7, LFPG Jul −5, LTFM Jul −5, LEBL Jul −4, EDDM/EHAM/LSZH Jul −3,
EGLL Jul −2, EGLL Jan −1, EHAM Jan +2, EDDF Jul +1, LEBL Jan 0 (c −0.33). Per-cell table: [cells_v12077_e218.csv](cells_v12077_e218.csv).

## Forecast check (pre-registered №455, commit `0bf06cec`)

E218 board model recomputed on v12077 ([forecast.json](forecast.json)): ΔSSE = Σ n_P (c² + 2c·β(gap − emu_hat)), β = 0.932 ± 0.067.

| scenario | Δ, s | score |
|---|---:|---:|
| β̂ = 0.932 (central) | −0.073 | 230.408 |
| β ± 2σ (corridor) | −0.088 … −0.059 | 230.394 … 230.422 |
| LIRF gap is an artefact (b_LIRF = 0, others β̂) | −0.033 | 230.448 |
| gap is noise (β = 0) | +0.027 | 230.508 |
| **observed** | **−0.0334** | **230.4478** |

**✗ outside the corridor** (0.025 s above its upper edge), on the scenario "LIRF gap is an artefact" to the fourth decimal (230.4479 model,
230.4467 with the rounded row shifts). Decomposition: 230.4478² − 230.4812² = −15.40 s² = RMS²(Δ) 12.47 + 2·cov −27.87 s²
(Σ Δ·(p − y)/N = −13.93 s², corr(Δ, residual) −0.017). One board number cannot separate the cells; two readings fit it exactly:
- uniform β over all 14 cells: **β ≈ 0.56** (2025: 0.93);
- non-LIRF cells at their 2025 β: **β_LIRF ≈ −0.03** (LIRF shift neither helped nor hurt beyond its c² cost), or equivalently LIRF b = 0 and
  **β_non-LIRF ≈ 0.92** — the 2025 value.

The second reading is the pre-registered artefact scenario; the first is not excluded. Either way the 2025 relation transfers only partly,
and the LIRF part (40 % of the β̂ expectation here, 71 % of the blind-month expectation in the final) is the doubtful one.

## Decision rule (written before the upload, №455)

score ≤ 230.4512 (v12077 − 0.03) ⇒ the coordinator rebuilds the final with E218 enabled; worse than v12077 ⇒ out; in between ⇒ out.
**230.4478 ≤ 230.4512 ⇒ rule says: rebuild the final with E218 enabled** (`E218_PRU_LEVEL=1`, new make-contract and seal). The margin is
0.0034 s. For the final, the E218 estimate (№452, board.json) under the scenario this board matches is −0.025 s on Jan/Jul, −0.006 s on
Feb/Jun and −0.016 s on all four months (vs −0.130 s at β̂): small but not negative, because the non-LIRF cells pay for the LIRF shifts.

## Delivery

Transport `scripts/submission_transport_v12081_20261011.py` (SHA `2968a3ea…`), the v12078 wrapper pattern with new pins; registration
`0bf06cece237a5248454182891706eb86b75d869` (journal №455, builder, transport, forecast/parity/cell files in one commit). Pinned: candidate,
builder, build report, member receipt `e5910d44`, parent v12077 `f06c66bf`, v12075, template, patch, both PRU files, ranking, guard/e185/
single-put/canary helpers, difference vs v12077 (changed 215682, RMS 3.531720, max 8), an independent cell check (every moved row lies in one
of 13 pinned airport-months and moved by exactly its integer shift), stop rule RMS ∈ [0.5; 30] s. Five negative scenarios stopped on the
Mac (tampered candidate, altered cell table, upload from the Mac, foreign commit/proof, seal before registration). Host **Hetzner**
`/root/prc-2026`, `.venv-upload/bin/python`; SHA of every copied file verified; preflight passed ([preflight_hetzner.json](preflight_hetzner.json)).
Quota before the PUT: 0 slots used on 11.10 (shared ledger + public board; scorer live, last other-team item 08:48:43Z). Exactly **one PUT**,
08:50:19 → 08:50:34 UTC, no retry. Ledger lines (`intent`/`uploaded`, `used_before` 0) in [canonical_ledger.jsonl](canonical_ledger.jsonl);
stdout in [upload_stdout.log](upload_stdout.log). Registration proof (SHA `c79be1e1…`) kept on both hosts, not committed. PRC keys were
neither read nor printed by the agent.

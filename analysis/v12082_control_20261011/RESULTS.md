# v12082 → 231.1445 s: Jan/Jul rows of the final with E218, −0.0360 s against v12080; ✓ corridor; the E218 file becomes the final

**v12082: 231.1445 s RMSE**, **−0.0360 s** against v12080 (231.1805, the Jan/Jul rows of the 09.10 final `76c32426`). Processed
11.10.2026 **09:10:50.734824 UTC**, all **344841** pairs, 35 s after the PUT. Submission ID `a5c59dc1-d149-4daa-a437-f151c74d4db0`;
raw board item in [score_response.json](score_response.json). Best of the team unchanged: v12081 230.4478. Slots on 11.10: **2 of 5**.

Journal: pre-registration №457 (commit `f9694de3`), build result + control pre-registration №458 (commit `4598d0fa`), verdict №459.

## What was rebuilt (№457)

- `scripts/final_tail_assembly_20261008.py` 9ceab78e → **21f46756**: the E218 patch
  (`analysis/e218_pru_level_recheck_20261010/final_tail_e218_pru_level_stage.patch`, 0613fdd9) applied unchanged, plus contract pinning:
  with `E218_PRU_LEVEL=1` `make-contract` pins `data/taxi_out_additional_time_{2025,2026}.csv` (4aedf195 / b72040e0) and writes a
  `pru_level` block; `admit` refuses when flag and contract disagree; `check_plan` checks the stage parameters (k 0.5, xs_max 0.005).
  Flag unset: `plan` byte-identical, a fresh `make-contract` equals contract 6af6c4ca except `created_unix`. Tests 22/22 with and without
  the flag (4 new).
- `E218_PRU_LEVEL=1 make-contract` (same arguments as №436/438) → `data/final_tail_20261011/contract.json` (7a1cde30, 4027 files) →
  `prepare` → `assemble --repeat-e156`: 231 s, peak 10.8 GB, 0 fits.
- Output **`data/final_tail_20261011/work/gentle-igloo_final.parquet` SHA `bbae2cc5…34a1`**, 670 790 rows, 2 145 576 bytes —
  byte-identical to the E218 dry-run of №452.
- Receipts 00–11 equal the 09.10 build field by field except `unix/seconds/job_sha256` and three explained fields
  ([receipt_parity.json](../final_tail_20261011/receipt_parity.json)): premix parquet row-group split (values/order/schema and
  `parts_v970100.npz` identical), E156 second-run time, E188 receipt keyed by absolute workspace paths. Stage outputs v970101…v970115
  byte-identical; `half` output = 76c32426; E156 repeat bitwise; the 62 E156 rows unchanged.
- Only `pru_level` changes rows: 415 697 rows, RMS 4.687 s, mean −0.456 s, max 21 s; 29/40 cells eligible; Jan/Jul 201 598 rows
  (RMS 3.80 s), Feb/Jun 214 099 (RMS 5.47 s). Per cell: [rows_changed_by_cell.csv](../final_tail_20261011/rows_changed_by_cell.csv),
  [pru_level_cells.csv](../final_tail_20261011/pru_level_cells.csv).
- Validator against v12081 (230.4478): ALL HARD CHECKS PASS ([validate_vs_v12081.json](../final_tail_20261011/validate_vs_v12081.json)).
- Replay in a second fresh workspace `data/final_tail_20261011/work_replay`: same SHA bbae2cc5, receipts 00–12 equal.
- [MANIFEST.json](../final_tail_20261011/MANIFEST.json) (MANIFEST_KEYS of the transport, `builder` = `scripts/final_tail_assembly_20261008.py`).

## Forecast check (pre-registered №458, commit `4598d0fa`)

v12082 − v12080 is exactly the E218 shift on Jan/Jul rows (12 cells, 201 598 rows) on the same labels, so the score difference is an exact
measurement of E218 on the final's rows. [forecast.py](forecast.py) / [forecast.json](forecast.json):
ΔSSE = Σd² + 2Σd(F − v12077) + 2Σd(v12077 − y); the first two terms are label-free, the third uses v12077 cell biases fitted to the v12081
projection.

| reading | score |
|---|---:|
| (a) one β for all cells (0.562 from v12081) | 231.1393 |
| (b) non-LIRF at the 2025 β̂ 0.932, LIRF fitted | 231.1478 |
| **point (mean of a, b); corridor ±0.15** | **231.144 [230.994; 231.294]** |
| β̂ in every cell (2025) | 231.0975 |
| β = 0 (gap is noise) | 231.2028 |
| naive transfer of −0.0334 | 231.1471 |
| **observed** | **231.1445** |

**✓ corridor**, 0.0005 s from the point. Decomposition: 231.1445² − 231.1805² = −16.644 s² = Σd²/N 14.438 + 2·cov −31.082;
cov −15.541 s² = label-free Σd(F − v12077)/N −2.067 + Σd(v12077 − y)/N −13.474. The second projection gives a uniform β 0.516 (v12081: 0.562)
or, with non-LIRF cells at 0.932, λ_LIRF +0.036 (v12081: −0.026). Both board numbers agree: non-LIRF cells transfer the 2025 relation almost
fully, the LIRF shift is about neutral beyond its c² cost. Blind Feb/Jun expectation (board.json model of №452): −0.006 s under "LIRF ≈ 0"
up to −0.07 s under a uniform β 0.52; LIRF June (+21 s) is the main risk.

## Decision (rule written before the upload, №458)

v12082 > 231.1805 ⇒ the final stays 76c32426; otherwise the E218 file becomes the final. **231.1445 ≤ 231.1805 ⇒ the final is now
`data/final_tail_20261011/work/gentle-igloo_final.parquet`, SHA `bbae2cc544989a0faf68abb85dbb4a9aab5ea38eec98d01fa9282c332b4234a1`.**

New final manifest `analysis/final_transport_20261009/manifest_final_v2.json` (SHA `3648f566…`, day 2026-10-12, builder 21f46756,
MANIFEST.json d676f93c, validator report 340a5226); sealed, committed, proof exported, copied to Hetzner, **preflight only** — no final PUT.
The old `manifest_final.json` (76c32426, builder 9ceab78e) no longer validates on Hetzner because the builder there is now 21f46756; the old
builder is kept at `/root/prc-final-backup-20261009/final_tail_assembly_20261008.py.9ceab78e` (and in git at `185df32`).

## Delivery of v12082

Transport `scripts/final_transport_20261009.py` (831712bf), mode `control`, manifest
`analysis/final_transport_20261009/manifest_control_v12082.json` (seal 09:08:32Z), registration `4598d0fa5cb5511af26d4474fb513f6bc150eda9`,
proof `registration_proof.json` (69dd416e; kept on both hosts, not committed; on Hetzner as `analysis/final_transport_20261009/proof_control_v12082.json`).
Copied to Hetzner `/root/prc-2026` and verified by SHA: builder, the E218 final, its complete.json / MANIFEST.json / validator report, the
control file e87b7978, manifest, proof; preflight passed on the Mac and on Hetzner ([preflight_hetzner.json](preflight_hetzner.json)).
Quota before the PUT: 1/5 used on 11.10 (v12081); scorer live (last other-team item 08:58:02Z). Exactly **one PUT**, 09:10:05 → 09:10:16 UTC,
`/root/prc-2026/.venv-upload/bin/python`, no retry. Ledger lines in [canonical_ledger.jsonl](canonical_ledger.jsonl), stdout in
[upload_stdout.log](upload_stdout.log), stderr empty. PRC keys were neither read nor printed by the agent.

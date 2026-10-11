# v12083 → 231.1749 s: removing the E218 shift on LIRF rows costs +0.0304 s; the LIRF cells carry 84 % of the E218 gain on Jan/Jul; the final stays bbae2cc5

**v12083: 231.1749 s RMSE**, **+0.0304 s** against v12082 (231.1445) and −0.0056 s against v12080 (231.1805). Processed
11.10.2026 **09:29:17.212774 UTC**, all **344841** pairs, 35 s after the PUT. Submission ID `5f792d66-2b3e-431b-849a-d20b690bb083`;
raw board item in [score_response.json](score_response.json). Best of the team unchanged: v12081 230.4478 (rank 18/242 at 09:30 UTC;
leader strong-crane 212.8457, third jolly-lobster 213.2898). Slots on 11.10: **3 of 5**.

Journal: pre-registration №460 (commit `a80bb0a8`), verdict №461.

## The single change

Builder `scripts/v12083_lirf_e218_exclusion_build_20261011.py` (SHA `6fd11a6b…`), 0 fits, no labels. Every LIRF row of the board
(ADEP_mvt = LIRF in `data/prc-2026-datasets/ranking.parquet`) takes its v12080 value (= the Jan/Jul rows of the 09.10 final `76c32426`,
i.e. the same row before the E218 stage); every other row keeps v12082 (= the Jan/Jul rows of the E218 final `bbae2cc5`).
Checks in the builder: v12082 − v12080 is exactly the 12 Jan/Jul E218 cells, one integer shift each (= rint(c) of
[pru_level_cells.csv](../final_tail_20261011/pru_level_cells.csv)); v12080/v12082 are the Jan/Jul rows of 76c32426/bbae2cc5;
v12083 − v12082 = 25 017 rows (LIRF Jan 10 549 × −9, LIRF Jul 14 468 × −11), RMS 2.7485 s, max 11; all 26 899 LIRF rows equal v12080,
the other 317 942 rows equal v12082; row by row v12083 = the Jan/Jul rows of "bbae2cc5 with the LIRF rows of 76c32426" (that swap
changes 49 433 rows of the four-month final — exactly the LIRF rows E218 moved). Candidate `submissions/gentle-igloo_v12083.parquet`
SHA `a834327581cec4b3cff90666ebb808b27c8511449b9be902086443f2a123df1c`, build report `submissions/gentle-igloo_v12083.manifest.json` (`2bda5dd3…`).

## Forecast check (pre-registered №460, commit `a80bb0a8`)

[forecast.py](forecast.py) / [forecast.json](forecast.json): ΔSSE(83 − 82) = −Σe² − 2Σd(v12080 − v12077) − 2T over the LIRF rows;
Σe² = 2 605 097 and Σd(v12080 − v12077) = −850 942 s² are label-free; T = Σ_LIRF d·(v12077 − y) is the only label term.

| reading (pre-registered) | score | Δ vs v12082 |
|---|---:|---:|
| 1 "LIRF ≈ neutral": non-LIRF at the 2025 β̂ 0.932, LIRF = rest of v12082 | 231.1407 | −0.0038 |
| 1′ same, LIRF λ from v12081 / T = 0 | 231.1375 / 231.1388 | −0.0070 / −0.0057 |
| 2 one β for all cells, 0.516 (v12082) / 0.562 (v12081) | 231.1659 / 231.1683 | +0.0214 / +0.0238 |
| β̂ 0.932 in LIRF too (2025 relation) | 231.1877 | +0.0432 |
| **point (mean of 1 and 2); corridor** | **231.1533 [231.130; 231.174]** | |
| **observed** | **231.1749** | **+0.0304** |

**✗ corridor** by 0.0009 s (upper edge 231.174). Reading 1 is refuted (miss 0.034 s); reading 2 has the right sign and is the closer
one, but the LIRF effect is larger than a uniform β gives.

## What the board now says exactly

v12080 → v12083 adds only the non-LIRF E218 cells; v12083 → v12082 adds only the LIRF cells, on the same labels. The three boards split
the E218 gain on Jan/Jul exactly (to the 4-decimal rounding), [verdict.py](verdict.py) / [verdict.json](verdict.json):

| part of v12082 − v12080 | ΔSSE, s² | Δ score |
|---|---:|---:|
| non-LIRF cells (10 cells, 176 581 rows) | −892 859 | −0.0056 |
| LIRF cells (2 cells, 25 017 rows) | −4 846 571 | −0.0304 |
| total E218 | −5 739 430 | −0.0360 |

- Measured label term T = −2 874 892 s²: in the v12077 parametrisation the LIRF cells transfer **λ_LIRF = 0.688** (rounding band
  0.686…0.690) of their gap; the non-LIRF cells of v12082 then sit at **β = 0.367** (with v12081's cells and that λ: 0.483). The picture
  of №456/459 ("non-LIRF cells transfer the 2025 relation, LIRF ≈ neutral") was one of two readings that one number could not separate;
  the separating measurement picks the other side: **LIRF carries most of the E218 gain, the non-LIRF cells transfer less than half of
  the 2025 relation.**
- In the parametrisation of the final (residual of a LIRF cell = β_F·(−2c)), Jan/Jul give **β_F = 0.74**. Removing the LIRF cells from
  the final would cost on Jan/Jul 0.0156 s in final-file units (measured) and, if Feb/Jun follow the same β_F, **+0.045 s** on Feb/Jun
  (model, not measured); break-even for Feb/Jun is β_F = 0.25. The June shift (+21 s) is still the largest single bet; Jan/Jul LIRF
  (+9/+11 s) is now the best-measured part of E218, not its doubtful part.

## Decision (rule written before the upload, №460)

v12083 ≤ 231.1475 (v12082 + 0.003) ⇒ rebuild the final with E218 excluding LIRF cells; otherwise the final stays bbae2cc5.
**231.1749 > 231.1475 ⇒ the final stays `data/final_tail_20261011/work/gentle-igloo_final.parquet`, SHA
`bbae2cc544989a0faf68abb85dbb4a9aab5ea38eec98d01fa9282c332b4234a1`, manifest `analysis/final_transport_20261009/manifest_final_v2.json`.**
No exclusion code, no new contract, no `manifest_final_v3.json`. The final v2 preflight was re-run on Hetzner after this upload and
still passes ([preflight_final_v2_recheck_hetzner.json](preflight_final_v2_recheck_hetzner.json)); the 12.10 command is unchanged:

    cd /root/prc-2026 && .venv-upload/bin/python scripts/final_transport_20261009.py upload --root /root/prc-2026 \
      --manifest analysis/final_transport_20261009/manifest_final_v2.json \
      --registration-proof analysis/final_transport_20261009/proof_final_v2.json \
      --registration-commit 6b130265f467b050182600cfe0d7feb6bb72100e \
      --source-sha256 831712bfd165c3e94a20f85a801047514ab6f462231e1c8cdfedc6264650ab4f

## Delivery

Transport `scripts/submission_transport_v12083_20261011.py` (SHA `36250f77…`), the v12081 wrapper pattern with new pins; registration
`a80bb0a87d15f349216cfffe975280b616e8bb21` (journal №460, builder, transport, forecast in one commit, pushed before the upload). Pinned:
candidate, builder, build report, v12082 `e87b7978`, v12080 `ae1680cb`, ranking `9db75e3a`, template, guard/e185/single-PUT/canary helpers;
difference vs v12082 (changed 25017, RMS 2.748543, max 11); an independent check (LIRF rows = v12080, all other rows = v12082, moved cells
exactly LIRF-1 −9 / LIRF-7 −11); stop rule RMS ∈ [0.5; 30] s. Negative scenarios stopped on the Mac: tampered candidate, altered cell
table (via the report and in the independent check), another airport, upload from the Mac, seal before registration, foreign
commit and foreign proof. Host **Hetzner** `/root/prc-2026`, `.venv-upload/bin/python`; SHA of every copied file verified; preflight
passed ([preflight_hetzner.json](preflight_hetzner.json)). Quota before the PUT: 2/5 used on 11.10 (v12081, v12082; shared ledger +
public board; scorer live, last other-team item 09:21:31Z). Exactly **one PUT**, 09:28:29 → 09:28:42 UTC, no retry. Ledger lines
(`intent`/`uploaded`, `used_before` 2) in [canonical_ledger.jsonl](canonical_ledger.jsonl); stdout in [upload_stdout.log](upload_stdout.log),
stderr empty. Registration proof (SHA `027e9920…`) kept on both hosts, not committed. PRC keys were neither read nor printed by the agent.

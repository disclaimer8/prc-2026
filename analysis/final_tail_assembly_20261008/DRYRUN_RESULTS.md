# Jan/Jul dry run of the uniform tail on v12075-era artifacts (8 October 2026)

Technical readiness run, not a submission and not a scientific experiment. Output
`data/final_tail_dryrun_janjul_20261008/work/gentle-igloo_v12090.parquet` (SHA-256
`180a185e232bfb5c13fe4f2dd518558f30005fc716024b2c77498665e9ed5fe3`), **not uploaded**; version
12090 is reserved for this file (range 12090–12099 for tail-assembly dry runs).

Inputs: old ranking/template (344,841 DEP), `features_fin1b/e106/e113` of the v12075 chain,
members `cat_fin1b_seeds15`, `lgb_reg_seeds_v870`, `e130_{raw,lobt,sched,eobt,aobt}f_rank`,
`data/e203_rank_20261007/prediction.npz` (3 seeds), `e185_motion_rank`, classifier `clf_v870.txt`;
62 historical Jan/Jul 2026 daily products per family (old MVT; acceptable only for this dry run);
3,481 files pinned in `contract.json` (SHA `acff8a55…`). Adapter SHA at run `9ff246b8…`
(sealed.json = committed file). Receipts copied to [dryrun_janjul/](dryrun_janjul/).

## Stage receipts

| Stage | Changed rows | RMS change, s | Evidence |
|---|---:|---:|---|
| premix | — | — | all five parts fields and the parquet **byte-identical** to the archived v12075 premix `parts_v940100.npz` / `gentle-igloo_v940100.parquet` |
| E185 | 4,768 | 6.41 | off-subset identity asserted by the frozen source |
| E167 | 178 | 14.35 | |
| E156 | 42 | 88.04 | 42 served IDs (LIRF, no NM, bins 4-8h/8-12h/12-30h); 10 seeds, `n_estimators=auto`, 91 s; **repeat run bitwise identical** (max |Δp| = 0 for all seeds, 0 of 42 rounded values differ, 45 s) |
| E141 | 222,706 | 92.72 | 5 cached models replayed on exact signature, 0 fits |
| E64 | 49,042 | 26.65 | 5 cached models replayed, 0 fits |
| E188 | 11,016 | 4.82 | 12,236 target rows, off-mask unchanged |
| E152 | 199 | 5.15 | |
| LOBT | 95 | 5.86 | |
| protect | 10 | 3.01 | all 42 served IDs restored (10 had been moved by later stages) |
| LSZH | 14,182 | 18.23 | 1 cached model replayed; changed rows outside LSZH = 0 |
| half | 14,057 | 9.11 | |

Total 141 s, `fits_performed = 0`, `e81_applied = false`, `donor_rows_copied = 0`.

## Against v12075 (`validate_final_submission.py`, all HARD checks PASS)

- 313,616 of 344,841 rows changed (90.95 %), RMS difference **18.209 s**, mean −0.059 s, max 2,058 s.
- Outside the 42 E156 rows: 313,574 changed, RMS 16.42 s — this is the removal of E81
  (whole-file blend with v44 at .1214 and v60 at .0270). On the 42 E156 rows the fresh 10-seed
  ensemble differs from the v361 single-seed donors by RMS 713.5 s (all 42 changed).
- Gram-identity range for the Jan/Jul score: **[212.7300, 249.1474] s**; uncorrelated-change
  estimate 231.6554 s against v12075 = 230.9387 s. Historical evidence: E81 alone was worth
  about −0.66 s on the board (v98 identity), so a board result near 231.6 s would be consistent
  with that; this is not a forecast of the final four-month score.
- Per-airport RMS difference: LIRF 44.6 s (E156 + E81), EHAM 17.1, LTFM 16.6, LFPG 15.0, EDDF 14.4,
  LSZH 13.6, EGLL 12.4, LEBL 11.3, LEMD 10.4, EDDM 8.5.

## Limits

The dry run used the old daily products of the old MVT, the old members and the old ranking; it
proves the adapter, the frozen stage execution, cached-model replay and E156 reproducibility, not
the final input. Whether the Jan/Jul candidate should take a regular-board slot is the
coordinator's decision.

# Final tail assembly adapter — protocol (8 October 2026)

Protocol `final-uniform-tail-E156fresh-noE81-fourmonths-20261008-v1`. Implements tasks 1–3 of
[PLAN.md](PLAN.md). Code: `scripts/final_tail_assembly_20261008.py`; tests:
`tests/test_final_tail_assembly_20261008.py` (15 synthetic tests, no competition rows opened).
Dry-run evidence on the v12075-era Jan/Jul artifacts: [DRYRUN_RESULTS.md](DRYRUN_RESULTS.md).

## Recipe (literal `stage_plan()`, checked by `check_plan`)

| # | Stage | Code executed | Versions (b = 970100) |
|---|---|---|---|
| 0 | premix | native `premix_arrays` = frozen `blend.py` arithmetic; raw probabilities recomputed from the saved classifier (`classifier.txt`, model feature order, stored category levels) — never the masked `base.proba` | parts/sub v b |
| 1 | E185 | frozen `e185_apply.py`, `E185_W = .35*2/3`, mask LOBT null ∧ ADEP ≠ LIRF | b+1 |
| 2 | E167 | frozen `e167_nm_window_rule.py` against the provided ARR rows of the sealed ranking | b+2 |
| 3 | E156 | frozen `lirf_tabpfn.py`, `TABPFN_EXTRA=pfx`, seeds 0..9, V2 CPU, `n_estimators=auto`, explicit checkpoint path (SHA cf8c519c…), `HF_HUB_OFFLINE=1`; capture wrapper records per-seed fit/query fingerprints, classes, probabilities and the served-ID manifest | b+3 |
| 4 | E141 | frozen `e141_apply.py`, seeds 42,1,2,3,4, keep-all, D7; cached models replayed only on exact training signature | b+4 |
| 5 | E64 | frozen `e64_apply.py`, same seeds, keep-all, D3; cached replay | b+5 |
| 6 | E188 | `scripts/e188_rank_20261006.py` transformed: `ROWS` = sealed row count, `validate_registration` replaced by a hash-bound bridge (original source SHA 5836e11f… = fit registration d45cb738…); apply arithmetic unchanged | b+6 |
| 7 | E152 | frozen `e152_stage_gate.py` gate 0.95 (pre = b+3) | b+10 |
| — | E81 | **absent** (b+11 unused) | — |
| 8 | LOBT | frozen `lobt_edge_lift.py`, `LOBT_CLIP=1` | b+12 |
| 9 | protect | native `protect()`: every served E156 ID restored from v b+3, including rows whose rounded value did not change | b+13 |
| 10 | LSZH | frozen `e159b_apply.py` with one hash-bound transformation (`344841` → sealed row count), `E159_APS=LSZH`, months all, shift 0, base `cv_preds_e150c_e64_keep`; cached replay; no OPDI match ⇒ row unchanged (`scope_check`: changed rows ⊆ LSZH) | b+14 |
| 11 | half | frozen `blend_submissions.py`, `.5·(b+13) + .5·(b+14)`, `np.rint` | b+15 |

Finalisation: template order, template schema, ID set, finite non-negative integers, served E156
values re-checked after LSZH; output `<work>/<output_name>`; `complete.json` with all stage receipts.
Upload is outside this adapter.

## Inputs (contract.json, every file pinned by SHA-256)

- `inputs`: `features_fin1b/e106/e113`, `ranking`, `template` (final: the sealed WORK №385 files).
- `members`: `cat`, `lgb`, `raw/lobt/sched/eobt/aobt`, `motion` (npz with `mvt_id/reg/per_seed`,
  full seed set 15/5/5/5, `reg` = seed mean), `nn` (npz `mvt_id/reg`, completion receipt listing
  seeds 0,1,2), `base` (`classifier.txt` + optional `state.json`). Every member must cover every
  prediction ID exactly once; partial/mixed seed sets are refused.
- `derivatives`: output root of `scripts/final_adsb_derivatives_20261008.py`; `complete.json`
  with `input_missing = 0`, 120 day receipts with status `processed`/`processed_no_matches`,
  `final_ranking.sha256` equal to the sealed ranking; all 7 × 120 product files pinned.
- Historical 2025 inventories (364 files per product, no 2025-12-31), OOF parents
  (`cv_preds_e116prod`, `cv_preds_nomonth`, `cv_preds_e150c_e64_keep`), training files, 44 OPDI
  event files, the 11 cached stage models with `complete.json` signatures, the 5 E188 models, the
  frozen archive closure (`data/today_chain_20261006/src`, 23 files), the TabPFN checkpoint.

## Cached stages

`_stage` worker (fresh LightGBM-only process) intercepts `lgb.train`: in `replay` the full
signature (params, rounds, data/label/weight SHA, categorical spec, feature names, package
versions, `sys.version`) must equal the cached call or the run stops before any fit; cached
model bytes are verified. `--refit e141 e64 lszh` is the explicit fixed-refit action: models are
trained by the unchanged stage source and stored under `<work>/models/<stage>` with the fresh
signature and a `cached_signature_match` flag. Default mode never trains.

## TabPFN qualification (Mac, `.venv` TabPFN 8.5.0, Torch 2.14.0, 6 threads)

`~/Library/Caches/tabpfn` was purged by macOS cache deletion on 08.10 ~13:40 UTC (free disk rose
9.5 → 14 GiB; `com.apple.cache_delete` touched). The checkpoint was restored from Hetzner
(`/root/.cache/tabpfn/…`, identical SHA cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49)
into the repo at `data/tabpfn_v2_checkpoint/` (pinned by the contract) and into the Mac cache.
Offline run with explicit `model_path`: 10 seeds on 1,441 training rows / 42 queries in 46–91 s;
two consecutive runs on the same input were bitwise identical in all ten probability arrays and
in the 42 rounded outputs (see DRYRUN_RESULTS.md).

## Commands

```sh
PY=.venv/bin/python
# dry run (Jan/Jul, v12075-era members, E81 removed, fresh E156) — done 08.10, see DRYRUN_RESULTS.md
$PY scripts/final_tail_assembly_20261008.py make-contract --mode janjul --out data/final_tail_dryrun_janjul_20261008/contract.json
$PY scripts/final_tail_assembly_20261008.py prepare --contract data/final_tail_dryrun_janjul_20261008/contract.json --work data/final_tail_dryrun_janjul_20261008/work
$PY scripts/final_tail_assembly_20261008.py assemble --work data/final_tail_dryrun_janjul_20261008/work --repeat-e156

# final (after cat/lgb/anchors/motion/base merged, the NN "models" stage and the derivatives complete.json exist)
$PY scripts/final_tail_assembly_20261008.py make-contract --mode final \
    --feature-work data/final_feature_work_mac_20261008 --tabular-root models/final_tabular \
    --nn-prediction <nn models stage>/prediction.npz --nn-complete <nn models stage>/complete.json \
    --derivatives-work data/final_adsb_derivatives_20261008 --out data/final_tail_assembly_20261008/contract.json
$PY scripts/final_tail_assembly_20261008.py prepare --contract data/final_tail_assembly_20261008/contract.json --work data/final_tail_assembly_20261008/work
nohup caffeinate -i $PY -u scripts/final_tail_assembly_20261008.py assemble --work data/final_tail_assembly_20261008/work --repeat-e156 > data/final_tail_assembly_20261008/assemble.log 2>&1 &
$PY scripts/validate_final_submission.py --file data/final_tail_assembly_20261008/work/gentle-igloo_final.parquet \
    --reference submissions/gentle-igloo_v12075.parquet --json data/final_tail_assembly_20261008/final_validation.json
```

A failed workspace keeps its partials and `failure.json`; a second attempt needs a fresh `--work`.
Resources on the dry run: ~141 s wall, ≤6 OpenMP threads, workspace 257 MB (links + ~20 small
parquet/npz). The final run reads the 2.76 M-row FIN1b once (premix) and the 2025 stage inputs
in the E141/E64/LSZH children; the v2 replay of the same stages peaked at 7.9 GiB.

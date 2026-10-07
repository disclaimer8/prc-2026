# Independent v12075 saved-component replay: memory correction

Protocol: `v12075-independent-saved-component-replay-zero-fit-lowmem-v2`.

This is a technical successor to journal 368 / registration
`05a1b2eb7adfa5bd1abc47bc03a94095e0a39d4e`. The first attempt stopped in
`blend.py` before the initial classifier result: the owned group reached
19267698688 bytes RSS and its 16 GiB monitor terminated it. Both original
sources and failed `data/reproduced_v12075` / launcher state are preserved.
Elevated numeric `ps` subsequently confirmed no process in its group 38246.

The original blend loads all 210 feature columns, computes target-encoding
tables despite the registered recipe dropping every `te_`, `ter_`, `tq_`
column, and materializes a full training Pandas matrix whose values it never
uses. The new helper executes the unchanged frozen `blend.py`, with three
strictly bounded adapters: project only its classifier and mixture inputs;
skip excluded encoding computation only when all three prefixes are dropped
and regime rates are disabled; obtain the exact Pandas sorted/dropna category
levels one column at a time. Ranking conversion and all classifier, mixture,
rounding and tail arithmetic remain the original code. The helper verifies
feature names and categorical levels against the saved classifier, rejects
unexpected call counts, and disallows model training.

Before proceeding beyond the regenerated premix, all fields of its saved
parts (`mvt_id`, `reg`, `proba`, `values`, `pred`, including dtype/order and
NaNs) must exactly equal the original E203 stage-zero control; its complete
344841-row premix Parquet must have identical bytes. These two original
intermediates are pinned separately in `inventory_v2.json/comparison_only`
and are never prediction inputs. The qualification receipt is
`data/reproduced_v12075_v2/premix_parity.json`; the helper receipt is
`data/reproduced_v12075_v2/lowmem_blend.json`.

The final target remains v12075 SHA256
`90d3cd0b5a68cc8008f2142f1801857752574b525b2ff046ad6bf4089a70aee2`.
Expected result/gate: 344841 exact predictions, unchanged schema/IDs and all
42 TabPFN donors, byte equality, tolerance 0, zero new fits and zero uploads.
The 3711 original replay dependencies remain individually pinned. Current
best/candidate/final-original prediction files are excluded from replay
inputs and opened only in the final comparison after new output sealing.

Fresh directories are `data/reproduced_v12075_v2` and
`data/reproduce_v12075_v2_launch_20261007`; the previous output is never reused.
Limits remain nice10, maximum 6 OMP threads, 16 GiB combined sampled RSS,
2-second monitoring, 4-hour deadline, 4 GiB initial disk admission and 2.5 GiB
disk floor. No fallback fit, automatic retry, environment update or upload.
The group-existence check uses numeric `ps` metadata, because macOS reported
EPERM for `killpg(pgid, 0)` after the first group's termination. Only the
supervisor's own recorded process group may be terminated.

All new sources, tests, this protocol and inventory must be committed with
the technical registration before this new run. Preparation/contract tests
do not execute a target fit or production replay.

```bash
.venv/bin/python scripts/reproduce_v12075_v2_launch_20261007.py --registration FULL_REGISTRATION_COMMIT --launch
```

Require both `data/reproduced_v12075_v2/complete.json` and
`data/reproduce_v12075_v2_launch_20261007/complete.json` for success. The result
remains saved-component replay, not raw-data retraining or independent nested CV.

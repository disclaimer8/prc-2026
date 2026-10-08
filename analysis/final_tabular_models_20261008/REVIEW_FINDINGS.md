# Independent tabular adapter review

Status: READY for preregistration; zero open Critical/Important findings. Both
reported admission defects are closed on the pins below. Read-only review; no competition
rows, production fit, feature build, remote job or environment mutation.

## Closed Important 1 — incomplete seed inventories passed merge

`merge()` validates only files listed by each seed receipt. On the reviewed
source, five tiny synthetic `raw/seed_0..4` directories, each containing only
`predictions.npz` plus a provenance-matching `complete.json` with `files={}`,
produced a successful merged completion. No model or state file existed and
not even the predictions file was hash-bound. The harness replaced only the
unrelated real-data `admit()` boundary; it exercised the real `merge()`.

Require exact per-family inventories (`model.cbm` or `model.txt`, `state.json`,
`predictions.npz`, plus `classifier.txt` for base), then validate every hash and
state family/seed/rounds identity before publishing a merge. Keep the full seed
set and ordered ID checks. This implements the protocol's saved-model promise.

## Closed Important 2 — accepted disabled options broke the original caller

`configure()` admits OPTIONAL values `None`, empty string or `0`, but leaves
those environment values set. The original `train.py` interprets truthy
`REG_OVERRIDE` as JSON mapping and truthy `ADV_WEIGHT` as a filename. A zero-fit
reproduction set `REG_OVERRIDE=0`, called the real configure, then imported the
unchanged train module: `TypeError: 'int' object is not a mapping`.
`ADV_WEIGHT=0` analogously selects a file named `0` instead of disabled weights.

After rejecting active alternatives, remove disabled OPTIONAL entries before
importing the original helpers, or enforce the exact per-key unset semantics.

## Final delta verification

The fixed merge requires the exact per-kind file set, all artifact hashes, and
state kind/seed/rounds/engine identity. Base now persists seed0. The original
empty-inventory reproduction rejects. Independent additional harness: raw, Cat
and base each accepted a correct synthetic inventory and rejected altered
seed, rounds, engine, kind or prediction hash (18 cases PASS). These placeholder
model files test admission only; native model reload is covered by the tests.

`configure` removes every admitted disabled OPTIONAL entry. The actual original
train import succeeds after REG_OVERRIDE=0/ADV_WEIGHT=0 are cleared, and
`T.adv_weights(None)` returns None. No scientific/model recipe changed.

## Verified scope and resolved draft gaps

Read the complete adapter, tests and protocol. Traced base classifier/proxy/mix
through `src/train.py`, Cat through `src/catboost_seeds.py`, LGB5 through
`src/seed_avg.py`, anchors through `src/e130_rank.py`, and E185 through
`src/e185_motion_view.py`/`src/e185_motion_rank.py`. Fixed rounds, seed schedules,
category populations, ordinary-row masks, clipping, proxy fallback, feature
mask and motion replacements match these callers. Training/final row admission
and final-label rejection are explicit. Serialization tests exercise native
LightGBM/CatBoost save/reload and the original proxy arithmetic.

The author delta records all seven package versions and thread settings, drops
the unconditional CPU-Linux Torch import, and enforces Linux cgroup CPU/RAM,
swap0, nice10 and disk floor. These resolve the earlier draft concerns. The
worker is Linux-production-only. Exact host allocation, RuntimeMaxSec and
owned InvocationID monitoring remain root's actual preregistration boundary;
this review does not qualify a Mac production supervisor or a real runtime.

Independent verification: `env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 .venv/bin/python
-m unittest discover -s tests -p test_final_tabular_models_20261008.py` — 9/9 PASS (1.915s independent repeat).
Initial synthetic reproductions confirmed both reported defects; final delta
checks now reject both original failure cases.
No new scientific claim or RMSE evaluation.

## Reviewed final pins

- `scripts/final_tabular_models_20261008.py`: `ba9a5e3aef9ef93b6d692806cba844c8fb0c1d075a6346245ac60c9ddf1bea37`
- `tests/test_final_tabular_models_20261008.py`: `425aa74eb4ead77b7f55be953d2d1e183e3ae334572cb28896b141e59ad5783c`
- `analysis/final_tabular_models_20261008/PROTOCOL.md`: `291281b11c9eb02248d31c227911aad5af28404882768ffbbd077e3d1ea78b06`

Updated UTC: 2026-10-08T11:40:52.167385+00:00

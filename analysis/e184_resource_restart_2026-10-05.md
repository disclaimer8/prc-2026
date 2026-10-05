# E184: resource-only restart after confirmed OpenMP spinning

The original registered model/gate/forecast remain journal254. No scored E184 fold
has been obtained: `data/cv_preds_e184_hz` contains zero NPZ files. The interrupted
attempt's input, empty output, logs and controller/worker identity must be preserved.
Root, not this implementation agent, owns any stop or restart.

## Current evidence

Read-only `/proc` samples confirmed controller2894663 and first-fold worker2894664
(`1,7`, Python3.14 from the existing server `.venv`). The worker has63 threads, but
exactly10 remain runnable. Their aggregate utilization is about600%; individual
threads spend16–51% of elapsed time waiting in the scheduler and undergo60–168
involuntary context switches/sec. RSS11.43GiB, process swap0, available RAM43–44GiB;
over6seconds no increase in swap-in/out or major faults, memory/IO pressure near0.
The empty fold log is expected because the original controller captures child output
until the entire fold returns; it is not a reliable iteration-progress measurement.

Topology from both `lscpu` and sysfs: SMT pairs0–1,2–3,4–5,6–7,8–9,10–11. Old affinity
0–5 used only3 physical cores. Root changed all64 controller/worker TIDs to
0,2,4,6,8,10 while retaining the six-logical-CPU budget. Three-second perf samples:

| Sample | CPU usage | Minimum cycles in spin loops | Complete folds |
|---|---:|---:|---:|
| Before affinity change | 599.8–601.0% | 94.91% | 0/6 |
| After affinity change | 600.1–600.8% | 92.79% | 0/6 |

Both hot addresses are in `/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0`:
0x26a82 and0x26c72 immediately follow x86 `PAUSE` instructions at0x26a80/0x26c70,
inside repeated counter/shared-state checks. No samples were lost. These are short
operational samples, not estimates of useful model progress across the whole run.

The original shell exports OMP6/POLARS6, but the nested controller overrides child
POLARS_MAX_THREADS to10 and train.py explicitly sets LightGBM num_threads10. Merely
changing affinity or OMP_NUM_THREADS does not fix those already initialized limits.

## Isolated launcher

`scripts/e184_resource_cv.py` snapshots the four actual server source files:
train.py, features.py, target_encoding.py, cv_fold_preds_nested.py. No existing
shared source or E185 frozen manifest is modified. The snapshot records source,
feature-table and launcher SHA256, registration commit, environment and interpreter.
The feature table is individually symlinked and its SHA rechecked before every fold
and after CV; the snapshot's source modules are real copies.

The original controller's literal RUNNER is extracted without executing the
controller. Exactly two assignments are inserted immediately after its train import:
REG_PARAMS.num_threads=6 and CLF_PARAMS.num_threads=6. The rest of the worker source,
all six month pairs, inner round choice,1.1 multiplier, targets, features, seeds,
hyperparameters and OOF format remain the original ones. The original model flags
were read from the whitelisted actual controller environment and are preserved:
PROXY_SCALE=1, CLF_COST_WEIGHT=1, CLF_COST_MIN_SPREAD=3600 and the complete production
DROP_PREFIX. All other model switches were unset and remain unset in the clean env.

Runtime settings: OMP_NUM_THREADS=6, OMP_THREAD_LIMIT=6, OMP_WAIT_POLICY=PASSIVE,
GOMP_SPINCOUNT=0, POLARS_MAX_THREADS=6, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1.
The launcher requires affinity0,2,4,6,8,10, refuses live retired PIDs, and refuses
existing snapshot/output directories. No automatic restart or score-based selection.
Worker stdout/stderr is unbuffered and streamed to both the service log and individual
fold logs. On failure the launcher stops; completion means six files saved, not that
the scientific gate passed. A resource restart loses the unsaved first-fold work.

After root commits operational amendment268 and this code together, reviews the
snapshot mechanism, and stops/reaps the original controller and worker:

```sh
cd /root/prc-2026
taskset -c 0,2,4,6,8,10 nice -n 10 .venv/bin/python -u scripts/e184_resource_cv.py \
  --registration-commit FULL_40_CHARACTER_AMENDMENT_COMMIT \
  --retired-pids 2894663,2894664
```

Defaults: snapshot`data/e184_resource_20261005`, new OOF`data/cv_preds_e184_hz_6t`.
Copy only the new launcher into `scripts/`; the original server `src/` stays intact.
No installation/environment rebuild is required. After launch, verify six active
LightGBM threads, recorded startup settings and actual iteration progress. Repeat
the brief perf check if high CPU returns without progress. No runtime ETA is claimed.

After six completed folds, run the existing strict audit against FIN1b and then
apply the original254 gate: decided lower in at least4/6, trimmed<0, body y<=7200<0.
Do not replace those criteria with a different gate because of this resource repair.

```sh
.venv/bin/python src/strict_oof_gate.py data/cv_preds_fin1b_hz data/cv_preds_e184_hz_6t \
  --features data/features_fin1b.parquet --audit-only
```

Operational preregistration wording for root: E184 repeat of254 after0/6 completed
OOF folds; perf confirmed≥94.91% spin before and≥92.79% after affinity correction.
Only resources change: six actual LightGBM threads, passive/no-spin OpenMP, Polars6,
BLAS1, six distinct physical cores, exact server-code snapshot, new OOF directory.
Model, nested CV, features, seeds, gate and forecast of254 unchanged. Old attempt
preserved; no E184 metrics were available for selecting this amendment.

## Validation

The five new resource-launcher tests first failed because the module was absent,
then passed after implementation. They check clean environment construction, the
two parameter-only overrides, safe extraction of the original worker, isolated
snapshots and hashes, exact CPU affinity, and refusal while retired PIDs exist.
The full repository suite passed: 124 tests in 6.240 seconds. The original nested
RUNNER was also verified byte-for-byte after removing the two inserted assignments.
No test trained an E184 model. Independent code review found no launch blocker.

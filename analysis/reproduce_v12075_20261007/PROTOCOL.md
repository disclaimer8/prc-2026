# Independent v12075 saved-component replay

Protocol: `v12075-independent-saved-component-replay-zero-fit-v1`.

The fixed target is v12075, SHA256
`90d3cd0b5a68cc8008f2142f1801857752574b525b2ff046ad6bf4089a70aee2`,
344841 rows, processed leaderboard RMSE 230.9387 s. This technical replay has
no new scientific hypothesis, model fitting, weight selection, CV evaluation,
leaderboard request, or upload. The acceptance gate is exactly zero changed
predictions, matching schema and IDs, all 42 protected TabPFN rows unchanged,
and byte-for-byte SHA256 equality. Expected outcome: exact equality; tolerance 0.

The original scientific recipe remains E203 registration
`6e95e3a65936a3401e41499bb340f2d54a1a2121`. The saved ordered trajectory member
`data/e203_rank_20261007/prediction.npz` replaces PLE at fixed weight 0.25.
The full FIN1b Cat15/LGB5 and five-anchor mixture, E185, E167, TabPFN donors,
E141, E64/E188, E152 0.95, E81, LOBT/protect42 and half-LSZH sequence is
recomputed in a fresh `data/reproduced_v12075`, versions 990200 through 990215.

`inventory.json` retains every original `setup.json` input and source entry
unchanged, adding SHA256-pinned replay helper, E203 member, manifests and cached
model dependencies. Immutable production sources are copied into the private
workspace; input files are linked individually. No original source, manifest,
member, model or output is overwritten. New state and output must not exist.
The current best, E203 candidate and original final chain output are excluded
from inputs, including symlink/hardlink aliases. The new final output is sealed
in `assembled_before_oracle.json` before opening the reference predictions.

The E203 exact-control cache supplies 11 saved models (E141 5, E64 5, LSZH 1).
The unchanged original replay worker validates each original training-call
signature: feature names/order, matrix, labels, weights, parameters, rounds,
runtime versions and model SHA. The new wrapper permits only replay mode;
capture mode fails before invoking that worker. Five existing E188 models are
loaded through its registered `apply` path. Expected new fits: exactly zero.

The launcher runs one fresh process group under `caffeinate` and nice 10, with
OMP thread limit 6, Polars 2 and BLAS 1 (E188's registered worker uses 4).
Admission requires at least 16 GiB reclaimable memory, 10% system memory-free
indicator, and 4 GiB free disk. It monitors the owned group's combined RSS
every 2 seconds and terminates only that group above 16 GiB, below 2.5 GiB
free disk, or after 4 hours including admission waits. This is a sampled RSS
guard, not an OS cgroup hard limit on macOS. All partials/logs survive failure;
there is no automatic retry or fallback fit.

Preparation and contract tests may run before registration. The runner,
launcher, this protocol, inventory and tests must be committed with the journal
entry before replay. The launcher verifies those exact committed bytes.

```bash
.venv/bin/python scripts/reproduce_v12075_launch_20261007.py --registration FULL_REGISTRATION_COMMIT --launch
```

Success requires both `data/reproduced_v12075/complete.json` and
`data/reproduce_v12075_launch_20261007/complete.json`. The result is explicitly
saved-component reproduction, not full retraining from public raw data, and
does not remove the existing selection-informed/full-tail-OOF limitations.

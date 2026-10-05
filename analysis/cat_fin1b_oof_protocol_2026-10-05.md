# CatFIN1b CPU component OOF execution protocol

This is retrospective validation of one production component of v12048. It is
not a new model hypothesis, independent holdout, complete assembled v12048 OOF,
or a leaderboard candidate. No quality gate, parameter search or upload is part
of this job. Register this code and the journal entry together before fitting.

The fixed schedule is seed 0 on all six pairs (1,7), (2,8), (3,9), (4,10),
(5,11), (6,12), then seed 1 on all six, through seed 14: exactly 90 CPU fits.
Each uses the unchanged `src/catboost_seeds.py` recipe: 5600 iterations,
depth 8, learning rate .08, RMSE, regular-regime fit rows, clipped residual to
the calibrated NM proxy. CatBoost 1.2.10 CPU defaults are retained. GPU is a
different recipe and is not supported by this wrapper. No interim quality
selection or seed-count choice is allowed.

The pinned FIN1b table SHA256 is
`f061a441f6c97c232d9a403d612471e45020eb4ad56f501c2acfd7220a7d3052`.
Preparation keeps only its 2,085,047 training rows in original order, preserves
every feature, and changes only `split` plus the excluded target field. For
each pair, the heldout rows become `rank`, their target becomes null, and the
other ten months remain `train`. Actual 2026 ranking rows never enter a view.
Heldout ID/target/month are saved separately. The unchanged trainer recomputes
regime labels, encodings and proxy on the ten fit months only. Feature names
and category names must equal the pinned E182 lineage (170 and 11 columns).

Four original modules are copied to an isolated source snapshot and checked
against hardcoded SHA256 values before and after fitting. The manifest records
these hashes, the wrapper hash, package/Python versions, exact environment,
column lists, all view/metadata hashes, schedule and registration commit. An
existing work directory is never silently repurposed. Changed packages,
sources, inputs, wrapper, registration or artifacts invalidate resume.

Every child uses `sys.executable`, a clean scientific environment and its own
working directory (`catboost_info` cannot collide). One fit runs at a time,
with 24 CatBoost/OpenMP threads, Polars 4 and BLAS 1. Root must launch the
controller in a service cgroup with MemoryMax=26G and stop/finish the earlier
diagnostic before fitting. The wrapper does not create a cgroup itself.
There is no artificial wall-time cutoff; the first fit prints measured time
and an explicitly provisional remaining-time estimate. Prior Mac fits took
43–81 minutes each; the Ryzen rate remains unmeasured.

Preparation materializes six fold tables (allow roughly 6 GiB of free disk,
plus attempts/logs), so do not start it on a Mac with 3.5 GiB free. This version
has one full schedule on one host. Distributing disjoint folds would require
a separately reviewed assignment/merge protocol; no ad hoc split is enabled.

Example after registration, with an existing isolated environment:

```bash
.venv-cat-oof/bin/python scripts/cat_fin1b_oof.py prepare \
  --root /home/denys/prc-2026 \
  --work /home/denys/prc-2026/data/cat_fin1b_oof_20261005 \
  --registration-commit FULL_40_HEX_REGISTRATION_COMMIT
.venv-cat-oof/bin/python scripts/cat_fin1b_oof.py run \
  --work /home/denys/prc-2026/data/cat_fin1b_oof_20261005 \
  --registration-commit FULL_40_HEX_REGISTRATION_COMMIT
```

Use the actual home directory, not the illustrative `/home/denys` if different.
Preparation and execution both hold a nonblocking single-writer lock. A seed
is published only after IDs, order, shape and finite predictions pass; its
NPZ and manifest appear together through atomic directory rename. A completed
seed is hash/lineage checked and read without rewriting on resume. Interrupted
attempts and unfinished publication directories move to quarantine on resume,
preserving diagnostics. Completed attempts remain in `logs/`.

Only all 15 seeds produce `fold_A_B/complete/prediction.npz` with
`mvt_id,reg2,per_seed,y,month`; `reg2` is the exact equal-weight mean. Every
seed is revalidated during merge. `complete.json` appears after all six folds.
These component artifacts intentionally do not masquerade as complete mixture
OOF accepted by `strict_oof_gate.py`; classifier, other regressors, five-seed
anchors, motion/MLP and the exact final stages still need separate accounting.

Synthetic contract tests cover label exclusion, exact row order, official
integral Float64 IDs, malformed populations, seed ordering, exclusive locking,
no overwrite, hash/lineage tampering, incomplete quarantine, all-15-only merge
and environment contamination. No model was fitted in implementation tests.

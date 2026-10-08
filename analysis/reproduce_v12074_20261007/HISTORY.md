# Historical reproduction notes: v12074 replay and replay operations (moved from `docs/reproduction.md` on 8 October 2026)

The current guide is [docs/reproduction.md](../../docs/reproduction.md). The
text below is kept for provenance; only missing spaces were corrected. The
v12074 `capture` mode of `scripts/reproduce_best.py` and the queue
`scripts/reproduce_best_queue_20261007.py` compare `docs/reproduction.md` with
their registration commit `c62911c96d8631b99e1a8443f7dd002d153578f0`, so they
must be run from a checkout of that commit. Plain `replay` does not check it.

## Operational notes on the v12075 v2 replay

The v2 helper projects 173 of 210 original columns, retaining exactly the 170
saved-classifier features, skips encoding columns excluded by the fixed recipe,
and obtains the same 11 categorical dictionaries without building an unused
training Pandas matrix. Its complete initial mixture was independently checked
against the original E203 intermediate: every array value, dtype, row order and
NaN position, plus Parquet bytes, matched exactly before the tail was allowed to
continue. Eleven cached E141/E64/LSZH models and five saved E188 models are reused.

The launcher uses nice 10, at most six OpenMP threads, a four-hour deadline and
a sampled 16 GiB combined-RSS guard with two-second polling. Disk admission is
4 GiB, with a 2.5 GiB runtime floor. It never retries automatically. The first
attempt under journal 368 exceeded the memory guard in the original initial
blend and was stopped; its sources, logs and partial output remain preserved.
The documented v2 run resolved that resource issue and passed the unchanged
exact-output gate.

## Historical v12074 reference

The remaining sections preserve the earlier v12074 reproduction and commands.
They concern the historical control, not the current best v12075.

The verified archived reference is `gentle-igloo_v12074.parquet`. Its
SHA-256 is `e5c78ab061492b69fe72d4a46d2e286f7a2f6b3129ebf2ae4a34731bb662374b`.
It contains 344,841 departure predictions for January and July 2026. The original
assembly was registered at `cd3598e15a25db9bd6a401d38b4ad9bee6c2e862` and is preserved
in `scripts/today_submissions_20261006.py` and its input/source receipt.

**Scope:** the command below recomputes the mixture and complete production tail
from archived feature tables, fitted model/member outputs, and historical donor
predictions. It does not retrain every model from raw data. It never copies the
best file into the prediction chain, changes competition submissions, uploads,
downloads, or reads upload credentials. The reference is read for hash checking
and the final exact comparison only.

## Status and commands

The reproduction wrapper has been implemented. Archive verification passed for
all 3,464 input files and 207 frozen sources. A synthetic 256-row, three-round
LightGBM check proved exact save/load replay and rejection of changed labels.
The real replay completed on October 7 at 16:47:44 UTC using the eleven verified
E203 control models. All 344,841 predictions and the Parquet bytes match v12074
exactly; output SHA equals the reference above. Replay took approximately 140.16 s
and performed no additional model fits. The completed receipt is
`data/reproduced_v12074/complete.json`. This proves archived-component reproduction,
not a raw-data rebuild from a clean clone.

Use the existing qualified Python environment, from the repository root:

The local CPU environment inspected on October 7 is macOS 27.0.1 arm64 with
Python 3.12.14 (`Sep 1 2026, 14:09:38`, Clang 22.1.3), NumPy 2.5.3,
Polars 1.44.2, LightGBM 4.7.0, scikit-learn 1.9.1, pandas 3.0.5,
PyArrow 25.0.1 and CatBoost 1.2.10. These are installed metadata, not a claim of
cross-platform equivalence. The archive and synthetic replay checks passed in
this environment; the full v12074 replay also passed as recorded above. No environment
was changed for this verification.

```bash
.venv/bin/python scripts/reproduce_best.py check
```

After the fixed tail models have been captured and verified, the one-command
replay is:

```bash
.venv/bin/python scripts/reproduce_best.py replay --output data/reproduced_v12074 --model-cache data/v12074_replay_models
```

If E203 has already completed its exact v12074 control capture, reuse that cache
and avoid the eleven duplicate fits:

```bash
.venv/bin/python scripts/reproduce_best.py replay --output data/reproduced_v12074 --model-cache data/e203_chain_20261007/models
```

The wrapper verifies E203's registered control receipt, all eleven models and
the control artifact before admitting this cache. An in-progress or failed E203
capture is rejected. Do not start a separate capture while E203 may be writing
its own; only schedule the standalone capture after E203 finishes without one.

Replay forbids new LightGBM model fits: eleven calls in the original production code are replaced
by loading their saved models. It verifies the exact training arrays, labels,
weights, feature order, categorical specification, rounds, parameters, and package
versions against the capture receipt before accepting each model. The result is
`data/reproduced_v12074/reproduced_v12074.parquet`, with per-stage logs and a final
receipt. A used output directory is rejected; use a new path for another run.
The original training-derived target encodings and closed-form proxy calibration
statistics are still recomputed from the archived training feature table.

The October 6 assembly did not save its eleven tail models. One initial capture
therefore requires eleven fixed LightGBM fits: E141 five seeds, E64 five seeds,
and the LSZH OPDI stage one seed. It reuses the five saved E188 models and the
archived E190 ordinary-member predictions. After review/registration, run:

```bash
.venv/bin/python scripts/reproduce_best.py capture --registration REGISTERED_COMMIT --output data/v12074_capture --model-cache data/v12074_replay_models
```

`REGISTERED_COMMIT` must contain the exact wrapper, documentation, manifest and a
journal entry naming `v12074-archived-component-reproduction-v1`. Capture is a
reproducibility check of an existing fixed recipe, not a new model selection or
quality claim. It uses at most six OpenMP threads in fresh LightGBM-only stage
processes. Do not overlap its heavy stages with another memory-heavy Mac job.
Allow at least 4 GiB of free disk at startup; each stage keeps a 2.5 GiB floor.

The comparison requires identical schemas, all 344,841 unique IDs and every
integer prediction to match v12074 exactly. Parquet byte equality is recorded
separately because serialization can differ without changing predictions. A
failed run keeps its logs and partial files and never retries automatically.

### October 7 automatic scheduling

`scripts/reproduce_best_queue_20261007.py` is an operational successor, separate
from the portable replay command. It binds E203 controller PID 51200, start time
`2026-10-07T14:59:25.379082+00:00` and registration `6e95e3a...`. It waits for both
the durable terminal receipt and process exit. Successful E203 capture is reused.
If E203 ends with its negative scientific gate and no capture directory, the
queue runs one independent fixed 11-model capture followed by one replay. A
technical failure, unexpected partial capture or missing terminal receipt blocks
this automatic fallback and preserves all artifacts.

After review/registration, the specific durable queue is launched once with:

```bash
.venv/bin/python scripts/reproduce_best_queue_20261007.py --registration REGISTERED_COMMIT --launch
```

The registration covers both scripts, this document and the manifest. It uses a
new state directory `data/reproduce_best_queue_20261007`, per-child PID/exit
receipts, logs, and an 18-hour overall bound. Only its own child process group may
be stopped. Startup requires 16 GiB reclaimable Mac memory and 4 GiB disk; sustained
system-wide free-memory pressure below 10% (three 10-second samples) or less than
2.5 GiB disk stops that child without retry. No E203/GPU/remote worker is signalled.
The queue's automatic transition does not depend on this chat remaining open.
SIGTERM/SIGINT enter the same failure cleanup. The queue retains the child's
owned process-group ID until the entire group has exited, including descendants
whose parent has already exited; successful leader exit alone cannot complete an
action. Cleanup allows 30 seconds after TERM, then 15 seconds after KILL. A cleanup
error is preserved alongside the original failure, and signal handlers are restored.
Synthetic process tests passed for both a live leader and a leader confirmed exited
before SIGTERM: a TERM-ignoring descendant was removed by the bounded KILL fallback,
an unrelated sentinel stayed alive, and the original signal handler was restored.

## Required archive

`docs/reproduction_manifest.json` pins the best reference, original setup receipt,
neural-member receipt, E188 model receipt and E188 application source. The original
setup receipt `data/today_chain_20261006/setup.json` pins 3,464 input files and 207
frozen source files. All are checked before and after reconstruction. These
inputs total approximately 5.11 GB; neural checkpoint files add further space.
They are currently local artifacts, not distributed by `git clone`.

Important prerequisites are:

| Component | Archived artifact / source |
|---|---|
| Challenge data | `data/prc-2026-datasets/training_*.parquet`, `ranking.parquet`, `submitting.parquet` |
| Feature tables | `data/features_fin1b.parquet`, `data/features_e106.parquet` |
| Classifier and base regressor | `submissions/clf_v870.txt`, `reg_v870.txt`; no classifier retraining in replay |
| Ordinary tabular members | `cat_fin1b_seeds15.npz`, `lgb_reg_seeds_v870.npz`, `e130_{raw,lobt,sched,eobt,aobt}f_rank.npz` |
| Neural ordinary member | `data/e190_rank_20261006/complete/prediction.npz`; three saved checkpoints and seed arrays also verified |
| Movement-only member | `data/e185_motion_rank.npz` |
| Surface inputs | Original daily `adsblol`, `adsblol_kt1`, `adsblol_sw_kt1` extractions for training/ranking |
| Stage training parents | Six-fold `cv_preds_nomonth`, `cv_preds_e116prod`, `cv_preds_e150c_e64_keep` arrays |
| E188 | `data/e188_rank_20261006_v2/fit_manifest.json` and `seed{42,1,2,3,4}.txt` |
| OPDI | Pinned `data/opdi/ev_*.parquet` event extracts |
| Historical donors | `gentle-igloo_v44.parquet`, `v60.parquet`, `v291.parquet`, `v361.parquet` |
| Frozen application source | `data/today_chain_20261006/src/`; its private E64 import truncation is part of the recorded source hashes |

The setup also contains older reference/intermediate files used by historical
builders. These are retained for provenance, not silently treated as fresh model
training. The frozen source is deliberately separate from the latest `src/`:
current working-tree edits cannot silently change the archived recipe. The E188
application retains its original Git-registration guard, so its registered commit
must be present in the local repository (a shallow export may omit it).

# Reproducing the archived best v12075 submission

**v12075 (230.9387 s) was independently reproduced on October 7 at 22:33:00 UTC.**
All 344,841 predictions, schema, IDs and Parquet bytes matched exactly; all 42
protected TabPFN donor values remained intact. The output SHA-256 is
`90d3cd0b5a68cc8008f2142f1801857752574b525b2ff046ad6bf4089a70aee2`.
The bounded run took 101.19 seconds and performed zero new model fits. Its
sampled process-group RSS peak was 8,485,355,520 bytes (7.90 GiB).

The current runner is `scripts/reproduce_v12075_v2.py`, registered with its
launcher, helper, tests, protocol and inventory at
`70aff7199731c381945c45754ee70c7e3a964aa5` (journal 372). The original E203
scientific recipe remains registration `6e95e3a65936a3401e41499bb340f2d54a1a2121`.
It recomputes the complete fixed chain from archived features, member predictions
and saved models; this is not retraining every model from public raw data or a
new independent nested-CV result. No upload or environment change occurs.

Read-only verification of the 3,711 pinned dependencies:

```bash
.venv/bin/python scripts/reproduce_v12075_v2.py check
```

The exact one-command replay, run once in a fresh populated workspace with the
existing qualified Python environment, is:

```bash
.venv/bin/python scripts/reproduce_v12075_v2_launch_20261007.py --registration 70aff7199731c381945c45754ee70c7e3a964aa5 --launch
```

The completed local output/state already exist and intentionally prevent a
repeat launch. The output is
`data/reproduced_v12075_v2/reproduced_v12075.parquet`; success requires both
`data/reproduced_v12075_v2/complete.json` and
`data/reproduce_v12075_v2_launch_20261007/complete.json`. All stage logs and the
pre-reference assembly receipt are retained. The final reference is opened only
after the generated output is sealed.

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

See the [replay results and independent verification](../analysis/reproduce_v12075_20261007/RESULTS.md),
[v2 protocol](../analysis/reproduce_v12075_20261007/PROTOCOL_V2.md),
[pinned inventory](../analysis/reproduce_v12075_20261007/inventory_v2.json), and
[submission receipt](../analysis/e203_upload_20261007/RESULTS.md).

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
The real replay completed on October7 at16:47:44UTC using the eleven verified
E203 control models. All344,841 predictions and the Parquet bytes match v12074
exactly; output SHA equals the reference above. Replay took approximately140.16s
and performed no additional model fits. The completed receipt is
`data/reproduced_v12074/complete.json`. This proves archived-component reproduction,
not a raw-data rebuild from a clean clone.

Use the existing qualified Python environment, from the repository root:

The local CPU environment inspected on October 7 is macOS27.0.1 arm64 with
Python3.12.14 (`Sep 1 2026, 14:09:38`, Clang22.1.3), NumPy2.5.3,
Polars1.44.2, LightGBM4.7.0, scikit-learn1.9.1, pandas3.0.5,
PyArrow25.0.1 and CatBoost1.2.10. These are installed metadata, not a claim of
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
from the portable replay command. It binds E203 controller PID51200, start time
`2026-10-07T14:59:25.379082+00:00` and registration `6e95e3a...`. It waits for both
the durable terminal receipt and process exit. Successful E203 capture is reused.
If E203 ends with its negative scientific gate and no capture directory, the
queue runs one independent fixed11-model capture followed by one replay. A
technical failure, unexpected partial capture or missing terminal receipt blocks
this automatic fallback and preserves all artifacts.

After review/registration, the specific durable queue is launched once with:

```bash
.venv/bin/python scripts/reproduce_best_queue_20261007.py --registration REGISTERED_COMMIT --launch
```

The registration covers both scripts, this document and the manifest. It uses a
new state directory `data/reproduce_best_queue_20261007`, per-child PID/exit
receipts, logs, and an18-hour overall bound. Only its own child process group may
be stopped. Startup requires16GiB reclaimable Mac memory and4GiB disk; sustained
system-wide free-memory pressure below10% (three10-second samples) or less than
2.5GiB disk stops that child without retry. No E203/GPU/remote worker is signalled.
The queue's automatic transition does not depend on this chat remaining open.
SIGTERM/SIGINT enter the same failure cleanup. The queue retains the child's
owned process-group ID until the entire group has exited, including descendants
whose parent has already exited; successful leader exit alone cannot complete an
action. Cleanup allows30seconds after TERM, then15seconds after KILL. A cleanup
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

## Exact recipe

1. Read FIN1b, reproduce its feature matrix, target encodings, classifier
   probabilities and candidate feasibility masks with frozen production code.
2. Form the ordinary candidate from CatBoost15 / LightGBM5 at 0.65 / 0.35. Inside
   that candidate, mix FIN1b anchored members at raw `1/30`, LOBT/SCHED/EOBT/AOBT
   `0.10` each, and the E190 PLE neural member at `0.25`; the remainder stays in
   the CatBoost/LightGBM blend. These are weights within the ordinary candidate,
   not weights applied independently to the final submission.
3. Apply E185 movement-only member at `0.35 × 2/3` on the missing-LOBT,
   non-LIRF subset (`LOBT_flt IS NULL` and `ADEP_mvt != 'LIRF'`), then E167
   NM-window rules. This mask does not require the NM flight ID to be missing.
4. Insert the 42 TabPFN donor rows identified by v361 versus v291.
5. Run E141 own-track correction (five seeds 42,1,2,3,4; keep all training rows;
   seven reception-density features dropped) and E64 previous-visit correction
   (same seeds; keep all; three reception features dropped). Their original
   historical OOF training parents remain unchanged.
6. Replace E64 outputs only for the registered natural LIRF/NM missing-ground
   subset using E188 five saved models and the E190 branch's own pre-E64 base.
7. E152 gate 0.95 → historical E81 donor mixture v44 `0.1214`, v60 `0.0270` →
   LOBT edge lift/window clipping → restore the 42 protected TabPFN values.
8. Fit/apply E159b on LSZH, all 2025 training months, historical staged OOF parent,
   shift zero; blend its output with the pre-LSZH file at 0.5. Preserve each
   original stage's rounding order.

The final submission's manifest explicitly says `quality_validated_on_full_chain:
false`. Later component and pre-E81 checks do not turn it into a fully nested
end-to-end validation. Reproduction demonstrates execution fidelity, not an
independent accuracy estimate.

## Why the older commands do not reproduce this file

`src/run_final_any_period.sh` remains the older v808-family implementation. Its
feature path ends at E106, it defaults to 25 CatBoost seeds and a different anchor
mixture, and it contains no E190 PLE member, E185 insertion, E188 replacement or
half-LSZH stage. Its new-period TabPFN fit is also distinct from reusing v12074's
fixed 42-row draw. It is useful historical code and is not rewritten or relabelled
as the best model's entry point.

The README's former `features_v4 → train.py submit` example produces an earlier
baseline, not v12074. The older README also describes E14/E36 as the final tail,
whereas the best line uses the more recent stages above.

## Remaining fresh-training and distribution work

Rebuilding the full solution from raw data requires a separate, tested orchestration
layer. Existing producer code is available, but these dependencies are scattered:

* FIN1b is the deterministic merge in `hz_queue/20_fin1b_cv.sh`: E163 + E51 + E157
  + E165 feature tables. The upstream feature builders, external snapshots and
  joins must be ordered and their column order/hash reproduced. The older
  any-period command does not perform this merge.
* v870 classifier/LGB producers are in `hz_queue/40_fin1b_base.sh`; CatBoost15
  comes from `src/mac_e175_catfin1b.sh` plus `src/mac_e175b_seeds.sh`, 5,600 rounds,
  seeds0–14. Historical scripts' partial-average/ready-marker behavior should not
  be presented as an all-seeds completeness guarantee.
* Five anchored members use `e130_rank.py`, five seeds, rounds raw2565, LOBT3065,
  SCHED2958, EOBT2736, AOBT2738, as recorded in the October5 ranking launchers.
* E190 has its own saved preparation/training receipt and three seeds, eight
  epochs, PLE48 bins/12-dimensional embeddings, width512. Its qualified runtime
  was Python3.12.15, Torch2.13.0+rocm10.0.0, rtdl-num-embeddings0.0.12 on AMD7900XTX.
  This GPU stack is not fully specified by `pyproject.toml`/`uv.lock`; a fresh
  machine must not infer an equivalent environment from the generic dependency
  list. Replay of archived member predictions needs no GPU or Torch.
* E185 has separate movement-feature and ranking producers. E188 training needs
  its 365-day masked previous-visit extraction, original screen manifest, and
  natural/synthetic weighting; it is not ordinary E64 retraining.
* Historical v44/v60 and fixed TabPFN donor values require either their archived
  prediction files or an independently verified historical rebuild. E201's newer
  donor-OOF reconstruction is a scoped validation project, not an exact replacement
  for those existing ranking files. The legacy TabPFN draw was not guaranteed
  bitwise reproducible, so its fixed donor values must be retained for exact replay.
* Three historical OOF parents and daily ADS-B extractions must be supplied or
  rebuilt before fitting the tail. A complete input archive needs a public
  distribution location or an explicit acquisition/build manifest; local file
  existence and a licence table alone do not make a clean clone runnable.

No public archive location is claimed here. Publishing source/history or data is a
separate release action. External datasets retain their own licences and acquisition
conditions; the OPDI use in the final LSZH stage must appear in the source inventory.

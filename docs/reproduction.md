# Reproducing the v12075 and v12077 submissions

v12077 is the team's best submission: 230.4812 s RMSE on the ranking period
(January and July 2026), scored on 8 October 2026
([scoring receipt](../analysis/v12077_upload_20261008/RESULTS.md)). It is a
zero-fit rebuild of v12075 (230.9387 s, 7 October 2026,
[scoring receipt](../analysis/e203_upload_20261007/RESULTS.md)) with the neural
member averaged over ten seeds and its weight raised to 0.35. This guide covers
[verifying the files](#1-verify-the-submitted-file),
[replaying v12075 from archived components](#2-replay-v12075-from-archived-components),
[rebuilding v12076 and v12077 on the same path](#2b-rebuild-v12076-and-v12077-from-the-same-components),
[retraining from raw data](#3-retrain-from-raw-data) and the
[software environment](#4-environment).

Experiment IDs (E-numbers) and terms are explained in the [glossary](glossary.md);
the model itself is described in [model design](model-design.md). Large data,
fitted models and submission files are not in Git, and no public download
location for them exists yet.

## 1. Verify the submitted file

| Property | Value |
|---|---|
| File | `submissions/gentle-igloo_v12075.parquet` |
| SHA-256 | `90d3cd0b5a68cc8008f2142f1801857752574b525b2ff046ad6bf4089a70aee2` |
| Rows | 344,841 departures, all `MVT_ID_mvt` unique |
| Schema | `MVT_ID_mvt: Float64`, `TAXITIME_SEC_mvt: Int32` |
| Score | 230.9387 s RMSE, submission ID `025020aa-aa58-4d1e-8d1c-3a2cea94a48c` |

The two files built from it on 8 October 2026 have the same schema and ID set:

| File | SHA-256 | Score, submission ID |
|---|---|---|
| `submissions/gentle-igloo_v12076.parquet` | `c22de523cd392fb035f9223c1ac2222173de0967ef5a8909ee9641876828e1b0` | 230.8183 s, `5be7ef97-8244-4f9a-94cb-5027f50c20d6` |
| `submissions/gentle-igloo_v12077.parquet` | `f06c66bfa3388b17f5b6ba2362fa45a7da1be8fd5ae2d37de47cfe0e6bf2ba44` | 230.4812 s, `ecc78bca-21ed-4ce4-89ff-5b9765d3f74c` |

Sources: [scoring receipt](../analysis/e203_upload_20261007/RESULTS.md),
[replay results](../analysis/reproduce_v12075_20261007/RESULTS.md),
[v12076 record](../analysis/v12076_upload_20261008/RESULTS.md),
[v12077 record](../analysis/v12077_upload_20261008/RESULTS.md). Check the
hash with `shasum -a 256 submissions/gentle-igloo_v12075.parquet` (Linux:
`sha256sum`), or run the script below, which also compares the schema and IDs
with the organiser's template `submitting.parquet` (for v12077 substitute its
path and hash):

```python
import hashlib
import polars as pl

path = "submissions/gentle-igloo_v12075.parquet"
digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
assert digest == "90d3cd0b5a68cc8008f2142f1801857752574b525b2ff046ad6bf4089a70aee2"

sub = pl.read_parquet(path)
template = pl.read_parquet("data/prc-2026-datasets/submitting.parquet")
assert sub.schema == template.schema  # MVT_ID_mvt: Float64, TAXITIME_SEC_mvt: Int32
assert sub.height == sub["MVT_ID_mvt"].n_unique() == 344_841
assert set(sub["MVT_ID_mvt"]) == set(template["MVT_ID_mvt"])
assert sub["TAXITIME_SEC_mvt"].null_count() == 0
print("v12075 verified")
```

## 2. Replay v12075 from archived components

The replay recomputes the complete fixed v12075 chain from archived feature
tables, member predictions and saved models. The run finished on 7 October 2026
at 22:33:00 UTC. All 344,841 predictions, the schema, the IDs and the Parquet
bytes matched v12075 exactly, and the 42 protected TabPFN values were unchanged.
It took 101.19 s, peaked at 7.90 GiB of memory and fitted no model
([results](../analysis/reproduce_v12075_20261007/RESULTS.md),
[independent check](../analysis/reproduce_v12075_20261007/independent_verification.json),
[protocol](../analysis/reproduce_v12075_20261007/PROTOCOL_V2.md)).

**Recomputed** by the frozen production code: regime-classifier probabilities,
candidate feasibility masks, the regime mixture, and every tail stage including
the input features of each surface stage.

**Reused, not refitted:** the saved regime classifier and base regressor
(`clf_v870.txt`, `reg_v870.txt`); archived ranking predictions of the CatBoost
(15 seeds), LightGBM (5 seeds), five anchored, neural (E203) and movement-only
(E185) members; 11 saved LightGBM tail models (own-track stage E141 and
previous-visit stage E64 with five seeds each, LSZH OPDI stage E159b with one);
the five saved models of the LIRF previous-visit stage (E188); and the donor
files v44 and v60 (whole-file blend) and v291 and v361 (42 TabPFN rows). Before
loading a tail model, the runner checks its training arrays, labels, weights,
feature order, categorical specification, rounds, parameters and package
versions against the receipt written when it was trained.

v12075 itself is never an input: the runner opens it only after the new output
is sealed. Before the tail stages run, the regenerated initial mixture must equal
the archived production intermediate in every field and byte. The replay is not
retraining from raw data, a new nested cross-validation or a test on another
platform; the production manifest records `full_tail_oof_validated: false`
([status receipt](../analysis/e203_reproduction_status_20261007.json)). It shows
that the recipe runs as recorded, not how accurate it is.

### What you need

**Availability.** The archive is not in Git and is not published as a whole,
because it contains files we have no right to redistribute: the challenge data
(330 MB), OPDI extracts (1.5 GB), and feature tables and out-of-fold arrays that
embed challenge rows and labels (about 2.3 GB). The rest is our own output (saved
models, member predictions, receipts, frozen sources) or ODbL-derived adsb.lol
extractions. The complete archive, with its hash inventory, is available to the
challenge organisers on request through the team's registered contact.

- The 3,711 files listed with their SHA-256 in
  [`inventory_v2.json`](../analysis/reproduce_v12075_20261007/inventory_v2.json),
  about 5.2 GB (5,187,588,486 bytes), at the same relative paths
  ([size report](../analysis/reproduction_bundle_20261007/RESULTS.md)),
  including 207 frozen source files under `data/today_chain_20261006/src/`.
- Git history containing commit `70aff7199731c381945c45754ee70c7e3a964aa5`
  (replay sources) and `d45cb738fb7671cf289ae8a400a30b5b46dfbb3d` (E188
  application). The runner compares its sources with these commits, so a copy
  without `.git` fails.
- macOS: the launcher calls `caffeinate`, `memory_pressure` and `vm_stat`.
- The exact CPU interpreter and packages in [section 4](#4-environment).
- Up to six OpenMP threads, 4 GiB of free disk at the start (2.5 GiB must stay free),
  and memory headroom: the launcher stops the run above 16 GiB.

### Commands

From the repository root, check all 3,711 pinned files (read-only):

```bash
.venv/bin/python scripts/reproduce_v12075_v2.py check
```

Run the replay once:

```bash
.venv/bin/python scripts/reproduce_v12075_v2_launch_20261007.py --registration 70aff7199731c381945c45754ee70c7e3a964aa5 --launch
```

The output is `data/reproduced_v12075_v2/reproduced_v12075.parquet`. The run
succeeded only if both `data/reproduced_v12075_v2/complete.json` and
`data/reproduce_v12075_v2_launch_20261007/complete.json` exist. Stage logs stay
in `data/reproduced_v12075_v2/`. Both directories must be absent before the
launch, so a second run needs a fresh workspace. Failed runs are not retried and
keep their partial files.

### Exact recipe

1. Read the final feature table (FIN1b). Rebuild its feature matrix, the
   regime-classifier probabilities and the candidate feasibility masks with the
   frozen production code.
2. Form the regular candidate from CatBoost (15 seeds) and LightGBM (5 seeds) at
   0.65 / 0.35. Inside that candidate, mix the anchored members at raw `1/30`
   and LOBT, schedule, EOBT and AOBT `0.10` each, and the ordered-trajectory
   neural member (E203) at `0.25`; the remainder stays with the CatBoost/LightGBM
   blend. These weights apply inside the regular candidate, not to the final file.
3. Apply the movement-only member (E185) at `0.35 × 2/3` to rows without LOBT
   outside Rome (`LOBT_flt IS NULL` and `ADEP_mvt != 'LIRF'`); the NM flight ID
   need not be missing. Then apply the LOBT-window rules (E167).
4. Insert the 42 TabPFN tail values from v361 (the rows where v361 differs from v291).
5. Run the own-track surface stage (E141: seeds 42, 1, 2, 3, 4; all training rows
   kept; seven reception-density features dropped) and the previous-visit stage
   (E64: same seeds; all rows kept; three reception features dropped). Their
   historical out-of-fold training parents are unchanged.
6. For Rome departures with an NM record, ground coverage below 2 and a previous
   visit, replace the E64 output: add the mean residual of the five saved E188
   models to this chain's own pre-E64 prediction.
7. Keep the pre-surface prediction where a planning regime has probability ≥ 0.95
   (E152). Blend the whole file with the historical pipelines v44 (`0.1214`) and
   v60 (`0.0270`) (E81). Apply the LOBT edge lift and window clipping, then
   restore the 42 protected TabPFN values.
8. Fit and apply the OPDI stage (E159b) at LSZH only: all 2025 training months,
   historical out-of-fold parent `cv_preds_e150c_e64_keep`, shift zero. Blend its
   output with the pre-LSZH file at 0.5. Every stage keeps its original rounding order.

The earlier v12074 replay and this replay's operational limits are kept in
[analysis/reproduce_v12074_20261007/HISTORY.md](../analysis/reproduce_v12074_20261007/HISTORY.md).

## 2b. Rebuild v12076 and v12077 from the same components

Both files were produced on 8 October 2026 by the v12075 replay path of section 2
(`reproduce_v12075_v2.py` imported as a library, the same archived features,
member predictions, saved models and frozen sources), each with exactly one input
or constant changed and no model fitted. Each builder compares its output with
its parent only after the output is sealed, and never uploads.

**The ten-seed neural member.** The seeds 0–9 fitted for the final file (section
6 of [final submission](final-submission.md)) are fits on the four-month feature
view and cannot be applied to the January/July queries of E203. Seeds 3–9 for
the ranking file were therefore fitted on the frozen E203 ranking pack
(`data/e203_rank_inputs_20261007`, `data/e203_rank_prepared_20261007`) with the
unchanged `S.fit(arm='ordered', seed)` on the Home GPU
(`scripts/e203_rank_seeds3to9_20261008.py`, launcher
`scripts/e203_rank_seeds3to9_run_20261008.sh`; 7 fits of about 225 s). Controls,
all exact: pack and archived-output hashes, PLE bins equal to the archived
`bins.npz`, the archived three-seed mean equal to `prediction.npz`, the archived
`model_0.pt` reproducing `seed_0.npy` through the same inference path, and every
new checkpoint reproducing its seed array. Output
`data/e203_rank_seeds3to9_20261008/models/prediction_10seed.npz` (float32 mean of
ten seeds, same IDs, order and dtype as the archived member) with the receipt
`models/complete.json` (SHA-256 `e5910d44…`), which both builders require.

**v12076** (`scripts/v12076_nn10seed_build_20261008.py`): the replay with
`data/e203_rank_20261007/prediction.npz` replaced by `prediction_10seed.npz`;
weights, E81 donors, the 42 TabPFN rows and the cached stage models unchanged.
Output `data/v12076_nn10seed_20261008/`, candidate
`submissions/gentle-igloo_v12076.parquet` (SHA `c22de523…`), build report
`submissions/gentle-igloo_v12076.manifest.json`; 1 min 57 s on the Mac, 0 fits;
premix `mvt_id` and `proba` equal to the archived v12075 parts; 250,190 rows
differ from v12075, RMS 4.849 s, largest 909 s.

**v12077** (`scripts/v12077_nn035_build_20261008.py`): the v12076 build with the
regular-candidate weights set to `WEIGHTS['final']` of
`scripts/final_tail_assembly_20261008.py` (raw `0.028888888888888857`, the four
anchors `0.08666666666666667` each, neural member `0.35`, tree pair the
remainder). Output `data/v12077_nn035_20261008/`, candidate
`submissions/gentle-igloo_v12077.parquet` (SHA `f06c66bf…`), build report
`submissions/gentle-igloo_v12077.manifest.json`; 1 min 38 s, 0 fits; 286,560
rows differ from v12076, RMS 7.422 s, largest 909 s; RMS 8.682 s against v12075.

```bash
.venv/bin/python scripts/v12076_nn10seed_build_20261008.py check
.venv/bin/python scripts/v12076_nn10seed_build_20261008.py build
.venv/bin/python scripts/v12077_nn035_build_20261008.py check
.venv/bin/python scripts/v12077_nn035_build_20261008.py build
```

`check` verifies every pinned input (the section 2 archive, the ten-seed member
and its receipt, and for v12077 the v12076 candidate and its report) without
writing; `build` refuses to run if the output directory or the candidate exists.
Records: [v12076](../analysis/v12076_upload_20261008/RESULTS.md),
[v12077](../analysis/v12077_upload_20261008/RESULTS.md); journal 416, 417, 426, 427.

## 3. Retrain from raw data

A clean end-to-end run of this list has not been completed: the producer code
exists, but no tested script runs it in order. Start from the organiser's files
in `data/prc-2026-datasets/`. "Not measured" means the repository has no record.

| # | Step | Script | Main inputs | Output | Machine | Wall-clock |
|---|---|---|---|---|---|---|
| 1 | Daily ADS-B surface extraction (adsb.lol) | `scripts/get_adsb_day.sh`, `src/adsb_offblock_v{2,3,5}.py` (default, 1-knot and 1-knot stand-walk settings), `src/adsb_visit.py`, `src/e69_visit_merge.py`, `src/e134_head_features.py`; order as in `adsb_day` of `src/run_final_any_period.sh` | adsb.lol daily archives, challenge data | `data/adsblol/`, `data/adsblol_kt1/`, `data/adsblol_sw_kt1/` | CPU | ≈10–20 min per day, three days in parallel on 6 cores (estimate, [audit §3.3](../analysis/final_any_period_audit_2026-09-26.md)) |
| 2 | OPDI ground events | `scripts/get_opdi_events.sh`, `scripts/get_opdi_events_2025.sh` | OPDI event files | `data/opdi/ev_*.parquet` | CPU | not measured |
| 3 | Base feature table (E106) | `src/features.py`, `src/e28_forward_neighbours.py`, `src/e77_filed_twins.py`, E106 step; order as in `src/run_final_any_period.sh` | challenge data, external snapshots | `data/features_e106.parquet` | CPU | not measured |
| 4 | Final feature table (FIN1b) | `src/e163_build_features.py`, `src/e51_add_route_features.py`, `src/e157_wake_features.py`, `src/e165_build_features.py`, merged by `src/build_final_table.py` in `hz_queue/20_fin1b_cv.sh` | E106 table, ATFM aggregates, OSM taxiway graph | `data/features_fin1b.parquet`; column order and hash must match | CPU | not measured |
| 5 | Regime classifier, base regressor, LightGBM 5 seeds (v870) | `hz_queue/40_fin1b_base.sh` (`src/train.py`, `src/seed_avg.py`) | FIN1b | `submissions/clf_v870.txt`, `reg_v870.txt`, `lgb_reg_seeds_v870.npz` | CPU | not measured |
| 6 | CatBoost, 15 seeds (0–14), 5,600 rounds | `src/mac_e175_catfin1b.sh`, `src/mac_e175b_seeds.sh` (`src/catboost_seeds.py`) | FIN1b | `submissions/cat_fin1b_seeds15.npz` | CPU | ≈1 h 45 min per seed on 6 cores (estimate, [audit §3.3](../analysis/final_any_period_audit_2026-09-26.md)) |
| 7 | Five anchored members, 5 seeds each; rounds raw 2565, LOBT 3065, schedule 2958, EOBT 2736, AOBT 2738 | `src/e130_rank.py` via `hz_queue/hp_e182_rank_20261005.sh` (raw, LOBT, schedule) and `src/mac_e182_rank_20261005.sh` (EOBT, AOBT) | FIN1b | `submissions/e130_{raw,lobt,sched,eobt,aobt}f_rank.npz` | CPU | not measured |
| 8 | Out-of-fold parents of the tail stages | `src/run_e24.sh` (`cv_preds_nomonth`), `src/mac_e151_prodcv.sh` (`cv_preds_e116prod`), `src/e150c_e64_keep_cv.py` (`cv_preds_e150c_e64_keep`) | feature tables, ADS-B extracts | six-fold arrays `data/cv_preds_*/` | CPU | not measured |
| 9 | Movement-only member (E185) | `src/mac_e185.sh` (`src/e185_motion_features.py`, `src/e185_motion_rank.py`) | FIN1b, ADS-B extracts | `data/e185_motion_rank.npz` | CPU | not measured |
| 10 | Neural member (E203): query tensors, ADS-B visit sequences, three seeds × eight epochs | `scripts/e190_rank_20261006.py` (query tensors), `scripts/e194_trajectory_sequence.py` (427 daily sequence archives), `scripts/e203_trajectory_production.py` | FIN1b, ADS-B extracts | `data/e203_rank_20261007/prediction.npz` | CPU, then GPU | GPU fits ≈269 s per seed ([status receipt](../analysis/e203_reproduction_status_20261007.json)); preparation not measured |
| 11 | LIRF previous-visit models (E188): 365-day masked previous-visit extraction, natural and synthetic rows | `scripts/e188_rank_20261006.py fit` | `cv_preds_nomonth`, `data/visit_mask_20261005_v2` | `data/e188_rank_20261006_v2/seed{42,1,2,3,4}.txt` | CPU | not measured |
| 12 | Historical donors: v44 (per-airport models), v60 (earlier global chain), TabPFN rows (v291 → v361) | `src/run_final_e38.sh` + `src/after_v42_stage.sh` (v44); `src/run_final_e52.sh` (v60); `src/lirf_tabpfn.py 291` (v361) | historical feature tables | `submissions/gentle-igloo_v{44,60,291,361}.parquet` | CPU | v44 + v60 ≈6–7 h (estimate, [audit §3.3](../analysis/final_any_period_audit_2026-09-26.md)) |
| 13 | Tail fits (the 11 LightGBM models) and final assembly, recipe steps 1–8 | `scripts/e203_trajectory_production.py` with the frozen sources in `data/today_chain_20261006/src/` | all of the above | final Parquet file | CPU | not measured |

- Steps 3–4 must reproduce FIN1b's exact column order and hash, so the upstream
  builders, external snapshots and joins must run in the original order.
- The CatBoost launchers merge whatever seeds exist after each seed; check that
  all 15 finished.
- TabPFN on the tail rows is not bitwise reproducible
  ([journal row 118](../HYPOTHESES_LOG.md)), so exact reproduction needs the
  archived v361 values. The same holds for v44 and v60 unless a verified rebuild
  exists; the later donor reconstruction (E201) is not an exact replacement.
- External data keep their own licences and acquisition conditions; see
  [SOURCES.md](../SOURCES.md), which also covers OPDI for the LSZH stage.

### Applying the model to a new period

This is not a single command. The 42 frozen TabPFN tail rows and the archived v44
and v60 donor files are specific to the ranking period, so a new period needs a
new TabPFN draw and rebuilt donors (or a recipe without them). The older
`src/run_final_any_period.sh` implements the v808 line (233.5877 s,
[leaderboard table](tables/leaderboard_gentle-igloo.csv)), not v12075: its
features stop at E106, it uses 25 CatBoost seeds and a different anchor mixture,
and it has no E203 neural member, E185 insertion, E188 replacement or
half-weight LSZH stage. Its fresh TabPFN fit also differs from reusing v12075's
fixed 42-row draw. The README's former `features_v4 → train.py submit` example
produces an earlier baseline, not v12075.

## 4. Environment

**CPU stack.** `pyproject.toml` requires Python ≥ 3.12; `uv.lock` pins the
packages (NumPy 2.5.3, Polars 1.44.2, LightGBM 4.7.0, scikit-learn 1.9.1,
pandas 3.0.5, PyArrow 25.0.1, CatBoost 1.2.10, TabPFN 8.5.0, XGBoost 3.4.1).
`uv sync --frozen` creates this environment. `uv.lock` contains torch 2.14.0 only
as a dependency of `tabpfn`, and does not contain `rtdl-num-embeddings`.

The replay ran on macOS 27.0.1 (arm64, as recorded in the history file) with
Python 3.12.14 (`main, Sep 1 2026, 14:09:38`, Clang 22.1.3) and the versions
above. The saved tail
models record the full `sys.version` string and the NumPy, Polars and LightGBM
versions, and the replay rejects any other interpreter build: the Linux GPU host
(Python 3.12.15, same packages) would fail this check
([size report](../analysis/reproduction_bundle_20261007/RESULTS.md)).

**GPU stack (neural member only).** The E203 ranking fits ran on an AMD Radeon RX
7900 XTX with Python 3.12.15, PyTorch 2.13.0+rocm10.0.0 (HIP 7.15.26333) and
`rtdl-num-embeddings` 0.0.12
([status receipt](../analysis/e203_reproduction_status_20261007.json)). The
versions and AMD's wheel index URL are in
[`requirements-gpu.txt`](../requirements-gpu.txt). Replaying v12075 from the
archived member predictions needs neither a GPU nor torch.

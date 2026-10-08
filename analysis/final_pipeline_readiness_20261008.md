# Final-period v12075-style pipeline: implementation readiness, 8 October 2026

Read-only source/artifact inspection by Codex. No feature extraction, model fit,
CV, final inference, large copy, or submission was performed for this note.
Existing calculations are taken from [Claude's strategy](final_strategy_20261008.md)
and [period-transfer report](final_period_transfer_20261008.md), not recomputed.
This is a build plan, not a runnable end-to-end pipeline or a quality result.

## Decision and critical path

Root's selected default is one procedure on **all 670,790 final DEP rows**:
fresh FIN1b from 2025 + the four-month `final_ranking`; Cat15/LGB5/five anchors;
ordered E203; E185; E167; fresh deterministic E156 TabPFN ensemble;
E141/E64; **retain E188**; E152; LOBT lift/clip; protect the new TabPFN segment;
LSZH half correction. **No E81, no copied January/July forecasts, no historical
42-row donor copy.** E207 is outside this default unless separately accepted
through both gates in time. This changes the deployment recipe and needs its
own preregistration; historical best replay remains separate evidence.

The fastest defensible path is **prepare and seal the fresh feature view once,
then parallelize independent full-2025 fits and cache every fitted object**.
CatBoost is the longest measured fit family. It should start as soon as the
fresh feature contract passes, while raw sensor derivatives and neural tensors
are prepared elsewhere. Do not run the legacy `validate`/nested-CV queues again:
their rounds and recipe are already selected.

Important source finding: `src/e165_build_features.py` takes route/type medians
over **training + ranking together**; `src/e157_wake_features.py` similarly
learns the aircraft-type wake fallback from their union. Replacing ranking can
therefore change training features even though the 2025 raw files are unchanged.
Root chose fresh, consistent training rather than an unregistered frozen lookup
from the old ranking. Thus the default **refits classifier/regressor and three
ordered neural seeds too**. Availability of their old models does not prove
they are the fitted models of this fresh view.

## Existing artifacts and the exact missing code

| Component | Existing evidence / fit entry point | Final action |
|---|---|---|
| Base classifier + single proxy regressor | Local `submissions/clf_v870.txt` (13.6 MB) and `reg_v870.txt` (118.8 MB); saved by `src/train.py submit`. Model headers: classifier 1,080 trees / 6 classes = **180 rounds**, regressor **2,319 rounds**; 170 features. | New fixed-round train/save/infer adapter; same `fit_predict` arithmetic, no re-selection of rounds. Saved originals remain immutable. |
| Cat15 | `src/catboost_seeds.py::fit_one`; `src/mac_e175_catfin1b.sh`, `mac_e175b_seeds.sh`: seeds **0..14**, **5,600** rounds, lr .08, depth 8. Only NPZ predictions are saved; no `.cbm` is written by this producer. | **15 fits**, save `.cbm` plus exact columns/categories/proxy recipe/fit fingerprint. Require all 15 seeds; old merge silently accepts a partial set. |
| LGB5 | `src/seed_avg.py rank 5 <rounds>`; `hz_queue/40_fin1b_base.sh` uses the same selected regressor rounds as v870. It saves only predictions. | **5 fits × 2,319 rounds**; seeds 1000+s, bagging 2000+s, feature fraction 3000+s. Save each Booster and preprocessing. |
| Five anchors | `src/e130_rank.py`; launchers `hz_queue/hp_e182_rank_20261005.sh` and `src/mac_e182_rank_20261005.sh`: raw **2565**, lobt **3065**, sched **2958**, eobt **2736**, aobt **2738**, five seeds each. Producer does not save Boosters. | **25 fits**. Preserve seed schedule, ordinary-row selection, clip/anchor arithmetic, column order. `RankArtifactWriter` defaults to **344841**, so existing CLI cannot accept final rows unchanged. Adapter must supply final template count/IDs and save models. |
| E185 motion member | `src/e185_motion_rank.py`; actual `data/e185_rank.log`: **5 × 2481** rounds, 148 features, 1,944,054 ordinary rows in the historical fit. Only per-seed predictions persisted. | **5 fits** after fresh `src/e185_motion_features.py`; preserve `e185_motion_view` category ordering and hidden-feature contract. Save models. Ordinary-row count is a check of historical recipe, not a substitute for final fit-ID validation. |
| E203 ordered neural member | Local `data/e203_rank_20261007/model_{0,1,2}.pt` (7.72 MB each), `bins.npz`, full receipt. `scripts/e203_trajectory_production.py` + `scripts/e197_trajectory_encoder.py::fit`: three seeds 0,1,2; 8 epochs; batch 4096; original optimizer/schedule/network. | Default **3 fresh fits** on fresh feature preprocessing. Persist quantile knots, numeric-column/missing-indicator plan, category maps, residual scale/proxy state, trajectory normalization, PLE bins, checkpoints. New period-sized adapter; current E190/E203 guards bind hashes, dates and row counts to old ranking. |
| E141 + E64 + LSZH | Local `data/e203_chain_20261007/models/{e141,e64,lszh}/`: **5 + 5 + 1** models and complete training signatures. | Reuse only after matching each exact training feature/label/weight/order/round signature. Otherwise explicitly register **11 fixed-recipe refits**, not silently bypass a guard. Apply to new final features/base. |
| E188 | Local `data/e188_rank_20261006_v2/seed{42,1,2,3,4}.txt`, `fit_manifest.json`; five **400-round** models already trained with the masked training augmentation. | Keep these five if fit/source signature remains exact. New **inference** adapter over final natural rows; existing `apply` has `ROWS=344841`. New final ranking does **not itself require another 365-day masked training extraction**. If training signature actually changes, register five refits explicitly. |
| E156 TabPFN | `src/lirf_tabpfn.py`: V2 classifier, CPU, `random_state=seed`, seeds from environment; stable prefix tie-break sorts `[count, airline_prefix]`. | Fixed **seeds 0..9**, `TABPFN_EXTRA=pfx`, original bins/candidate logic; recompute all final tail rows uniformly. Capture probabilities, IDs and the exact protected segment rather than copying donors. Do not hard-code the strategy's 62/63-row estimate: derive the source predicate after features are sealed. |

Default count before optional stage refits: **55 fitted production models**
(15 + 5 + 25 + 5 + 2 + 3), plus ten TabPFN classifier fit/predict calls.
The worst listed fixed-stage refit addition is 16 models. These are full-2025
production fits, not new CV. No updated RMSE is inferred from this count.

The base mixture arithmetic remains the frozen v12075 order:
Cat/LGB .65/.35; anchors raw 1/30, other four .10 each; ordered E203 .25;
E185 .35×2/3 on its original no-NM/outside-LIRF mask. The ordinary candidate
weights must not be confused with the downstream regime-probability mixture.
Use `scripts/reproduce_v12075.py:210` and
`scripts/e203_trajectory_production.py::branch` as the arithmetic specification.

## Data and preprocessing boundary

Create a **new WORK root**, with explicit copied/sealed
`data/prc-2026-datasets/ranking.parquet = final_ranking.parquet` and
`submitting.parquet = final_submitting.parquet`; manifest their source hashes.
Only read-only 2025/static inputs may be linked. Do not symlink writable `data`,
`submissions` or current production source directories into WORK. Frozen source
copies are preferable to the legacy script's mutable `src` link.

Existing feature producer order is available; the full wrapper is missing:

1. `features.py` → `e28_forward_neighbours.py build` → `e77_filed_twins.py` →
   E106 airline-prefix coalesce (inline in `run_final_any_period.sh:287`) →
   `e113_prefix_lookup.py` for the TabPFN prefix table.
2. `e163_build_features.py`; `e51_add_route_features.py` on E106;
   `e157_wake_features.py`; `e165_build_features.py`; then `build_final_table.py`
   in the exact order in `hz_queue/20_fin1b_cv.sh`.
3. Fresh E185 motion view; E194's original **ordered/unfilled** sampler and
   extraction semantics for final E203 query dates; fresh training-only neural
   normalizers. E207's fill sampler is a different candidate, not a portability fix.

Use **final** movement times throughout all four months. January/July raw ADS-B
points can be reused, but movement-linked offblock/visit/head outputs and E194
tokens must be recomputed because official final MVT times changed. Do not reuse
old Jan/Jul derived predictions by ID. Fresh final ARR times must likewise feed
context features; do not substitute the old provided ARR fields.

Claude's existing 175 prefetch `/home/a1/claude_adsb_2026/` owns the 58 February/June
surface days. Do not repeat those downloads. Its final manifest/58-day receipt is
an input admission gate; the earlier ETA is not proof it finished. Derive the
exact needed UTC dates/boundaries from the final rows; do not equate four calendar
months with an automatically complete previous-day context inventory.

Retain unchanged 2025 OOF parents for fixed tail training where their exact
signature still matches; do not revive cancelled E201 or claim full-tail nested
OOF. For OPDI, the prior report establishes incomplete Feb/Jun coverage. Acquire
the missing allowed events or explicitly seal the original rule **no matched
event ⇒ no correction**; never invent an event or use a month-specific weight.
ATFM/METAR/static snapshots need date-coverage admission before feature execution.

## Zero-fit neural alternative: available, but not the selected default

The old checkpoints could serve a separately frozen-preprocessing inference
recipe. Local old query arrays exist: xnf 2.115 GB, xcf 218 MB, xnp 375 MB,
xcp 39 MB. `scripts/mlp_lowmem_prepare.py` does **not** save category maps and
enumerates unsorted Polars `value_counts`; rerunning it need not reproduce codes.
Maps could be recovered from original training categories + saved xcf with an
exact one-value/one-code check. Quantile plans could be recreated from original
fit features and proved against xnf; saved bins/checkpoints then suffice.
Trajectory `normalization.json` is absent locally at the original prepared path
and would need read-only retrieval/verification from its preparation host.
Old-rank per-seed prediction parity would be required before any final inference.
Root selected three fresh fits instead, avoiding this extra contract and keeping
the fresh final feature view internally consistent. Do not call this alternative
already implemented or qualified.

## TabPFN availability and determinism

Read-only local package metadata: TabPFN **8.5.0**, Torch **2.14.0**, LightGBM
**4.7.0**, CatBoost **1.2.10** in the existing Mac `.venv`. Cached V2 classifier:
`~/Library/Caches/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`,
29,009,539 bytes, SHA256
`cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49`.
No import/model initialization/download/fit was run in this audit. Existing
checkpoint presence is not a fresh runtime qualification. Keep the current
environment unchanged and qualify offline under the preregistered resource cap.
Source seeds and prefix tie-break are deterministic choices; prove repeated
probability/output equality on the sealed input rather than assuming all runtime
operations are bitwise deterministic. Persist the explicit checkpoint hash and
the resolved estimator configuration.

`protect_tabpfn_rows.py` already supports `TAB_BASE`; its old default infers
42 donor rows and is inappropriate. For final, retain a declared predicate/ID
manifest of **all served TabPFN rows**, including any whose rounded prediction
happens to equal the preceding value; change-detection alone omits such rows.

## Bounded resource allocation and timing evidence

Historical measurements, not forecasts: Mac Cat logs contain completed seeds
from **42 min to 1 h 42 min** with 12-thread launch configuration (mixed load);
E182 Mac anchors **11.9–13.6 min/seed**, Home sched anchors **11.2–12.4 min/seed**
at 24 threads. E185 Mac five seeds took **36.8 min total**. E203 full-rank GPU
receipt: **268–269 s/seed**, peak **14.15 GB GPU**, **4.23 GiB RSS**. Larger final
prediction population and resource sharing can change these times. LGB5 FIN1b
wall time was not found locally; E163's 431–436 s/seed is a different feature
view and must not be labeled FIN1b measurement.

Suggested launch allocation after preregistration and live-idle/resource check:

| Host | Work | Admission / cap proposal |
|---|---|---|
| Mac 48 GiB | Cat seeds 0..7 serial; later small TabPFN run | 12 threads, process-group RSS guard 20 GiB; disk floor ≥4 GiB. No full fresh WORK/tensor staging on scarce Mac disk. |
| Home 32 GiB | Cat seeds 8..14 serial; ordered NN after E207 relinquishes GPU | Cat 12 threads/MemoryMax12G; E207's 12G reservation plus Cat12G leaves OS room. This is a proposal, pending actual first-fit RSS admission; never start two large prepared-frame jobs together. NN sole GPU owner, 2CPU/12G CPU RAM as existing recipe, full-shape smoke for new rows. |
| Hetzner 62 GiB | Fresh feature view, then one LGB fit worker: base2 + LGB5 + anchors25 + E1855; stage reuse/refit checks | One worker at a time; 6 **physical** cores (`0,2,4,6,8,10`), num_threads≤6, nice10, passive OMP. Feature build first, free its frame before fitting. |
| 171 / 175, ~14 GiB each | Day-at-a-time sensor derivatives / E194, bounded tensor preparation | 2CPU/3–6GiB/day worker; 175 after its existing prefetch; reuse E207's proven owned-unit lifecycle, no new scheduler. No parallel heavy model matrices here. |

Cat split above is approximately **6–14 h on the slower shard at observed Mac
per-seed times**, not a Home benchmark. LGB/feature work runs alongside it. Budget
**12–24 h for first complete candidate after code/data admission**, with explicit
technical-failure slack; this is an engineering estimate, not a guaranteed ETA.
Freeze executable design before the repository deadline; record every subsequent
fixed-recipe execution separately. Do not promise unused machines are training.

Disk planning: new full query count = 2,085,047 + 670,790 = **2,755,837** rows.
Existing trajectory layout uses 4,736 bytes/query (values float16 + known uint8 +
present uint8), about **12.16 GiB** for one all-row table before small metadata;
query tensors, feature intermediates, saved Boosters and receipts are additional.
Reserve **≥40 GiB** on a build host, **≥25 GiB** on the GPU host for one such
final table + query arrays + checkpoints, and retain the ≥4GiB floor. These are
capacity proposals, not fresh free-space measurements. Avoid Mac bulk copies and
extra LAN transfers while E207's existing fold transport is active.

## Minimal implementation ownership proposal

No code was added by this note. Four bounded responsibilities are enough:

1. **Final feature/workspace adapter:** immutable input/source manifest, fresh
   WORK, exact producer order above, final ID/schema/null-label boundaries,
   external coverage receipts, fixed feature order and model-specific views.
2. **Train/save/predict adapter:** base, Cat, LGB5, anchors, motion; reuse existing
   fit functions/arithmetic, persist every model and fitted preprocessing;
   strict complete seed sets, new output namespaces, no implicit resume/overwrite.
3. **Ordered neural adapter:** period-sized E194/E190/E203 preparation and three
   fixed fits; serialize all currently missing fitted transforms; CPU/RAM/GPU
   smoke and same-source synthetic shape tests; no sampler/model redesign.
4. **Tail/assembly adapter:** cached-stage training-signature gate or explicit
   fixed refit, dynamic ID counts for E188/LSZH, E156 served-row manifest,
   E81 omission, E152/LOBT/half-LSZH arithmetic, final template-order validator.

Tests should cover the actual gaps: final 670790/dynamic population rather than
old344841; exact model-save/load predictions on synthetic data; all15/all5 seeds;
feature/category order and unknown levels; no final targets in fitting;
normalizers fit only training; all-month identical routing; TabPFN protection
when its rounded value equals base; missing OPDI leaves base unchanged;
immutable old input/output hashes. Historical exact replay is not an acceptance
criterion for a deliberately new final-input recipe.

## Command-level handoff

Existing feature commands below are ready **only after the new WORK has been
sealed and preregistered**. `$PY` is the existing qualified absolute interpreter,
not an environment-install command; execute with `cwd=$WORK` and captured logs:

```sh
FEATURES_OUT=data/features_final.parquet "$PY" src/features.py
E28_IN=data/features_final.parquet E28_OUT=data/features_e28.parquet "$PY" src/e28_forward_neighbours.py build
FEATURES=data/features_e28.parquet FEATURES_OUT=data/features_e77.parquet "$PY" src/e77_filed_twins.py
# Registered adapter performs the exact E106 coalesce from run_final_any_period.sh.
"$PY" src/e113_prefix_lookup.py
"$PY" src/e163_build_features.py
FEATURES=data/features_e106.parquet FEATURES_OUT=data/features_e106_e51.parquet "$PY" src/e51_add_route_features.py
"$PY" src/e157_wake_features.py
"$PY" src/e165_build_features.py
"$PY" src/build_final_table.py data/features_fin1b.parquet data/features_e163.parquet data/features_e106_e51.parquet data/features_e157.parquet data/features_e165.parquet
"$PY" src/e185_motion_features.py
```

Do **not** launch the old Cat/anchor/seed/`train.py submit` commands for final yet:
they lack the required model persistence, dynamic guards or fixed-round adapter.
The new adapter CLI and exact resource-unit commands belong in its preregistration,
not an invented ready command here. Reuse `src/run_final_any_period.sh` for its
workspace/producer ordering only; executing it end to end produces the old v808
line with different defaults.

The existing final validator entry point, once a sealed candidate exists:

```sh
"$PY" scripts/validate_final_submission.py --file "$WORK/submissions/gentle-igloo_final.parquet" --ranking "$WORK/data/prc-2026-datasets/ranking.parquet" --template "$WORK/data/prc-2026-datasets/submitting.parquet" --json "$WORK/final_validation.json"
```

Publication and the single final upload remain separate authorized actions.

ARR pilot was stopped at metadata inspection when priorities changed. The
read-only 171 check found 11 of 12 requested surface/neighbor dates (not
2024-12-31), expected surface schema, but no original `training_*.parquet` files
in that host's canonical raw-dataset directory. No ARR helper, extraction,
evaluation or model was created; absence on that host is not global absence.

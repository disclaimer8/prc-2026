# E205 — current cloud CB beyond existing weather, 7 October 2026

Protocol `E205-current-cloud-CB-matched-residual-sixpair-preE81-v1`.
Preregistration includes this document, `scripts/night_cb_20261007.py`,
`scripts/night_cb_20261007_queue.py`, input receipt and journal, before any real
target fitting or new error comparison. This is a bounded 12-fit CPU experiment.

## Mechanism and observed reach

The current thunder flag searches `wxcodes`, while cumulonimbus can be present
only in the raw METAR cloud groups. Current convective clouds may affect runway
capacity and ground handling even when the report has no thunder weather code.
This is a concrete additional input, not weather history or a repeat of E193.

On all 2,429,888 FIN1b flights, the label-free join reproduces the existing
`wx_thunder` and `metar_age_min` exactly. Train 2025: 45,461 flights with CB,
40,154 missed by the existing flag, of 2,085,047. Ranking January/July 2026:
5,611 CB, 4,513 missed, of 344,841. No unknown raw reports; airport/month
counts and source SHA are in `input_receipt.json`. These are coverage numbers,
not performance estimates. Ranking is read only for target-free coverage.

Each flight takes the latest report at or before its observed `AOBT_3_flt`, or
`MVT_TIME_UTC_mvt - 900s` if missing, with inclusive 3-hour tolerance. Airport
boundaries are isolated; future weather is prohibited. Same-time source
duplicates deterministically use the final CSV row. Counts are recorded; all
ten current files have zero duplicate timestamps. No temporal history is added.

Four added features are fixed: `cloud_cb`, `cloud_cb_missed` (cloud CB with no
existing TS/CB code), minimum CB base in feet, maximum CB cloud amount encoded
FEW=1/SCT=2/BKN=3/OVC=4. The regex is
`\b(FEW|SCT|BKN|OVC)(\d{3})CB\b`. TCU and free-text remarks do not count.
Missing raw METAR gives four NaNs. Valid METAR with no CB gives flags 0 and
base/amount NaN, never a fictitious zero-foot cloud. CB base is not called a
meteorological ceiling.

## Frozen control and model

Control is `ordered` in `data/e203_pre_e81_20261007/predictions.parquet`, whose
SHA is pinned through the original `result.json` and this input receipt.
All 2,085,047 IDs are preserved; original `evaluation_mask` retains 2,084,921,
with exactly the same 126 excluded TabPFN rows. This current E203 comparison
ends before E81. Historical upstream predictors were not rebuilt inside each
outer residual fit, and E203 was selected on these folds. Therefore this is
selection-informed, not fully nested CV or full production-tail validation.

Two arms differ only by the four CB inputs: existing current weather/context
(`weather`) and the same inputs plus CB (`cb`). Exact 29 shared columns:

```
wx_temp wx_spread wx_wind wx_gust wx_vis_mi wx_ceiling_ft
wx_snow wx_freezing wx_rain wx_thunder wx_fog wx_icing
wx_snow_recent wx_icing_recent metar_age_min
airport_code has_nm local_hour d_mvt_aobt d_mvt_eobt d_mvt_lobt
d_mvt_iobt d_mvt_sched rwy_dep_before_30 nb_ap_30_0_n
n_dep_rwys n_arr_rwys own_rwy_share_hour ordered
```

Airport is categorical, encoded using the fixed ten ICAOs. No ID, month, raw
target, target-derived time, or fresh target encoding is a feature. The common
`ordered` feature is the frozen historical OOF and carries the limitation above.

Each held-out pair is `(1,7)` through `(6,12)`. Fit only the other ten months,
included rows, `y<=7200`, and decided rows. Decided excludes LIRF without NM
and `d_mvt_sched>=14400`. The body restriction uses training labels explicitly;
held-out labels never decide application, correction magnitude or features.
The target is `clip(y - ordered, -900, 900)`.

Both arms: LightGBM squared-error regression, 400 rounds, learning rate .03,
15 leaves, max depth4, min leaf500, L2=100, max_bin127, full rows/features,
seed20307, deterministic/force_col_wise, at most6 actual threads. No early
stopping, tuning or seed selection. Apply to included and decided held-out rows:
`max(0, ordered + 0.5 * clip(residual_prediction,-120,120))`.
All other rows exactly retain `ordered`. No held-out body filter is applied.
No ranking model is trained by this protocol.

## Gates and interpretation

Require **both** CB versus matched weather and CB versus untouched ordered to:

- have pooled body (`y<=7200`, included) SSE difference below zero;
- have pooled decided SSE difference below zero;
- have decided SSE difference below zero in at least4 of6 pairs;
- have negative totals of within-fold decided symmetric trim3 and trim20 SSE.

Body harm rejects the experiment. Pooled included and weather versus
base are descriptive; no extra threshold, subset/airport selection or weight
search follows. Empty primary groups or nonfinite metrics fail. Zero SSE is
well-defined and a tie cannot pass the strict improvement gate. Pooled trimmed
SSE does not replace the within-fold trim gate.

Expert prior before new comparisons: body delta center0s, uncalibrated range
[-0.5,+0.5]s. This is not a CI, transfer estimate or promise. A pass only supports
a separately registered full-chain/ranking step; 42 production donor rows must
remain exact in that future step. This protocol never changes them or uploads.

## Integrity and operation

Source/input SHA and config are frozen in the registered receipt. `pack` verifies
the registration commit and source/journal before reading targets. Pack validates
ID/month/airport/NM/D/y alignment against FIN1b and original protected support.
The portable bundle is one compact parquet plus script, protocol, receipt and
manifest. Workers verify all hashes and their own source before fit and after.
Each arm saves its model; each fold saves predictions and independent metric
inputs, model hashes, runtime versions and a completion receipt. The collector
requires each of six folds exactly once, checks metadata/support/SHA and
recomputes metrics before reporting the gate. Partial outputs remain on failure;
no automatic restart, environment mutation or fallback is allowed.

Assignments are disjoint: Hetzner folds1/4, laptop171 folds2/5, laptop175 folds3/6.
Per-node sequential arms/folds. Hetzner system service: physical CPUs
0,2,4,6,8,10, CPUQuota600%, MemoryMax12G, swap0, Nice10. Laptops user services:
CPUQuota600%, MemoryMax9G, swap0, Nice10. Six-thread OpenMP hard limit, passive
waiting, zero GOMP spin and Polars2. RuntimeMaxSec6h per node. Existing runtime
only; never install packages. No Home CPU/GPU or local Mac fit.

Durable queue on Mac handles pack/transfer/launch/collection, tied to this
registration. It waits at most6h15m, records InvocationID and terminal exit;
failure stops collection/evaluation without killing independent foreign jobs.
Exact systemd properties are checked after launch. The portable verifier can
be run independently on ASUS with target-free query columns+METAR under its
separate audit protocol; ASUS has no duplicate fold assignment here.

Commands, after root supplies the registration commit:

```
.venv/bin/python scripts/night_cb_20261007_queue.py launch --root . --registration-commit SHA
```

The launch writes PID/state/log once and runs under caffeinate. Completion is
`analysis/night_cb_20261007/execution/evaluation.json`, not process existence.
Before actual InvocationID/PID and FIT log, report only prepared or waiting.

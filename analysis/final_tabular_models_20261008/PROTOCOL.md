# Final tabular production portability, 8 October 2026

Draft, not preregistered or executed. Protocol `final-four-month-tabular-fixed-recipe-save-models-v1`.
This adapter trains the already selected tabular recipe on the fresh feature view
of 2025 plus all four final months. It is not a new CV experiment and makes no
claim about final RMSE. An engineering pass requires the exact input IDs/schema,
fixed rounds/seeds/feature recipe, saved fitted transforms and model reload parity.

## Frozen recipe proposed for registration

Base: one regime classifier, 180 rounds, and one proxy regressor,2319 rounds,
using unchanged train.fit_predict, original cost weighting/minimum spread3600.
Cat15: seeds0..14,5600 rounds,depth8,lr.08. LGB5:2319 rounds.
Five five-seed anchors:raw2565,lobt3065,sched2958,eobt2736,aobt2738.
Motion:E185,5seeds×2481 rounds. All LGB seeded members use1000+s/2000+s/3000+s.
Production DROP_PREFIX and PROXY_SCALE=1/ridge1000 are explicit; competing
optional recipes/weights are refused. Total52 tabular fitted models;three
ordered neural models belong to the separate final neural adapter.

Classifier/raw-anchor categories use all training rows; LGB5 categories use
ordinary training rows; Cat uses original NA/string conversion. Proxy state
stores per-airport medians and per-airport/source scales, with original unseen
source fallback1.0. E185 uses its unchanged NM-hiding/movement matrix helper.
All target encodings computed by original tabular callers are removed by this
recipe's feature mask, so no used TE transform is omitted. The neural recipe
has its own used TE transforms and separate serialization.

Every model saves its parameters, ordered features, categories, proxy when
used, native model file, predictions and hashed completion receipt. Save/load
prediction equality is required; the base mixture allows1e-8 absolute floating
rounding in reconstructing proxy arithmetic. Native seed predictions require
array equality. All15/all5 seeds must be complete before a family mean; partial,
duplicate, mixed-input or changed model artifacts fail. No output overwrite or
automatic retry/resume. Root may assign disjoint seed shards to separate hosts.

The feature owner supplies feature_complete.json. Root binds its feature SHA,
ordered train/final ID hashes, final template SHA, motion SHA, helper source
hashes and receipt itself in a separate contract committed with this worker,
tests, protocol and journal entry before any real fit. The Mac creates a Git
admission receipt; Home verifies the fixed receipt and all bound files without
copying private Git history. This is root-controlled transport, not an external
cryptographic signature. No contract has yet been created for actual final fits.

## Execution boundaries

Only2025 labels may enter fits. Exactly2085047 training and670790 final DEP
IDs are checked; all final targets/BLOCK are null. The same learned family
applies to all four months, without old Jan/Jul predictions. No CV, tuning,
ranking upload or E81/donor handling exists in this adapter. MVT-derived
features come from the new final input via the feature-owner build.

Use existing environments. Import Torch before LightGBM. Synthetic Mac tests
required OMP_NUM_THREADS=1/OMP_THREAD_LIMIT=1 before native library initialization:
without them a native LightGBM predict call segfaulted; the bounded repeat
passed. Do not mutate environments to fix this. Production native threads are
set before importing ML packages; fitted models store all seven package versions
and thread settings. CPU-only Linux workers do not import or require Torch.

Initial deployment after live resource admission is Linux-only: Cat
Home12threads/12GiB, one seed at a time. Mac production fits are explicitly
refused until a separately reviewed RSS/time supervisor exists. Hetz one LightGBM worker,
6physicalcores0,2,4,6,8,10/nice10/passiveOMP,≤32GiB. The worker enforces Linuxcgroup CPU≤requested threads, memory≤12GiBCat/32GiBLGB,
swap0/nice10 and4GiB disk floor before each seed; root must supply RuntimeMaxSec
and owned unit receipts in the actual preregistration. This script does not start detached processes or
implement a second scheduler. Feature build and E207 memory reservations take
precedence. No production launch is authorized by this draft.

## Verification so far

Seven synthetic tests pass, including saved LightGBM and CatBoost prediction
parity, original proxy parity, unknown/null categories, label boundary,
strict seed sets, anchors, overwrite and finite checks. No competition rows
were decoded by these tests. Further independent review and final feature
contract admission remain before scientific execution.

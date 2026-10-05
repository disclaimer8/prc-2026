# FIN1b E181a split Mac preparation / ROCm fits, pre-registration287

Fixed retrospective completion of the production-configuration MLP component
OOF: all6 month pairs, each seeds0/1/2,8epochs,width512. No new model, quality
search, upload or full-v12048 validation. Historical backend is uncertain;
matching configuration is not bitwise historical ranking reproduction.

## Why split

Registered284 full-data prepare-only probe was OOM-killed at its12GiB cgroup
limit during target encoding, before numeric preparation and before any fit.
Before-encoding RSS was5.986GiB. The failed attempt and exact runtime/source
manifest remain evidence. Cat continued running. We do not raise the memory cap
or retry the same allocation pattern. Mac has48GiB RAM and prepares one fold;
home GPU reads only prepared matrices, estimated2.2GiB, under the same12GiB cap.

## Producer and exact arithmetic

`scripts/mlp_gpu_component_oof.py prepare` creates one fresh immutable fold
bundle. Mac uses the unchanged four E100/train/target_encoding/features sources
at pinned SHA and FIN1b SHA
`f061a441f6c97c232d9a403d612471e45020eb4ad56f501c2acfd7220a7d3052`.
The sequence exactly follows original`regular_candidate`: fit-only regime labels,
fit-month-out target encoding, original feature selection, calibrated proxy,
same nested target clipping, residual standard deviation, and low-memory prepare
already parity-tested in284. Original category mapping is shared by all3 seeds.
No network, optimizer or fit runs on Mac.

Bundle stores Float32 numeric train/pred matrices, Int64 categorical matrices,
Float64 original`y/sd`, Float64 prediction proxy, sd, category cardinalities,
held-out original Float64 integral IDs/y/month and validation→control permutation.
Held-out labels only verify metadata; they never feed training. Every input
array/source/helper/metadata/evidence has SHA; manifest publication is atomic.
Control must match the entire exact held-out population and targets.

MLP DROP_PREFIX is`day_,tq_,wxr_,month`, retaining te_/ter_, not base170features.
PROXY_SCALE1, PROXY_SCALE_RIDGE1000, AOBT_GATE1, REG_CLIP_HI7200, REGIME_TOL6,
REGIME_EXCLUDE EDDF/EHAM/LSZH, SCHED_TOL_LONG0, REGIME_RATES0. Producer and consumer
use a clean fixed child environment; no unrelated experiment switches survive.

## Consumer, resources and fixed schedule

`run` verifies all bundle SHA and admitted exact home runtime from284 manifest,
then validates matrix dimensions/dtypes/finite values, category code ranges,
positive finite sd, exact ID permutation and metadata. It never reads the
original parquet or computes target encodings. Original`fit_nn` and`predict`
are called unchanged for seeds0,1,2, including original4096 batch, AdamW,
OneCycleLR, dropout, shuffling and Float32 training. Original mean×sd+proxy and
control-row permutation produce the component. No AMP or CPU fallback.

Same RX7900XTX cuda:0 backend/runtime for all6 folds. Dedicated unit:
MemoryMax12G, MemorySwapMax0, CPUQuota200%, Nice10; Polars/OMP2, BLAS1. Actual
cgroup checked before training; torch imports before LightGBM. Tiny64×64 GPU
warmup initializes context. Cat health and host memory are checked by root
before launch; no working environment or active Cat service is modified.

Only all3 completed seed fits may publish a fold. Atomic `prediction.npz` has
reg2, original ID/y/month plus manifest with exact input digest, runtime, source
lineage and memory usage. Finished output is verified read-only on resume and
never overwritten. Each command holds an exclusive sidecar writer lock. Failed
partial bundles/publications remain for diagnosis. No metrics choose whether to
continue the fixed6×3 schedule.

## One-fold disk pipeline

Mac has21GiB free at registration. `scripts/mlp_split_queue.py` prepares onefold
(~2.2GiB), transfers it to home, runs destination`verify` and compares spec SHA,
then waits for the dedicated GPU unit to finish successfully. It copies and
verifies the finished component on Mac before preparing the next fold. All six
input bundles are retained on both machines; nothing is deleted. Each new Mac
preparation requires at least7GiB free, leaving over4GiB after an expected bundle.
Queue and fold writer locks prevent concurrent duplicate runs. Failure stops
the queue with artifacts retained. All six folds use the same fixed recipe.

Example producer arguments: `prepare --root /Users/denyskolomiiets/prc-2026
--input <owned-fold-directory> --fold 1 --registration-commit <287-commit>
--capacity-evidence analysis/mlp_gpu_prepare_failure_2026-10-05.json
--gpu-runtime data/mlp_gpu_prepare_20261005/manifest.json`.
Home consumer: `run --input <verified-fold-directory> --output <complete-fold>
--registration-commit <287-commit>` inside the dedicated cgroup.

Synthetic checks cover tensor roundtrip/invalid values and codes, source/input
tampering, same matrices across original3seed calls, unchanged fit function,
exact scaling/ID permutation, no parquet read on consumer, immutable completed
resume, device/version refusal and environment isolation. Registration287
commits code+this protocol+journal before any real fold preparation or fit.

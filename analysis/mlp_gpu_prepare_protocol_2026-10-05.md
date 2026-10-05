# FIN1b E181a ROCm prepare-only memory probe, pre-registration 284

Purpose: measure whether the existing E181a production configuration can produce
the missing three-seed component OOF on the home RX 7900 XTX concurrently with
the running CatFIN1b CPU job. This probe cannot fit a model or write predictions.
It does not evaluate quality or choose a new model. Cat and Python environments
must remain untouched.

## Fixed transformation and source lineage

`scripts/mlp_lowmem_prepare.py` replaces only E100 `prepare`: first compute the
same float64 257-point quantiles and missingness decisions, then allocate exact
float32 matrices and write each transformed column directly. Interpolation is
still float64, cast to float32 at the same semantic boundary. Numeric feature
order, selection thresholds, missing indicators, category count threshold20,
category ordering from `value_counts`, int64 codes, rare/unseen zero and card
counts match the original. Categories are allocated directly rather than stacked.
No sorting is introduced in production. Polars `value_counts` hash order already
varies between original calls; synthetic byte parity injects the same stable
counts order into BOTH reference and candidate only in the test fixture. It
does not establish identical randomly initialized embeddings between two natural
calls with differing category maps.

The probe snapshots four unchanged trainer/helper source files with pinned SHA,
the adapter and probe launcher, fixed environment and ROCm runtime. FIN1b SHA is
`f061a441f6c97c232d9a403d612471e45020eb4ad56f501c2acfd7220a7d3052`.
Torch is imported before LightGBM. Source snapshots and live feature SHA are
verified before work. Shared `src` and working environments are never modified.

## Memory experiment and gate

Only month pair1+7 is prepared, using original E100 `main` and `regular_candidate`.
This preserves outer training/validation dataframe residency and target encoding
allocations, so a small artificial standalone adapter peak cannot masquerade as
the real process peak. The first `fit_nn` call is replaced by an unconditional
stop before any forward pass/optimizer/model construction. A unique fresh output
directory and exclusively created `probe.json` prevent overwriting evidence.

Dedicated systemd unit: `MemoryMax=12G`, `MemorySwapMax=0`, `CPUQuota=200%`,
`Nice=10`; Polars/OMP2 and BLAS1. The worker checks actual cgroup constraints.
The existing Cat service is not changed. Probe reports startup, target encoding,
numeric preprocessing and before-fit peak RSS from Linux `ru_maxrss`, plus
runtime, cgroup and source manifest digest. An OOM/failure is a failed capacity
probe, not a model result. No automatic retry with a larger RAM allowance.
Before loading data, a64×64 synthetic GPU matrix multiplication initializes the
ROCm context/GEMM library so its host-memory cost enters the probe peak. This
does not instantiate or fit a model. A single-writer lock prevents concurrent
`run` invocations in the same prepared directory.

Historical E181a folds have1.607–1.633M regular-fit rows,270–272 numeric columns
and14 categories. The resulting fit+prediction arrays are estimated2.2GiB.
Original accumulated float64 vectors/stack/cast add several GiB; retained Polars
frames may still dominate. Expected whole-process RSS8–11GiB is an estimate.

Proceed only if measured peak RSS **<11GiB**, the Cat process remains healthy,
and host available RAM is rechecked with margin. Root then registers a separate
fixed training launcher/protocol before any real fit. Intended task is all six
month pairs × seeds0,1,2, width512,8epochs; use one ROCm backend for all folds,
not a mixed MPS/ROCm OOF. This remains matching configuration with historical
backend uncertainty, not bitwise ranking reproduction or complete v12048 OOF.
No model gains, quality gate, leaderboard upload or new hypothesis is authorized
by the prepare-only registration.

## Validation before the real-data probe

Reference function is extracted from the actual unchanged E100 source by AST.
Synthetic byte parity covers finite/nonfinite numeric data, all-null and
constant columns,99/100 valid-value boundary,257 quantile edge points, repeated
values, missing indicators, categories with19/20 observations, null/unseen
categories, multiple categorical column ordering, four random seeds and the
original empty-feature-family failure. Root commits code, this protocol and
journal row284 together before preparing any real fold.

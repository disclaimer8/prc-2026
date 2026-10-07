# v12075 independent saved-component replay — PASS

Completed **2026-10-07 22:33:00 UTC**, registration
`70aff7199731c381945c45754ee70c7e3a964aa5`, protocol
`v12075-independent-saved-component-replay-zero-fit-lowmem-v2`.

| Check | Measured result |
|---|---|
| Complete output | 344,841 rows and 344,841 unique IDs |
| Schema | `MVT_ID_mvt: Float64`, `TAXITIME_SEC_mvt: Int32`, identical to reference |
| Prediction differences | 0 rows, maximum absolute difference 0 |
| Byte equality | PASS |
| Protected TabPFN donor rows | 42 of 42 identical |
| New fits / uploads | 0 / 0 |
| Cached tail / E188 models | 11 / 5 |
| Elapsed time | 101.191959 seconds |
| Sampled combined RSS peak | 8,485,355,520 bytes, 7.90 GiB |
| Input and frozen-source checks | 3,711 SHA256-pinned files, checked before and after replay |

Both reference and freshly produced output SHA256:
`90d3cd0b5a68cc8008f2142f1801857752574b525b2ff046ad6bf4089a70aee2`.
The stored leaderboard value is 230.9387 s; this replay made no leaderboard
request and no new submission.

The runner recomputed the complete fixed v12075 chain in
`data/reproduced_v12075_v2` using versions 990200–990215. It used the saved
ordered-trajectory member at weight 0.25 and the existing model/member/donor
artifacts. The current best and original final candidate were not prediction
inputs. `assembled_before_oracle.json` records the generated output before the
final reference was opened.

The initial low-memory blend was qualified against the original E203 premix on
all 344,841 rows. Every NPZ field (`mvt_id`, `reg`, `proba`, `values`, `pred`),
including dtype, ordering and NaNs, matched exactly; premix Parquet bytes also
matched. The helper retained the same 170 classifier features and 11 categorical
dictionaries, read 173 necessary columns from the 210-column table, and skipped
only target-encoding columns excluded by the fixed production recipe.

After both completion receipts appeared, a separate read-only verification
re-read the output and best, checked schema/IDs/every prediction and byte hashes,
independently derived the protected set from v361 versus v291, and validated the
supervisor-to-runner receipt hash. Its result is
[independent_verification.json](independent_verification.json).

Durable evidence:

- `data/reproduced_v12075_v2/complete.json`
- `data/reproduce_v12075_v2_launch_20261007/complete.json`
- `data/reproduced_v12075_v2/premix_parity.json`
- `data/reproduced_v12075_v2/lowmem_blend.json`
- `data/reproduced_v12075_v2/assembled_before_oracle.json`
- `data/reproduced_v12075_v2/jobs/*.complete.json` and per-stage logs.

The first attempt (journal 368, registration 05a1b2e) was stopped at the initial
blend when its group exceeded the 16 GiB sampled-RSS guard (19,267,698,688 bytes
observed). Its failed output and launcher state are preserved. Its original
sources remain unchanged. The v2 correction was separately reviewed and
registered before execution. Numeric process metadata confirmed the old group
had no surviving process; v2's supervisor confirmed its own group had exited.

The targeted v2 tests passed (10 tests). The full project suite at preparation
passed 367 tests with two skips: the Home-only PLE package and unavailable system
libxxhash. The independent review also checked excluded-encoding semantics on
shuffled synthetic rows; the production premix gate then verified the full data.

This establishes exact **saved-component reproduction** in the existing qualified
environment. It does not establish full retraining from public raw data,
cross-platform equivalence, new nested-CV validity, or public acquisition of every
local artifact. The original E203 selection-informed and full-tail-OOF limitations
are unchanged. See [the current guide](../../docs/reproduction.md) for commands,
and [the immutable v2 protocol](PROTOCOL_V2.md) for limits and scope.

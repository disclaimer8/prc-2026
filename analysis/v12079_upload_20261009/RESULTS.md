# v12079 (fresh E156 tail rows on v12077): uploaded 08:58:31 UTC, **not processed by the board as of 10:36 UTC**

**Status:** one PUT at 08:58:18 → 08:58:31 UTC (ledger `intent`/`uploaded`, `used_before` 1), the object
`gentle-igloo_v12079.parquet` is in the team bucket with `last_modified` 2026-10-09T08:58:31.170Z (authenticated read-only
listing on Hetzner). The public leaderboard has **no item for it 97 minutes later**, and it has processed **no file of any team since
08:54:51 UTC** (last four: bold-ladder_v224 08:08, zesty-sailboat_v50 08:32, smart-jigsaw_v56 08:43, gentle-igloo_v12078 08:54:51) —
an organiser-side stall of the scorer, not something specific to this file. Nothing was re-uploaded; a second PUT of the same bytes or
of the same content under another name is forbidden without the coordinator's decision. The file counts as **2 of 5** slots on 09.10
by our guard's rule (bucket objects of the UTC day), whatever the scorer does.

File: `submissions/gentle-igloo_v12079.parquet`, SHA-256 `d3111a7f1978e838db3c31490dba2ef923b003ee59f67d4cc815334f4660bb7e`;
build report `submissions/gentle-igloo_v12079.manifest.json` (SHA `87cd69f9…`). Submission ID: none yet; `score_response.json` will be
added when (if) the board scores it.

## Diagnosis without a new PUT

1. **Authenticated status/quota endpoint:** our tooling has none. Quota checks use the public leaderboard API (processed items only;
   it exposes no queued/rejected state) plus the authenticated S3 listing of the team bucket. The listing shows the object; the API
   shows nothing — consistent with "queued, scorer stalled", indistinguishable from "silently rejected" by these two sources.
2. **Validator on the exact uploaded bytes** (`scripts/validate_final_submission.py`, ordinary template): every HARD check passes —
   columns `['MVT_ID_mvt', 'TAXITIME_SEC_mvt']`, dtypes Float64/Int32, 344 841 rows, no null/NaN/inf, 0 duplicate IDs, ID set identical to
   the template, value range 0…109 663, all integer seconds; the only failure is the name check for `gentle-igloo_final.parquet`, which
   does not apply to a regular submission. Parquet metadata identical in kind to v12077/v12078 (writer Polars, ZSTD, 3 row groups,
   double/int32), row order identical to the template and to v12078 (which scored 45 s after its PUT). v12078 max value 110 987 >
   v12079 max 109 663, so the value range is not the difference either.
3. The transport's preflight (same guard chain as v12076/v12077/v12078) passed on both hosts before the PUT.

## The single change (for the record)

The v12077 build path (`scripts/v12078_v12079_builds_20261009.py --variant freshe156`, zero fit, 08:49–08:51 UTC) with the 42 frozen
TabPFN tail rows (v361 vs v291) replaced by the fresh ten-seed E156 values the 08.10 canary computed on the same Jan/Jul inputs
(`analysis/final_tail_assembly_20261008/dryrun_janjul/e156_served.json`, SHA `80f5f261…`, same 42 IDs — verified); the values enter where the
donor values entered (b+2 → b+3) and are restored by the unchanged `protect_tabpfn_rows.py`; E81, premix (byte-identical to the v12077
premix), every other stage and the cached models unchanged. Difference against v12077: exactly 42 rows, RMS 7.874 s
(= 713.5 s × √(42/344841)), mean +0.045 s, max |Δ| 2058 s.

## Forecast (pre-registered №433) and decomposition so far

Central **231.17** (+0.69 s vs v12077 = canary +1.2546 minus the measured E81 price +0.5676), corridor **[230.83; 231.48]**.
**Not checkable yet.** Known half: E81 price on this chain = v12078 − v12077 = **+0.5676 s** (№432). If v12079 scores at the forecast,
fresh E156 ≈ +0.69 s and the two parts add to the canary's +1.25 s within the weight/member interaction.

## Delivery

Transport `scripts/submission_transport_v12079_20261009.py` (SHA `4df36c06…`; registration `6963463cf5ac6bbae21caafea475208544849cc1`,
after a one-line fix of the served-row check committed before seal), pins: candidate, builder `4a396236…`, build report, member receipt
`e5910d44`, served rows `80f5f261`, parent v12077 `f06c66bf`, difference vs v12077 (changed 42, RMS 7.874374, max 2058), stop rule
RMS ∈ [0.5; 30] s; five negative scenarios stop on the Mac. Host Hetzner `.venv-upload`; SHA of every copied file verified; preflight
passed ([preflight_hetzner.json](preflight_hetzner.json)); exactly one PUT, no retry ([upload_stdout.log](upload_stdout.log),
[canonical_ledger.jsonl](canonical_ledger.jsonl)). Registration proof (SHA `7af72882…`) kept on both hosts, not committed.

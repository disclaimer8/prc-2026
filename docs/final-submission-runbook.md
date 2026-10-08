# Final submission runbook — `gentle-igloo_final.parquet`

One file, one PUT, no retry. Written 2026-10-08 from the organiser's announcement (Discord 08.10 07:34Z/07:46Z,
`docs/organiser-clarifications.md`), the site (`ranking.html`, `data.html`) and the repo's frozen upload tooling.
This document describes the procedure; it does not run anything.

## 1. What is uploaded

| Item | Value |
|---|---|
| Object name | `gentle-igloo_final.parquet` in bucket `prc-2026-gentle-igloo` (`s3.opensky-network.org`) — exactly one |
| Template | `data/prc-2026-datasets/final_submitting.parquet`: 670,790 rows, `MVT_ID_mvt` Float64, `TAXITIME_SEC_mvt` Int32 |
| Inputs | `data/prc-2026-datasets/final_ranking.parquet` (1,341,197 rows; DEP for 2026-01/02/06/07 = 152,719 / 144,158 / 181,791 / 192,122) |
| Rejection rules (site) | any `MVT_ID_mvt` does not match; rows missing; extra rows |
| Deadline | **2026-10-14 10:00Z**. The site says "12:00 CET (+02:00)"; Europe is on summer time, so treat it as 10:00Z and do not rely on the extra hour |
| Target | file validated and uploaded **by the evening of 13.10** (≤ 20:00Z), leaving a full night and morning for a verified result |

The ranking is blind: no score is published for the final file before the deadline. Nothing can be learned from the
upload itself, so there is no reason to upload late or twice.

## 2. Pre-upload checks (all on the machine that holds the candidate; read-only)

1. **Validator** (hard gate, exit code 0 required):
   ```
   .venv/bin/python scripts/validate_final_submission.py \
       --file <dir>/gentle-igloo_final.parquet \
       --reference submissions/gentle-igloo_v12075.parquet \
       --json analysis/final_submission_<date>/validate.json | tee analysis/final_submission_<date>/validate.txt
   ```
   Hard checks: name, column names and order, dtypes (Int32 is *reported*, never cast), row count, ID set and
   duplicates, no null/NaN/inf, 0 ≤ value ≤ 172,800, integer seconds. Read the INFO lines as well:
   - per month × airport table — Feb/Jun medians and p95 should look like Jan/Jul, not like a constant or a shifted scale;
   - fallback-regime share (`pred == take-off − SCHED/IOBT/LOBT/EOBT/AOBT_3 ± 15 s`) — on v12075 this is ≈2.4–3.5 %
     (SCHED…EOBT) and ≈6 % (AOBT_3) per month; a Feb/Jun value far from that indicates a changed regime mixture;
   - reference block — RMS and max |diff| against v12075 on the 344,841 Jan/Jul IDs and the Gram-identity range
     `[|230.9387 − RMS|, 230.9387 + RMS]`. If the Jan/Jul rows were copied from v12075, RMS must be exactly 0.
2. **SHA-256 recorded**: the validator prints it; also run `shasum -a 256 <file>` independently and write both into
   the manifest. Any later copy (scp to the upload host) is verified against this value before the PUT.
3. **Manifest** `analysis/final_submission_<date>/MANIFEST.json`: file SHA-256, size, template SHA-256, ranking SHA-256,
   git commit of the pipeline that produced it, journal number (HYPOTHESES_LOG), builder script path, machine, UTC time.
   The v12075 reference anchor is `90d3cd0b5a68cc8008f2142f1801857752574b525b2ff046ad6bf4089a70aee2` (Mac file and the
   Hetzner ledger agree, checked 08.10).
4. **Last look** (no new model work): `polars.read_parquet(file).head()` — IDs look like `2026xxxxx.0`, values are
   plausible seconds. The 2025 edition lost eleven of thirty teams to a positional reorder that passes every format check
   (`src/upload.py` docstring); the reference RMS test is the only guard against that, so do not skip it.

## 3. Single-PUT rule and how to reuse the existing transport

The repo already has a registered one-PUT transport; the final upload must reuse its mechanism, not `src/upload.py`
(which retries silently through MinIO's default pool). The pieces:

| Piece | Where | What it gives |
|---|---|---|
| `OnePutHTTP` | `src/e185_single_put_upload.py` (same class in `src/e159_fixed_single_put_upload.py`) | `urllib3.PoolManager(retries=False)`; a second PUT in the same process raises; `connect 10 s / read 60 s / total 90 s`; `fput_object(..., num_parallel_uploads=1)` under a 90 s `SIGALRM` |
| `single_attempt` | `src/submission_guard.py` | exclusive lock on the ledger, `intent → uploaded | ambiguous` events in `data/submission_upload_ledger.jsonl`, quota from leaderboard + bucket listing + ledger |
| `require_window`, `check_quota` | `src/submission_guard.py` | hard-coded `DAY` and the 5-per-UTC-day rule |
| `load_upload_credentials`, `bucket_inventory` | `src/e185_submission.py` | keys from `.prc-upload-credentials.json` (mode 600) or `.env`; bucket listing |
| Dated wrapper | `scripts/submission_transport_20261007.py` | the pattern: pin `DAY`/`CUTOFF`, pin source SHA-256s, `inspect / seal / preflight / upload` sub-commands, no network in `inspect` |

**The frozen helpers do not accept the final name as they are.** Three places are hard-wired to
`gentle-igloo_v\d+\.parquet`: `bucket_inventory` raises `ambiguous team bucket object` for any other `.parquet` in the
bucket (so after the final file exists the old transports stop working for everyone), `read_ledger` rejects a ledger
line with another filename, and `submission_guard.ROWS = 344841` / `validate_submission` expect the old template.
Therefore the final upload needs a **new dated wrapper** `scripts/submission_transport_final_<date>.py`, built like the
07.10 one, that:

- imports `OnePutHTTP`, `single_attempt`, `exclusive_lock`, `load_upload_credentials` unchanged (pin their SHA-256s),
- defines its own `DAY`, `CUTOFF = 2026-10-14T10:00Z` minus a margin, `FINAL = 'gentle-igloo_final.parquet'`,
  its own bucket inventory that tolerates both name patterns, and a ledger record for the `_final` name,
- runs `scripts/validate_final_submission.py` (or imports `run()`) as its preflight, with exit code 0 as the gate,
- checks `0 < size ≤ 5 MiB` (single S3 PUT; the synthetic file is 1.66 MB),
- verifies the file SHA-256 from the manifest immediately before `fput_object`,
- is registered (journal number + commit) before any network action, as every transport so far.

Operational rules carried over from the 05.10 and 07.10 deliveries (`HANDOFF.md`):
canonical upload host is **Hetzner**, interpreter **`.venv-upload/bin/python`** (Python 3.14 with `minio`, `certifi`,
`urllib3`; never install `minio` into the main `.venv`), the ledger lives at `/root/prc-2026/data/submission_upload_ledger.jsonl`
(mode 600), the candidate is copied with `scp` and its SHA-256 compared on the host before the PUT. There is no `mc`
binary on the Mac; the MinIO CLI route from the site is a fallback, not the plan.

Whether the final object counts toward the 5-per-UTC-day quota is not stated anywhere. Plan the PUT on a UTC day with
at least one regular slot unused, so the question cannot bite.

## 4. Timing

| When (UTC) | Step |
|---|---|
| ≤ 12.10 | candidate built by the pipeline (Codex), validator run, manifest written, wrapper registered |
| 13.10 morning | fresh validator run on the exact bytes to be uploaded; copy to Hetzner; SHA compared |
| **13.10 ≤ 20:00Z** | the single PUT |
| 13.10 20:00Z – 14.10 09:00Z | post-upload verification (section 5); if it fails, the one permitted reconciliation |
| 14.10 10:00Z | hard deadline (site: "12:00 CET (+02:00)") — nothing is planned for 14.10 |

## 5. Post-upload verification

1. **Ledger**: exactly one `intent` and one `uploaded` line for `gentle-igloo_final.parquet` with the manifest SHA-256.
2. **Bucket listing** (read-only, from the upload host with the competition keys): `list_objects('prc-2026-gentle-igloo')`
   shows `gentle-igloo_final.parquet` with the expected size and a `last_modified` matching the ledger time.
   Download it to a scratch directory and compare SHA-256 with the manifest — the bytes in the bucket are what is scored.
3. **Result file**: for regular submissions the organiser's scorer has never written anything back into our bucket; the
   score appears only on the leaderboard API (`src/leaderboard.py`, `processedAt` within ~minutes). For the final file the
   announcement says the ranking is blind, so **no score and no result object is expected before the deadline**. If an
   object other than our own uploads appears in the bucket (e.g. `*_final*.csv`/`*.json`/an error report), do not delete it;
   download and read it. Expect the final result on the public site/Discord after 14.10 10:00Z, not in the bucket.
4. Record 1–3 in `analysis/final_submission_<date>/POST_UPLOAD.md` and a line in `HANDOFF.md`.

## 6. If the upload fails

- **Exception before `fput_object` returned** (timeout, TLS, reset): the ledger holds an `ambiguous` event. Do **not** PUT
  again blindly. List the bucket: if the object is there with the right size, download it, compare SHA-256; if it matches,
  append a manual `uploaded` reconciliation note (ledger + HANDOFF) and stop. If the object is absent or the hash differs,
  that is the one case for a second PUT — with a fresh wrapper run after the ledger has been reconciled by hand.
- **Guard stop** (`STOP: …` from `require_window`, quota, SHA mismatch): fix the cause (wrong day constant, stale copy),
  re-run the validator on the exact bytes, and repeat the same registered procedure. No code change to the frozen helpers.
- **Wrong file uploaded** (validator was skipped or the wrong candidate was copied): S3 PUT to the same name overwrites.
  Upload the correct file to the same name once, then verify by download + SHA-256. Record both events in the ledger.
  Never upload a second final name (`_final_v2`, `_final2`): the site asks for one unique final file.
- **Credentials or host unavailable**: the Mac holds `.env` with the same two keys (`src/upload.py` reads it); a Mac-side
  run of the same wrapper is acceptable if the Hetzner ledger is fetched first and written back afterwards, so the ledger
  stays single.
- Nothing above is urgent before 13.10 20:00Z; all of it is routine before 14.10 09:00Z. After 09:00Z do not start
  anything new — a half-finished overwrite at the deadline is worse than the file already in the bucket.

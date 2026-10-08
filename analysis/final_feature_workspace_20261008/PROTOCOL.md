# Fresh final feature WORK — protocol v1, 2026-10-08

Scope: one immutable feature workspace for all final months, using unchanged
feature producer code. No model fits, CV, ADS-B extraction, sensor downloads,
submission, or network transport in this helper. Full data sealing, copying and
feature construction require root's source/protocol/journal preregistration.
Only synthetic tests and small read-only metadata checks were run beforehand.

## Scientific contract

Inputs: the twelve official training files (2,085,047 DEP rows), official
`final_ranking.parquet` (670,790 DEP rows in Jan/Feb/Jun/Jul 2026), and official
`final_submitting.parquet`. ARR inputs remain available for the unchanged context
features. Final DEP TAXITIME and BLOCK must be null; template targets must be null;
IDs must be finite exact integers, unique within each population, disjoint across
train/final, and exactly cover the template. Sources use the original raw schema
and order captured in `source_contract.json`.

Fresh train+final union for E165 medians and E157 wake-type fallback is deliberate.
No old Jan/Jul feature values or predictions are copied into new final rows.
No fit occurs in this stage. Later trainers must consume the new complete feature
hash and perform their own registered training; old classifier/NN model reuse is
not authorized by this protocol.

Producer order is fixed:

1. `features.py` -> `features_final.parquet`.
2. `e28_forward_neighbours.py build` -> `features_e28.parquet`.
3. `e77_filed_twins.py` -> `features_e77.parquet`.
4. E106 airline prefix: exact coalesce of CALLSIGN's first three letters and
   `^([A-Z]{3})\d` from FLIGHT, copied from the existing any-period script.
5. `e113_prefix_lookup.py` -> separate TabPFN prefix table; does not replace E106.
6. `e163_build_features.py` -> destination daily ATFM additions.
7. `e51_add_route_features.py`, input E106 -> E106+route.
8. `e157_wake_features.py` -> E106+wake.
9. `e165_build_features.py` -> E106+plan residuals.
10. `build_final_table.py` merges E163 + E106/E51 + E157 + E165, in that order.
11. `e185_motion_features.py` produces the movement-only replacement view.

All 11 Python source files (including `geometry.py`) are copied verbatim and
hash-bound to both `source_contract.json` and the preregistration Git commit.
Only the E106 inline expression is implemented by the adapter. The model helpers
`train.py`, `target_encoding.py`, NN code and root's tabular worker are separate
owners' inputs; this workspace does not pretend to seal those files.

## Input allowlist and external admission

Exactly twelve training Parquets and two final Parquets are staged; the final
files are copied under WORK's conventional `ranking.parquet`/`submitting.parquet`
names. The original repository inputs remain untouched. Extra globs, whole
directories, credentials, `.git`, old feature values, model files and old ranking
are not included. The fixed external list has 35 files:

| Files | Required fields / admission |
|---|---|
| 10 `data/metar/<AP>.csv` | Producer consumes station, valid, tmpc, dwpc, sknt, gust, vsby, skyl1, wxcodes. Metadata check requires correct station and an observation on every date used by MVT/base-weather/motion-weather queries. |
| 10 `data/metar_wind/<AP>.csv` | Producer consumes station, valid, drct, sknt. Same station/date admission. |
| 3 `data/ext_pru/apt_dly_{2024,2025,2026}.csv` | APT_ICAO, FLT_DATE, FLT_ARR_1, DLY_APT_ARR_1, FLT_ARR_1_DLY, DLY_APT_ARR_W_1, DLY_APT_ARR_C_1, DLY_APT_ARR_G_1; every query MVT calendar date must exist globally. Destination absence remains the original `dst_known=0`/null behavior. UTF-8-lossy parsing matches the producer. |
| 10 `analysis/sources/samples/osm_overpass/<AP>.json` | Nonempty elements; unchanged geometry parser decides stand/runway matches. |
| `data/taxi_route.parquet` | Existing route snapshot, unchanged join/fallback semantics. |
| `data/ext/openflights_airlines.dat` | Existing OpenFlights snapshot consumed by E113. |

`source_contract.json` contains exact source hashes, external paths, full raw
schema and the ordered schemas of E106/E113/FIN1b/motion, read from existing local
Parquet footers. It removes any need to transfer historical large feature files
to the build host just to obtain their schemas. No historical feature values
are read by the new build.

Small pre-registration check: `external_availability.json` reports all ten
airports × 58 Feb/Jun dates × both weather CSV families present, and no missing
Feb/Jun ATFM calendar date. This is date availability, **not per-flight weather
matching quality**. Full seal additionally checks all actual 2025/final query
dates. Missing file/date/schema causes a named failure before staging. Existing
producer nulls, 3h weather tolerances and geometry unmatched rows remain unchanged.

## Actions and immutability

`seal` runs on Mac, where Git and the allowed inputs already exist. It checks
the registered worker/test/protocol/source-contract and all producer bytes,
hashes the allowlisted inputs before metadata reads, projects only seven raw
columns, checks identity/label/calendar contracts, then rehashes everything.
Only `manifest.json` is written, exclusively; no data is copied or extracted.
The manifest contains hashes, schemas and counts; no raw target values. Keep
the real manifest as a private execution artifact, not a Git data dump.

`stage` runs on the selected Linux host after separately authorized normal-tool
transport. It requires the exact trusted manifest SHA passed by root. It verifies
all local source files, then copies them into a fresh WORK with exclusive files
and read-only mode 0444. It uses neither hardlinks nor symlinks and never chmods
originals. It verifies source and copied bytes again. No overwrite/resume; a
failed partial WORK remains evidence and is not reused automatically.

`build` rechecks the copied manifest/input/source hashes and executing helper,
rejects any existing feature outputs, then runs the fixed sequential subprocess
list in WORK. Ambient modeling environment is cleared; X_FEATS=0/NO_AOBT=0,
producer Polars/OMP≤6, BLAS1, passive OMP, no bytecode writing. Each successful
producer publishes a small SHA receipt. A failure stops downstream execution.
Four final outputs must match the sealed ordered schemas, exact DEP ID sets,
train/rank splits and final null-label boundary. Training labels must also match
the original training ID/label fingerprint. All immutable inputs are rechecked
before `feature_complete.json` is created. Incomplete directories are not fit
inputs. `verify` rechecks input/output hashes and contracts without rerunning ETL.

## Resources and host selection

Root's read-only snapshot at or before 11:18 UTC: Home has ~617 GiB free and all
twelve training files; Hetz has ~8.5 GiB free, Mac ~14 GiB. These are observations,
not current launch admission. **Home is the intended WORK host.** No cleanup or
40GiB requirement is imposed on Hetz; it will later receive only the completed
FIN1b and required small model/proxy inputs through a separately authorized step.

Mac metadata seal: Polars2, RLIMIT_CPU900s, SIGALRM900s, and peak-RSS rejection
above 3GiB after each phase. The phase checks record true process high-water RSS;
they are **not a kernel memory.max limit**. Only seven raw columns, small external
CSV date/field checks, schemas and streaming hashes are processed; no model or
full feature table is materialized. If it fails this budget, preserve the failure
and stop; no automatic broader-resource retry. It needs no new supervisor.

Linux stage/verify: cgroup≤2CPU/3GiB/swap0, RuntimeMaxSec900. Home build:
**≤6CPU/24GiB/swap0**, Nice10, RuntimeMaxSec7200, KillMode=control-group,
RemainAfterExit=yes. Build starts **after E207 releases resources**; 16GiB
co-resident operation has not been qualified. The helper verifies cgroup caps;
the caller must set RuntimeMaxSec and owned-unit lifecycle as shown below.
Stage requires payload bytes +4GiB disk floor; build admission requires25GiB
free, then preserves a4GiB floor between stages. Actual build peak RAM/time has
not yet been measured. One queue, no retries and no nested scheduler.

## Commands after preregistration

Root substitutes full `REG` and a new manifest path; these commands are not yet
authorized by a journal entry for this implementation:

```sh
.venv/bin/python scripts/final_feature_workspace_20261008.py seal --root /Users/denyskolomiiets/prc-2026 --registration-commit "$REG" --manifest /Users/denyskolomiiets/prc-2026/data/final_feature_manifest_20261008.json
```

Root then uses normal approved copy tools to deliver only missing allowlisted
files plus the manifest to Home's repository, after checking transfer permission
and E207 network occupancy. The helper contains no SSH/network path. The earlier
auto-review denial for final data transfer to175 is **not** bypassed by this tool.

On Home, use separately named owned transient units and retained InvocationID:

```sh
systemd-run --user --unit=prc-final-feature-stage-20261008 --property=RemainAfterExit=yes --property=CPUQuota=200% --property=MemoryMax=3G --property=MemorySwapMax=0 --property=RuntimeMaxSec=900 --property=KillMode=control-group --property=Nice=10 --working-directory=/home/a1/prc-2026 /home/a1/prc-2026/.venv/bin/python scripts/final_feature_workspace_20261008.py stage --root /home/a1/prc-2026 --work /home/a1/prc-final-features-20261008 --manifest /home/a1/prc-2026/data/final_feature_manifest_20261008.json --manifest-sha256 "$MANIFEST_SHA"
systemd-run --user --unit=prc-final-feature-build-20261008 --property=RemainAfterExit=yes --property=CPUQuota=600% --property=MemoryMax=24G --property=MemorySwapMax=0 --property=RuntimeMaxSec=7200 --property=KillMode=control-group --property=Nice=10 --working-directory=/home/a1/prc-2026 /home/a1/prc-2026/.venv/bin/python scripts/final_feature_workspace_20261008.py build --work /home/a1/prc-final-features-20261008 --manifest /home/a1/prc-2026/data/final_feature_manifest_20261008.json --manifest-sha256 "$MANIFEST_SHA"
```

Launch build only after exact owned stage success and output receipt verification;
never treat unit presence or a stale ready file as progress. Actual progress is
FEATURE_STAGE and its completed per-stage hash receipt. Cleanup only owned unit
InvocationIDs; no broad process-kill patterns. Root owns monitoring and launch.

## Consumer API

WORK/`manifest.json`: protocol, registration_commit, files with source/path/bytes/
sha256, contract, expected_output_schemas, external_coverage, steps.

WORK/`feature_complete.json`:

- `feature_path`: `data/features_fin1b.parquet`; `feature_sha256`.
- `train_rows`=2085047, `rank_rows`=670790.
- `train_ordered_id_sha256`, `rank_ordered_id_sha256`: SHA256 of contiguous
  **little-endian float64 bytes**, in each final feature split's actual row order.
- `template_sha256`; `source_sha256` map of staged Python producers/helper/tests.
- `manifest_sha256`, `registration_commit`, `outputs` (each output SHA/schema/
  rows and split-order ID hashes where present), `fits_performed`=0.

Downstream root tabular and neural adapters must pin this completion SHA and
recheck their consumed file hashes. They independently seal their own model
source closures. Training label fingerprints are integrity checks, not published
labels; final targets never enter a fitting routine here.

`seal` requires this exact protocol identifier in the registered commit's
`HYPOTHESES_LOG.md` before reading real metadata, in addition to exact source,
test, protocol and source-contract blobs. `verify` requires exactly the four
declared output paths and rederives every completion identity field, output
record, FIN1b alias, template/source hash and zero-fit assertion from the pinned
manifest and current files. Empty or partial output inventories fail admission.

## Verification before review

Synthetic tests cover hidden final labels/BLOCK, train/final overlap, schema and
month changes, missing weather files/days and missing ATFM dates, exclusive
read-only copies without changing historical inodes, manifest traversal/
duplicates/drift, exact E106 coalesce, schema-order/ID/label integrity, full
subprocess dependency order, and immediate downstream stop after failure.
Two admission regressions cover the actual verify CLI with empty/partial output
inventories and altered aliases, plus registration rejection before metadata
when a mocked commit has all exact code blobs but no journal protocol entry.
No real producer was executed; no feature-quality/transfer result is claimed.

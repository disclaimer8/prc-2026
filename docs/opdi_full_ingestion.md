# OPDI full ingestion

Run from the repository using its existing interpreter. This only ingests external data; it does not train or read PRC labels, MVT tables, predictions or submissions.

```sh
.venv/bin/python scripts/opdi_full_ingest.py \
  --flights-dir data/opdi \
  --output-root data/opdi_full_v002 \
  --threads 2
```

The default covers all 45 official windows intersecting 2025 and January/July 2026. Monthly originals in `--flights-dir` are copied intact and hashed; missing required boundary months (December 2024, February and June 2026) are downloaded. Every locally available monthly FLIGHT list participates in the airport filter. August 2026 lies beyond this release's published July 31 horizon, even though the last archive name ends August 8. No event-time trimming is applied.

FLIGHT sources are preserved in `flights/`; EVENT and MEASUREMENT subsets are written to `events/` and `measurements/`. All native source columns are preserved. EVENT adds `opdi_event_id` and `opdi_flight_id`; MEASUREMENT adds `opdi_event_id`. Integer IDs become lossless decimal strings; existing string IDs remain opaque, including leading zeros and prefixes. Native callsigns and timestamps remain intact; downstream users should normalize callsigns and parse UTC times explicitly. IDs never pass through float or signed 64-bit casts.

Only `adep IN airports OR ades IN airports` filters flights. All their event types and all measurement types survive, including arrival-only flights. A MEASUREMENT match uses equality of its canonical event key with the retained EVENT's canonical key. Modern integer keys and opaque string EVENT keys use identity. Legacy date-prefixed MEASUREMENT keys paired with uint64 EVENT IDs use the narrowly defined compatibility conversion below. No measurement value is inferred for missing matches. Each manifest records mapping counts, matched IDs, missing measurements, missing distance/time types and duplicate event/type pairs. Conflicting values for the same event/type reject the stage.

Transfers are sequential; one original EVENT or MEASUREMENT is staged at a time, filtered in 65,536-row batches, then removed. Memory scales with the eligible FLIGHT-ID set and current window's EVENT IDs, not with the complete raw archive history. PyArrow uses at most two CPU and I/O workers. HTTP retries restart the current archive because the source server rejected Range requests during the catalog audit. The entire approximately 26 GB raw collection is never stored at once.

Repeat the same command to resume. The output root is locked against concurrent writers. Completed stages are skipped only after output size, SHA-256, schema, row count and dependency hashes verify. Every FLIGHT input and the ingestion code hash contribute to the dependency hash, so changed input/code forces re-extraction. Different window selections require separate output roots. Resume pins already recorded remote downloads: it does not re-HEAD every completed source to discover upstream revisions; choose a new root when intentionally refreshing the release.

The legacy-ID fix can reuse completed integer EVENT outputs from the exact previous ingestion code SHA `e372d1a05d1e2fb7b83c0e7089c922f47430a6ff53205dc86fe119f1cb6a5f8a`. Reuse requires unchanged full FLIGHT hashes/airport policy, the old dependency hash, native integer EVENT/flight IDs, and successful output hash/schema/row validation. The manifest records `reused_from`; altered inputs or unknown code versions cannot take this migration path. The compatibility record survives interrupted migration. MEASUREMENT stages are re-extracted.

`manifest.json` is authoritative and atomically published. Exit **0** means all requested archives were ingested; exit **2** means incomplete or invalid. An interrupted process leaves completed checkpoints usable. A nonempty Parquet alone is never considered complete, and zero selected EVENT rows fail closed for investigation. A stage's `complete` status describes archive ingestion, not continuous source coverage; monthly manifests list observed and missing `first_seen` days.

The catalog currently has a known EVENT 404 at `20251002_20251012`. Its MEASUREMENT source is still downloaded and audited for bytes/SHA/schema/types, then deleted because its IDs cannot be linked without EVENT. The stage is `blocked_missing_events`, retained/matched counts are null, and the overall dataset remains **incomplete**, with exit 2. It is never represented as a successfully extracted empty table.

To validate the small real pair without downloading it again:

```sh
.venv/bin/python scripts/opdi_full_ingest.py \
  --flights-dir /Users/denyskolomiiets/prc-2026/data/opdi \
  --output-root /private/tmp/opdi_full_sample_20261004 \
  --windows 20251231_20260110 \
  --raw-cache /private/tmp/opdi_full_catalog --threads 2
```

The optional cache is read-only and must contain original files named `flight_events_<window>.parquet` and `measurements_<window>.parquet`. Cached/local source hashes are established locally; the manifest does not claim their remote bytes were reverified.

Offline regression checks:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

A small real validation on 2026-10-04 preserved **467,537 EVENT rows**, **17,674 flight IDs**, **72,767 nonempty info rows** and **477,160 MEASUREMENT rows**. Measurements match **238,580** of the retained event IDs; **228,957** events lack measurements. Both distance and time have 238,580 rows, with no duplicate event/type pairs. Native `info` includes OSM object IDs, airport references, aeroway types and time-in-use. These are data-availability observations, not model-quality results.

An independent direction check matched every retained event back to FLIGHT: **236,429 events / 9,558 departing flights** plus **231,108 events / 8,116 arrival-only flights**, with zero unmatched IDs. Building the index from all 14 local monthly originals and processing the pair took about 20 seconds and peaked at about 1.0 GB RSS on the Mac; larger windows add more event-ID state. January 2026's original FLIGHT has no `first_seen` rows on January 2–10 and 23; the manifest exposes those source gaps.

Original sample SHA-256:

- EVENT, 40,715,808 bytes: `a1b3b8496d3f0618eaad059da9df9816f2d06b342ecb9b0960332b303e85f98d`.
- MEASUREMENT, 32,144,541 bytes: `6d68c0f796451769f8389c647651fa9470b23aba1a9e87220864ff263aef7406`.

## Legacy MEASUREMENT compatibility

The initial December 2025/January 2026 sample used integer measurement keys and did not reveal an older schema. A full run found 2025 MEASUREMENT files retaining opaque keys such as `20241201_1005022351812`, while corresponding EVENT files republished in April 2026 contain uint64 IDs. Merely accepting the strings would produce no matches.

For the observed `YYYYMMDD_<1–20 digits>` format **only when the associated EVENT id type is uint64**, the canonical measurement join key is the unsigned decimal result of **XXH64 over the original ASCII/UTF-8 string, seed 0**. The native `id` and `event_id` columns remain unchanged. Modern signed integer EVENT/MEASUREMENT keys are never hashed. An existing system `libxxhash` is used when present; a tested short-input implementation of the [XXH64 specification](https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md) provides a dependency-free fallback.

This is an **empirically verified compatibility rule**, not a claim about the documented OPDI exporter. The [official April exporter](https://github.com/euctrl-pru/OPDI/blob/8bf973d0997d3ef202e44e3ecc6426c714de0086/python/v2.0.0/05_extract_OPDI.py#L345-L347) uses Spark `xxhash64`, whose seed is 42. That seed produced zero matches in the first 200,000 legacy measurement rows; seed 0 produced 54,854. No official publisher script explaining the seed-0 rewrite was found.

Full validation of `20241226_20250105` read all **8,624,020** original MEASUREMENT rows (SHA-256 `b37eab2a859f7eaa3d835bf85d3491c7138e82ab27dcaf385f6501fbf3a76935`, 153,428,068 bytes). Against the **2,376,478** airport-filtered EVENT rows, it retained **2,348,098** measurements for **1,174,049** distinct events, with exactly one distance and one time row per matched event, no hash collisions, duplicate event/type rows or conflicting values. Unmatched raw measurements may belong to flights outside the airport filter; they are not automatically erroneous. All retained original IDs survive unchanged. The fixed output SHA-256 is `a4bd6f63a6154a7ca28c866481903391514167c3c56610831bae3036be826ace`.

The matched legacy events in this sample are altitude crossings, level start/end, take-off, landing and top-of-climb/descent. **No parking/taxiway/runway entry/exit measurements were matched**. Therefore availability of MEASUREMENT alone does not establish a measured ground-path distance signal.

An independent time check compared each `Time Passed (s)` with `EVENT.event_time − first_seen` of the same flight. **1,096,679 / 1,174,049** matched exactly. All **1,096,570** landing/take-off/level/top-of-climb/descent rows matched exactly, but only **109 / 77,479 altitude-crossing rows** did. The latter require investigation of source consistency/semantics before use; all original rows remain preserved and the compatibility manifest explicitly warns about this limitation. Exact ID joins alone do not certify physical validity. Distances in the retained sample were finite, 0–9,519.59 NM. Diagnostic files on home PC are `/tmp/opdi_legacy_debug/{hash_join_validation,fixed_filter_validation,legacy_semantic_validation}.json`.

## Scoped December flight-key repair

The full ingestion finished 43/45 windows on 2026-10-04 at 23:06 UTC: **113,140,967 EVENT rows**, **124,882,662 MEASUREMENT rows**, and **62,441,331 matched event IDs**. The 17 intact FLIGHT months contain 22,454,373 rows. October 2–12 remains a publisher EVENT 404. December 1–11 failed closed because its uint64 `EVENT.flight_id` values have zero direct or signed/unsigned-bitcast matches to the native int64 December FLIGHT IDs.

The December mismatch has a deterministic compatibility explanation. The original full track key is:

```text
SHA256(UTF8(concat_ws('', icao24, flt_id))).hexdigest() + '_' + offset + '_2025_12'
```

The full **64-character** SHA256 and untrimmed callsign are essential. The recipe is pinned to [official source commit d09e4ccb, track construction lines 223–252](https://github.com/euctrl-pru/OPDI/blob/d09e4ccb0327d3a7fdf4b40a48d7c6d61f76f98b/python/v2.0.0/02_osn_tracks_etl.py#L223-L252). An earlier April implementation used a truncated digest and cannot explain these files. For each source group, candidate offsets 0–4095 are accepted **only** when `XXH64(key, seed=42)` matches the exact native FLIGHT ID bits. The same verified preimage hashed with seed 0 provides the uint64 EVENT alias. The recipe is official source code; this relationship between the two exported hashes is **empirically verified compatibility**, not a documented publisher contract. No timestamp proximity or fuzzy callsign match participates.

An exhaustive source-only audit reconstructed **all 1,166,309 distinct December FLIGHT IDs** and matched **all 378,773 distinct flight IDs in all 8,739,132 raw December EVENT rows**, with zero unrecovered or unmapped flight IDs. There are 1,174 duplicated rows in the full December FLIGHT source; native IDs sharing a group can be deduplicated for the mapping. A conflicting group for one native ID rejects the repair. Source EVENT and FLIGHT `first_seen` timestamps can differ; neither is replaced or inferred.

The separate repair command leaves `opdi_full_ingest.py` unchanged, locks the dataset, verifies recorded source hashes and schemas, and processes only `20251201_20251211`. It checks every crosswalk row against both hashes, requires complete native-FLIGHT coverage and complete raw-EVENT flight-key coverage, then applies the same airport OR filter. All original EVENT columns, including uint64 `flight_id`, remain unchanged. Only added `opdi_flight_id` becomes the native signed FLIGHT ID encoded as a decimal string, consistent with the other windows. MEASUREMENT uses the existing checked legacy event-ID rule.

```sh
.venv/bin/python scripts/opdi_repair_december.py \
  --output-root data/opdi_full_v002 \
  --raw-cache /tmp/opdi_legacy_debug \
  --crosswalk /tmp/opdi_legacy_debug/december_flight_crosswalk_extended.parquet
```

Omit `--crosswalk` to reconstruct it independently from the verified full December FLIGHT source. The optional candidate is copied and fully revalidated; its auxiliary columns are never trusted for airport selection. No package installation is needed. An existing `libxxhash` accelerates hashing; `opdi_id_hashes.py` provides a tested arbitrary-length fallback. PyArrow uses two CPU and I/O workers, and each source is processed in 65,536-row batches. Memory scales with one month's flight registry/crosswalk and one window's event keys; full production peak RSS is to be measured on the authorized repair run.

Both output files are staged before publication. A source-manifest backup, a persistent crosswalk addressed by SHA256, the method source commit, repair/helper code hashes, original producer/input hashes, and full mapping/count audits accompany the repair. The 43 completed windows are neither reprocessed nor rewritten. The original ingestor verifies and reuses the repaired checkpoints because its code and dependency hashes are unchanged. Conflicting measurement values, altered IDs, incomplete mappings, and stale resume dependencies reject the repair. A normal exception before manifest commit rolls back published output files. A hard kill during the publication interval can leave uncommitted files; a retry fails closed and identifies the files for inspection instead of assuming success.

The scoped command exits 0 when this repair succeeds or an identical verified repair already exists. **That does not mean the whole dataset is complete:** successful repair should bring archive completion to **44/45**, while October remains explicitly incomplete. Raw cache files remain intact; any newly downloaded December source also remains available for investigation.

Recorded source-only audit artifacts (2026-10-05, before production repair):

- Raw December EVENT: 251,261,371 bytes, SHA256 `ce8fe8ec415f660bc666e9a9c31323f14c76b2f75631667458c2a0eb16b58b4a`.
- Raw December MEASUREMENT: 192,394,322 bytes, SHA256 `dbe38985fe167ce04e0d761ffbd497858ac4cec69c4b7cf221d7b64f7e826e2e`.
- Candidate crosswalk: 55,731,034 bytes, SHA256 `55a9214fd766aa795b79644c7a3c6f3f183b4850dc3a17880bb0f2512cc64165`.
- Home audit files: `/tmp/opdi_legacy_debug/december_crosswalk_extended.json` and `/tmp/opdi_legacy_debug/full_inventory_20261005.json`.

Offline tests cover the hash preimages, preservation of arrival-only flights/native fields, all 43 unaffected checkpoints, primary-ingestor reuse, source/crosswalk tampering, unmapped flight IDs, duplicate mappings, measurement conflicts, locking, publication rollback, and stale dependency rejection. These are ingestion validations; they do not establish predictive value or physical validity of measurements.

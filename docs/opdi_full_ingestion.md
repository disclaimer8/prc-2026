# OPDI full ingestion

Run from the repository using its existing interpreter. This only ingests external data; it does not train or read PRC labels, MVT tables, predictions or submissions.

```sh
.venv/bin/python scripts/opdi_full_ingest.py \
  --flights-dir data/opdi \
  --output-root data/opdi_full_v002 \
  --threads 2
```

The default covers all 45 official windows intersecting 2025 and January/July 2026. Monthly originals in `--flights-dir` are copied intact and hashed; missing required boundary months (December 2024, February and June 2026) are downloaded. Every locally available monthly FLIGHT list participates in the airport filter. August 2026 lies beyond this release's published July 31 horizon, even though the last archive name ends August 8. No event-time trimming is applied.

FLIGHT sources are preserved in `flights/`; EVENT and MEASUREMENT subsets are written to `events/` and `measurements/`. All native source columns are preserved. EVENT adds `opdi_event_id` and `opdi_flight_id`; MEASUREMENT adds `opdi_event_id`. These decimal strings support lossless joins with signed/unsigned source IDs. Native callsigns and timestamps remain intact; downstream users should normalize callsigns and parse UTC times explicitly. FLIGHT IDs are matched with `str(int(id))`, without float conversion or signed 64-bit casts.

Only `adep IN airports OR ades IN airports` filters flights. All their event types and all measurement types survive, including arrival-only flights. A MEASUREMENT match is exact equality of its `event_id` and the retained EVENT's `id`. No measurement value is inferred for missing matches. Each manifest records matched IDs, missing measurements, missing distance/time types and duplicate event/type pairs.

Transfers are sequential; one original EVENT or MEASUREMENT is staged at a time, filtered in 65,536-row batches, then removed. Memory scales with the eligible FLIGHT-ID set and current window's EVENT IDs, not with the complete raw archive history. PyArrow uses at most two CPU and I/O workers. HTTP retries restart the current archive because the source server rejected Range requests during the catalog audit. The entire approximately 26 GB raw collection is never stored at once.

Repeat the same command to resume. The output root is locked against concurrent writers. Completed stages are skipped only after output size, SHA-256, schema, row count and dependency hashes verify. Every FLIGHT input and the ingestion code hash contribute to the dependency hash, so changed input/code forces re-extraction. Different window selections require separate output roots. Resume pins already recorded remote downloads: it does not re-HEAD every completed source to discover upstream revisions; choose a new root when intentionally refreshing the release.

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

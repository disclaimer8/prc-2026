# Full OPDI ingestion implementation plan

**Goal:** Preserve every published EVENT column and associated MEASUREMENT for flights departing or arriving at the ten PRC airports, across 2025 and January/July 2026.

**Architecture:** One sequential CLI copies the full available monthly FLIGHT inputs into a separate dataset, fills required boundary months, filters one raw EVENT/MEASUREMENT archive at a time, then deletes that archive. Fixed-size PyArrow batches and sets of eligible flight IDs/current-window event IDs bound memory; atomic JSON manifests and Parquet files establish resumability.

**Tech stack:** Existing Python 3.12, PyArrow, standard library. No installation, training, labels, MVT filtering, or existing dataset mutation.

**Spec:** User-authorized ingestion scope and main-checkout `analysis/opdi_full_catalog_2026-10-04.md`, `analysis/opdi_local_inventory_2026-10-04.md`.

1. Write offline Parquet/network-substitution tests in `tests/test_opdi_full_ingest.py` for arrival-only selection, all source columns, event-to-measurement IDs, signed/unsigned IDs, out-of-window timestamps, boundary flight downloads, server gaps, corrupted resumes and atomic publication. Run them before implementation.
2. Implement `scripts/opdi_full_ingest.py`: official grid from 2022-01-01; 45 selected windows; release horizon 2026-07-31; complete FLIGHT input hashes; sequential HTTP transfers with retries; source/output schema, bytes and SHA-256; validated atomic checkpoints; explicit incomplete exit 2.
3. Run the complete local unittest suite with two threads, validate the authorized small real EVENT/MEASUREMENT pair, inspect source schemas and exact match counts, and record a reproducible CLI command and sample evidence.

Operational checks: existing nonempty or empty files never establish completion; zero selected events are suspicious and incomplete; missing FLIGHT boundary inputs block filtering; event times are never truncated to archive names; missing EVENT blocks MEASUREMENT linkage; October 2–12 remains incomplete even when it is a known 404. Source availability is not continuous temporal coverage: manifests expose observed/missing FLIGHT first_seen days.

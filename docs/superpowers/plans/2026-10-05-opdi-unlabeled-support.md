# OPDI unlabeled incremental-support gate

**Goal:** distinguish source coverage, validated same-object intervals, unique PRC matches, and sensor support below existing ADS-B coverage2; no targets, predictions, OOF or models.

**Approved design:** canonical OPDI IDs; exact flight+OSM-object+airport pairs; positive entry-to-exit duration matching both source durations; airport geometry; fixed callsign/ADEP/public-anchor matching with both-direction ambiguity rejection. Report 2025 monthly and Jan/Jul2026 separately, explicit source-gap/censor flags, and takeoff versus weaker fallback anchors.

**Architecture:** one source window/month at a time;16 temporary disk shards permit global event-ID duplication checks and global same-flight pairing without keeping all ground rows in RAM. Native FLIGHT index is sharded using canonical string IDs, including repaired December IDs. PRC/ADS-B readers project explicit allowed columns before data access; provenance hashes those projected copies. Native DOUBLE PRC IDs require finite exact integers within2^53; OPDI float IDs reject. Matching expands day buckets lazily and prunes the full candidate graph globally.

**Execution:** existing Polars/PyArrow, CPU2, root runs under3GiB cgroup after review. Worktree code/tests only; no SSH or production mutation by implementer.

- [x] TDD pair validation, duplicated IDs across windows, malformed info, geography, exact ID normalization, forbidden-column projection and source gaps.
- [x] Independent matching helper: fixed inclusive[-900,+1800]seconds, day boundaries, takeoff/fl50/first_seen breakdown, no nearest tie-breaking.
- [x] Driver: read-only OPDI/PRC/ADS-B inputs, temporary shards, stable report of source→pair→MVT→incremental-sensor stages.
- [x] Synthetic end-to-end smoke with label sentinels excluded and cross-window object pairs.
- [x] Independent review, focused/full tests, explicit-path commit; root reviews/copies/executes.

The existing full archive profile does not validate all object pairs. This gate does not estimate residual information, prediction improvement, or causal runway delay. Unknown traffic coverage is never coded as a free runway.

Verified before commit: global source conflict fix reviewed independently; complete synthetic suite passes. Root owns real-data execution;3GiB peak not inferred from synthetic tests. Source semantics update: v002≤2000ft absolute polygon events do not prove ground occupancy; altitude/context flags added and residual proposal withdrawn.

#!/usr/bin/env python3
"""Repair only OPDI v002 EVENT/MEASUREMENT window 20251201_20251211.

The original track key is reconstructed and checked against BOTH published ID
representations. No time, callsign-similarity, or probabilistic join is used.
The existing ingestion producer and all other windows remain untouched.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

import opdi_full_ingest as I
import opdi_id_hashes as H

WINDOW = "20251201_20251211"
SOURCE_COMMIT = "d09e4ccb0327d3a7fdf4b40a48d7c6d61f76f98b"
PRODUCER_SHA = "23ce263a3793953cb2c84ebda3f9624ee20160f900098f053613503edc4007ac"
MASK = (1 << 64) - 1
BATCH_SIZE = 65536
MAX_OFFSET = 4095
CROSSWALK_SCHEMA = pa.schema([("id", pa.int64()), ("event_flight_id", pa.uint64()),
                             ("offset", pa.int64()), ("legacy_key", pa.string())])


def flight_registry(path):
    """Native ID -> original SHA256 group and eligibility from full FLIGHT."""
    source = pq.ParquetFile(path)
    I.require_columns(source.schema_arrow, ("id", "icao24", "flt_id", "adep", "ades"))
    if source.schema_arrow.field("id").type != pa.int64():
        raise ValueError("December FLIGHT must have native int64 IDs")
    result = {}
    for batch in source.iter_batches(batch_size=BATCH_SIZE, columns=["id", "icao24", "flt_id", "adep", "ades"]):
        for row in batch.to_pylist():
            ident = row["id"]
            if ident is None:
                raise ValueError("Null FLIGHT ID")
            icao, callsign = row["icao24"], row["flt_id"]
            if any(x is not None and not isinstance(x, str) for x in (icao, callsign)):
                raise ValueError("FLIGHT icao24 and flt_id must be strings or null")
            # Source concat_ws('', icao24, flt_id) ignores nulls; do not strip spaces.
            group = hashlib.sha256(((icao or "") + (callsign or "")).encode("utf-8")).hexdigest()
            eligible = row["adep"] in I.AIRPORTS or row["ades"] in I.AIRPORTS
            if ident in result:
                previous_group, previous_eligible = result[ident]
                if previous_group != group:
                    raise ValueError(f"Conflicting FLIGHT groups for native ID {ident}")
                eligible |= previous_eligible
            result[ident] = (group, eligible)
    if not result:
        raise ValueError("Empty December FLIGHT source")
    return result


def build_crosswalk(flights, destination):
    """Enumerate a bounded original offset, accepting only exact seed42 hits."""
    groups = {}
    for native, (group, _) in flight_registry(flights).items():
        groups.setdefault(group, {})[native & MASK] = native
    destination = Path(destination)
    tmp = destination.with_suffix(".parquet.part")
    buffer = []
    try:
        with pq.ParquetWriter(tmp, CROSSWALK_SCHEMA, compression="zstd") as writer:
            for group, targets in groups.items():
                for offset in range(MAX_OFFSET + 1):
                    key = f"{group}_{offset}_2025_12"
                    data = key.encode("utf-8")
                    match = H.xxh64_bytes(data, 42)
                    if match not in targets:
                        continue
                    buffer.append({"id": targets.pop(match), "event_flight_id": H.xxh64_bytes(data, 0),
                                   "offset": offset, "legacy_key": key})
                    if len(buffer) >= BATCH_SIZE:
                        writer.write_table(pa.Table.from_pylist(buffer, schema=CROSSWALK_SCHEMA))
                        buffer.clear()
                    if not targets:
                        break
                if targets:
                    raise ValueError(f"Unrecovered FLIGHT IDs for group {group}: {len(targets)}; offset bound {MAX_OFFSET}")
            if buffer:
                writer.write_table(pa.Table.from_pylist(buffer, schema=CROSSWALK_SCHEMA))
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)


def validate_crosswalk(flights, crosswalk):
    """Validate all rows and full FLIGHT coverage; never trust cached metadata."""
    registry = flight_registry(flights)
    source = pq.ParquetFile(crosswalk)
    I.require_columns(source.schema_arrow, CROSSWALK_SCHEMA.names)
    for field in CROSSWALK_SCHEMA:
        if source.schema_arrow.field(field.name).type != field.type:
            raise ValueError(f"Crosswalk {field.name} must have type {field.type}")
    native_seen, aliases, eligible = set(), {}, set()
    maximum = 0
    for batch in source.iter_batches(batch_size=BATCH_SIZE, columns=CROSSWALK_SCHEMA.names):
        for row in batch.to_pylist():
            native, alias, offset, key = [row[c] for c in CROSSWALK_SCHEMA.names]
            if any(x is None for x in (native, alias, offset, key)):
                raise ValueError("Null crosswalk value")
            if native in native_seen or alias in aliases:
                raise ValueError("Duplicate or conflicting crosswalk ID")
            if native not in registry:
                raise ValueError(f"Crosswalk native ID absent from FLIGHT: {native}")
            group, keep = registry[native]
            if not 0 <= offset <= MAX_OFFSET or key != f"{group}_{offset}_2025_12":
                raise ValueError(f"Crosswalk preimage mismatch for {native}")
            data = key.encode("utf-8")
            if H.xxh64_bytes(data, 42) != native & MASK or H.xxh64_bytes(data, 0) != alias:
                raise ValueError(f"Crosswalk hash mismatch for {native}")
            canonical = str(native)
            native_seen.add(native)
            aliases[alias] = canonical
            if keep:
                eligible.add(canonical)
            maximum = max(maximum, offset)
    if len(native_seen) != len(registry):
        raise ValueError(f"Unmapped FLIGHT IDs in crosswalk: {len(registry) - len(native_seen)}")
    return aliases, eligible, {"crosswalk_rows": len(aliases), "native_flight_ids_verified": len(native_seen),
                               "eligible_native_flight_ids": len(eligible), "max_offset": maximum,
                               "duplicate_or_conflicting_aliases": 0}


def filter_events(raw, destination, aliases, eligible):
    source = pq.ParquetFile(raw)
    I.require_columns(source.schema_arrow, ("id", "flight_id", "type", "event_time", "info"))
    if source.schema_arrow.field("id").type != pa.uint64() or source.schema_arrow.field("flight_id").type != pa.uint64():
        raise ValueError("December EVENT must have uint64 id and flight_id")
    if {"opdi_event_id", "opdi_flight_id"} & set(source.schema_arrow.names):
        raise ValueError("Raw EVENT already contains reserved normalized columns")
    schema = source.schema_arrow.append(pa.field("opdi_event_id", pa.string())).append(pa.field("opdi_flight_id", pa.string()))
    source_types, kept_types, versions, days = Counter(), Counter(), Counter(), Counter()
    event_ids, source_flights, kept_flights = set(), set(), set()
    rows = info_nonempty = 0
    with pq.ParquetWriter(destination, schema, compression="zstd") as writer:
        for batch in source.iter_batches(batch_size=BATCH_SIZE):
            table = pa.Table.from_batches([batch])
            foreign = table["flight_id"].to_pylist()
            if any(ident not in aliases for ident in foreign):
                raise ValueError("Unmapped source EVENT flight_id; refuse partial crosswalk repair")
            source_flights.update(foreign)
            canonical = [aliases[ident] for ident in foreign]
            mask = [ident in eligible for ident in canonical]
            I.counter_update(source_types, table["type"])
            if "version" in table.column_names:
                I.counter_update(versions, table["version"])
            chosen = table.filter(pa.array(mask))
            chosen_native = [ident for ident, keep in zip(canonical, mask) if keep]
            chosen_ids = I.keys(chosen["id"])
            for ident in chosen_ids:
                if ident in event_ids:
                    raise ValueError(f"Duplicate retained EVENT id: {ident}")
                event_ids.add(ident)
            kept_flights.update(chosen_native)
            chosen = chosen.append_column("opdi_event_id", pa.array(chosen_ids, type=pa.string()))
            chosen = chosen.append_column("opdi_flight_id", pa.array(chosen_native, type=pa.string()))
            writer.write_table(chosen)
            rows += chosen.num_rows
            I.counter_update(kept_types, chosen["type"])
            days.update("<null>" if t is None else str(t)[:10] for t in chosen["event_time"].to_pylist())
            info_nonempty += sum(v not in (None, "", "{}", "null") for v in chosen["info"].to_pylist())
    if not rows:
        raise ValueError("Zero retained December EVENT rows")
    output = I.metadata(destination)
    if output["rows"] != rows:
        raise ValueError("Written EVENT count differs from selected rows")
    audit = {"source_rows": source.metadata.num_rows, "retained_rows": rows,
             "excluded_rows": source.metadata.num_rows - rows, "source_types": dict(source_types),
             "retained_types": dict(kept_types), "source_row_versions": dict(versions),
             "unique_event_ids": len(event_ids), "unique_flight_ids": len(kept_flights),
             "source_unique_flight_ids": len(source_flights), "unmapped_source_flight_ids": 0,
             "info_nonempty_rows": info_nonempty, "event_utc_days": dict(sorted(days.items())),
             "flight_key_mapping": "verified original preimage: uint64 XXH64(seed0) -> native int64 XXH64(seed42)",
             "timestamp_warning": "Original EVENT and FLIGHT first_seen can differ; neither timestamp was replaced"}
    return output, audit, event_ids


def check_source(path, expected, label):
    if not I.verified(path, expected):
        raise ValueError(f"{label} source bytes/hash/schema/rows differ from manifest")


def repair(root, raw_cache, *, crosswalk=None):
    root = Path(root).resolve()
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    with (root / ".ingest.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("An ingestion already owns this output root") from error
        return _repair(root, raw_cache, crosswalk)


def _repair(root, raw_cache, crosswalk):
    manifest_path = root / "manifest.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    if (manifest.get("code_sha256") != PRODUCER_SHA or I.digest(I.__file__) != PRODUCER_SHA
            or manifest.get("pipeline_version") != 2 or manifest.get("release") != "OPDI/v002"
            or manifest.get("airports") != sorted(I.AIRPORTS)):
        raise ValueError("Repair requires the verified v002 pipeline2 producer and airport selection")
    expected_input = I.object_hash({"flights": {m: s["sha256"] for m, s in manifest["flight_sources"].items()},
                                   "airports": sorted(I.AIRPORTS), "pipeline": 2, "code": PRODUCER_SHA})
    if manifest.get("input_hash") != expected_input:
        raise ValueError("Manifest FLIGHT input hash does not match recorded sources")
    stages = manifest["windows"][WINDOW]
    flights = root / "flights/flight_list_202512.parquet"
    if manifest["flight_sources"]["202512"].get("url") != f"{I.BASE}/flight_list/flight_list_202512.parquet":
        raise ValueError("Unexpected December FLIGHT source URL")
    check_source(flights, manifest["flight_sources"]["202512"], "FLIGHT")
    destinations = {kind: root / kind / f"{kind}_{WINDOW}.parquet" for kind in ("events", "measurements")}
    if any(s.get("status") == "complete" for s in stages.values()):
        for kind, destination in destinations.items():
            stage = stages[kind]
            provenance = stage.get("repair", {})
            if (stage.get("status") != "complete" or provenance.get("method_source_commit") != SOURCE_COMMIT
                    or provenance.get("algorithm_version") != 1):
                raise ValueError("Refuse to overwrite an unexpected completed December stage")
            dependency = expected_input if kind == "events" else I.object_hash(
                {"input_hash": expected_input, "event_sha256": stages["events"]["output"]["sha256"]})
            if (stage.get("dependency_hash") != dependency or provenance.get("input_hash") != expected_input
                    or provenance.get("flight_source_sha256") != manifest["flight_sources"]["202512"]["sha256"]
                    or provenance.get("source_sha256") != {k: stages[k]["source"]["sha256"] for k in destinations}):
                raise ValueError("Repaired checkpoint lineage differs from current manifest inputs")
            check_source(destination, stage.get("output", {}), f"repaired {kind}")
            check_source(Path(provenance.get("crosswalk_path", "")), provenance.get("crosswalk", {}), "crosswalk")
        return {"status": "already_repaired", "window": WINDOW, "dataset_status": manifest["status"]}
    for kind, destination in destinations.items():
        if destination.exists():
            raise ValueError(f"Uncommitted destination exists; inspect before retry: {destination}")
        expected_url = f"{I.BASE}/{'flight_events' if kind == 'events' else kind}/{'flight_events' if kind == 'events' else kind}_{WINDOW}.parquet"
        if stages[kind].get("source", {}).get("url") != expected_url:
            raise ValueError(f"Unexpected December {kind} source URL")
    ingestor = I.Ingestor(root, root / "flights", windows=[WINDOW], raw_cache=raw_cache)
    raw = {}
    for kind in destinations:
        raw[kind], _, _ = ingestor.acquire(kind, WINDOW)
        check_source(raw[kind], stages[kind]["source"], kind)
    with tempfile.TemporaryDirectory(prefix=".december-repair-", dir=root) as staging:
        staging = Path(staging)
        staged_crosswalk = staging / "crosswalk.parquet"
        if crosswalk is None:
            build_crosswalk(flights, staged_crosswalk)
        else:
            shutil.copyfile(crosswalk, staged_crosswalk)
        aliases, eligible, crosswalk_audit = validate_crosswalk(flights, staged_crosswalk)
        crosswalk_meta = I.metadata(staged_crosswalk)
        stored_crosswalk = root / "crosswalks" / f"december_flight_ids_{crosswalk_meta['sha256']}.parquet"
        event_path, measurement_path = staging / "events.parquet", staging / "measurements.parquet"
        event_output, event_audit, event_ids = filter_events(raw["events"], event_path, aliases, eligible)
        del aliases, eligible
        measurement_output, measurement_audit = ingestor.filter_source(
            "measurements", raw["measurements"], measurement_path, event_ids, pa.uint64())
        del event_ids
        if manifest_path.read_bytes() != original:
            raise ValueError("Manifest changed during repair; refuse publication")
        # Source and helper digests stay explicit; leave original producer/input hashes
        # intact so its normal resume validation reuses all 44 completed windows.
        provenance = {"method_source_commit": SOURCE_COMMIT,
                      "method_source_url": f"https://github.com/euctrl-pru/OPDI/blob/{SOURCE_COMMIT}/python/v2.0.0/02_osn_tracks_etl.py#L223-L252",
                      "mapping_status": "empirically_verified_export_compatibility",
                      "algorithm_version": 1, "repair_code_sha256": I.digest(__file__),
                      "hash_helper_sha256": I.digest(H.__file__), "original_producer_sha256": PRODUCER_SHA,
                      "input_hash": expected_input, "crosswalk_path": str(stored_crosswalk),
                      "crosswalk": crosswalk_meta, "crosswalk_audit": crosswalk_audit,
                      "source_sha256": {k: stages[k]["source"]["sha256"] for k in destinations},
                      "flight_source_sha256": manifest["flight_sources"]["202512"]["sha256"],
                      "key_recipe": "SHA256 UTF8 concat_ws('',icao24,flt_id), full 64 hex + '_' + offset + '_2025_12'; preserve spaces",
                      "native_check": "XXH64 UTF8 key seed42 bits equal signed native FLIGHT.id",
                      "alias_check": "XXH64 UTF8 key seed0 equals unsigned source EVENT.flight_id"}
        for kind, output, audit, dependency in (
                ("events", event_output, event_audit, expected_input),
                ("measurements", measurement_output, measurement_audit,
                 I.object_hash({"input_hash": expected_input, "event_sha256": event_output["sha256"]}))):
            stages[kind] = {"status": "complete", "dependency_hash": dependency,
                            "source": stages[kind]["source"], "output": output, "audit": audit,
                            "completed_at": I.now(), "repair": provenance}
        manifest["updated_at"] = I.now()
        manifest["completed_windows"] = sum(all(s.get("status") == "complete" for s in w.values())
                                            for w in manifest["windows"].values())
        manifest["status"] = "complete" if manifest["completed_windows"] == len(manifest["requested_windows"]) else "incomplete"
        backups = root / "repairs"
        backups.mkdir(exist_ok=True)
        backup = backups / f"before_december_{hashlib.sha256(original).hexdigest()}.json"
        if not backup.exists():
            backup.write_bytes(original)
        stored_crosswalk.parent.mkdir(exist_ok=True)
        if stored_crosswalk.exists():
            check_source(stored_crosswalk, crosswalk_meta, "stored crosswalk")
        else:
            os.replace(staged_crosswalk, stored_crosswalk)
        published = []
        try:
            for kind, staged in (("events", event_path), ("measurements", measurement_path)):
                os.replace(staged, destinations[kind])
                published.append(destinations[kind])
            I.atomic_json(manifest_path, manifest)
        except BaseException:
            # Until the manifest commits, outputs have no valid checkpoint.
            if manifest_path.read_bytes() == original:
                for path in published:
                    path.unlink(missing_ok=True)
            raise
    return {"status": "repaired", "window": WINDOW, "dataset_status": manifest["status"],
            "completed_windows": manifest["completed_windows"], "event_rows": event_output["rows"],
            "measurement_rows": measurement_output["rows"], "crosswalk": str(stored_crosswalk)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/opdi_full_v002"))
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, help="Optional candidate; every row is independently verified")
    args = parser.parse_args(argv)
    try:
        result = repair(args.output_root, args.raw_cache, crosswalk=args.crosswalk)
    except (OSError, ValueError, KeyError, RuntimeError, pa.ArrowException) as error:
        print(json.dumps({"status": "error", "window": WINDOW, "error": str(error)}), flush=True)
        return 2
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

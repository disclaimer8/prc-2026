#!/usr/bin/env python3
"""Resumable OPDI v002 ingestion. Never reads PRC labels or movement tables.

Full FLIGHT originals are retained. EVENT/MEASUREMENT preserve native columns
and add decimal string keys for joins across signed/unsigned Parquet schemas.
Exit 0 means requested archives were ingested; exit 2 means incomplete/error.
It does not mean that the published source covers every day or every flight.
"""
import argparse
import calendar
from collections import Counter
import ctypes
import ctypes.util
from datetime import date, datetime, timedelta, timezone
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import pyarrow as pa
import pyarrow.parquet as pq

BASE = "https://www.eurocontrol.int/performance/data/download/OPDI/v002"
AIRPORTS = frozenset("EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF LTFM LSZH".split())
EPOCH = date(2022, 1, 1)
RELEASE_END = date(2026, 8, 1)  # Exclusive published horizon, despite August window suffix.
MONTHS = [f"2025{m:02d}" for m in range(1, 13)] + ["202601", "202607"]
KNOWN_EVENT_GAPS = {"20251002_20251012"}
PIPELINE_VERSION = 2
MEASUREMENT_TYPES = ("Distance flown (NM)", "Time Passed (s)")
COMPATIBLE_INTEGER_EVENT_CODE = "e372d1a05d1e2fb7b83c0e7089c922f47430a6ff53205dc86fe119f1cb6a5f8a"
LEGACY_EVENT_ID = re.compile(r"[0-9]{8}_[0-9]{1,20}\Z")
LEGACY_MAPPING = {"algorithm": "XXH64(UTF-8 source event_id, seed=0), unsigned uint64",
                  "status": "empirically_verified_compatibility_not_official_export_contract",
                  "evidence_window": "20241226_20250105",
                  "source_measurement_sha256": "b37eab2a859f7eaa3d835bf85d3491c7138e82ab27dcaf385f6501fbf3a76935",
                  "retained_event_ids_matched": 1174049,
                  "reference_time_rows_exact": 1096679,
                  "reference_time_rows_checked": 1174049,
                  "warning": "Reference altitude-crossing Time Passed mostly disagrees with event_time minus first_seen; preserve as source data, do not assume physical validity"}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def window_dates(window):
    if not re.fullmatch(r"\d{8}_\d{8}", window):
        raise ValueError(f"Invalid window: {window}")
    start, end = [datetime.strptime(x, "%Y%m%d").date() for x in window.split("_")]
    if (start - EPOCH).days % 10 or end - start != timedelta(days=10):
        raise ValueError(f"Window is not on the official ten-day grid: {window}")
    if start < EPOCH or start >= RELEASE_END:
        raise ValueError(f"Window outside published v002 release: {window}")
    return start, end


def target_windows(months=MONTHS):
    windows = set()
    for month in months:
        start = datetime.strptime(month, "%Y%m").date()
        end = start + timedelta(days=calendar.monthrange(start.year, start.month)[1])
        cursor = EPOCH + timedelta(days=((start - EPOCH).days // 10) * 10)
        while cursor < end:
            window = f"{cursor:%Y%m%d}_{cursor + timedelta(days=10):%Y%m%d}"
            window_dates(window)
            windows.add(window)
            cursor += timedelta(days=10)
    return sorted(windows)


def required_months(windows):
    months = set()
    for window in windows:
        cursor, end = window_dates(window)
        while cursor < min(end, RELEASE_END):
            months.add(cursor.strftime("%Y%m"))
            cursor += timedelta(days=1)
    return sorted(months)


def keys(array):
    """Preserve opaque strings; integer IDs never pass through float/int64."""
    if not (pa.types.is_integer(array.type) or pa.types.is_string(array.type)
            or pa.types.is_large_string(array.type)):
        raise ValueError(f"Non-integer ID type: {array.type}")
    result = []
    for value in array.to_pylist():
        if value is None:
            raise ValueError("Null ID in source")
        if isinstance(value, str) and not value:
            raise ValueError("Empty ID in source")
        result.append(value if isinstance(value, str) else str(value))
    return result


@lru_cache(maxsize=1)
def native_xxhash64():
    """Use an existing system library when available; never install packages."""
    library = ctypes.util.find_library("xxhash")
    if not library:
        return None
    try:
        function = ctypes.CDLL(library).XXH64
        function.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulonglong]
        function.restype = ctypes.c_ulonglong
        return function
    except (OSError, AttributeError):
        return None


def legacy_xxhash64(value, *, portable=False):
    """XXH64 seed 0 for observed ASCII legacy keys (<32 bytes).

    The portable short-input algorithm follows xxHash's BSD-licensed XXH64
    specification: https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md
    This is an empirically validated publisher compatibility rule, not Spark's
    xxhash64 (which uses seed 42). Never apply it to modern signed integer IDs.
    """
    if not LEGACY_EVENT_ID.fullmatch(value):
        raise ValueError(f"Unsupported legacy event ID format: {value!r}")
    data = value.encode("ascii")
    native = None if portable else native_xxhash64()
    if native:
        return native(data, len(data), 0)
    p1, p2, p3 = 11400714785074694791, 14029467366897019727, 1609587929392839161
    p4, p5, mask = 9650029242287828579, 2870177450012600261, (1 << 64) - 1

    def rotate(number, bits):
        number &= mask
        return ((number << bits) | (number >> (64 - bits))) & mask

    h, offset = (p5 + len(data)) & mask, 0
    while offset + 8 <= len(data):
        lane = int.from_bytes(data[offset:offset + 8], "little")
        lane = (rotate(lane * p2, 31) * p1) & mask
        h = (rotate(h ^ lane, 27) * p1 + p4) & mask
        offset += 8
    if offset + 4 <= len(data):
        h ^= (int.from_bytes(data[offset:offset + 4], "little") * p1) & mask
        h = (rotate(h, 23) * p2 + p3) & mask
        offset += 4
    for byte in data[offset:]:
        h = (rotate(h ^ ((byte * p5) & mask), 11) * p1) & mask
    h = ((h ^ (h >> 33)) * p2) & mask
    h = ((h ^ (h >> 29)) * p3) & mask
    return h ^ (h >> 32)


def measurement_keys(array, event_id_type):
    result, modes = [], Counter()
    for ident in keys(array):
        if pa.types.is_uint64(event_id_type) and LEGACY_EVENT_ID.fullmatch(ident):
            result.append(str(legacy_xxhash64(ident)))
            modes["xxh64_utf8_seed0_uint64"] += 1
        else:
            result.append(ident)
            modes["identity"] += 1
    return result, modes


def metadata(path):
    parquet = pq.ParquetFile(path)
    return {"bytes": Path(path).stat().st_size, "sha256": digest(path),
            "rows": parquet.metadata.num_rows, "schema": str(parquet.schema_arrow),
            "row_groups": parquet.metadata.num_row_groups}


def verified(path, record):
    try:
        actual = metadata(path)
        return all(actual[k] == record[k] for k in ("bytes", "sha256", "rows", "schema"))
    except (OSError, ValueError, KeyError, pa.ArrowException):
        return False


def require_columns(schema, names):
    missing = set(names) - set(schema.names)
    if missing:
        raise ValueError(f"Missing required source columns: {sorted(missing)}")


def counter_update(counter, array):
    counter.update("<null>" if x is None else str(x) for x in array.to_pylist())


class Ingestor:
    def __init__(self, root, flights_dir, *, windows=None, raw_cache=None,
                 batch_size=65536, retries=3, opener=urlopen, threads=2):
        self.root = Path(root).resolve()
        self.flights_dir = Path(flights_dir).resolve()
        if self.root == self.flights_dir or self.flights_dir in self.root.parents:
            raise ValueError("Use a separate output root outside the original OPDI directory")
        self.windows = sorted(set(target_windows() if windows is None else windows))
        if not self.windows:
            raise ValueError("No windows requested")
        self.required = required_months(self.windows)
        self.raw_cache = Path(raw_cache).resolve() if raw_cache else None
        self.batch_size, self.retries, self.opener = batch_size, retries, opener
        if batch_size <= 0 or retries <= 0 or not 1 <= threads <= 2:
            raise ValueError("Require positive batch-size/retries and one or two threads")
        pa.set_cpu_count(threads)
        pa.set_io_thread_count(threads)
        self.code_sha = digest(__file__)
        self.report = {}
        self.compatible_event_dependency = None

    def save(self):
        self.report["updated_at"] = now()
        atomic_json(self.root / "manifest.json", self.report)

    def download(self, url, destination):
        """Resume at archive boundaries: this host does not support HTTP Range."""
        tmp = destination.with_suffix(destination.suffix + ".part")
        for attempt in range(self.retries):
            try:
                request = Request(url, headers={"User-Agent": "Mozilla/5.0 OPDI-research-ingest/1"})
                with self.opener(request, timeout=120) as response, tmp.open("wb") as stream:
                    headers = dict(response.headers.items())
                    expected = response.headers.get("Content-Length")
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                if expected is not None and tmp.stat().st_size != int(expected):
                    raise OSError(f"Incomplete transfer: {tmp.stat().st_size} != {expected}")
                source = metadata(tmp)  # Reject HTML and corrupt Parquet before publication.
                source.update(url=url, fetched_at=now(), origin="http", http_status=200,
                              response_headers={k: v for k, v in headers.items()
                                                if k.lower() in ("content-length", "etag", "last-modified")})
                os.replace(tmp, destination)
                return source
            except Exception as error:
                tmp.unlink(missing_ok=True)
                if isinstance(error, HTTPError) and error.code in (403, 404, 410):
                    raise
                if attempt + 1 == self.retries:
                    raise
                time.sleep(min(2 ** attempt, 8))

    def flight_inputs(self, old):
        sources = dict(old.get("flight_sources", {}))
        local = {}
        for pattern in ("flight_list_*.parquet", "flights_*.parquet"):
            for path in sorted(self.flights_dir.glob(pattern)):
                match = re.fullmatch(r"(?:flight_list|flights)_(\d{6})\.parquet", path.name)
                if match:
                    month = match[1]
                    if month in local and digest(local[month]) != digest(path):
                        raise ValueError(f"Conflicting full FLIGHT originals for {month}")
                    local[month] = path
        old_sources = old.get("flight_sources", {})
        months = sorted(set(local) | set(self.required) | set(old_sources))
        for month in months:
            destination = self.root / "flights" / f"flight_list_{month}.parquet"
            url = f"{BASE}/flight_list/flight_list_{month}.parquet"
            try:
                if month in local:
                    source = metadata(local[month])
                    source.update(url=url, origin="local_original", local_path=str(local[month]),
                                  remote_bytes_verified=False)
                    if not verified(destination, source):
                        tmp = destination.with_suffix(".parquet.part")
                        shutil.copyfile(local[month], tmp)
                        if not verified(tmp, source):
                            raise OSError(f"FLIGHT source changed while copying: {month}")
                        os.replace(tmp, destination)
                elif month in old_sources and verified(destination, old_sources[month]):
                    source = old_sources[month]
                else:
                    source = self.download(url, destination)
                require_columns(pq.ParquetFile(destination).schema_arrow, ("id", "adep", "ades", "first_seen"))
                if source["rows"] == 0:
                    raise ValueError("Empty full FLIGHT source")
                sources[month] = source
            except Exception as error:
                sources.pop(month, None)
                self.report["flight_errors"][month] = dict(url=url, error=str(error),
                                                          http_status=getattr(error, "code", None))
            self.report["flight_sources"] = sources
            self.save()
        return sources

    def flight_keys(self, sources):
        eligible = set()
        for month, source in sources.items():
            parquet = pq.ParquetFile(self.root / "flights" / f"flight_list_{month}.parquet")
            columns = [c for c in ("id", "adep", "ades", "first_seen", "version") if c in parquet.schema_arrow.names]
            days, versions = Counter(), Counter()
            kept = arrivals = 0
            for batch in parquet.iter_batches(batch_size=self.batch_size, columns=columns):
                table = pa.Table.from_batches([batch])
                ids = keys(table["id"])
                departures, destinations = table["adep"].to_pylist(), table["ades"].to_pylist()
                for ident, dep, arr in zip(ids, departures, destinations):
                    if dep in AIRPORTS or arr in AIRPORTS:
                        eligible.add(ident)
                        kept += 1
                        arrivals += dep not in AIRPORTS and arr in AIRPORTS
                days.update("<null>" if t is None else str(t)[:10] for t in table["first_seen"].to_pylist())
                if "version" in columns:
                    counter_update(versions, table["version"])
            year, m = int(month[:4]), int(month[4:])
            expected_days = [f"{year:04d}-{m:02d}-{d:02d}" for d in range(1, calendar.monthrange(year, m)[1] + 1)]
            source["audit"] = {"eligible_rows": kept, "arrival_only_rows": arrivals,
                               "first_seen_utc_days": dict(sorted(days.items())),
                               "missing_first_seen_days": [d for d in expected_days if not days[d]],
                               "row_versions": dict(versions)}
        self.report["eligible_unique_flight_ids"] = len(eligible)
        self.save()
        if not eligible:
            raise ValueError("No eligible flights in full FLIGHT sources")
        return eligible

    def acquire(self, kind, window):
        directory, prefix = ("flight_events", "flight_events") if kind == "events" else ("measurements", "measurements")
        name = f"{prefix}_{window}.parquet"
        url = f"{BASE}/{directory}/{name}"
        if self.raw_cache and (self.raw_cache / name).is_file():
            path = self.raw_cache / name
            source = metadata(path)
            source.update(url=url, origin="local_raw_cache", local_path=str(path),
                          remote_bytes_verified=False)
            return path, source, False
        path = self.root / "raw" / name
        return path, self.download(url, path), True

    def filter_source(self, kind, raw, destination, accepted, event_id_type=None):
        parquet = pq.ParquetFile(raw)
        required = ("id", "flight_id", "type", "event_time", "info") if kind == "events" else ("id", "event_id", "type", "value")
        require_columns(parquet.schema_arrow, required)
        added = ("opdi_event_id", "opdi_flight_id") if kind == "events" else ("opdi_event_id",)
        if set(added) & set(parquet.schema_arrow.names):
            raise ValueError("Source already contains reserved normalized key columns")
        schema = parquet.schema_arrow
        for column in added:
            schema = schema.append(pa.field(column, pa.string()))
        tmp = destination.with_suffix(".parquet.part")
        selected_ids, selected_flights = set(), set()
        by_type = {}
        source_types, kept_types, versions, days = Counter(), Counter(), Counter(), Counter()
        key_mapping = Counter()
        rows = duplicates = info_nonempty = 0
        try:
            with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
                for batch in parquet.iter_batches(batch_size=self.batch_size):
                    table = pa.Table.from_batches([batch])
                    foreign = keys(table["flight_id" if kind == "events" else "event_id"])
                    if kind == "measurements":
                        foreign, modes = measurement_keys(table["event_id"], event_id_type)
                        key_mapping.update(modes)
                    mask = [key in accepted for key in foreign]
                    counter_update(source_types, table["type"])
                    if "version" in table.column_names:
                        counter_update(versions, table["version"])
                    chosen = table.filter(pa.array(mask))
                    chosen_foreign = [key for key, keep in zip(foreign, mask) if keep]
                    if kind == "events":
                        event_ids = keys(chosen["id"])
                        for ident in event_ids:
                            if ident in selected_ids:
                                raise ValueError(f"Duplicate retained EVENT id: {ident}")
                            selected_ids.add(ident)
                        selected_flights.update(chosen_foreign)
                        chosen = chosen.append_column("opdi_event_id", pa.array(event_ids, type=pa.string()))
                        chosen = chosen.append_column("opdi_flight_id", pa.array(chosen_foreign, type=pa.string()))
                        days.update("<null>" if t is None else str(t)[:10] for t in chosen["event_time"].to_pylist())
                        info_nonempty += sum(x not in (None, "", "{}", "null") for x in chosen["info"].to_pylist())
                    else:
                        selected_ids.update(chosen_foreign)
                        for ident, value_type, value in zip(chosen_foreign, chosen["type"].to_pylist(), chosen["value"].to_pylist()):
                            seen = by_type.setdefault(str(value_type), {})
                            duplicates += ident in seen
                            if ident in seen and seen[ident] != value:
                                raise ValueError(f"Conflicting measurement values for EVENT {ident}, type {value_type}")
                            seen[ident] = value
                        chosen = chosen.append_column("opdi_event_id", pa.array(chosen_foreign, type=pa.string()))
                    counter_update(kept_types, chosen["type"])
                    writer.write_table(chosen)
                    rows += chosen.num_rows
            if rows == 0:
                raise ValueError(f"Zero retained {kind} rows: source/key coverage must be investigated")
            output = metadata(tmp)
            if output["rows"] != rows:
                raise ValueError("Written row count differs from selected rows")
            os.replace(tmp, destination)
        finally:
            tmp.unlink(missing_ok=True)
        audit = {"source_rows": parquet.metadata.num_rows, "retained_rows": rows,
                 "excluded_rows": parquet.metadata.num_rows - rows,
                 "source_types": dict(source_types), "retained_types": dict(kept_types),
                 "source_row_versions": dict(versions)}
        if kind == "events":
            audit.update(unique_event_ids=len(selected_ids), unique_flight_ids=len(selected_flights),
                         info_nonempty_rows=info_nonempty, event_utc_days=dict(sorted(days.items())))
        else:
            audit.update(matched_event_ids=len(selected_ids), retained_event_ids=len(accepted),
                         events_without_measurements=len(accepted - selected_ids),
                         duplicate_event_type_rows=duplicates,
                         conflicting_event_type_values=0,
                         join_key_mapping_rows=dict(key_mapping),
                         legacy_mapping=LEGACY_MAPPING if key_mapping["xxh64_utf8_seed0_uint64"] else None,
                         matched_ids_by_type={k: len(v) for k, v in by_type.items()},
                         events_without_type={k: len(accepted.difference(by_type.get(k, {}))) for k in MEASUREMENT_TYPES})
        return output, audit

    def process(self, kind, window, accepted, dependency_hash, previous, event_id_type=None):
        destination = self.root / kind / f"{kind}_{window}.parquet"
        if (kind == "events" and self.compatible_event_dependency is not None
                and previous.get("dependency_hash") == self.compatible_event_dependency
                and previous.get("status") == "complete" and previous.get("output", {}).get("rows", 0) > 0
                and verified(destination, previous.get("output", {}))):
            schema = pq.ParquetFile(destination).schema_arrow
            if pa.types.is_integer(schema.field("id").type) and pa.types.is_integer(schema.field("flight_id").type):
                return {**previous, "dependency_hash": dependency_hash,
                        "reused_from": {"code_sha256": COMPATIBLE_INTEGER_EVENT_CODE,
                                        "dependency_hash": self.compatible_event_dependency,
                                        "reason": "Numeric EVENT extraction unchanged; verified output and FLIGHT inputs"}}
        if (previous.get("status") == "complete" and previous.get("dependency_hash") == dependency_hash
                and previous.get("output", {}).get("rows", 0) > 0
                and verified(destination, previous.get("output", {}))):
            return previous
        source = None
        owned = False
        raw = None
        try:
            raw, source, owned = self.acquire(kind, window)
            output, audit = self.filter_source(kind, raw, destination, accepted, event_id_type)
            return {"status": "complete", "dependency_hash": dependency_hash,
                    "source": source, "output": output, "audit": audit, "completed_at": now()}
        except Exception as error:
            result = {"status": "missing" if isinstance(error, HTTPError) and error.code == 404 else "error",
                      "error": str(error), "http_status": getattr(error, "code", None),
                      "expected_missing": kind == "events" and window in KNOWN_EVENT_GAPS
                                          and isinstance(error, HTTPError) and error.code == 404,
                      "source": source, "dependency_hash": dependency_hash}
            prefix = "flight_events" if kind == "events" else "measurements"
            result["url"] = f"{BASE}/{prefix}/{prefix}_{window}.parquet"
            return result
        finally:
            if owned and raw:
                raw.unlink(missing_ok=True)

    def blocked_measurements(self, window, dependency_hash, previous, reason):
        """Audit the 45th source too, but never fabricate an empty linked table."""
        if (previous.get("status") == "blocked_missing_events"
                and previous.get("dependency_hash") == dependency_hash
                and previous.get("source_scan_complete")):
            return previous
        result = {"status": "blocked_missing_events", "reason": reason,
                  "url": f"{BASE}/measurements/measurements_{window}.parquet",
                  "dependency_hash": dependency_hash, "source_scan_complete": False}
        raw, owned = None, False
        try:
            raw, source, owned = self.acquire("measurements", window)
            result["source"] = source
            parquet = pq.ParquetFile(raw)
            require_columns(parquet.schema_arrow, ("id", "event_id", "type", "value"))
            counts, versions = Counter(), Counter()
            columns = [c for c in ("event_id", "type", "version") if c in parquet.schema_arrow.names]
            for batch in parquet.iter_batches(batch_size=self.batch_size, columns=columns):
                table = pa.Table.from_batches([batch])
                keys(table["event_id"])
                counter_update(counts, table["type"])
                if "version" in columns:
                    counter_update(versions, table["version"])
            result.update(source_scan_complete=True,
                          audit={"source_rows": source["rows"], "retained_rows": None,
                                 "source_types": dict(counts), "source_row_versions": dict(versions),
                                 "matched_event_ids": None})
        except Exception as error:
            result.update(error=str(error), http_status=getattr(error, "code", None))
        finally:
            if owned and raw:
                raw.unlink(missing_ok=True)
        return result

    def load_event_keys(self, window):
        result = set()
        path = self.root / "events" / f"events_{window}.parquet"
        for batch in pq.ParquetFile(path).iter_batches(batch_size=self.batch_size, columns=["opdi_event_id"]):
            result.update(batch.column(0).to_pylist())
        return result

    def run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".ingest.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Another ingestion owns {self.root}") from error
            try:
                return self._run()
            except (Exception, KeyboardInterrupt) as error:
                if self.report:
                    self.report.update(status="incomplete", error=type(error).__name__ + ": " + str(error))
                    self.save()
                raise

    def _run(self):
        self.compatible_event_dependency = None
        for folder in ("flights", "events", "measurements", "raw"):
            (self.root / folder).mkdir(exist_ok=True)
        manifest = self.root / "manifest.json"
        old = json.loads(manifest.read_text()) if manifest.exists() else {}
        if old and old.get("requested_windows") != self.windows:
            raise ValueError("Output root belongs to a different window selection; use a separate root")
        compatibility = old.get("event_checkpoint_compatibility")
        if old.get("code_sha256") == COMPATIBLE_INTEGER_EVENT_CODE and old.get("pipeline_version") == 1:
            compatibility = {"code_sha256": COMPATIBLE_INTEGER_EVENT_CODE, "pipeline_version": 1,
                             "input_hash": old.get("input_hash")}
        self.report = {"status": "running", "created_at": old.get("created_at", now()),
                       "release": "OPDI/v002", "release_end_exclusive": RELEASE_END.isoformat(),
                       "requested_months": MONTHS, "requested_windows": self.windows,
                       "required_flight_months": self.required, "airports": sorted(AIRPORTS),
                       "filter": "FLIGHT.adep in airports OR FLIGHT.ades in airports; no event-time/type/MVT/label filter",
                       "pipeline_version": PIPELINE_VERSION, "code_sha256": self.code_sha,
                       "event_checkpoint_compatibility": compatibility,
                       "flight_sources": old.get("flight_sources", {}), "flight_errors": {},
                       "windows": old.get("windows", {}),
                       "completion_meaning": "Available archive ingestion, not continuous source-day coverage"}
        self.save()
        sources = self.flight_inputs(old)
        if self.report["flight_errors"] or not set(self.required) <= set(sources):
            self.report["status"] = "incomplete"
            self.save()
            return self.report
        eligible = self.flight_keys(sources)
        input_hash = object_hash({"flights": {m: s["sha256"] for m, s in sources.items()},
                                  "airports": sorted(AIRPORTS), "pipeline": PIPELINE_VERSION,
                                  "code": self.code_sha})
        if (compatibility and compatibility.get("code_sha256") == COMPATIBLE_INTEGER_EVENT_CODE
                and compatibility.get("pipeline_version") == 1):
            old_dependency = object_hash({"flights": {m: s["sha256"] for m, s in sources.items()},
                                          "airports": sorted(AIRPORTS), "pipeline": 1,
                                          "code": COMPATIBLE_INTEGER_EVENT_CODE})
            integer_flight_sources = all(
                pa.types.is_integer(pq.ParquetFile(self.root / "flights" / f"flight_list_{month}.parquet").schema_arrow.field("id").type)
                for month in sources)
            if compatibility.get("input_hash") == old_dependency and integer_flight_sources:
                self.compatible_event_dependency = old_dependency
        self.report["input_hash"] = input_hash
        for index, window in enumerate(self.windows, 1):
            print(f"[{index}/{len(self.windows)}] {window}", flush=True)
            previous = old.get("windows", {}).get(window, {})
            event = self.process("events", window, eligible, input_hash, previous.get("events", {}))
            self.report["windows"][window] = {**previous, "events": event}
            self.save()  # Commit EVENT before downloading MEASUREMENT.
            if event["status"] != "complete":
                measurement = self.blocked_measurements(window, input_hash,
                                                       previous.get("measurements", {}), event["status"])
            else:
                event_ids = self.load_event_keys(window)
                dependencies = object_hash({"input_hash": input_hash, "event_sha256": event["output"]["sha256"]})
                measurement = self.process("measurements", window, event_ids, dependencies,
                                           previous.get("measurements", {}),
                                           pq.ParquetFile(self.root / "events" / f"events_{window}.parquet").schema_arrow.field("id").type)
                del event_ids
            self.report["windows"][window]["measurements"] = measurement
            self.save()
            print(f"  events={event['status']} measurements={measurement['status']}", flush=True)
        complete = all(stage["status"] == "complete" for w in self.report["windows"].values() for stage in w.values())
        self.report["status"] = "complete" if complete else "incomplete"
        self.report["completed_windows"] = sum(all(s["status"] == "complete" for s in w.values())
                                               for w in self.report["windows"].values())
        self.save()
        return self.report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flights-dir", type=Path, default=Path("data/opdi"))
    parser.add_argument("--output-root", type=Path, default=Path("data/opdi_full_v002"))
    parser.add_argument("--windows", help="Comma-separated official windows; default all 45 target windows")
    parser.add_argument("--raw-cache", type=Path, help="Read-only original EVENT/MEASUREMENT cache")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--list-windows", action="store_true")
    args = parser.parse_args(argv)
    if args.list_windows:
        print("\n".join(target_windows()))
        return 0
    try:
        result = Ingestor(args.output_root, args.flights_dir,
                          windows=args.windows.split(",") if args.windows else None,
                          raw_cache=args.raw_cache, batch_size=args.batch_size,
                          retries=args.retries, threads=args.threads).run()
        print(json.dumps({"status": result["status"], "completed_windows": result.get("completed_windows", 0),
                          "requested_windows": len(result["requested_windows"]),
                          "manifest": str(args.output_root / "manifest.json")}))
        return 0 if result["status"] == "complete" else 2
    except Exception as error:
        print(f"Ingestion failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

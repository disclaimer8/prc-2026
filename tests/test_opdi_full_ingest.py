"""Network is substituted; all Parquet filtering, joins and manifests are real."""
import importlib.util
import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from urllib.error import HTTPError

import pyarrow as pa
import pyarrow.parquet as pq

SPEC = importlib.util.spec_from_file_location(
    "opdi_full", Path(__file__).parents[1] / "scripts/opdi_full_ingest.py")
M = importlib.util.module_from_spec(SPEC)
if Path(SPEC.origin).exists():
    SPEC.loader.exec_module(M)

W = "20251231_20260110"


def parquet_bytes(rows, schema=None):
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), sink)
    return sink.getvalue().to_pybytes()


class Response(io.BytesIO):
    def __init__(self, body):
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body)), "ETag": '"fixture"'}


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.flights = self.base / "input"
        self.flights.mkdir()
        self.root = self.base / "output"
        self.payloads = {}
        self.flight("202512", [(1, "EDDF", "KJFK", "2025-12-31"),
                               (2, None, "LSZH", "2025-12-31"),
                               (3, "KJFK", "KLAX", "2025-12-31")])
        self.flight("202601", [(4, "EHAM", "EDDF", "2026-01-01")])
        self.events = [dict(id=i * 10, flight_id=i, type="landing" if i == 2 else "exit-taxiway",
                            event_time="2026-01-11 00:01:00", info='{"aerodrome":"kept"}',
                            longitude=8.0, latitude=47.0, altitude=0.0,
                            source="OSN", version="v0.0.2", extra="future column")
                       for i in (1, 2, 3, 4)]
        self.measurements = [dict(id=i * 100 + j, event_id=i * 10, type=t,
                                  value=3.5, version="v0.0.2")
                             for i in (1, 2, 3, 4) for j, t in enumerate(
                                 ("Distance flown (NM)", "Time Passed (s)"))]
        self.set_sources(W)

    def flight(self, month, values, remote=False):
        body = parquet_bytes([dict(id=i, adep=a, ades=b, icao24="abc123", flt_id=" TEST  ",
                                   first_seen=t, last_seen=t, version="v0.0.2", extra="raw")
                              for i, a, b, t in values])
        name = f"flight_list_{month}.parquet"
        if remote:
            self.payloads[name] = body
        else:
            (self.flights / name).write_bytes(body)

    def set_sources(self, window):
        self.payloads[f"flight_events_{window}.parquet"] = parquet_bytes(self.events)
        self.payloads[f"measurements_{window}.parquet"] = parquet_bytes(self.measurements)

    def opener(self, request, timeout=None):
        name = request.full_url.rsplit("/", 1)[-1]
        if name not in self.payloads:
            raise HTTPError(request.full_url, 404, "fixture gap", {}, None)
        return Response(self.payloads[name])

    def run_ingest(self, windows=(W,), **kwargs):
        return M.Ingestor(self.root, self.flights, windows=windows, opener=self.opener,
                          batch_size=2, retries=1, **kwargs).run()

    def read(self, kind, window=W):
        return pq.read_table(self.root / kind / f"{kind}_{window}.parquet")

    def test_departures_arrival_only_source_columns_and_both_measurements_survive(self):
        report = self.run_ingest()
        self.assertEqual(report["status"], "complete")
        events = self.read("events").to_pylist()
        self.assertEqual([r["id"] for r in events], [10, 20, 40])
        self.assertEqual(events[1]["info"], '{"aerodrome":"kept"}')
        self.assertEqual(events[1]["event_time"], "2026-01-11 00:01:00")
        self.assertEqual(events[1]["extra"], "future column")
        self.assertEqual(events[1]["opdi_flight_id"], "2")
        measurements = self.read("measurements").to_pylist()
        self.assertEqual([r["event_id"] for r in measurements], [10, 10, 20, 20, 40, 40])
        self.assertEqual({r["type"] for r in measurements}, {"Distance flown (NM)", "Time Passed (s)"})
        stage = report["windows"][W]["measurements"]
        self.assertEqual(stage["audit"]["matched_event_ids"], 3)
        self.assertEqual(stage["audit"]["events_without_measurements"], 0)
        self.assertEqual(pq.read_table(self.root / "flights/flight_list_202512.parquet").num_rows, 3)
        self.assertFalse(list((self.root / "raw").glob("*.parquet")))

    def test_zero_filtered_file_is_incomplete_not_success(self):
        self.events = [dict(self.events[0], flight_id=999)]
        self.set_sources(W)
        result = self.run_ingest()
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["windows"][W]["events"]["status"], "error")
        self.assertFalse((self.root / "events" / f"events_{W}.parquet").exists())

    def test_resume_verifies_hash_and_rebuilds_corrupt_output(self):
        self.run_ingest()
        path = self.root / "events" / f"events_{W}.parquet"
        path.write_bytes(b"corrupt existing nonempty file")
        report = self.run_ingest()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(self.read("events").column("id").to_pylist(), [10, 20, 40])
        self.payloads = {}
        self.assertEqual(self.run_ingest()["status"], "complete", "verified outputs resume offline")

    def test_input_hash_change_invalidates_previous_filter(self):
        self.run_ingest()
        self.flight("202512", [(1, "EDDF", "KJFK", "2025-12-31"),
                               (2, None, "KLAX", "2025-12-31")])
        report = self.run_ingest()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(self.read("events").column("id").to_pylist(), [10, 40])
        self.assertEqual(self.read("measurements").num_rows, 4)

    def test_missing_event_known_404_is_explicit_and_measurements_blocked(self):
        window = "20251002_20251012"
        self.flight("202510", [(1, "EDDF", "KJFK", "2025-10-03")])
        self.payloads[f"measurements_{window}.parquet"] = parquet_bytes(self.measurements)
        report = self.run_ingest(windows=[window])
        self.assertEqual(report["status"], "incomplete")
        stage = report["windows"][window]
        self.assertEqual(stage["events"]["status"], "missing")
        self.assertTrue(stage["events"]["expected_missing"])
        self.assertEqual(stage["events"]["http_status"], 404)
        self.assertEqual(stage["measurements"]["status"], "blocked_missing_events")
        self.assertEqual(stage["measurements"]["source"]["rows"], 8)
        self.assertEqual(stage["measurements"]["audit"]["retained_rows"], None)

    def test_interrupt_after_event_checkpoint_resumes_without_downloading_events(self):
        original = M.atomic_json

        def interrupt_after_event(path, report):
            original(path, report)
            if report.get("windows", {}).get(W, {}).get("events", {}).get("status") == "complete":
                raise KeyboardInterrupt()

        with patch.object(M, "atomic_json", side_effect=interrupt_after_event):
            with self.assertRaises(KeyboardInterrupt):
                self.run_ingest()
        self.payloads.pop(f"flight_events_{W}.parquet")
        report = self.run_ingest()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(self.read("measurements").num_rows, 6)

    def test_early_restart_interrupt_does_not_erase_completed_window_checkpoints(self):
        self.run_ingest()
        with patch.object(M.Ingestor, "flight_keys", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_ingest()
        self.payloads = {}
        self.assertEqual(self.run_ingest()["status"], "complete")

    def test_restart_interrupt_preserves_remote_boundary_input_checkpoint(self):
        window = "20260629_20260709"
        self.flight("202607", [(4, "EHAM", "EDDF", "2026-07-01")])
        self.flight("202606", [(2, "KJFK", "LSZH", "2026-06-30")], remote=True)
        self.set_sources(window)
        self.assertEqual(self.run_ingest(windows=[window])["status"], "complete")
        with patch.object(M.Ingestor, "flight_inputs", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_ingest(windows=[window])
        self.payloads = {}
        self.assertEqual(self.run_ingest(windows=[window])["status"], "complete")

    def test_restart_interrupt_after_verified_event_preserves_measurement_checkpoint(self):
        self.run_ingest()
        original = M.Ingestor.save

        def interrupt(injest):
            original(injest)
            if injest.report.get("windows", {}).get(W, {}).get("events", {}).get("status") == "complete" and injest.report.get("input_hash"):
                raise KeyboardInterrupt()

        with patch.object(M.Ingestor, "save", interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_ingest()
        self.payloads = {}
        self.assertEqual(self.run_ingest()["status"], "complete")

    def test_missing_boundary_month_is_downloaded_before_event_filtering(self):
        window = "20260629_20260709"
        self.flight("202607", [(4, "EHAM", "EDDF", "2026-07-01")])
        self.flight("202606", [(2, "KJFK", "LSZH", "2026-06-30")], remote=True)
        self.set_sources(window)
        report = self.run_ingest(windows=[window])
        self.assertEqual(report["status"], "complete")
        self.assertIn("202606", report["flight_sources"])
        self.assertIn(20, self.read("events", window).column("id").to_pylist())

    def test_missing_boundary_month_prevents_false_complete(self):
        report = self.run_ingest(windows=["20260629_20260709"])
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("202606", report["flight_errors"])

    def test_failed_download_never_publishes_partial_parquet(self):
        self.payloads[f"flight_events_{W}.parquet"] = b"not a parquet"
        report = self.run_ingest()
        self.assertEqual(report["status"], "incomplete")
        self.assertFalse((self.root / "events" / f"events_{W}.parquet").exists())

    def test_truncated_http_body_is_rejected_before_any_output(self):
        normal = self.opener

        def truncated(request, timeout=None):
            response = normal(request, timeout)
            response.headers["Content-Length"] = str(len(response.getvalue()) + 1)
            return response

        self.opener = truncated
        report = self.run_ingest()
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("Incomplete transfer", report["windows"][W]["events"]["error"])
        self.assertFalse(list((self.root / "raw").glob("*.part")))

    def test_float_id_source_and_duplicate_event_id_fail_closed(self):
        for invalid in ([dict(self.events[0], flight_id=1.0)], [self.events[0], self.events[0]]):
            self.events = invalid
            self.set_sources(W)
            report = self.run_ingest()
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["windows"][W]["events"]["status"], "error")
            self.assertFalse((self.root / "events" / f"events_{W}.parquet").exists())

    def test_atomic_rename_failure_never_publishes_events(self):
        original = M.os.replace

        def fail_events(source, target):
            if Path(target).parent.name == "events":
                raise OSError("simulated filesystem failure")
            return original(source, target)

        with patch.object(M.os, "replace", side_effect=fail_events):
            report = self.run_ingest()
        self.assertEqual(report["status"], "incomplete")
        self.assertFalse((self.root / "events" / f"events_{W}.parquet").exists())
        self.assertFalse(list((self.root / "events").glob("*.part")))

    def test_cli_exit_distinguishes_complete_and_incomplete_dataset(self):
        cache = self.base / "cache"
        cache.mkdir()
        for name, body in self.payloads.items():
            (cache / name).write_bytes(body)
        args = ["--flights-dir", str(self.flights), "--windows", W, "--raw-cache", str(cache)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(M.main(args + ["--output-root", str(self.root)]), 0)
        (cache / f"flight_events_{W}.parquet").write_bytes(b"bad")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(M.main(args + ["--output-root", str(self.base / "other")]), 2)

    def test_large_unsigned_and_negative_ids_join_without_float_conversion(self):
        big = 18446744073709551601
        body = parquet_bytes([dict(id=big, adep="EDDF", ades="KJFK", first_seen="2025-12-31",
                                   version="v2.0.0")], pa.schema([("id", pa.uint64()), ("adep", pa.string()),
                                   ("ades", pa.string()), ("first_seen", pa.string()), ("version", pa.string())]))
        (self.flights / "flight_list_202512.parquet").write_bytes(body)
        self.events = [dict(self.events[0], id=big, flight_id=big)]
        event_schema = pa.Table.from_pylist([dict(self.events[0], id=1, flight_id=1)]).schema
        event_schema = event_schema.set(0, pa.field("id", pa.uint64())).set(1, pa.field("flight_id", pa.uint64()))
        self.payloads[f"flight_events_{W}.parquet"] = parquet_bytes(self.events, event_schema)
        rows = [dict(self.measurements[0], id=-17, event_id=big)]
        schema = pa.schema([("id", pa.int64()), ("event_id", pa.uint64()), ("type", pa.string()),
                            ("value", pa.float64()), ("version", pa.string())])
        self.payloads[f"measurements_{W}.parquet"] = parquet_bytes(rows, schema)
        report = self.run_ingest()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(self.read("events")["opdi_event_id"].to_pylist(), [str(big)])
        self.assertEqual(self.read("measurements")["opdi_event_id"].to_pylist(), [str(big)])
        self.assertEqual(self.read("measurements")["id"].to_pylist(), [-17])

    def legacy_pair(self):
        old = "20241201_1005022351812"
        event = dict(self.events[0], id=10842447142759660653)
        schema = pa.Table.from_pylist([self.events[0]]).schema.set(0, pa.field("id", pa.uint64()))
        self.payloads[f"flight_events_{W}.parquet"] = parquet_bytes([event], schema)
        rows = [dict(id=f"measurement_{n}", event_id=old, type=typ, value=3.5, version="v0.0.2")
                for n, typ in enumerate(("Distance flown (NM)", "Time Passed (s)"))]
        self.payloads[f"measurements_{W}.parquet"] = parquet_bytes(rows)
        return rows

    def test_legacy_measurement_strings_use_verified_seed_zero_and_preserve_source_ids(self):
        self.legacy_pair()
        report = self.run_ingest()
        self.assertEqual(report["status"], "complete")
        rows = self.read("measurements").to_pylist()
        self.assertEqual(rows[0]["event_id"], "20241201_1005022351812")
        self.assertEqual(rows[0]["opdi_event_id"], "10842447142759660653")
        self.assertEqual(rows[0]["id"], "measurement_0")
        audit = report["windows"][W]["measurements"]["audit"]
        self.assertEqual(audit["join_key_mapping_rows"], {"xxh64_utf8_seed0_uint64": 2})
        self.assertEqual(audit["matched_event_ids"], 1)

    def test_opaque_event_keys_are_identity_when_event_source_is_string(self):
        self.events = [dict(self.events[0], id="opaque_event_001")]
        self.measurements = [dict(self.measurements[0], event_id="opaque_event_001")]
        self.set_sources(W)
        report = self.run_ingest()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(self.read("events")["opdi_event_id"].to_pylist(), ["opaque_event_001"])
        self.assertEqual(self.read("measurements")["opdi_event_id"].to_pylist(), ["opaque_event_001"])

    def test_legacy_and_numeric_aliases_detect_conflicting_measurement_values(self):
        rows = self.legacy_pair()
        rows.append(dict(rows[0], id="conflict", event_id="10842447142759660653", value=999.0))
        self.payloads[f"measurements_{W}.parquet"] = parquet_bytes(rows)
        report = self.run_ingest()
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("Conflicting measurement", report["windows"][W]["measurements"]["error"])

    def test_known_previous_integer_event_checkpoint_reused_after_measurement_fix(self):
        old_sha = "e372d1a05d1e2fb7b83c0e7089c922f47430a6ff53205dc86fe119f1cb6a5f8a"
        ingestor = M.Ingestor(self.root, self.flights, windows=[W], opener=self.opener, retries=1)
        ingestor.code_sha = old_sha
        with patch.object(M, "PIPELINE_VERSION", 1):
            self.assertEqual(ingestor.run()["status"], "complete")
        self.payloads.pop(f"flight_events_{W}.parquet")
        report = self.run_ingest()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["windows"][W]["events"]["reused_from"]["code_sha256"], old_sha)

    def test_changed_flight_inputs_prevent_previous_event_checkpoint_migration(self):
        old_sha = "e372d1a05d1e2fb7b83c0e7089c922f47430a6ff53205dc86fe119f1cb6a5f8a"
        ingestor = M.Ingestor(self.root, self.flights, windows=[W], opener=self.opener, retries=1)
        ingestor.code_sha = old_sha
        with patch.object(M, "PIPELINE_VERSION", 1):
            self.assertEqual(ingestor.run()["status"], "complete")
        self.flight("202601", [(4, "KJFK", "KLAX", "2026-01-01")])
        self.payloads.pop(f"flight_events_{W}.parquet")
        self.assertEqual(self.run_ingest()["status"], "incomplete")

    def test_event_compatibility_migration_survives_interruption_before_first_window(self):
        old_sha = "e372d1a05d1e2fb7b83c0e7089c922f47430a6ff53205dc86fe119f1cb6a5f8a"
        ingestor = M.Ingestor(self.root, self.flights, windows=[W], opener=self.opener, retries=1)
        ingestor.code_sha = old_sha
        with patch.object(M, "PIPELINE_VERSION", 1):
            self.assertEqual(ingestor.run()["status"], "complete")
        with patch.object(M.Ingestor, "flight_inputs", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_ingest()
        self.payloads.pop(f"flight_events_{W}.parquet")
        self.assertEqual(self.run_ingest()["status"], "complete")


class WindowTests(unittest.TestCase):
    def test_string_ids_are_opaque_and_invalid_id_types_reject(self):
        values = ["20241201_1005022349580", "000123", "+123", "123"]
        self.assertEqual(M.keys(pa.array(values)), values)
        for array in (pa.array([None], type=pa.string()), pa.array([1.5]), pa.array([True])):
            with self.assertRaises(ValueError):
                M.keys(array)

    def test_portable_legacy_hash_matches_independent_xxhash_vectors(self):
        self.assertEqual(M.legacy_xxhash64("20241201_1005022351812", portable=True), 10842447142759660653)
        self.assertEqual(M.legacy_xxhash64("20241201_1005022354001", portable=True), 2581909214055292463)
        self.assertEqual(M.legacy_xxhash64("20241201_1005022354889", portable=True), 5289981801981732348)

    def test_grid_bounds_and_release_horizon(self):
        windows = M.target_windows()
        self.assertEqual(len(windows), 45)
        self.assertEqual(windows[0], "20241226_20250105")
        self.assertIn("20260130_20260209", windows)
        self.assertIn("20251002_20251012", windows)
        self.assertEqual(windows[-1], "20260729_20260808")
        self.assertEqual(M.required_months([windows[-1]]), ["202607"])
        self.assertIn("202412", M.required_months(windows))
        self.assertIn("202602", M.required_months(windows))
        self.assertIn("202606", M.required_months(windows))
        with self.assertRaises(ValueError):
            M.required_months(["20260101_20260111"])


if __name__ == "__main__":
    unittest.main()

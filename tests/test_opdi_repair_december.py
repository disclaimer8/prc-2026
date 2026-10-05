import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import opdi_full_ingest as I

SPEC = importlib.util.spec_from_file_location("december_repair", SCRIPTS / "opdi_repair_december.py")
R = importlib.util.module_from_spec(SPEC)
if Path(SPEC.origin).exists():
    SPEC.loader.exec_module(R)

W = "20251201_20251211"
GROUP = "39c17d7746133c5da0f6d727d149f81f819970f731f9c93451814290397394d4"
NATIVE = -7024018821582472598
ALIAS = 16574437519372709528


class DecemberRepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root, self.cache = self.base / "dataset", self.base / "cache"
        for folder in (self.root / "flights", self.root / "events", self.root / "measurements", self.root / "raw", self.cache):
            folder.mkdir(parents=True, exist_ok=True)
        self.flight = self.root / "flights/flight_list_202512.parquet"
        pq.write_table(pa.Table.from_pylist([
            dict(id=NATIVE, icao24="4ba8d4", flt_id="TKJ8GW  ", adep=None, ades="EDDF",
                 first_seen="2025-12-01 00:00:00", last_seen="2025-12-01 00:32:20")]), self.flight)
        self.raw_event = self.cache / f"flight_events_{W}.parquet"
        schema = pa.schema([("id", pa.uint64()), ("flight_id", pa.uint64()), ("type", pa.string()),
                            ("event_time", pa.string()), ("info", pa.string()), ("version", pa.string()),
                            ("future_column", pa.string())])
        pq.write_table(pa.Table.from_pylist([
            dict(id=10842447142759660653, flight_id=ALIAS, type="landing", event_time="2025-12-01 00:20:00",
                 info='{"osm_airport":"EDDF"}', version="v0.0.2", future_column="preserve")], schema=schema), self.raw_event)
        self.raw_measurement = self.cache / f"measurements_{W}.parquet"
        pq.write_table(pa.Table.from_pylist([
            dict(id=f"raw_{i}", event_id="20241201_1005022351812", type=typ, value=value, version="v0.0.2")
            for i, (typ, value) in enumerate((("Time Passed (s)", 1200.0), ("Distance flown (NM)", 75.0)))]), self.raw_measurement)
        self.other_paths = []
        windows = {}
        for other in I.target_windows():
            if other in (W, "20251002_20251012"):
                continue
            windows[other] = {}
            for kind in ("events", "measurements"):
                path = self.root / kind / f"{kind}_{other}.parquet"
                path.write_bytes((kind + other).encode())
                self.other_paths.append(path)
                windows[other][kind] = {"status": "complete", "output": {"sha256": I.digest(path)}}
        windows["20251002_20251012"] = {"events": {"status": "missing"}, "measurements": {"status": "blocked_missing_events"}}
        event_source = {**I.metadata(self.raw_event), "url": f"{I.BASE}/flight_events/flight_events_{W}.parquet"}
        ms_source = {**I.metadata(self.raw_measurement), "url": f"{I.BASE}/measurements/measurements_{W}.parquet"}
        windows[W] = {"events": {"status": "error", "source": event_source},
                      "measurements": {"status": "blocked_missing_events", "source": ms_source}}
        source = I.metadata(self.flight)
        source["url"] = f"{I.BASE}/flight_list/flight_list_202512.parquet"
        self.manifest = {"status": "incomplete", "pipeline_version": 2,
                         "code_sha256": "23ce263a3793953cb2c84ebda3f9624ee20160f900098f053613503edc4007ac",
                         "release": "OPDI/v002", "requested_windows": I.target_windows(),
                         "airports": sorted(I.AIRPORTS), "flight_sources": {"202512": source},
                         "windows": windows, "completed_windows": 43}
        self.manifest["input_hash"] = I.object_hash({"flights": {"202512": source["sha256"]},
                                                   "airports": sorted(I.AIRPORTS), "pipeline": 2,
                                                   "code": self.manifest["code_sha256"]})
        I.atomic_json(self.root / "manifest.json", self.manifest)
        self.crosswalk = self.base / "crosswalk.parquet"
        self.crosswalk_rows = [dict(id=NATIVE, event_flight_id=ALIAS, offset=0, legacy_key=f"{GROUP}_0_2025_12")]
        self.write_crosswalk(self.crosswalk_rows)

    def write_crosswalk(self, rows):
        schema = pa.schema([("id", pa.int64()), ("event_flight_id", pa.uint64()),
                            ("offset", pa.int64()), ("legacy_key", pa.string())])
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), self.crosswalk)

    def run_repair(self, crosswalk=None):
        return R.repair(self.root, self.cache, crosswalk=crosswalk)

    def test_repair_canonicalizes_flight_join_and_preserves_all_43_other_windows(self):
        before = {p: p.read_bytes() for p in self.other_paths}
        result = self.run_repair(self.crosswalk)
        self.assertEqual(result["status"], "repaired")
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "incomplete")
        self.assertEqual(manifest["completed_windows"], 44)
        self.assertEqual(manifest["input_hash"], self.manifest["input_hash"])
        events = pq.read_table(self.root / "events" / f"events_{W}.parquet").to_pylist()
        self.assertEqual(events[0]["flight_id"], ALIAS)
        self.assertEqual(events[0]["opdi_flight_id"], str(NATIVE))
        self.assertEqual(events[0]["future_column"], "preserve")
        self.assertEqual(events[0]["event_time"], "2025-12-01 00:20:00")
        measurements = pq.read_table(self.root / "measurements" / f"measurements_{W}.parquet").to_pylist()
        self.assertEqual(len(measurements), 2)
        self.assertEqual(measurements[0]["event_id"], "20241201_1005022351812")
        stage = manifest["windows"][W]["events"]
        self.assertEqual(stage["repair"]["method_source_commit"], "d09e4ccb0327d3a7fdf4b40a48d7c6d61f76f98b")
        self.assertEqual(stage["audit"]["unmapped_source_flight_ids"], 0)
        self.assertEqual(stage["dependency_hash"], self.manifest["input_hash"])
        self.assertTrue(Path(stage["repair"]["crosswalk_path"]).is_file())
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        for window, stages in self.manifest["windows"].items():
            if window != W:
                self.assertEqual(manifest["windows"][window], stages)
        self.assertEqual(self.raw_event.stat().st_size, self.manifest["windows"][W]["events"]["source"]["bytes"])

    def test_crosswalk_builder_uses_full_sha_and_preserves_callsign_spaces(self):
        output = self.base / "built.parquet"
        R.build_crosswalk(self.flight, output)
        self.assertEqual(pq.read_table(output).to_pylist(), self.crosswalk_rows)
        self.assertEqual(self.run_repair()["status"], "repaired")

    def test_tampered_alias_preimage_and_conflicting_native_id_reject_without_changes(self):
        variants = [dict(self.crosswalk_rows[0], event_flight_id=ALIAS - 1),
                    dict(self.crosswalk_rows[0], legacy_key=f"{GROUP}_1_2025_12"),
                    dict(self.crosswalk_rows[0], id=NATIVE + 1)]
        before = (self.root / "manifest.json").read_bytes()
        for row in variants:
            self.write_crosswalk([row])
            with self.assertRaises(ValueError):
                self.run_repair(self.crosswalk)
            self.assertEqual((self.root / "manifest.json").read_bytes(), before)
            self.assertFalse((self.root / "events" / f"events_{W}.parquet").exists())

    def test_unmapped_raw_event_flight_rejects_before_publication(self):
        table = pq.read_table(self.raw_event)
        table = table.set_column(table.schema.get_field_index("flight_id"), "flight_id", pa.array([7], type=pa.uint64()))
        pq.write_table(table, self.raw_event)
        self.manifest["windows"][W]["events"]["source"].update(I.metadata(self.raw_event))
        I.atomic_json(self.root / "manifest.json", self.manifest)
        before = (self.root / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "Unmapped"):
            self.run_repair(self.crosswalk)
        self.assertEqual((self.root / "manifest.json").read_bytes(), before)
        self.assertFalse((self.root / "events" / f"events_{W}.parquet").exists())

    def test_input_source_hash_mismatch_and_duplicate_crosswalk_fail_closed(self):
        self.write_crosswalk(self.crosswalk_rows * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_repair(self.crosswalk)
        self.write_crosswalk(self.crosswalk_rows)
        self.manifest["flight_sources"]["202512"]["sha256"] = "0" * 64
        I.atomic_json(self.root / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "FLIGHT"):
            self.run_repair(self.crosswalk)

    def test_successful_repair_resumes_only_with_verified_outputs_and_crosswalk(self):
        self.run_repair(self.crosswalk)
        self.assertEqual(self.run_repair(self.crosswalk)["status"], "already_repaired")
        manifest = json.loads((self.root / "manifest.json").read_text())
        path = Path(manifest["windows"][W]["events"]["repair"]["crosswalk_path"])
        path.write_bytes(b"tampered crosswalk")
        with self.assertRaises(ValueError):
            self.run_repair(self.crosswalk)

    def test_conflicting_measurement_values_reject_both_outputs(self):
        rows = pq.read_table(self.raw_measurement).to_pylist()
        rows.append(dict(rows[0], id="other_raw_id", value=1201.0))
        pq.write_table(pa.Table.from_pylist(rows), self.raw_measurement)
        self.manifest["windows"][W]["measurements"]["source"].update(I.metadata(self.raw_measurement))
        I.atomic_json(self.root / "manifest.json", self.manifest)
        before = (self.root / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "Conflicting measurement"):
            self.run_repair(self.crosswalk)
        self.assertEqual((self.root / "manifest.json").read_bytes(), before)
        for kind in ("events", "measurements"):
            self.assertFalse((self.root / kind / f"{kind}_{W}.parquet").exists())

    def test_manifest_publication_failure_rolls_back_both_outputs_and_allows_retry(self):
        before = (self.root / "manifest.json").read_bytes()
        with mock.patch.object(I, "atomic_json", side_effect=OSError("test disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.run_repair(self.crosswalk)
        self.assertEqual((self.root / "manifest.json").read_bytes(), before)
        for kind in ("events", "measurements"):
            self.assertFalse((self.root / kind / f"{kind}_{W}.parquet").exists())
        self.assertEqual(self.run_repair(self.crosswalk)["status"], "repaired")

    def test_primary_ingestor_reuses_repaired_checkpoints_without_downloading(self):
        self.run_repair(self.crosswalk)
        manifest = json.loads((self.root / "manifest.json").read_text())
        ingestor = I.Ingestor(self.root, self.root / "flights", windows=[W])
        with mock.patch.object(ingestor, "acquire", side_effect=AssertionError("unexpected download")):
            for kind in ("events", "measurements"):
                previous = manifest["windows"][W][kind]
                result = ingestor.process(kind, W, set(), previous["dependency_hash"], previous, pa.uint64())
                self.assertEqual(result, previous)

    def test_builder_and_validator_reject_missing_native_preimage(self):
        row = pq.read_table(self.flight).to_pylist()[0]
        pq.write_table(pa.Table.from_pylist([dict(row, id=NATIVE + 1)]), self.flight)
        with self.assertRaisesRegex(ValueError, "Unrecovered"):
            R.build_crosswalk(self.flight, self.base / "incomplete.parquet")
        self.assertFalse((self.base / "incomplete.parquet").exists())

    def test_active_ingestion_lock_prevents_repair(self):
        import fcntl
        with (self.root / ".ingest.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                self.run_repair(self.crosswalk)

    def test_resume_rejects_changed_flight_eligibility_even_with_updated_input_hash(self):
        self.run_repair(self.crosswalk)
        rows = pq.read_table(self.flight).to_pylist()
        rows[0]["ades"] = "KJFK"
        pq.write_table(pa.Table.from_pylist(rows), self.flight)
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["flight_sources"]["202512"].update(I.metadata(self.flight))
        manifest["input_hash"] = I.object_hash({"flights": {"202512": I.digest(self.flight)},
                                              "airports": sorted(I.AIRPORTS), "pipeline": 2,
                                              "code": manifest["code_sha256"]})
        I.atomic_json(self.root / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "lineage"):
            self.run_repair(self.crosswalk)


if __name__ == "__main__":
    unittest.main()

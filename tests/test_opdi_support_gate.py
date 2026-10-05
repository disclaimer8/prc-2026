import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("support_gate", SCRIPTS / "opdi_support_gate.py")
G = importlib.util.module_from_spec(SPEC)
if Path(SPEC.origin).exists():
    SPEC.loader.exec_module(G)


def event(ident, action, second, *, flight="-7024018821582472598", osm=123, duration=30, lon=8., airport="EDDF"):
    return {"opdi_event_id": str(ident), "opdi_flight_id": flight, "type": action + "-runway",
            "event_time": f"2026-01-11 00:00:{second:02d}", "longitude": lon, "latitude": 50.,
            "info": json.dumps({"osm_id": osm, "osm_aeroway": "runway", "osm_ref": "07/25",
                                "osm_airport": airport, "opdi_time_in_use_s": duration})}


class SupportGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.flights = pl.DataFrame({"opdi_flight_id": ["-7024018821582472598"], "adep": ["EDDF"],
                                     "ades": ["LSZH"], "month": ["202601"]})
        self.centers = pl.DataFrame({"osm_airport": ["EDDF", "LSZH"], "center_lat": [50., 47.], "center_lon": [8., 8.5]})

    def pairs(self, rows):
        return G.validate_pairs(pl.DataFrame(rows), self.flights, self.centers)

    def test_valid_pair_preserves_signed_flight_id_and_object_identity(self):
        pairs, audit = self.pairs([event(1, "entry", 10), event(2, "exit", 40)])
        row = pairs.row(0, named=True)
        self.assertTrue(row["valid"])
        self.assertEqual(row["opdi_flight_id"], "-7024018821582472598")
        self.assertEqual(row["osm_id"], 123)
        self.assertEqual(row["exit_s"] - row["entry_s"], 30)

    def test_same_flight_different_objects_are_censored_not_combined(self):
        pairs, _ = self.pairs([event(1, "entry", 10), event(2, "exit", 40, osm=124)])
        self.assertEqual(pairs["valid"].sum(), 0)
        self.assertEqual(pairs["one_sided"].sum(), 2)

    def test_repeated_visits_negative_time_and_duration_disagreement_reject(self):
        for rows in ([event(1, "entry", 10), event(2, "entry", 11), event(3, "exit", 40)],
                     [event(1, "entry", 40), event(2, "exit", 10)],
                     [event(1, "entry", 10), event(2, "exit", 40, duration=31)]):
            pairs, _ = self.pairs(rows)
            self.assertEqual(pairs["valid"].sum(), 0)

    def test_wrong_airport_far_coordinates_and_malformed_info_fail_closed(self):
        pairs, _ = self.pairs([event(1, "entry", 10, lon=20.), event(2, "exit", 40)])
        self.assertFalse(pairs["valid"][0])
        pairs, _ = self.pairs([event(1, "entry", 10, airport="LSZH"), event(2, "exit", 40, airport="LSZH")])
        self.assertFalse(pairs["valid"][0])
        bad = dict(event(1, "entry", 10), info="not json")
        pairs, audit = self.pairs([bad, event(2, "exit", 40)])
        self.assertEqual(audit["invalid_info_rows"], 1)
        self.assertEqual(pairs["valid"].sum(), 0)

    def test_global_duplicate_event_id_rejected_across_window_files(self):
        a, b = self.root / "a.parquet", self.root / "b.parquet"
        pl.DataFrame({"opdi_event_id": ["9"], "opdi_flight_id": ["-1"]}).write_parquet(a)
        pl.DataFrame({"opdi_event_id": ["9"], "opdi_flight_id": ["2"]}).write_parquet(b)
        with self.assertRaisesRegex(ValueError, "Duplicate EVENT"):
            G.check_global_event_ids([a, b])

    def test_altitude_does_not_create_ground_validity_and_phase_is_explicit(self):
        rows = [dict(event(1, "entry", 10), altitude=1800.), dict(event(2, "exit", 40), altitude=1900.),
                dict(event(3, "entry", 20), type="take-off", info=None, altitude=1850.)]
        pairs, _ = self.pairs(rows)
        self.assertTrue(pairs["valid"][0])  # Coherent polygon pair, not ground occupancy.
        self.assertEqual(pairs["altitude_min"][0], 1800.)
        self.assertEqual(pairs["altitude_max"][0], 1900.)
        self.assertTrue(pairs["straddles_takeoff"][0])
        self.assertFalse(pairs["ends_before_takeoff"][0])

    def test_float_osm_id_is_not_silently_truncated(self):
        pairs, audit = self.pairs([event(1, "entry", 10, osm=123.5), event(2, "exit", 40, osm=123.5)])
        self.assertEqual(audit["invalid_info_rows"], 2)
        self.assertEqual(pairs["valid"].sum(), 0)

    def test_malformed_extra_event_quarantines_apparently_valid_pairs(self):
        pairs, _ = self.pairs([event(1, "entry", 10), event(2, "exit", 40), dict(event(3, "entry", 20), info="broken")])
        self.assertFalse(pairs["valid"][0])
        self.assertTrue(pairs["has_invalid_info"][0])

    def test_native_double_prc_ids_require_exact_integral_values(self):
        self.assertEqual(G.prc_ids(pl.Series("id", [123., 456.])).to_list(), ["123", "456"])
        for values in ([123.5], [float("nan")], [float("inf")], [float(2**53 + 2)], [None]):
            with self.assertRaises(ValueError):
                G.prc_ids(pl.Series("id", values))

    def test_public_anchor_priority_uses_literal_event_first_seen_only(self):
        events = pl.DataFrame({"opdi_flight_id": ["-1", "-1", "-2", "-3", "-4"],
                               "type": ["take-off", "first_seen", "first-xing-fl50", "first_seen", "landing"],
                               "event_s": [10, 5, 20, 30, 40]})
        flights = pl.DataFrame({"opdi_flight_id": ["-1", "-2", "-3", "-4"], "cs": ["A", "B", "C", "D"],
                                "adep": ["EDDF"]*4, "first_seen": [999]*4})
        anchors = G.extract_anchors(events, flights).sort("opdi_flight_id")
        self.assertEqual(anchors["opdi_flight_id"].to_list(), ["-1", "-2", "-3"])
        self.assertEqual(anchors["anchor_type"].to_list(), ["take-off", "first-xing-fl50", "first_seen"])
        self.assertEqual(anchors["anchor_s"].to_list(), [10, 20, 30])

    def test_prc_reader_projects_only_allowlisted_columns(self):
        path = self.root / "training.parquet"
        pq.write_table(pa.table({"MVT_ID_mvt": [123.], "PHASE_mvt": ["DEP"], "ADEP_mvt": ["EDDF"],
                                 "ADES_mvt": ["LSZH"], "MVT_TIME_UTC_mvt": ["2026-01-11 00:00:00"],
                                 "CALLSIGN_flt": [" ABC123 "], "FLIGHT_mvt": ["ABC123"],
                                 "BLOCK_TIME_UTC_mvt": ["FORBIDDEN"], "TAXITIME_SEC_mvt": [-999.],
                                 "pred": [-888.], "y": [-777.]}), path)
        frame = G.projected_frame(path, "prc")
        self.assertEqual(set(frame.columns), set(G.READ_COLUMNS["prc"]) & set(pq.ParquetFile(path).schema_arrow.names))
        self.assertFalse(set(frame.columns) & {"BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "pred", "y"})
        with self.assertRaises(ValueError):
            G.projected_frame(path, "predictions")

    def test_end_to_end_cross_window_pair_and_incremental_sensor_support(self):
        repo = self.root / "repo"
        opdi = repo / "data/opdi_full_v002"
        for name in ("flights", "events"):
            (opdi / name).mkdir(parents=True)
        prc = repo / "data/prc-2026-datasets"
        prc.mkdir()
        adsb = repo / "data/adsblol_kt1"
        adsb.mkdir()
        geo = repo / "analysis/sources/samples/osm_overpass"
        geo.mkdir(parents=True)
        for ap in "EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF LSZH LTFM".split():
            (geo / f"{ap}.json").write_text(json.dumps({"elements": [{"lat": 50., "lon": 8., "tags": {"aeroway": "parking_position"}}]}))
        flight = opdi / "flights/flight_list_202601.parquet"
        pl.DataFrame({"id": [-7024018821582472598], "flt_id": ["ABC123  "], "icao24": ["4ba8d4"],
                      "adep": ["EDDF"], "ades": ["LSZH"], "first_seen": ["2026-01-11 00:00:00"],
                      "last_seen": ["2026-01-11 00:59:00"]}).write_parquet(flight)
        import opdi_full_ingest as I
        manifest = {"flight_sources": {"202601": I.metadata(flight)}, "windows": {}, "requested_windows": []}
        for window, rows in (("20251231_20260110", [event(1, "entry", 10), dict(event(3, "entry", 0), type="first_seen", info=None)]),
                             ("20260110_20260120", [event(2, "exit", 40)])):
            path = opdi / "events" / f"events_{window}.parquet"
            pl.DataFrame(rows).write_parquet(path)
            manifest["windows"][window] = {"events": {"status": "complete", "output": I.metadata(path)}}
            manifest["requested_windows"].append(window)
        (opdi / "manifest.json").write_text(json.dumps(manifest))
        pl.DataFrame({"MVT_ID_mvt": [123., 124.], "PHASE_mvt": ["DEP", "DEP"], "ADEP_mvt": ["EDDF", "EDDF"],
                      "ADES_mvt": ["LSZH", "LSZH"], "MVT_TIME_UTC_mvt": ["2026-01-11 00:00:30", "2026-01-12 00:00:30"],
                      "CALLSIGN_flt": ["ABC123", "OTHER"], "FLIGHT_mvt": ["ABC123", "OTHER"],
                      "BLOCK_TIME_UTC_mvt": ["FORBIDDEN", "FORBIDDEN"], "TAXITIME_SEC_mvt": [-999., -999.]}).write_parquet(prc / "ranking.parquet")
        pl.DataFrame({"MVT_ID_mvt": [123., 123., 124.], "coverage": [0, 1, 2], "t_first": ["FORBIDDEN"]*3}).write_parquet(adsb / "offblock2_2026-01-11.parquet")
        report = G.run_gate(repo, self.root / "audit", months=["202601"])
        self.assertEqual(report["status"], "complete_unlabeled_audit")
        self.assertEqual(report["matching"]["unique_matches"], 1)
        row = next(x for x in report["mvt_support"] if x["airport"] == "EDDF")
        self.assertEqual(row["prc_departures"], 2)
        self.assertEqual(row["valid_runway_mvt"], 1)
        self.assertEqual(row["valid_runway_adsb_lt2_known"], 1)
        self.assertEqual(row["flight_source_day_missing"], 1)
        self.assertEqual(report["pair_audit"]["cross_window_object_groups"], 1)
        self.assertFalse(report["labels_predictions_models_read"])
        manifest["requested_windows"].append("20260120_20260130")
        (opdi / "manifest.json").write_text(json.dumps(manifest))
        missing = G.run_gate(repo, self.root / "audit_missing", months=["202601"])
        self.assertIn("20260120_20260130", missing["missing_event_windows"])
        original_flight = pl.read_parquet(flight)
        conflicting_flight = original_flight.with_columns(pl.lit("2026-01-11 12:00:00").alias("first_seen"),
                                                          pl.lit("2026-01-11 12:59:00").alias("last_seen"))
        pl.concat([original_flight, conflicting_flight]).write_parquet(flight)
        manifest["flight_sources"]["202601"] = I.metadata(flight)
        (opdi / "manifest.json").write_text(json.dumps(manifest))
        conflict = G.run_gate(repo, self.root / "audit_conflict", months=["202601"])
        self.assertEqual(conflict["flight_audit"]["conflicting_native_ids_excluded"], 1)
        self.assertEqual(conflict["matching"]["unique_matches"], 0)


if __name__ == "__main__":
    unittest.main()

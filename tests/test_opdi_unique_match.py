"""Synthetic, unlabeled examples of exact and mutually unique OPDI linkage."""
import importlib
from pathlib import Path
import sys
import unittest

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    M = importlib.import_module("opdi_unique_match")
except ModuleNotFoundError:
    M = None

PRC_SCHEMA = {"mvt_id": pl.String, "cs": pl.String, "airport": pl.String, "mvt_s": pl.Int64}
ANCHOR_SCHEMA = {"opdi_flight_id": pl.String, "cs": pl.String, "airport": pl.String,
                 "anchor_s": pl.Int64, "anchor_type": pl.String}


def prc(rows):
    return pl.DataFrame(rows, schema=PRC_SCHEMA, orient="row")


def anchors(rows):
    return pl.DataFrame(rows, schema=ANCHOR_SCHEMA, orient="row")


class UniqueMatchTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(M, "unlabeled unique-match helper is not implemented")

    def test_normalizes_all_whitespace_and_case_but_preserves_significant_characters(self):
        for raw, expected in ((None, None), ("", None), (" \t\n", None),
                              (" ab \t12\n3\r", "AB123"), ("\u2003a\u00a0b\x1cc", "ABC"),
                              (" a-b_1 ", "A-B_1")):
            with self.subTest(raw=raw):
                self.assertEqual(M.normalize_callsign(raw), expected)

    def test_inclusive_time_bounds_and_one_second_exclusions(self):
        p = prc([(f"m{i}", f"C{i}", "EDDF", 100000) for i in range(4)])
        a = anchors([(f"a{i}", f"C{i}", "EDDF", time, "takeoff")
                     for i, time in enumerate((99100, 101800, 99099, 101801))])
        matches, audit = M.unique_matches(p, a)
        self.assertEqual(matches.to_dicts(), [
            dict(mvt_id="m0", opdi_flight_id="a0", anchor_type="takeoff", anchor_delta_s=-900),
            dict(mvt_id="m1", opdi_flight_id="a1", anchor_type="takeoff", anchor_delta_s=1800)])
        self.assertEqual(audit["candidate_edges"], 2)
        self.assertEqual(audit["mvt_without_candidates"], 2)

    def test_cross_midnight_and_negative_epoch_days_are_not_lost(self):
        p = prc([("midnight", "C1", "EDDF", 86300), ("negative", "C2", "EDDF", -50)])
        a = anchors([("a", "C1", "EDDF", 86500, "takeoff"), ("b", "C2", "EDDF", 50, "firstseen")])
        matches, audit = M.unique_matches(p, a)
        self.assertEqual(matches["anchor_delta_s"].to_list(), [200, 100])
        self.assertEqual(audit["matched_anchor_types"], {"firstseen": 1, "takeoff": 1})

    def test_full_graph_degrees_prevent_hidden_opdi_reuse(self):
        # m1->{a1,a2}; m2->{a2}. Removing m1 first must not make m2-a2 unique.
        p = prc([("m1", "C", "EDDF", 0), ("m2", "C", "EDDF", 2000), ("m3", "D", "EDDF", 0)])
        a = anchors([("a1", "C", "EDDF", -800, "takeoff"), ("a2", "C", "EDDF", 1100, "takeoff"),
                     ("a3", "D", "EDDF", 0, "firstseen")])
        matches, audit = M.unique_matches(p, a)
        self.assertEqual(matches["mvt_id"].to_list(), ["m3"])
        self.assertEqual(audit["candidate_edges"], 4)
        self.assertEqual(audit["mvt_with_candidates"], 3)
        self.assertEqual(audit["ambiguous_mvt_ids"], 1)
        self.assertEqual(audit["reused_opdi_flight_ids"], 1)
        self.assertEqual(audit["unique_matches"], 1)

    def test_nearest_or_equal_distance_does_not_resolve_ambiguity(self):
        p = prc([("m", "C", "EDDF", 0)])
        for times in ((-100, 100), (0, 1500)):
            with self.subTest(times=times):
                a = anchors([(f"a{i}", "C", "EDDF", t, "takeoff") for i, t in enumerate(times)])
                matches, audit = M.unique_matches(p, a)
                self.assertEqual(matches.height, 0)
                self.assertEqual(audit["ambiguous_mvt_ids"], 1)

    def test_callsign_normalization_is_applied_to_both_inputs_and_airport_is_exact(self):
        p = prc([("m1", " a\u2003b\x1c12 ", "EDDF", 0), ("m2", " C12 ", "eddf", 0)])
        a = anchors([("a1", "AB12", "EDDF", 0, "takeoff"), ("a2", "C12", "EDDF", 0, "takeoff")])
        matches, _ = M.unique_matches(p, a)
        self.assertEqual(matches["mvt_id"].to_list(), ["m1"])
        self.assertEqual(p["cs"][0], " a\u2003b\x1c12 ")

    def test_missing_callsigns_are_filtered_and_audited(self):
        p = prc([("m0", None, "EDDF", 0), ("m1", " \t", "EDDF", 0), ("m2", "C", "EDDF", 0)])
        a = anchors([("a0", None, "EDDF", 0, "takeoff"), ("a1", "", "EDDF", 0, "takeoff"),
                     ("a2", "C", "EDDF", 0, "takeoff")])
        matches, audit = M.unique_matches(p, a)
        self.assertEqual(matches.height, 1)
        self.assertEqual(audit["prc_missing_callsign_rows"], 2)
        self.assertEqual(audit["anchor_missing_callsign_rows"], 2)

    def test_native_negative_and_large_canonical_ids_never_pass_through_float(self):
        p = prc([("-9223372036854775808", "C1", "EDDF", 0), ("9007199254740993", "C2", "EDDF", 0)])
        a = anchors([("18446744073709551615", "C1", "EDDF", 0, "takeoff"),
                     ("9007199254740992", "C2", "EDDF", 0, "takeoff")])
        matches, _ = M.unique_matches(p, a)
        self.assertEqual(matches["mvt_id"].to_list(), ["-9223372036854775808", "9007199254740993"])
        self.assertEqual(matches["opdi_flight_id"].to_list(), ["18446744073709551615", "9007199254740992"])

    def test_duplicate_ids_reject_even_if_callsign_is_missing(self):
        for p, a in ((prc([("m", "C", "EDDF", 0), ("m", None, "EDDF", 0)]), anchors([])),
                     (prc([]), anchors([("a", "C", "EDDF", 0, "takeoff"), ("a", None, "EDDF", 0, "takeoff")]))):
            with self.subTest(prc_rows=p.height):
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    M.unique_matches(p, a)

    def test_null_and_float_ids_or_times_are_rejected(self):
        p = prc([("m", "C", "EDDF", 0)])
        a = anchors([("a", "C", "EDDF", 0, "takeoff")])
        for side, column in (("p", "mvt_id"), ("p", "mvt_s"), ("a", "opdi_flight_id"), ("a", "anchor_s")):
            for replacement in (pl.lit(None), pl.lit(1.0)):
                with self.subTest(side=side, column=column, replacement=str(replacement)):
                    pp = p.with_columns(replacement.alias(column)) if side == "p" else p
                    aa = a.with_columns(replacement.alias(column)) if side == "a" else a
                    with self.assertRaises(ValueError):
                        M.unique_matches(pp, aa)

    def test_empty_inputs_have_stable_output_schema_and_zero_audit(self):
        matches, audit = M.unique_matches(prc([]), anchors([]))
        self.assertEqual(matches.schema, {"mvt_id": pl.String, "opdi_flight_id": pl.String,
                                         "anchor_type": pl.String, "anchor_delta_s": pl.Int64})
        for key in ("candidate_edges", "mvt_with_candidates", "ambiguous_mvt_ids", "reused_opdi_flight_ids", "unique_matches"):
            self.assertEqual(audit[key], 0)
        self.assertEqual(audit["matched_anchor_types"], {})

    def test_repeated_callsigns_are_bucketed_by_day_before_time_filter(self):
        p = prc([(f"m{i}", "DAILY", "EDDF", i*86400) for i in range(100)])
        a = anchors([(f"a{i}", "DAILY", "EDDF", i*86400+10, "takeoff") for i in range(100)])
        matches, audit = M.unique_matches(p, a)
        self.assertEqual(matches.height, 100)
        self.assertEqual(audit["candidate_edges"], 100)

    def test_int64_times_remain_exact_at_extremes_and_above_float_precision(self):
        times = [-(1 << 63), (1 << 63)-101, (1 << 53)+1000, (1 << 53)+1000]
        p = prc([(f"m{i}", f"C{i}", "EDDF", t) for i, t in enumerate(times)])
        a = anchors([(f"a{i}", f"C{i}", "EDDF", t+delta, "takeoff")
                     for i, (t, delta) in enumerate(zip(times, [100, 100, 1800, 1801]))])
        matches, audit = M.unique_matches(p, a)
        self.assertEqual(matches["anchor_delta_s"].to_list(), [100, 100, 1800])
        self.assertEqual(audit["unique_matches"], 3)


if __name__ == "__main__":
    unittest.main()

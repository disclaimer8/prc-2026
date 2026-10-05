"""Independent synthetic extraction/mask contracts; no real-data audit."""
import contextlib
import datetime as dt
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import visit_airborne_audit as A


def points(times, *, icao="abc", callsign="TEST1", alt=None, gs=None, ground=None):
    n = len(times)
    return pl.DataFrame({"icao": [icao] * n, "airport": ["EDDF"] * n,
                         "t": np.asarray(times, dtype=np.float64),
                         "alt": np.asarray(alt if alt is not None else [1500.] * n, dtype=np.float64),
                         "gs": np.asarray(gs if gs is not None else [100.] * n, dtype=np.float64),
                         "on_ground": ground if ground is not None else [False] * n,
                         "callsign": [callsign] * n})


def deps(takeoff=10000., mvt_id=1.):
    return pl.DataFrame({"MVT_ID_mvt": [mvt_id], "ADEP_mvt": ["EDDF"], "cs": ["TEST1"], "to": [takeoff]})


class VisitAirborneTests(unittest.TestCase):
    def test_fixed_calendar_and_departure_column_allowlist(self):
        expected = [f"{year}-{month:02}-{day:02}" for year in [2025, 2026]
                    for month in [1, 7] for day in range(1, 32)]
        self.assertEqual(A.DAYS, expected)
        self.assertEqual(len(A.DAYS), len(set(A.DAYS)))
        self.assertEqual(len(A.DAYS), 124)
        self.assertEqual(set(A.DEP_COLS), {"MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "CALLSIGN_flt",
                                         "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "FLIGHT_ID_mvt"})
        self.assertEqual(set(A.ACTIVE), set(A.V.NAMES) - {"vi_n_pts", "vi_prev_pts", "vi_alt_first"})

    def test_exact_canonical_ground_low_slow_null_alt_and_boundaries(self):
        # This intentionally retains alt=null at gs>40: exactly the source mask.
        frame = pl.DataFrame({"icao": ["abc"] * 9, "airport": ["EDDF"] * 9,
                              "t": np.arange(9, dtype=float), "callsign": ["TEST1"] * 9,
                              "on_ground": [True, False, False, False, False, False, False, False, None],
                              "gs": [200., 40., 40.0001, 40., None, 40., 41., 100., 100.],
                              "alt": [3000., 1164., 1164., 1164.0001, 100., None, None, 2000., 2000.]})
        self.assertEqual(A.airborne(frame)["t"].to_list(), [2., 3., 6., 7.])

    def test_all_visit_fields_recomputed_from_filtered_points(self):
        frame = points([1000, 1100, 5000, 5100, 9000, 9900, 10500],
                       alt=[1700, 1800, 1600, 10, 20, 2300, 2400],
                       gs=[130, 140, 120, 10, 15, 170, 180],
                       ground=[False, False, False, True, True, False, False])
        control, cid = A.extract(frame, deps())
        candidate, aid = A.extract(A.airborne(frame), deps())
        c, a = control.row(0, named=True), candidate.row(0, named=True)
        expected_c = {"vi_first_seen": 1000., "vi_turnaround": 4900., "vi_gap": 3900.,
                      "vi_n_pts": 3, "vi_prev_pts": 2, "vi_alt_first": 20., "vi_gs_first": 15., "vi_has_prev": 1}
        expected_a = {"vi_first_seen": 100., "vi_turnaround": 5000., "vi_gap": 4900.,
                      "vi_n_pts": 2, "vi_prev_pts": 1, "vi_alt_first": 2300., "vi_gs_first": 170., "vi_has_prev": 1}
        self.assertEqual({k: c[k] for k in expected_c}, expected_c)
        self.assertEqual({k: a[k] for k in expected_a}, expected_a)
        self.assertEqual(cid, {1.: "abc"}); self.assertEqual(aid, {1.: "abc"})

    def test_previous_visit_disappears_after_ground_deletion(self):
        frame = points([1000, 1100, 9000, 9900], ground=[True, True, False, False])
        candidate, identity = A.extract(A.airborne(frame), deps())
        row = candidate.row(0, named=True)
        self.assertEqual(row["vi_has_prev"], 0)
        self.assertEqual(row["vi_n_pts"], 2)
        for name in ["vi_turnaround", "vi_gap", "vi_prev_pts"]:
            self.assertIsNone(row[name])

    def test_identity_is_recomputed_without_ground_only_callsigns(self):
        full = pl.concat([points([9000, 9900], icao="air"),
                          points([1000, 1100], icao="ground", ground=[True, True])])
        control, old = A.extract(full, deps())
        candidate, new = A.extract(A.airborne(full), deps())
        self.assertIsNone(old[1.]); self.assertEqual(new[1.], "air")
        self.assertIsNone(control["vi_first_seen"][0])
        self.assertEqual(candidate["vi_first_seen"][0], 1000.)

    def test_ground_only_identity_is_lost_and_no_window_yields_nulls(self):
        frame = points([9000, 9900], ground=[True, True])
        candidate, identity = A.extract(A.airborne(frame), deps())
        self.assertIsNone(identity[1.])
        row = candidate.row(0, named=True)
        self.assertEqual(row["vi_has_prev"], 0)
        for name in A.V.NAMES[:-1]:
            self.assertIsNone(row[name])
        one, identity = A.extract(points([9900]), deps())
        self.assertEqual(identity[1.], "abc")
        self.assertIsNone(one["vi_first_seen"][0])

    def test_exact_lookback_and_after_takeoff_bounds(self):
        to = 100000.
        frame = points([to-43201, to-43200, to-100, to+600, to+601])
        result, _ = A.extract(frame, deps(to))
        row = result.row(0, named=True)
        self.assertEqual(row["vi_first_seen"], 100.)
        self.assertEqual(row["vi_n_pts"], 2)
        self.assertEqual(row["vi_prev_pts"], 1)
        self.assertEqual(row["vi_turnaround"], 43200.)
        self.assertEqual(row["vi_gap"], 43100.)

    def test_strict_visit_gap_and_airport_identity(self):
        frame = points([1000, 2800, 4600])
        result, _ = A.extract(frame, deps(5000.))
        self.assertEqual(result["vi_has_prev"][0], 0)
        self.assertEqual(result["vi_n_pts"][0], 3)
        foreign = points([9000, 9900], icao="foreign").with_columns(pl.lit("EGLL").alias("airport"))
        result, identity = A.extract(pl.concat([points([9000, 9900]), foreign]), deps())
        self.assertEqual(identity[1.], "abc")

    def test_inputs_and_row_order_are_preserved(self):
        frame = points([9900, 5000, 9000, 1000])
        dep = pl.concat([deps(mvt_id=3.), deps(mvt_id=1.)])
        before_pts, before_dep = frame.clone(), dep.clone()
        result, _ = A.extract(A.airborne(frame), dep)
        self.assertEqual(result["MVT_ID_mvt"].to_list(), [3., 1.])
        assert_frame_equal(frame, before_pts, check_exact=True)
        assert_frame_equal(dep, before_dep, check_exact=True)

    def test_control_matches_actual_original_main_without_writing(self):
        # Exercise the frozen original implementation, intercepting I/O in memory.
        spec = importlib.util.spec_from_file_location("independent_original_visit", ROOT / "src/adsb_visit.py")
        original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
        day = "2025-01-15"
        base = dt.datetime(2025, 1, 15, tzinfo=dt.timezone.utc).timestamp()
        frame = points(base + np.array([1000, 1100, 5000, 5100, 9000, 9900, 10500]))
        normalized = deps(base + 10000)
        raw = pl.DataFrame({"MVT_ID_mvt": [1.], "ADEP_mvt": ["EDDF"], "PHASE_mvt": ["DEP"],
                            "CALLSIGN_flt": ["TEST1"], "FLIGHT_mvt": ["IGNORED"],
                            "MVT_TIME_UTC_mvt": [dt.datetime.fromtimestamp(base + 10000, tz=dt.timezone.utc)]})
        captured = []
        with patch.object(original, "load_surface", return_value=(frame.sort("icao", "t"), frame)), \
             patch.object(original.pl, "read_parquet", return_value=raw), \
             patch.object(original.pl.DataFrame, "write_parquet", lambda self, *a, **k: captured.append(self.clone())), \
             patch.object(original.os, "makedirs"), patch.object(sys, "argv", ["adsb_visit.py", day]), \
             contextlib.redirect_stdout(io.StringIO()):
            original.main()
        actual, _ = A.extract(frame, normalized)
        self.assertEqual(len(captured), 1)
        assert_frame_equal(actual, captured[0], check_exact=True)

    def test_duplicate_timestamp_conflicting_values_match_original_load_surface_sort(self):
        # Unsorted 32-point fixture: with current Polars the historical one-sort
        # audit picked alt1008/gs108 for C0; the original double-sort picks
        # alt1018/gs118. Exercise the actual original load_surface, not a mock
        # that could accidentally hide its required global sort.
        rng = np.random.default_rng(1)
        codes = rng.integers(0, 3, 32)
        times = rng.integers(0, 8, 32).astype(float) + 9000
        day = "2025-01-15"
        base = dt.datetime(2025, 1, 15, tzinfo=dt.timezone.utc).timestamp()
        frame = pl.DataFrame({"icao": [f"ic{i}" for i in codes], "airport": ["EDDF"] * 32,
                              "t": base + times, "alt": np.arange(32, dtype=float) + 1000,
                              "gs": np.arange(32, dtype=float) + 100, "on_ground": [False] * 32,
                              "callsign": [f"C{i}" for i in codes]})
        normalized = pl.DataFrame({"MVT_ID_mvt": [1., 2., 3.], "ADEP_mvt": ["EDDF"] * 3,
                                   "cs": ["C0", "C1", "C2"], "to": [base + 10000] * 3})
        raw = pl.DataFrame({"MVT_ID_mvt": [1., 2., 3.], "ADEP_mvt": ["EDDF"] * 3,
                            "PHASE_mvt": ["DEP"] * 3, "CALLSIGN_flt": ["C0", "C1", "C2"],
                            "FLIGHT_mvt": ["unused"] * 3,
                            "MVT_TIME_UTC_mvt": [dt.datetime.fromtimestamp(base+10000, dt.timezone.utc)] * 3})
        spec = importlib.util.spec_from_file_location("original_visit_tie_sort", ROOT / "src/adsb_visit.py")
        original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
        original.PREV_DAY = False
        captured = []
        def read_original(path, columns=None):
            data = frame if "surface_" in str(path) else raw
            return data.select(columns) if columns is not None else data
        self.assertFalse(frame.equals(frame.sort("icao", "t")))
        self.assertLess(frame.select("icao", "airport", "t").unique().height, frame.height)
        with patch.object(original.pl, "read_parquet", side_effect=read_original), \
             patch.object(original.pl.DataFrame, "write_parquet", lambda self, *a, **k: captured.append(self.clone())), \
             patch.object(original.os, "makedirs"), patch.object(sys, "argv", ["adsb_visit.py", day]), \
             contextlib.redirect_stdout(io.StringIO()):
            original.main()
        actual, _ = A.extract(frame, normalized)
        self.assertEqual(len(captured), 1)
        assert_frame_equal(actual, captured[0], check_exact=True)

    def test_saved_control_parity_is_order_invariant_and_detects_all_vi_changes(self):
        frame = points([1000, 1100, 9000, 9900])
        control, _ = A.extract(frame, pl.concat([deps(mvt_id=1.), deps(mvt_id=2.)]))
        self.assertEqual(sum(A.compare_control(control, control.reverse()).values()), 0)
        for name in A.V.NAMES:
            changed = control.with_columns((pl.col(name).fill_null(0) + 1).alias(name))
            with self.subTest(name=name):
                self.assertEqual(A.compare_control(control, changed)[name], 2)
        with self.assertRaises(RuntimeError):
            A.compare_control(control, control.head(1))

    def test_run_identity_stays_keyed_when_joins_reorder_every_sidecar(self):
        # Synthetic full run with one day per fixed month. Force arbitrary join
        # ordering to expose positional ICAO attachment after hash joins.
        days = [f"{m}-15" for m in A.MONTHS]
        tables = {}
        for month, day in zip(A.MONTHS, days):
            base = dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp()
            raw = pl.DataFrame({"MVT_ID_mvt": [1., 2.], "PHASE_mvt": ["DEP", "DEP"],
                                "ADEP_mvt": ["LIRF", "LIRF"], "CALLSIGN_flt": ["TEST1", "TEST2"],
                                "FLIGHT_mvt": ["TEST1", "TEST2"], "FLIGHT_ID_mvt": [11., 22.],
                                "MVT_TIME_UTC_mvt": [dt.datetime.fromtimestamp(base+10000, dt.timezone.utc)] * 2})
            pts = pl.concat([points(base+np.array([1000, 1100, 9000, 9900]), icao="one"),
                             points(base+np.array([1200, 1300, 9200, 9950]), icao="two", callsign="TEST2")]).with_columns(pl.lit("LIRF").alias("airport"))
            normalized = pl.DataFrame({"MVT_ID_mvt": [1., 2.], "ADEP_mvt": ["LIRF", "LIRF"],
                                       "cs": ["TEST1", "TEST2"], "to": [base+10000] * 2})
            saved, _ = A.extract(pts, normalized)
            tables[("departures", month)] = raw
            tables[("surface", day)] = pts
            tables[("visit", day)] = saved.reverse()
            tables[("coverage", day)] = pl.DataFrame({"MVT_ID_mvt": [2., 1.], "coverage": [0, 2]})
        def projected_read(path, *, columns=None, **kwargs):
            self.assertIsNotNone(columns, "every input must use an explicit projection")
            self.assertFalse(set(columns) & {"TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt"})
            return tables[(Path(path).parent.name, Path(path).stem)].select(columns)
        original_join = pl.DataFrame.join
        def reverse_join(self, *args, **kwargs):
            return original_join(self, *args, **kwargs).reverse()
        with tempfile.TemporaryDirectory() as tmp:
            bundle, output = Path(tmp) / "bundle", Path(tmp) / "output"
            bundle.mkdir(); (bundle / "manifest.json").write_text("{}")
            with patch.object(A, "DAYS", days), patch.object(A, "verify", return_value={}), \
                 patch.object(A.pl, "read_parquet", side_effect=projected_read), \
                 patch.object(A.pl.DataFrame, "join", reverse_join), contextlib.redirect_stdout(io.StringIO()):
                A.run(bundle, output, "a" * 40)
            for day in days:
                result = pl.read_parquet(output / "days" / f"{day}.parquet")
                identities = {r["MVT_ID_mvt"]: (r["base_icao"], r["air_icao"]) for r in result.iter_rows(named=True)}
                self.assertEqual(identities, {1.: ("one", "one"), 2.: ("two", "two")})
            self.assertEqual(json.loads((output / "report.json").read_text())["parity_mismatched_cells"], 0)


if __name__ == "__main__":
    unittest.main()

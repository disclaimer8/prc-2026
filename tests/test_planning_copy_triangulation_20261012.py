"""Scientific-contract tests; synthetic data only, no research outcomes loaded."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

PATH = Path(__file__).resolve().parents[1] / "scripts/planning_copy_triangulation_20261012.py"
spec = importlib.util.spec_from_file_location("triangulation", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_common_denominator_missingness_and_multiple_anchor_union():
    residuals = np.array([[0, 0, np.nan, np.nan], [60, np.nan, np.nan, np.nan],
                          [np.nan]*4, [7, np.nan, np.nan, np.nan]])
    eligible, original, multiple = m.matches(residuals, 6, 0)
    shifted_eligible, shifted, _ = m.matches(residuals, 6, 60)
    assert eligible.tolist() == [True, True, False, True]
    assert np.array_equal(eligible, shifted_eligible)
    assert original.tolist() == [True, False, False, False]
    assert shifted.tolist() == [False, True, False, False]
    assert multiple.tolist() == [True, False, False, False]


def test_tolerance_edges_and_offset_sign():
    r = np.array([[-6], [6], [7], [-60], [60]], dtype=float)
    assert m.matches(r, 6, 0)[1].tolist() == [True, True, False, False, False]
    assert m.matches(r, 1, -60)[1].tolist() == [False, False, False, True, False]
    assert m.matches(r, 1, 60)[1].tolist() == [False, False, False, False, True]


def test_minute_grid_neutral_controls_and_injected_alignment():
    grid = np.repeat(np.array(m.OFFSETS, dtype=float), 10).reshape(-1, 1)
    rates = [m.matches(grid, 6, off)[1].mean() for off in m.OFFSETS]
    assert max(rates) == min(rates)
    injected = np.vstack([grid, np.zeros((70, 1))])
    rates2 = {off: m.matches(injected, 6, off)[1].mean() for off in m.OFFSETS}
    assert rates2[0] > max(v for k, v in rates2.items() if k)


def test_touching_30second_windows_are_not_disjoint_claims():
    r = np.array([[30.], [np.nan]])
    e0, h0, _ = m.matches(r, 30, 0)
    e1, h1, _ = m.matches(r, 30, 60)
    assert np.array_equal(e0, e1)
    assert h0.tolist() == h1.tolist() == [True, False]


def test_ids_reject_duplicates_and_fractional_ids():
    for ids in ([1., 1.], [1., 1.5], [1., float("nan")]):
        try:
            m.checked_ids(pl.DataFrame({"MVT_ID_mvt": ids}), "fixture")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid IDs accepted")
    assert m.checked_ids(pl.DataFrame({"MVT_ID_mvt": [1., 2.]}), "fixture")["MVT_ID_mvt"].dtype == pl.Int64


def test_timestamp_units_and_duration_load_contract():
    import datetime as dt
    t = dt.datetime(2025, 1, 1, 1, 0, 0, 500000, tzinfo=dt.timezone.utc)
    raw = {"MVT_ID_mvt": [1.], "PHASE_mvt": ["DEP"], "ADEP_mvt": ["EDDF"],
           "MVT_TIME_UTC_mvt": [t], "BLOCK_TIME_UTC_mvt": [t-dt.timedelta(seconds=600)],
           "TAXITIME_SEC_mvt": [600], "ADEP_flt": [None], "CALLSIGN_flt": ["ABC1"],
           "FLIGHT_mvt": ["ABC1"], "STAND_mvt": ["A1"]}
    raw.update({p: [t-dt.timedelta(seconds=606)] for p in m.PLANS})
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = root/"fixture.parquet"
        pl.DataFrame(raw).write_parquet(path)
        df = m.load_inputs(root, {"files": [{"kind": "training", "path": path.name}]})
        assert df["takeoff"][0] == t.timestamp()
        assert df["block_implied"][0] == df["block_raw"][0]
        assert df["block_implied"][0] - df["sched"][0] == 6
        assert m.matches(np.array([[df["block_implied"][0]-df["sched"][0]]]), 6)[1][0]


def test_primary_population_includes_all_airports_but_excludes_day_shifts():
    assert m.primary_population(np.array([0, 86399, 86400, -1, np.nan])).tolist() == [True, True, False, False, False]
    assert {"EDDF", "EHAM", "LSZH"}.issubset(m.AIRPORTS)
    assert m.OFFSETS == (-180, -120, -60, 0, 60, 120, 180)


def stand_rows():
    return pl.DataFrame({"takeoff": [10000.]*5, "gs_stand_known": [1.]*5,
        "gs_seen_at_stand": [1.]*5, "gs_d_first": [20., 20., 200., 20., 20.],
        "gs_d_firstmove": [30.]*5, "gs_t_leave": [9400.]*5,
        "gs_dwell_stand": [120., 10., 120., 120., 120.],
        "coverage": [3, 3, 3, 3, 3], "n_pts": [12]*5, "n_cand": [1, 1, 1, 2, 1],
        "y": [600., 600., 600., 600., 86400.], "sched": [9400.]*5})


def test_stand_eligibility_is_label_blind_and_rejects_weak_geometry():
    d = stand_rows()
    before = m.stand_eligibility(d)
    after = m.stand_eligibility(d.with_columns(pl.lit(-1e9).alias("y"), pl.lit(1e9).alias("sched")))
    assert np.array_equal(before, after)
    assert before.tolist() == [True, False, False, False, True]


def test_stand_temporal_validity_and_missing_fields_fail_closed():
    d = stand_rows().with_columns(pl.lit(float('nan')).alias("gs_t_leave"))
    assert not m.stand_eligibility(d).any()
    d = stand_rows().with_columns(pl.lit(10001.).alias("gs_t_leave"))
    assert not m.stand_eligibility(d).any()


def test_standardisation_uses_common_support_not_pooled_composition():
    # In each stratum copied records are +0.1; mixture proportions reverse the crude comparison.
    rows = [dict(cell="a", flagged=True, n=10, late=9), dict(cell="a", flagged=False, n=90, late=72),
            dict(cell="b", flagged=True, n=90, late=18), dict(cell="b", flagged=False, n=10, late=1),
            dict(cell="c", flagged=True, n=100, late=100)]
    r = m.standardized_contrast(rows, min_per_arm=10)
    assert r["cells"] == 2
    assert abs(r["risk_difference"] - .1) < 1e-12
    assert r["eligible_rows"] == 200
    assert r["discarded_rows"] == 100


def test_no_common_support_is_not_zero_effect():
    r = m.standardized_contrast([dict(cell="a", flagged=True, n=100, late=50)], 10)
    assert r["risk_difference"] is None and r["cells"] == 0


def test_retained_counts_and_months_ignore_nonoverlap():
    rows = [dict(cell="a", flagged=True, n=20, late=10, month=1),
            dict(cell="a", flagged=False, n=30, late=10, month=1),
            dict(cell="b", flagged=True, n=1000, late=10, month=7)]
    r = m.standardized_contrast(rows)
    assert r["retained_flagged_rows"] == 20 and r["retained_unflagged_rows"] == 30
    assert r["retained_months"] == [1] and r["discarded_rows"] == 1000


def test_synthetic_output_pipeline_handles_empty_physical_overlap():
    df = stand_rows().with_columns(
        pl.lit("LIRF").alias("airport"), pl.lit(1).alias("month"),
        pl.lit("0..1800").alias("delay_bin"), pl.lit(False).alias("nm_present"),
        pl.lit(3).alias("coverage_bin"), pl.lit(True).alias("has_stand_row"),
        pl.lit(True).alias("has_track_row"),
        (pl.col("takeoff")-pl.col("y")).alias("block_implied"),
        pl.col("sched").alias("iobt"), pl.col("sched").alias("lobt"), pl.col("sched").alias("eobt"))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        eligible, flag, _, table = m.alignment_tables(df, root)
        _, stand = m.triangulation(df, eligible, flag, root)
        assert stand["LIRF"]["risk_difference"] is None
        assert stand["LIRF"]["interpretation"] == "inconclusive_insufficient_common_support"
        zero = table.filter((pl.col("scope") == "all") & (pl.col("anchor") == "any") & (pl.col("tolerance_s") == 6))
        assert zero["eligible"][0] == 4


def test_run_requires_committed_preregistration():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            m.verify_registration(Path(tmp), "", {})
        except (RuntimeError, ValueError, FileNotFoundError):
            pass
        else:
            raise AssertionError("unregistered outcome calculation must fail closed")


def raw_track(times=None, speed=None):
    t = np.array([0., 25., 50., 75., 100., 125., 150., 175.]) if times is None else np.array(times, dtype=float)
    n = len(t)
    return {"t": t, "lat": np.array([41.]*n), "lon": np.array([12.]*n),
            "gs": np.array([0.]*n) if speed is None else np.array(speed, dtype=float),
            "surf": np.ones(n, dtype=bool), "stale": np.zeros(n, dtype=bool)}


def test_raw_prefix_missing_stale_key_fails_closed():
    track = raw_track()
    del track["stale"]
    assert not m.raw_stationary_prefix(track, 200., (41., 12.))["accepted"]


def test_raw_stationary_prefix_accepts_continuous_observed_dwell():
    r = m.raw_stationary_prefix(raw_track(), 200., (41., 12.))
    assert r["accepted"] and r["first_near"] == 0. and r["dwell_s"] == 175.
    assert r["n_stationary"] == 8


def test_raw_prefix_rejects_gap_missing_speed_and_moving_first_point():
    assert not m.raw_stationary_prefix(raw_track(times=[0, 25, 70, 95, 120, 145]), 200, (41, 12))["accepted"]
    assert not m.raw_stationary_prefix(raw_track(speed=[np.nan]*8), 200, (41, 12))["accepted"]
    assert not m.raw_stationary_prefix(raw_track(speed=[2, 0, 0, 0, 0, 0, 0, 0]), 200, (41, 12))["accepted"]


def test_raw_prefix_does_not_search_later_stationary_period():
    track = raw_track()
    track["lon"][1] += .001  # leaves the permitted displacement before dwell can qualify
    assert not m.raw_stationary_prefix(track, 200., (41, 12))["accepted"]


def test_raw_prefix_uses_terminal_ground_segment_not_inbound_dwell():
    track = raw_track(times=[0,25,50,75,100,1000,1025,1050])
    track["surf"][5] = False
    assert not m.raw_stationary_prefix(track, 1100, (41,12))["accepted"]


def test_raw_prefix_ignores_later_motion_after_sufficient_dwell_and_rejects_stale():
    track = raw_track(speed=[0,0,0,0,0,2,2,2])
    r = m.raw_stationary_prefix(track, 200., (41,12))
    assert r["accepted"] and r["dwell_s"] == 100 and r["n_stationary"] == 5
    track["stale"] = np.array([False, True, False, False, False, False, False, False])
    assert not m.raw_stationary_prefix(track, 200., (41,12))["accepted"]


def test_raw_supplement_end_to_end_and_missing_stale_schema():
    for present in (True, False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"src").mkdir()
            (root/"src/geometry.py").write_text(
                'import polars as pl\n'
                'def norm_stand(x): return x\n'
                'def strip_suffix(x): return x\n'
                'def load_tables():\n'
                ' return pl.DataFrame({"ADEP_mvt":["LIRF"],"stand_key":["A1"],"stand_lat":[41.],"stand_lon":[12.]}),pl.DataFrame()\n')
            day = m.RAW_DATES[0]
            track = raw_track()
            data = {k: track[k] for k in ("t", "lat", "lon", "gs")}
            data.update(icao=["abc123"]*8, airport=["LIRF"]*8, on_ground=[True]*8,
                        alt=[0.]*8, callsign=["ABC1"]*8)
            if present:
                data["stale"] = [False]*8
            pl.DataFrame(data).write_parquet(root/"raw.parquet")
            visits = pl.DataFrame(dict(MVT_ID_mvt=[1], day=[day], airport=["LIRF"], month=[1],
                delay_bin=["0..1800"], nm_present=[False], stand_eligible=[True], primary=[True],
                flagged=[True], callsign=["ABC1"], stand=["A1"], takeoff=[200.], block_implied=[-400.]))
            result = m.raw_supplement(root, {"files": [{"kind": "raw", "day": day, "path": "raw.parquet"}]}, visits, root)
            assert result["validated_episodes"] == int(present)
            assert result["scopes"]["LIRF"]["risk_difference"] is None
            audit = pl.read_csv(root/"raw_episode_audit.csv")
            assert audit["reason"][0] == ("validated_stationary_prefix" if present else "missing_stale_status")


if __name__ == "__main__":
    tests = [unittest.FunctionTestCase(v) for k, v in list(globals().items()) if k.startswith("test_")]
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(tests))
    raise SystemExit(not result.wasSuccessful())

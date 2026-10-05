"""Math, lineage, sensor-observability and atomic-output regression tests."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    D = importlib.import_module("current_error_mechanisms")
except ModuleNotFoundError:
    D = None


def fixture(root):
    folder = root / "data/e185_on_e182_w035_20261005"
    (folder / "candidate").mkdir(parents=True)
    rows = []
    for a in range(1, 7):
        ids = np.array([a, a + 6])
        z = dict(mvt_id=ids, month=ids, y=np.array([150., 150.]),
                 reg=np.array([100., 100.]), pred=np.array([207., 207.]),
                 values=np.tile([100., 300., np.nan, np.nan, np.nan, 86500.], (2, 1)),
                 proba=np.tile([.5, .5, 0., 0., 0., 0.], (2, 1)))
        np.savez(folder / "candidate" / f"fold_{a}_{a+6}.npz", **z)
        for ident in ids:
            rows.append(dict(MVT_ID_mvt=int(ident), month=int(ident), split="train",
                             TAXITIME_SEC_mvt=150., ADEP_mvt="LTFM", FLIGHT_ID_mvt=1,
                             d_mvt_sched=300., d_mvt_lobt=None))
    rows.append(dict(MVT_ID_mvt=99, month=7, split="rank", TAXITIME_SEC_mvt=None,
                     ADEP_mvt="LIRF", FLIGHT_ID_mvt=1, d_mvt_sched=300., d_mvt_lobt=None))
    pl.DataFrame(rows, schema_overrides={"d_mvt_lobt": pl.Float64}).write_parquet(root / "data/features_fin1b.parquet")
    for year, ids in ((2025, list(range(1, 13))), (2026, [99])):
        for directory, prefix, values in (
            ("adsblol_kt1", "offblock2", {"coverage": [0] * len(ids)}),
            ("adsblol", "visit", {"vi_has_prev": [0] * len(ids), "vi_first_seen": [None] * len(ids)}),
        ):
            dest = root / "data" / directory
            dest.mkdir(exist_ok=True)
            pl.DataFrame({"MVT_ID_mvt": ids, **values}).write_parquet(dest / f"{prefix}_{year}-01-01.parquet")
    manifest = {"input_sha256": {"data/features_fin1b.parquet": D.sha256(root / "data/features_fin1b.parquet")},
                "output_sha256": {f"candidate/{p.name}": D.sha256(p) for p in (folder / "candidate").glob("*.npz")},
                "scope": "synthetic pre-ADS-B fixture",
                "registration_commit": "6fb22b43c69465729d0ee3463c43c205771274c5",
                "weight": .35, "ranking_weight": .2333333333333333,
                "subset": "LOBT_flt IS NULL AND ADEP_mvt != LIRF",
                "control_source": "/synthetic/cv_preds_e172_rawf_lobtf_schedf_eobtf_aobtf_mlpa_f1b",
                "member_source": "/synthetic/cv_e130motion",
                "gate": {"passed": True, "flags": {"four_of_six": True, "trimmed_lower": True, "body_lower": True}},
                "code_sha256": {str(Path(module.__file__).resolve()): D.sha256(module.__file__)
                                for module in (D.E, D.G)}}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(D, "diagnostic runner is not implemented")

    def test_finite_oracle_excludes_nan_zero_and_production_oracle_includes_it(self):
        z = dict(reg=np.array([100.]), y=np.array([50.]), pred=np.array([57.]),
                 values=np.array([[100., np.nan, np.nan, np.nan, np.nan, 86500.]]),
                 proba=np.array([[.5, .5, 0., 0., 0., 0.]]))
        out = D.decompose(z, np.array([np.nan]))
        self.assertEqual(out["existing_correction"][0], 7.)
        self.assertEqual(out["all_regular"][0], 107.)
        self.assertEqual(out["finite_oracle"][0], 107.)
        self.assertEqual(out["production_oracle"][0], 50.)
        self.assertEqual(out["missing_probability_mass"][0], .5)
        self.assertTrue(out["current_outside_finite_interval"][0])

    def test_clipping_precedes_existing_delta_in_oracle_and_all_regular(self):
        z = dict(reg=np.array([10000., -100.]), y=np.array([4000., -30.]),
                 pred=np.array([3613., 7.]),
                 values=np.array([[10000., -100., np.nan, np.nan, np.nan, 96400.],
                                  [-100., -200., np.nan, np.nan, np.nan, 86300.]]),
                 proba=np.array([[1., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0.]]))
        out = D.decompose(z, np.array([0., np.nan]))
        np.testing.assert_array_equal(out["all_regular"], [3613., 7.])
        np.testing.assert_array_equal(out["finite_oracle"], [3613., 7.])

    def test_oracle_keeps_current_option_when_probability_sum_has_accepted_roundoff(self):
        z = dict(reg=np.array([100.]), y=np.array([90000.]), pred=np.array([86500.00001]),
                 values=np.array([[100., np.nan, np.nan, np.nan, np.nan, 86500.]]),
                 proba=np.array([[.0000001, 0., 0., 0., 0., 1.]]))
        out = D.decompose(z, np.array([np.nan]))
        self.assertEqual(out["production_oracle"][0], z["pred"][0])
        self.assertTrue(out["production_current_fallback"][0])

    def test_complete_run_preserves_inputs_and_reports_body_help_hurt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = fixture(root)
            before = D.sha256(folder / "candidate/fold_1_7.npz")
            result = D.run(root, root / "out", registration="a" * 40)
            row = result["tables"]["aggregate"]["NM_LTFM"]["body"]
            self.assertEqual(row["n"], 12)
            self.assertEqual(row["current_sse"], 12 * 57**2)
            self.assertEqual(row["all_regular_sse"], 12 * 43**2)
            self.assertEqual(row["allocation_hurts_n"], 12)
            self.assertEqual(row["finite_oracle_sse"], 0)
            self.assertAlmostEqual(row["current_top4_sse_share"], 1/3)
            self.assertEqual(row["current_top20_sse_share"], 1)
            self.assertEqual(result["ranking_counts"]["NM_LIRF_July"], 1)
            self.assertEqual(D.sha256(folder / "candidate/fold_1_7.npz"), before)
            self.assertTrue((root / "out/report.json").exists())
            with self.assertRaisesRegex(D.InvalidInput, "exists"):
                D.run(root, root / "out", registration="a" * 40)

    def test_stale_endpoints_invalid_probabilities_and_target_month_mismatch_rejected(self):
        for mutation in ("values", "proba", "y", "month", "mvt_id"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                folder = fixture(root)
                path = folder / "candidate/fold_1_7.npz"
                with np.load(path) as archive:
                    z = dict(archive)
                z[mutation] = z[mutation].copy()
                if mutation == "values":
                    z[mutation][0, 0] = 0
                elif mutation == "proba":
                    z[mutation][0, 0] = -.1
                else:
                    z[mutation][0] += 1
                np.savez(path, **z)
                manifest = json.loads((folder / "manifest.json").read_text())
                manifest["output_sha256"]["candidate/fold_1_7.npz"] = D.sha256(path)
                (folder / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(D.InvalidInput):
                    D.run(root, root / "out", registration="a" * 40)
                self.assertFalse((root / "out").exists())

    def test_manifest_mismatch_and_missing_fold_fail_before_publication(self):
        for mutation in ("manifest", "missing"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                folder = fixture(root)
                path = folder / "candidate/fold_1_7.npz"
                if mutation == "missing":
                    path.unlink()
                else:
                    path.write_bytes(path.read_bytes() + b"changed")
                with self.assertRaises(D.InvalidInput):
                    D.run(root, root / "out", registration="a" * 40)
                self.assertFalse((root / "out").exists())

    def test_six_files_with_incomplete_id_coverage_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = fixture(root)
            path = folder / "candidate/fold_1_7.npz"
            with np.load(path) as archive:
                z = {key: archive[key][1:] for key in archive.files}
            np.savez(path, **z)
            manifest = json.loads((folder / "manifest.json").read_text())
            manifest["output_sha256"]["candidate/fold_1_7.npz"] = D.sha256(path)
            (folder / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(D.InvalidInput, "full training ID"):
                D.run(root, root / "out", registration="a" * 40)
            self.assertFalse((root / "out").exists())

    def test_missing_records_and_nan_first_seen_do_not_invent_sensor_availability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            own_path = root / "data/adsblol_kt1/offblock2_2026-01-01.parquet"
            pl.read_parquet(own_path).head(0).write_parquet(own_path)
            visit_path = root / "data/adsblol/visit_2026-01-01.parquet"
            pl.read_parquet(visit_path).with_columns(pl.lit(float("nan")).alias("vi_first_seen")).write_parquet(visit_path)
            frames, _ = D.load_features(root, D.sensor_paths(root))
            masks = D.observed_masks(frames["ranking"])
            self.assertEqual(masks["NM_LIRF_July/missing_own_record"].sum(), 1)
            self.assertEqual(masks["NM_LIRF_July/no_own_no_previous"].sum(), 1)
            self.assertEqual(masks["NM_LIRF_July/airborne_seen"].sum(), 0)
            self.assertEqual(masks["NM_LIRF_July/own_only"].sum(), 0)

    def test_duplicate_sensor_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            path = root / "data/adsblol/visit_2026-01-01.parquet"
            frame = pl.read_parquet(path)
            pl.concat([frame, frame]).write_parquet(path)
            with self.assertRaisesRegex(D.InvalidInput, "duplicate sensor IDs"):
                D.load_features(root, D.sensor_paths(root))

    def test_different_parent_registration_weight_or_failed_gate_rejected(self):
        for key, value in (("registration_commit", "b" * 40), ("weight", .2),
                           ("gate", {"passed": False}), ("code_sha256", {})):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                folder = fixture(root)
                path = folder / "manifest.json"
                parent = json.loads(path.read_text())
                parent[key] = value
                path.write_text(json.dumps(parent))
                with self.assertRaisesRegex(D.InvalidInput, "parent lineage"):
                    D.run(root, root / "out", registration="a" * 40)
                self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()

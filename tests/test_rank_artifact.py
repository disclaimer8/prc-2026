"""Synthetic filesystem checks: final predictions must survive failed runs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import rank_artifact as artifact
except ModuleNotFoundError:
    artifact = None


class RankArtifactTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(artifact, "safe ranking artifact writer is not implemented")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.features = self.root / "features_fin1b.parquet"
        self.features.write_bytes(b"synthetic feature provenance")
        self.out = self.root / "raw_fin1b.npz"
        self.ids = np.array([31, 12, 29])
        self.predictions = np.array([[10., 20., 30.], [12., 24., 36.]])

    def writer(self, **changes):
        options = dict(output=self.out, expected_seeds=2, rounds=10,
                       features_path=self.features, mode="raw", expected_rows=3,
                       provenance={"source": "synthetic"})
        options.update(changes)
        return artifact.RankArtifactWriter(**options)

    def test_interruption_preserves_old_final_and_separate_incomplete_checkpoint(self):
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with self.writer() as writer:
                self.out.write_bytes(b"finished previous artifact")
                writer.checkpoint(self.ids, self.predictions[:1])
                raise RuntimeError("interrupted")
        self.assertEqual(self.out.read_bytes(), b"finished previous artifact")
        checkpoint = self.out.with_name("raw_fin1b.checkpoint.npz")
        with np.load(checkpoint, allow_pickle=False) as saved:
            self.assertEqual(saved["status"].item(), "checkpoint")
            self.assertFalse(saved["complete"].item())
            self.assertEqual(saved["completed_seeds"].item(), 1)
            np.testing.assert_array_equal(saved["reg"], [10., 20., 30.])
        self.assertFalse(self.out.with_name("raw_fin1b.npz.lock").exists())

    def test_incomplete_seeds_cannot_publish(self):
        with self.writer() as writer:
            with self.assertRaisesRegex(ValueError, "seed"):
                writer.publish(self.ids, self.predictions[:1])
        self.assertFalse(self.out.exists())

    def test_changed_feature_source_cannot_publish(self):
        with self.writer() as writer:
            writer.checkpoint(self.ids, self.predictions[:1])
            self.features.write_bytes(b"changed features during fitting")
            with self.assertRaisesRegex(ValueError, "features.*changed"):
                writer.publish(self.ids, self.predictions)
        self.assertFalse(self.out.exists())

    def test_success_has_exact_mean_ids_counts_and_provenance(self):
        with self.writer() as writer:
            writer.checkpoint(self.ids, self.predictions[:1])
            writer.publish(self.ids, self.predictions)
        with np.load(self.out, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["mvt_id"], [31, 12, 29])
            np.testing.assert_array_equal(saved["reg"], [11., 22., 33.])
            np.testing.assert_array_equal(saved["per_seed"], self.predictions)
            self.assertTrue(saved["complete"].item())
            self.assertEqual(saved["status"].item(), "complete")
            self.assertEqual(saved["expected_seeds"].item(), 2)
            self.assertEqual(saved["completed_seeds"].item(), 2)
            self.assertEqual(saved["rounds"].item(), 10)
            self.assertEqual(saved["mode"].item(), "raw")
            self.assertEqual(saved["features_path"].item(), str(self.features.resolve()))
            self.assertEqual(saved["features_sha256"].item(), hashlib.sha256(self.features.read_bytes()).hexdigest())
            self.assertEqual(json.loads(saved["provenance_json"].item())["source"], "synthetic")

    def test_preexisting_final_rejects_even_with_overwrite_permission(self):
        self.out.write_bytes(b"old")
        with self.assertRaises(FileExistsError):
            with self.writer():
                self.fail("existing final accepted")
        self.assertEqual(self.out.read_bytes(), b"old")
        with self.assertRaisesRegex(ValueError, "fresh E130_OUT"):
            with self.writer(overwrite=True):
                self.fail("unsafe overwrite accepted")
        self.assertEqual(self.out.read_bytes(), b"old")

    def test_collision_during_run_does_not_overwrite(self):
        with self.writer() as writer:
            self.out.write_bytes(b"other publisher")
            with self.assertRaises(FileExistsError):
                writer.publish(self.ids, self.predictions)
        self.assertEqual(self.out.read_bytes(), b"other publisher")

    def test_overwrite_permission_rejects_before_reading_features(self):
        with self.assertRaisesRegex(ValueError, "fresh E130_OUT"):
            with self.writer(overwrite=True, features_path=self.root / "missing.parquet"):
                self.fail("overwrite accepted")
        self.assertFalse(self.out.exists())

    def test_exclusive_lock_rejects_second_writer_and_releases_after_exception(self):
        with self.writer():
            with self.assertRaises(FileExistsError):
                with self.writer():
                    self.fail("second writer accepted")
        with self.writer() as writer:
            writer.publish(self.ids, self.predictions)

    def test_invalid_ids_predictions_and_row_counts_fail_closed(self):
        cases = [
            ([1, 1, 2], self.predictions),
            ([1, np.nan, 2], self.predictions),
            ([1, np.inf, 2], self.predictions),
            ([1, 1.5, 2], self.predictions),
            (["a", "b", "c"], self.predictions),
            ([1, 2], self.predictions[:, :2]),
            (self.ids, [[1., np.nan, 2.], [1., 2., 3.]]),
            (self.ids, [[1., np.inf, 2.], [1., 2., 3.]]),
            (self.ids, [1., 2., 3.]),
            (self.ids, np.zeros((3, 3))),
        ]
        with self.writer() as writer:
            for ids, predictions in cases:
                with self.subTest(ids=ids, predictions=predictions):
                    with self.assertRaises(ValueError):
                        writer.publish(ids, predictions)
        self.assertFalse(self.out.exists())

    def test_zero_or_negative_seed_and_round_counts_reject(self):
        for field in ("expected_seeds", "rounds"):
            for value in (0, -1):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.writer(**{field: value})

    def test_alias_to_fin1b_cannot_bypass_legacy_path_guard(self):
        alias = self.root / "features.parquet"
        alias.symlink_to(self.features)
        with self.assertRaisesRegex(ValueError, "E130_OUT"):
            artifact.resolve_output_path("raw", alias, None)

    def test_unsafe_floating_ids_reject_but_small_integral_floats_pass(self):
        with self.writer() as writer:
            with self.assertRaisesRegex(ValueError, "exact|safe"):
                writer.publish(np.array([float(2**53 + 2), 12., 29.]), self.predictions)
            self.assertFalse(self.out.exists())
            writer.publish(np.array([31., 12., 29.]), self.predictions)
        with np.load(self.out) as saved:
            np.testing.assert_array_equal(saved["mvt_id"], [31., 12., 29.])

    def test_default_production_row_count_rejects_small_fixture(self):
        with artifact.RankArtifactWriter(self.out, 2, 10, self.features, "raw") as writer:
            with self.assertRaisesRegex(ValueError, "344841"):
                writer.publish(self.ids, self.predictions)
        self.assertFalse(self.out.exists())

    def test_writer_requires_lock_before_save_and_rejects_republication(self):
        writer = self.writer()
        with self.assertRaises(RuntimeError):
            writer.publish(self.ids, self.predictions)
        with writer:
            writer.publish(self.ids, self.predictions)
            original = self.out.read_bytes()
            with self.assertRaises(RuntimeError):
                writer.publish(self.ids, self.predictions * 2)
            self.assertEqual(self.out.read_bytes(), original)

    def test_fin1b_requires_explicit_distinct_output_even_with_overwrite(self):
        legacy = Path("submissions/e130_raw_rank.npz")
        for override in (None, "", str(legacy), str(legacy.resolve())):
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, "E130_OUT"):
                    artifact.resolve_output_path("raw", self.features, override)
        self.assertEqual(artifact.resolve_output_path("raw", self.features, str(self.out)), self.out)
        self.assertEqual(artifact.resolve_output_path("lobt", "features_e106.parquet", None), Path("submissions/e130_lobt_rank.npz"))

    def test_script_fin1b_guard_precedes_feature_preparation(self):
        env = {**os.environ, "FEATURES": str(self.features), "E130_MODE": "raw", "E130_OVERWRITE": "1",
               "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "POLARS_MAX_THREADS": "2"}
        env.pop("E130_OUT", None)
        script = Path(__file__).resolve().parents[1] / "src/e130_rank.py"
        result = subprocess.run([sys.executable, str(script), "2", "10"], cwd=self.root, env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("E130_OUT", result.stderr)
        self.assertNotIn("Parquet", result.stderr)

    def test_script_existing_final_guard_precedes_feature_preparation(self):
        self.out.write_bytes(b"old final")
        env = {**os.environ, "FEATURES": str(self.root / "missing.parquet"), "E130_OUT": str(self.out),
               "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "POLARS_MAX_THREADS": "2"}
        env.pop("E130_OVERWRITE", None)
        script = Path(__file__).resolve().parents[1] / "src/e130_rank.py"
        result = subprocess.run([sys.executable, str(script), "2", "10"], cwd=self.root, env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("E130_OUT", result.stderr)
        self.assertEqual(self.out.read_bytes(), b"old final")

    def test_script_existing_final_with_overwrite_flag_rejects_before_preparation(self):
        self.out.write_bytes(b"competing complete final")
        env = {**os.environ, "FEATURES": str(self.root / "missing.parquet"), "E130_OUT": str(self.out),
               "E130_OVERWRITE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "POLARS_MAX_THREADS": "2"}
        script = Path(__file__).resolve().parents[1] / "src/e130_rank.py"
        result = subprocess.run([sys.executable, str(script), "2", "10"], cwd=self.root, env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fresh E130_OUT", result.stderr)
        self.assertNotIn("FileNotFoundError", result.stderr)
        self.assertEqual(self.out.read_bytes(), b"competing complete final")


if __name__ == "__main__":
    unittest.main()

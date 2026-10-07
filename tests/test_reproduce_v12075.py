"""Integrity boundaries for the independent saved-component replay."""
import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class ReplayContracts(unittest.TestCase):
    def runner(self):
        try:
            return importlib.import_module('reproduce_v12075')
        except ModuleNotFoundError:
            self.fail('the isolated v12075 replay runner is missing')

    def test_input_inventory_rejects_oracle_aliases_and_parent_escape(self):
        r = self.runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'best.parquet').write_bytes(b'private oracle')
            (root / 'safe.npz').write_bytes(b'archived member')
            (root / 'alias.npz').symlink_to(root / 'best.parquet')
            os.link(root / 'best.parquet', root / 'hardlink.npz')
            inventory = {'safe.npz': r.R.sha(root / 'safe.npz')}
            r.verify_inventory(root, inventory, ['best.parquet'])
            for name in ('best.parquet', 'alias.npz', 'hardlink.npz', '../escape'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    r.verify_inventory(root, {name: '0' * 64}, ['best.parquet'])

    def test_input_hash_drift_fails(self):
        r = self.runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'member.npz').write_bytes(b'first')
            inventory = {'member.npz': r.R.sha(root / 'member.npz')}
            (root / 'member.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                r.verify_inventory(root, inventory, [])

    def test_final_check_requires_predictions_schema_ids_and_bytes(self):
        r = self.runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = pl.DataFrame({r.ID: [8, 4, 1], r.Y: [13, 20, 5]}, schema={r.ID: pl.Int64, r.Y: pl.Int32})
            expected.write_parquet(root / 'oracle.parquet')
            expected.write_parquet(root / 'actual.parquet')
            digest = r.R.sha(root / 'oracle.parquet')
            result = r.compare_final(root / 'actual.parquet', root / 'oracle.parquet', digest, rows=3)
            self.assertTrue(result['byte_identical'])
            for altered in (expected.reverse(), expected.with_columns(pl.Series(r.Y, [13, 20, 6], dtype=pl.Int32)),
                            expected.with_columns(pl.Series(r.ID, [8, 8, 1])), expected.head(2)):
                altered.write_parquet(root / 'actual.parquet')
                with self.subTest(altered=altered.rows()), self.assertRaises(ValueError):
                    r.compare_final(root / 'actual.parquet', root / 'oracle.parquet', digest, rows=3)

    def test_capture_job_refused_before_original_stage_can_run(self):
        r = self.runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / 'job.json'
            r.R.save(job, {'mode': 'capture'})
            with self.assertRaisesRegex(ValueError, 'replay'):
                r.replay_stage(root, root, job)
            self.assertEqual(list(root.iterdir()), [job])

    def test_supervisor_counts_only_owned_group_and_fails_resource_limits(self):
        try:
            launcher = importlib.import_module('reproduce_v12075_launch_20261007')
        except ModuleNotFoundError:
            self.fail('bounded replay launcher is missing')
        sample = '11 99 2048 5.0\n12 99 4096 400.0\n13 100 99999999 700.0\n'
        result = launcher.group_resources(sample, 99)
        self.assertEqual(result, {'rss_bytes': 6291456, 'cpu_percent': 405.0, 'processes': 2})
        launcher.enforce_limits(result, elapsed=10, disk_free=4 * 1024**3)
        for rss, seconds, disk in ((17 * 1024**3, 10, 4 * 1024**3),
                                    (100, 4 * 3600 + 1, 4 * 1024**3),
                                    (100, 10, 1024**3)):
            with self.assertRaises(RuntimeError):
                launcher.enforce_limits({'rss_bytes': rss}, seconds, disk)

    def test_e188_rejects_unregistered_sensor_files_in_live_root(self):
        r = self.runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ['data/adsblol_kt1/offblock2_2026-01-01.parquet',
                     'data/adsblol/visit_2026-01-01.parquet']
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            verify = getattr(r, 'verify_e188_globs', None)
            self.assertIsNotNone(verify, 'the live-root E188 input set must be closed')
            verify(root, names)
            (root / 'data/adsblol/visit_2026-07-31.parquet').touch()
            with self.assertRaisesRegex(ValueError, 'sensor'):
                verify(root, names)


if __name__ == '__main__':
    unittest.main()

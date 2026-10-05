"""Synthetic wrapper contracts; never imports or runs a training module."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl


class LGBTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1]/'scripts/lgb_fin1b_oof.py'
        self.assertTrue(path.exists(), 'LGB component wrapper missing')
        spec = importlib.util.spec_from_file_location('lgb_oof', path)
        self.M = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.M)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def frame(self):
        return pl.DataFrame({'MVT_ID_mvt': [90., 11., 27., 12., 29.],
                             'split': ['rank', 'train', 'train', 'train', 'train'],
                             'month': [1, 1, 7, 2, 9], 'TAXITIME_SEC_mvt': [None, 100., 200., 300., 400.],
                             'feature': [9., 1., 2., 3., 4.]})

    def raw(self, **changes):
        seeds = np.arange(10, dtype=float).reshape(5, 2)
        payload = dict(mvt_id=np.array([11., 27.]), reg=seeds.mean(axis=0), per_seed=seeds)
        payload.update(changes)
        path = self.root/'raw.npz'
        np.savez(path, **payload)
        return path

    def test_views_exclude_ranking_and_hide_exact_heldout_targets_without_reordering(self):
        view, meta = self.M.fold_view(self.frame(), 1)
        self.assertEqual(view['MVT_ID_mvt'].to_list(), [11., 27., 12., 29.])
        self.assertEqual(view.schema['MVT_ID_mvt'], pl.Float64)
        self.assertEqual(view['split'].to_list(), ['rank', 'rank', 'train', 'train'])
        self.assertEqual(view['TAXITIME_SEC_mvt'].to_list(), [None, None, 300., 400.])
        self.assertEqual(meta['mvt_id'].tolist(), [11., 27.])
        self.assertEqual(meta['y'].tolist(), [100., 200.])
        self.assertEqual(meta['month'].tolist(), [1, 7])

    def test_view_rejects_fractional_duplicate_or_nonfinite_ids_and_bad_months(self):
        for expr in [pl.lit(1.).alias('MVT_ID_mvt'), (pl.col('MVT_ID_mvt')+.5).alias('MVT_ID_mvt'),
                     pl.lit(float('nan')).alias('MVT_ID_mvt'), pl.lit(2**54*1.).alias('MVT_ID_mvt'),
                     pl.lit(13).alias('month'), pl.lit(None, dtype=pl.Float64).alias('TAXITIME_SEC_mvt')]:
            with self.subTest(expr=str(expr)), self.assertRaises(self.M.Invalid):
                self.M.fold_view(self.frame().with_columns(expr), 1)
        self.assertFalse(self.M.integral_ids(np.array([2**63], dtype=np.uint64)))

    def test_raw_requires_exact_five_seed_mean_ids_order_and_finite_values(self):
        self.M.validate_raw(self.raw(), np.array([11., 27.]))
        for changes in [dict(per_seed=np.zeros((4, 2))), dict(reg=[0., 0.]),
                        dict(mvt_id=[27., 11.]), dict(mvt_id=[11., 11.]),
                        dict(per_seed=np.full((5, 2), np.nan)), dict(extra=[1])]:
            with self.subTest(changes=changes), self.assertRaises(self.M.Invalid):
                self.M.validate_raw(self.raw(**changes), np.array([11., 27.]))

    def test_publication_resume_and_tampering_preserve_metadata_and_seeds(self):
        _, meta = self.M.fold_view(self.frame(), 1)
        final = self.root/'complete'
        evidence = {'lineage': 'a'*64, 'fold': 1, 'rounds': 2319, 'seeds': [1000,1001,1002,1003,1004]}
        self.M.publish_fold(self.raw(), final, meta, evidence)
        self.M.verify_fold(final, meta, evidence)
        with np.load(final/'prediction.npz', allow_pickle=False) as z:
            self.assertEqual(set(z.files), {'mvt_id','y','month','reg2','per_seed'})
            np.testing.assert_array_equal(z['reg2'], np.arange(10).reshape(5,2).mean(axis=0))
            np.testing.assert_array_equal(z['y'], [100.,200.])
        with self.assertRaises(self.M.Invalid):
            self.M.publish_fold(self.raw(), final, meta, evidence)
        with self.assertRaises(self.M.Invalid):
            self.M.verify_fold(final, meta, {**evidence, 'fold': 2})
        with (final/'prediction.npz').open('ab') as f:
            f.write(b'changed')
        with self.assertRaises(self.M.Invalid):
            self.M.verify_fold(final, meta, evidence)

    def test_child_environment_uses_only_supported_thread_override_and_private_cwd_output(self):
        with patch.dict(os.environ, {'ADV_WEIGHT': '/bad', 'PROXY_AFFINE': '1', 'REG_OVERRIDE': '{}', 'PYTHONPATH': '/bad'}):
            env = self.M.child_environment(self.root/'view.parquet')
        self.assertEqual(json.loads(env['REG_OVERRIDE']), {'num_threads': 6})
        self.assertEqual(env['OMP_THREAD_LIMIT'], '6')
        self.assertEqual(env['OMP_WAIT_POLICY'], 'PASSIVE')
        self.assertEqual(env['GOMP_SPINCOUNT'], '0')
        self.assertFalse({'ADV_WEIGHT','PROXY_AFFINE','PYTHONPATH'} & set(env))
        self.assertEqual(self.M.command(self.root), [__import__('sys').executable, '-u',
                         str(self.root/'sources/seed_avg.py'), 'rank', '5', '2319'])

    def test_only_complete_five_seed_log_is_publishable(self):
        log = self.root/'fit.log'
        good = '\n'.join(f'seed {n} done (10s)' for n in range(5))+'\nsaved submissions/lgb_reg_seeds.npz\n'
        log.write_text(good)
        self.M.validate_log(log)
        for bad in [good.replace('seed 3 done (10s)\n',''), good.replace('seed 4','seed 3'), good+'failure\n']:
            log.write_text(bad)
            with self.assertRaises(self.M.Invalid):
                self.M.validate_log(log)

    def test_quarantine_preserves_failed_attempt_and_single_writer(self):
        attempt = self.root/'attempts/failed'
        attempt.mkdir(parents=True)
        (attempt/'fit.log').write_text('diagnostic evidence')
        self.M.quarantine_attempts(self.root)
        self.assertFalse(attempt.exists())
        self.assertEqual(next((self.root/'quarantine').glob('*/fit.log')).read_text(), 'diagnostic evidence')
        with self.M.writer_lock(self.root):
            with self.assertRaises(self.M.Invalid):
                with self.M.writer_lock(self.root):
                    pass

    def test_input_and_source_snapshots_reject_changed_bytes(self):
        folder = self.root/'inputs/fold_1_7'
        folder.mkdir(parents=True)
        for name in ['features.parquet', 'metadata.npz']:
            (folder/name).write_bytes(b'pinned bytes')
        record = {name: self.M.sha(folder/name) for name in ['features.parquet', 'metadata.npz']}
        self.M.verify_input(self.root, 'fold_1_7', record)
        (folder/'metadata.npz').write_bytes(b'replaced metadata')
        with self.assertRaises(self.M.Invalid):
            self.M.verify_input(self.root, 'fold_1_7', record)
        source = self.root/'seed_avg.py'
        source.write_text('original trainer')
        with patch.dict(self.M.SOURCES, {'seed_avg.py': self.M.sha(source)}, clear=True):
            self.M.verify_sources(self.root)
            source.write_text('modified trainer')
            with self.assertRaises(self.M.Invalid):
                self.M.verify_sources(self.root)

    def test_resume_rejects_changed_targets_even_with_valid_prediction_checksum(self):
        _, meta = self.M.fold_view(self.frame(), 1)
        final, evidence = self.root/'complete', {'lineage':'a'*64}
        self.M.publish_fold(self.raw(), final, meta, evidence)
        with self.assertRaises(self.M.Invalid):
            self.M.verify_fold(final, {**meta, 'y': meta['y']+1}, evidence)

    def test_resource_guard_requires_exact_cpu_mask_nice_and_bounded_cgroup(self):
        def content(path):
            if str(path) == '/proc/self/cgroup':
                return '0::/system.slice/lgb.service\n'
            if path.name == 'memory.max':
                return str(24*1024**3)
            if path.name == 'memory.swap.max':
                return '0'
            raise AssertionError(path)
        with patch.object(self.M.sys, 'platform', 'linux'), \
                patch.object(self.M.os, 'sched_getaffinity', return_value={0,2,4,6,8,10}, create=True) as affinity, \
                patch.object(self.M.os, 'nice', return_value=10) as nice, \
                patch.object(Path, 'read_text', content):
            self.assertEqual(self.M.resource_guard()['memory_max_bytes'], 24*1024**3)
            affinity.return_value = {0,1,2,3,4,5}
            with self.assertRaises(self.M.Invalid):
                self.M.resource_guard()
            affinity.return_value = {0,2,4,6,8,10}
            nice.return_value = 0
            with self.assertRaises(self.M.Invalid):
                self.M.resource_guard()
            nice.return_value = 10
            with patch.object(Path, 'read_text', lambda p: '0::/system.slice/lgb.service' if str(p)=='/proc/self/cgroup' else 'max'):
                with self.assertRaises(self.M.Invalid):
                    self.M.resource_guard()


if __name__ == '__main__':
    unittest.main()

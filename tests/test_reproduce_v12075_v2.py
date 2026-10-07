"""Low-memory execution preserves classifier inputs and pre-mixture arithmetic."""
import importlib
from pathlib import Path
import sys
import unittest

import numpy as np
import polars as pl
from tests import test_reproduce_v12075 as baseline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class V2ReplayContracts(baseline.ReplayContracts):
    def runner(self):
        try:
            return importlib.import_module('reproduce_v12075_v2')
        except ModuleNotFoundError:
            self.fail('isolated v2 runner is missing')

    def test_supervisor_counts_only_owned_group_and_fails_resource_limits(self):
        launcher = importlib.import_module('reproduce_v12075_v2_launch_20261007')
        result = launcher.group_resources('11 99 2048 5.0\n12 99 4096 400.0\n13 100 99999999 700.0\n', 99)
        self.assertEqual(result, {'rss_bytes': 6291456, 'cpu_percent': 405.0, 'processes': 2})
        launcher.enforce_limits(result, 10, 4 * 1024**3)
        with self.assertRaises(RuntimeError):
            launcher.enforce_limits({'rss_bytes': 17 * 1024**3}, 10, 4 * 1024**3)

    def test_group_probe_uses_numeric_ps_even_when_kill_zero_is_unavailable(self):
        launcher = importlib.import_module('reproduce_v12075_v2_launch_20261007')
        from unittest.mock import patch
        probe = getattr(launcher, 'group_exists', None)
        self.assertIsNotNone(probe, 'group existence must not depend on macOS killpg(0) permission')
        with patch('subprocess.check_output', return_value='11 99 2048 5.0\n12 100 4096 2.0\n'):
            self.assertTrue(probe(99))
            self.assertFalse(probe(101))


class LowMemoryBlendTests(unittest.TestCase):
    def helper(self):
        try:
            return importlib.import_module('reproduce_v12075_lowmem_blend')
        except ModuleNotFoundError:
            self.fail('low-memory replay helper is missing')

    def test_categories_match_original_pandas_null_and_sort_semantics(self):
        h = self.helper()
        frame = pl.DataFrame({'airport': ['Z', None, 'A', 'Z'], 'numeric': [2.0, float('nan'), 1.0, None],
                              'unused': ['not', 'a', 'feature', 'column'], 'value': [10, 20, 30, 40]})
        self.assertEqual(h.categories_only(frame, ['airport', 'numeric', 'absent', 'unused'],
                                          ['airport', 'numeric', 'value']),
                         {'airport': ['A', 'Z'], 'numeric': [1.0, 2.0]})

    def test_cannot_skip_encoding_that_model_uses_or_regime_rates(self):
        h = self.helper()
        h.require_unused_encodings(['value', 'ADEP_mvt'], 'day_,tq_,te_,ter_', False)
        for cols, prefix, rates in ((['te_stand', 'value'], 'te_,ter_,tq_', False),
                                    (['value'], 'te_,tq_', False),
                                    (['value'], 'te_,ter_,tq_', True)):
            with self.assertRaises(ValueError):
                h.require_unused_encodings(cols, prefix, rates)

    def test_intermediate_comparison_rejects_changed_members_and_order(self):
        h = self.helper()
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / 'expected.npz', mvt_id=np.array([2, 1]), reg=np.array([4., 5.]))
            np.savez(root / 'actual.npz', mvt_id=np.array([2, 1]), reg=np.array([4., 5.]))
            h.compare_parts(root / 'actual.npz', root / 'expected.npz')
            np.savez(root / 'actual.npz', mvt_id=np.array([1, 2]), reg=np.array([5., 4.]))
            with self.assertRaises(ValueError):
                h.compare_parts(root / 'actual.npz', root / 'expected.npz')


if __name__ == '__main__':
    unittest.main()

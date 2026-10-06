import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import cat_ally_windows as A


class AllyParityTests(unittest.TestCase):
    def test_no_tolerance_for_prediction_difference(self):
        ids = np.array([1., 2.]); pred = np.array([10., 20.])
        self.assertTrue(A.compare(ids, pred, ids, pred)['exact'])
        self.assertFalse(A.compare(ids, pred, ids, pred + 1e-10)['exact'])

    def test_rejects_wrong_population_nonfinite_and_duplicates(self):
        ids = np.array([1., 2.]); pred = np.array([10., 20.])
        for got, values in [(ids[::-1], pred), (ids, np.array([np.nan, 1.])),
                            (ids, pred[:1])]:
            with self.assertRaises(ValueError):
                A.compare(ids, pred, got, values)
        with self.assertRaises(ValueError):
            A.compare(np.array([1., 1.]), pred, np.array([1., 1.]), pred)

    def test_windows_guard_cannot_silently_run_on_other_os(self):
        if sys.platform != 'win32':
            with self.assertRaises(RuntimeError):
                A.windows_limit()

"""Synthetic tests: aggregation, constrained null, alignment, temporal signal."""
import importlib.util
import itertools
from pathlib import Path
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/regime_dependence_audit_20261012.py'


class RegimeDependenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = None
        if SCRIPT.exists():
            spec = importlib.util.spec_from_file_location('regime_dependence', SCRIPT)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def api(self):
        self.assertIsNotNone(self.module, 'regime dependence implementation is missing')
        return self.module

    def test_cluster_pair_sums_match_explicit_ordered_pairs(self):
        a = self.api()
        x = np.array([.3, -.1, .2, -.4, .5])
        day = np.array([0, 0, 0, 1, 1])
        hour = np.array([0, 0, 1, 24, 25])
        got = a.pair_components(x, day, hour, 2)
        for k, same in enumerate([lambda i,j: day[i] == day[j],
                                  lambda i,j: hour[i] == hour[j],
                                  lambda i,j: day[i] == day[j] and hour[i] != hour[j]]):
            wanted = sum(x[i]*x[j] for i in range(5) for j in range(5) if i != j and same(i,j))
            pairs = sum(1 for i in range(5) for j in range(5) if i != j and same(i,j))
            self.assertAlmostEqual(got['numerator'][:, k].sum(), wanted)
            self.assertEqual(got['pairs'][:, k].sum(), pairs)

    def test_finite_stratum_null_equals_exhaustive_permutations(self):
        a = self.api()
        # Two strata, two values each. Cross-stratum products average to zero;
        # the one same-stratum day pair has expected ordered product -2.
        x = np.array([-1., 1., -2., 2.])
        s = np.array([0, 0, 1, 1])
        day = np.array([0, 0, 0, 1])
        hour = np.array([0, 1, 0, 24])
        expected = a.finite_null(x, s, day, hour, 2)
        observed = []
        for p in itertools.permutations([0, 1]):
            for q in itertools.permutations([2, 3]):
                observed.append(a.pair_components(x[list(p)+list(q)], day, hour, 2)['numerator'])
        np.testing.assert_allclose(expected, np.mean(observed, axis=0), atol=1e-12)
        self.assertAlmostEqual(expected[:, 0].sum(), -2.)

    def test_permutation_keeps_whole_values_in_original_strata(self):
        a = self.api()
        x = np.arange(30, dtype=float)
        s = np.repeat(np.arange(3), 10)
        got = a.shuffle_strata(x, s, np.random.default_rng(5))
        for k in range(3):
            np.testing.assert_array_equal(np.sort(got[s == k]), x[s == k])
        self.assertFalse(np.array_equal(got, x))

    def test_injected_day_effect_exceeds_iid_and_permuted_null(self):
        a = self.api()
        rng = np.random.default_rng(144)
        day = np.repeat(np.arange(120), 40)
        hour = day*24 + np.tile(np.repeat([8, 9], 20), 120)
        s = np.zeros(len(day), dtype=int)
        iid = rng.normal(0, .2, len(day)); iid -= iid.mean()
        x = iid + np.repeat(rng.normal(0, .3, 120), 40); x -= x.mean()
        got = a.pair_components(x, day, hour, 120)
        base = a.pair_components(iid, day, hour, 120)
        covariance = got['numerator'][:, 0].sum()/got['pairs'][:, 0].sum()
        iid_cov = base['numerator'][:, 0].sum()/base['pairs'][:, 0].sum()
        null = a.finite_null(x, s, day, hour, 120)[:, 0].sum()/got['pairs'][:, 0].sum()
        self.assertGreater(covariance-null, .05)
        self.assertLess(abs(iid_cov), .003)

    def test_alignment_rejects_duplicates_missing_folds_and_unmatched_ids(self):
        a = self.api()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            a.exact_ids(np.array([1., 1.]), 'fixture')
        with self.assertRaisesRegex(ValueError, 'integral'):
            a.exact_ids(np.array([1., 1.5]), 'fixture')
        with self.assertRaisesRegex(ValueError, 'six'):
            a.validate_fold_names(['fold_1_7.npz'])
        with self.assertRaisesRegex(ValueError, 'ID set'):
            a.alignment_order(np.array([1, 2]), np.array([1, 3]))
        np.testing.assert_array_equal(a.alignment_order(np.array([3, 1, 2]), np.array([1, 2, 3])), [1, 2, 0])

    def test_probability_validation_rejects_nonfinite_and_wrong_simplex(self):
        a = self.api()
        for p in [np.array([[np.nan, 0, 0, 0, 0, 0]]), np.ones((1, 6)), np.array([[1.1, -.1, 0, 0, 0, 0]])]:
            with self.assertRaises(ValueError):
                a.check_probabilities(p)

    def test_stratum_filter_excludes_sparse_or_single_day_groups(self):
        a = self.api()
        strata = np.array([0]*24 + [1]*20 + [2]*8)
        day = np.array(list(range(4))*6 + [5]*20 + list(range(4))*2)
        keep = a.eligible_strata(strata, day, min_n=20, min_days=4)
        np.testing.assert_array_equal(keep, [True]*24+[False]*28)


if __name__ == '__main__':
    unittest.main()

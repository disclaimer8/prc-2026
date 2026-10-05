"""Compare against the actual original function, without torch/LightGBM imports."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from mlp_lowmem_prepare import prepare


def original(cats):
    tree = ast.parse((ROOT / "src/e100_nn_partner.py").read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare")
    scope = {"np": np, "pl": pl, "T": SimpleNamespace(CATEGORICAL=cats)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "original_prepare", "exec"), scope)
    return scope["prepare"]


def stable_counts():
    # Hash iteration order is nondeterministic in both implementations. Give
    # both calls the same counts ordering ONLY inside this comparison fixture.
    counts = pl.Series.value_counts
    def ordered(series, *args, **kwargs):
        return counts(series, *args, **kwargs).sort(series.name, nulls_last=False)
    return patch.object(pl.Series, "value_counts", ordered)


def compare(fit, pred, cats):
    expected = original(cats)(fit, pred, fit.columns)
    actual = prepare(fit, pred, fit.columns, categorical=cats)
    for a, b in zip(actual[:4], expected[:4]):
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        assert a.flags.c_contiguous
        assert a.tobytes() == b.tobytes()
    assert actual[4] == expected[4]
    return actual


def check_quantiles_nonfinite_constant_null_and_thresholds():
    rng = np.random.default_rng(127)
    n = 1025
    values = np.linspace(-1234567.891, 2345678.913, n)
    pred_values = np.r_[values[::4], [-np.inf, np.inf, np.nan, 1e100, -1e100]]
    count = len(pred_values)
    finite99 = np.full(n, np.nan); finite99[:99] = np.arange(99)
    finite100 = np.full(n, np.nan); finite100[:100] = np.arange(100)
    mixed = rng.normal(size=n); mixed[[0, 4, 800]] = [np.nan, np.inf, -np.inf]
    fit = pl.DataFrame({"cat": ["a"] * n, "q257": values, "ties": np.arange(n) % 4,
                        "mixed": mixed, "finite99": finite99, "finite100": finite100,
                        "constant": np.ones(n), "allnull": pl.Series([None] * n, dtype=pl.Float64)})
    pred = pl.DataFrame({"cat": ["a"] * count, "q257": pred_values,
                         "ties": np.linspace(-5, 10, count), "mixed": pred_values,
                         "finite99": pred_values, "finite100": pred_values,
                         "constant": np.full(count, 9.), "allnull": np.full(count, np.nan)})
    actual = compare(fit, pred, ["cat"])
    assert actual[0].shape[1] == 6


def check_category_20_threshold_null_unknown_and_category_order():
    cats = ["exact20"] * 20 + ["rare19"] * 19 + [None] * 20 + ["common"] * 101
    fit = pl.DataFrame({"cat1": cats, "num": np.arange(len(cats), dtype=float),
                        "cat2": ["x"] * len(cats)})
    pred = pl.DataFrame({"cat1": ["exact20", "rare19", None, "common", "unseen"],
                         "num": [0., np.nan, np.inf, -np.inf, 1.], "cat2": ["x"] * 5})
    result = compare(fit, pred, ["cat2", "cat1"])
    assert result[4] == [2, 4]
    assert result[3][1, 1] == result[3][4, 1] == 0
    assert result[3][0, 1] != 0 and result[3][2, 1] != 0


def check_randomized_byte_parity(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(601, 7)); x[rng.random(x.shape) < .11] = np.nan
    z = rng.normal(size=(197, 7)); z[rng.random(z.shape) < .15] = np.inf
    fit = pl.DataFrame({**{f"x{j}": x[:, j] for j in range(7)}, "cat": [str(i % 11) for i in range(len(x))]})
    pred = pl.DataFrame({**{f"x{j}": z[:, j] for j in range(7)}, "cat": [str(i % 13) for i in range(len(z))]})
    compare(fit, pred, ["cat"])


def check_empty_feature_family_preserves_failure(cats, constant):
    fit = pl.DataFrame({"x": np.ones(200) if constant else np.arange(200), "cat": ["a"] * 200})
    cols = fit.columns if cats else ["x"]
    with unittest.TestCase().assertRaises(ValueError):
        original(cats)(fit, fit, cols)
    with unittest.TestCase().assertRaises(ValueError):
        prepare(fit, fit, cols, categorical=cats)


class PrepareParityTests(unittest.TestCase):
    def setUp(self):
        self.counts = stable_counts()
        self.counts.start()
        self.addCleanup(self.counts.stop)

    def test_numeric_edges(self):
        check_quantiles_nonfinite_constant_null_and_thresholds()

    def test_category_edges(self):
        check_category_20_threshold_null_unknown_and_category_order()

    def test_randomized_byte_parity(self):
        for seed in [0, 1, 8, 126]:
            with self.subTest(seed=seed):
                check_randomized_byte_parity(seed)

    def test_empty_family(self):
        for cats, constant in [([], False), (["cat"], True)]:
            with self.subTest(cats=cats, constant=constant):
                check_empty_feature_family_preserves_failure(cats, constant)


if __name__ == "__main__":
    unittest.main()

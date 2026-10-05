"""Probe safety contracts: synthetic only, no torch/real feature loading."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mlp_gpu_prepare_probe as P


class ProbeTests(unittest.TestCase):
    def test_cgroup_requires_ram_cpu_and_no_swap(self):
        good = {"cgroup": "0::/probe.service\n", "memory.max": str(12 * 1024 ** 3),
                "memory.swap.max": "0", "cpu.max": "200000 100000"}
        def read(path):
            return good[path.name]
        with patch.object(Path, "read_text", read):
            self.assertEqual(P.verify_cgroup()["memory_max"], 12 * 1024 ** 3)
            for field, bad in [("memory.max", "max"), ("memory.max", str(13 * 1024 ** 3)),
                               ("memory.swap.max", "1"), ("cpu.max", "max 100000"),
                               ("cpu.max", "300000 100000")]:
                with self.subTest(field=field, bad=bad):
                    old, good[field] = good[field], bad
                    with self.assertRaises(ValueError):
                        P.verify_cgroup()
                    good[field] = old

    def test_worker_stops_before_any_real_fit(self):
        real_fit = Mock(side_effect=AssertionError("fitting forbidden"))
        trainer = SimpleNamespace(T=SimpleNamespace(CATEGORICAL=["cat"]),
                                  add_encodings=lambda *args: args, fit_nn=real_fit)
        trainer.main = lambda: trainer.fit_nn(np.zeros((3, 2)), np.zeros((3, 1)),
                                               np.ones(3), [2], seed=0)
        warm = Mock()
        warm.__matmul__ = Mock(return_value=warm)
        warm.sum.return_value.item.return_value = 0
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, synchronize=lambda: None),
                                version=SimpleNamespace(hip="test"), ones=lambda *a, **k: warm,
                                set_num_threads=lambda n: None, set_num_interop_threads=lambda n: None)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "manifest.json").write_text("{}")
            spec = {"runtime": {"device": "synthetic"}}
            with patch.object(P, "verify_spec", return_value=spec), \
                 patch.object(P, "verify_cgroup", return_value={"test": True}), \
                 patch.object(P, "child_environment", return_value={}), \
                 patch.dict(sys.modules, {"torch": torch, "e100_nn_partner": trainer}), \
                 contextlib.redirect_stdout(io.StringIO()):
                P.worker(work, "a" * 40)
            report = json.loads((work / "probe.json").read_text())
            self.assertEqual(report["fit_calls"], 0)
            self.assertFalse(report["training_authorized_by_this_probe"])
            self.assertEqual(report["fit_shape_numeric"], [3, 2])
            self.assertEqual(report["phases"][-1]["phase"], "before_first_fit_STOP")
            real_fit.assert_not_called()

    def test_report_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.json"
            P.write_json(path, {"first": True})
            with self.assertRaises(FileExistsError):
                P.write_json(path, {"second": True})
            self.assertEqual(json.loads(path.read_text()), {"first": True})


if __name__ == "__main__":
    unittest.main()

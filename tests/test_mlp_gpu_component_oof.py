"""Synthetic split pipeline contracts; no model training or real features."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mlp_gpu_component_oof as M


class SplitContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.reg = "a" * 40
        self.gpu = {"hip": "10", "device": "AMD Radeon RX 7900 XTX", "torch": "synthetic"}
        self.arrays = {"xnf": np.zeros((4096, 2), dtype=np.float32), "xcf": np.zeros((4096, 1), dtype=np.int64),
                       "xnp": np.zeros((3, 2), dtype=np.float32), "xcp": np.zeros((3, 1), dtype=np.int64),
                       "y_scaled": np.full(4096, 1.5, dtype=np.float64),
                       "base_p": np.array([100., 200., 300.]), "control_indices": np.array([2, 0, 1], dtype=np.int64)}
        self.meta = {"mvt_id": np.array([70., 10., 20.]), "y": np.array([310., 110., 210.]),
                     "month": np.array([7, 1, 1], dtype=np.int8)}
        self.spec = {"fold": 1, "sd": 2., "card": [3]}

    def bundle(self):
        path = self.root / "input"
        path.mkdir(); (path / "src").mkdir(); (path / "scripts").mkdir()
        for name in M.SOURCES:
            shutil.copy2(ROOT / "src" / name, path / "src" / name)
        for script in M.launcher_paths():
            shutil.copy2(script, path / "scripts" / script.name)
        for name, array in self.arrays.items():
            np.save(path / f"{name}.npy", array)
        np.savez(path / "metadata.npz", **self.meta)
        (path / "capacity_evidence.json").write_text('{"result":"oom-kill","fits":0}')
        spec = {**self.spec, "protocol": M.PROTOCOL, "registration_commit": self.reg,
                "feature_sha256": M.FEATURE_SHA, "source_sha256": M.SOURCES,
                "fixed_environment": M.FIXED_ENV, "seeds": [0, 1, 2], "gpu_runtime": self.gpu,
                "launcher_sha256": {p.name: M.C.sha(p) for p in M.launcher_paths()},
                "array_sha256": {n: M.C.sha(path / f"{n}.npy") for n in M.ARRAY_NAMES},
                "metadata_sha256": M.C.sha(path / "metadata.npz"),
                "capacity_evidence_sha256": M.C.sha(path / "capacity_evidence.json")}
        M.C.write_json(path / "manifest.json", {"spec": spec, "spec_sha256": M.C.digest(spec)})
        return path

    def test_valid_tensor_contract_and_exact_roundtrip(self):
        path = self.bundle()
        manifest = M.verify_bundle(path, self.reg)
        arrays, meta = M.load_arrays(path, manifest["spec"])
        for name, array in self.arrays.items():
            self.assertEqual(arrays[name].tobytes(), array.tobytes())
        for name, array in self.meta.items():
            self.assertEqual(meta[name].tobytes(), array.tobytes())

    def test_invalid_tensor_shapes_finiteness_codes_and_permutation(self):
        cases = [("xnf", np.full((4096, 2), np.nan, dtype=np.float32)),
                 ("xnp", np.zeros((3, 3), dtype=np.float32)),
                 ("xcf", np.full((4096, 1), 3, dtype=np.int64)),
                 ("xcp", np.full((3, 1), -1, dtype=np.int64)),
                 ("y_scaled", np.ones(4096, dtype=np.float32)),
                 ("base_p", np.array([1., np.inf, 3.])),
                 ("control_indices", np.array([0, 0, 1]))]
        for name, value in cases:
            with self.subTest(name=name), self.assertRaises(M.Invalid):
                M.validate_arrays({**self.arrays, name: value}, self.meta, self.spec)

    def test_invalid_metadata_and_normalization(self):
        for meta in ({**self.meta, "mvt_id": np.array([1., 1., 2.])},
                     {**self.meta, "month": np.array([1, 2, 7])},
                     {**self.meta, "y": np.array([1., np.nan, 2.])}):
            with self.assertRaises(M.Invalid):
                M.validate_arrays(self.arrays, meta, self.spec)
        for sd in [0., float("nan"), -1.]:
            with self.assertRaises(M.Invalid):
                M.validate_arrays(self.arrays, self.meta, {**self.spec, "sd": sd})

    def test_bundle_refuses_tensor_source_and_evidence_changes(self):
        path = self.bundle()
        for relative in ["xnf.npy", "src/e100_nn_partner.py", "capacity_evidence.json", "metadata.npz"]:
            original = (path / relative).read_bytes()
            (path / relative).write_bytes(original + b"changed")
            with self.subTest(relative=relative), self.assertRaises(M.Invalid):
                M.verify_bundle(path, self.reg)
            (path / relative).write_bytes(original)
        with self.assertRaises(M.Invalid):
            M.verify_bundle(path, "b" * 40)

    def test_consumer_original_calls_three_seeds_then_scale_and_id_permutation(self):
        path = self.bundle()
        output = self.root / "complete"
        calls = []
        def fit_nn(xn, xc, y, card, seed):
            calls.append((seed, id(xn), id(xc), id(y)))
            np.testing.assert_array_equal(y, self.arrays["y_scaled"])
            self.assertEqual(card, [3])
            return seed
        trainer = SimpleNamespace(DEV="cuda:0", EPOCHS=8, fit_nn=fit_nn,
                                  predict=lambda net, xn, xc: np.arange(net, net+3, dtype=np.float32))
        warm = Mock(); warm.__matmul__ = Mock(return_value=warm)
        torch = SimpleNamespace(ones=lambda *a, **k: warm, set_num_threads=lambda n: None,
                                set_num_interop_threads=lambda n: None,
                                cuda=SimpleNamespace(synchronize=lambda: None, memory_allocated=lambda: 10,
                                                     memory_reserved=lambda: 20))
        with patch.object(M, "runtime", return_value=self.gpu), patch.object(M, "verify_environment"), \
             patch.object(M, "verify_cgroup", return_value={"memory_max": 12 * 1024 ** 3}), \
             patch.dict(sys.modules, {"torch": torch, "e100_nn_partner": trainer}), \
             patch.object(M.pl, "read_parquet", side_effect=AssertionError("GPU must not read parquet")), \
             contextlib.redirect_stdout(io.StringIO()):
            M.consume(path, output, self.reg)
            first = (output / "prediction.npz").read_bytes()
            M.consume(path, output, self.reg)
        self.assertEqual([c[0] for c in calls], [0, 1, 2])
        self.assertEqual(len(set(c[1:] for c in calls)), 1)
        self.assertIs(trainer.fit_nn, fit_nn)
        with np.load(output / "prediction.npz") as z:
            np.testing.assert_array_equal(z["reg2"], [306., 102., 204.])
            np.testing.assert_array_equal(z["mvt_id"], self.meta["mvt_id"])
        self.assertEqual(first, (output / "prediction.npz").read_bytes())

    def test_output_no_overwrite_invalid_and_tampered(self):
        output, evidence = self.root / "output", {"seeds": [0, 1, 2]}
        for reg in [np.array([1., np.nan, 2.]), np.ones(2)]:
            with self.assertRaises(M.Invalid):
                M.publish(output, reg, self.meta, evidence, {})
            self.assertFalse(output.exists())
        M.publish(output, np.ones(3), self.meta, evidence, {})
        with self.assertRaises(M.Invalid):
            M.publish(output, np.ones(3), self.meta, evidence, {})
        with self.assertRaises(M.Invalid):
            M.verify_complete(output, self.meta, {"seeds": [0]})

    def test_runtime_refuses_cpu_backend_and_version_drift(self):
        M.validate_gpu_runtime(self.gpu, self.gpu)
        for value in [{**self.gpu, "hip": None}, {**self.gpu, "device": "cpu"}, {**self.gpu, "torch": "changed"}]:
            with self.assertRaises(M.Invalid):
                M.validate_gpu_runtime(value, self.gpu)

    def test_environment_drops_unrelated_flags_and_pins_specific_mlp_features(self):
        with patch.dict(os_environ := M.os.environ, {"PROXY_AFFINE": "1", "NN_SEEDS": "9", "PYTHONPATH": "/bad"}):
            env = M.child_environment()
        self.assertNotIn("PROXY_AFFINE", env); self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["NN_DEVICE"], "cuda:0")
        self.assertEqual(env["NN_SEEDS"], "3")
        self.assertEqual(env["DROP_PREFIX"], "day_,tq_,wxr_,month")


if __name__ == "__main__":
    unittest.main()

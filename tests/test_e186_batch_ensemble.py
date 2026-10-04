"""Numerical checks against explicit per-member matrices, not a second vectorized implementation."""
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import torch
import numpy as np
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class BatchEnsembleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)

    def implementation(self):
        self.assertIsNotNone(importlib.util.find_spec("e186_batch_ensemble"), "E186 implementation is missing")
        import e186_batch_ensemble as be
        return be

    def test_layer_matches_explicit_member_matrices_and_gradients(self):
        be = self.implementation()
        layer = be.BatchEnsembleLinear(3, 2, 4, first=True).double()
        x = torch.randn(5, 4, 3, dtype=torch.double, requires_grad=True)
        actual = layer(x)
        expected = torch.stack([
            F.linear(x[:, j], layer.weight * layer.r[j][None, :] * layer.s[j][:, None], layer.bias[j])
            for j in range(4)
        ], dim=1)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        tensors = (x, layer.weight, layer.r, layer.s, layer.bias)
        ga = torch.autograd.grad(actual.square().sum(), tensors, retain_graph=True)
        ge = torch.autograd.grad(expected.square().sum(), tensors)
        for a, e in zip(ga, ge):
            torch.testing.assert_close(a, e, rtol=1e-11, atol=1e-11)
            self.assertTrue(torch.isfinite(a).all())

    def test_loss_does_not_cancel_disagreeing_heads(self):
        be = self.implementation()
        p = torch.tensor([[1.0, -1.0]], requires_grad=True)
        loss = be.individual_mse(p, torch.zeros(1))
        self.assertEqual(loss.item(), 1.0)
        loss.backward()
        torch.testing.assert_close(p.grad, torch.tensor([[1.0, -1.0]]))

    def test_network_has_distinct_heads_and_finite_gradients(self):
        be = self.implementation()
        model = be.BatchEnsembleNet(5, [4, 7], k=4, widths=(12, 12, 6))
        model.eval()
        pred = model(torch.randn(8, 5), torch.stack([torch.randint(4, (8,)), torch.randint(7, (8,))], dim=1))
        self.assertEqual(tuple(pred.shape), (8, 4))
        self.assertGreater(pred.var(dim=1).mean().item(), 1e-7)
        be.individual_mse(pred, torch.randn(8)).backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_rejects_shape_mismatch(self):
        be = self.implementation()
        with self.assertRaises(ValueError):
            be.BatchEnsembleLinear(3, 2, 4)(torch.randn(8, 3))

    def test_gate_rebuilds_stale_regular_and_day_values_before_lobt_clip(self):
        from e186_member_gate import rebuild
        proba = np.array([[0.6, 0.3, 0.1], [0.6, 0.3, 0.1]])
        stale_values = np.array([[5.0, 200.0, 86405.0], [5.0, 200.0, 86405.0]])
        pred, values = rebuild(proba, stale_values, np.array([150.0, 150.0]), np.array([np.nan, 1000.0]))
        np.testing.assert_allclose(pred, [8805.0, 4606.0])
        np.testing.assert_allclose(values[:, 0], [150.0, 150.0])
        np.testing.assert_allclose(values[:, -1], [86550.0, 86550.0])

    def test_ambient_regime_rates_cannot_change_registered_preprocessing(self):
        import e186_nn_cv as cv  # No e100/train import or dataset access occurs here.
        self.assertTrue(hasattr(cv, "pin_preprocessing_environment"), "Missing preprocessing environment pin")
        with patch.dict(os.environ, {"REGIME_RATES": "1"}):
            recorded = cv.pin_preprocessing_environment("data/features_fin1b.parquet")
            self.assertEqual(cv.CONFIG["regime_rates"], "0")
            self.assertEqual(os.environ["REGIME_RATES"], "0")
            self.assertEqual(recorded["REGIME_RATES"], "0")


class BackendTests(unittest.TestCase):
    def runtime(self):
        self.assertIsNotNone(importlib.util.find_spec("e186_runtime"), "Missing portable E186 runtime")
        import e186_runtime
        return e186_runtime

    def test_auto_selects_largest_gpu_and_explicit_index_is_respected(self):
        rt = self.runtime()
        props = [SimpleNamespace(total_memory=512 * 1024 ** 2),
                 SimpleNamespace(total_memory=24 * 1024 ** 3)]
        # Hardware discovery must be faked on the CPU/MPS test host.
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "device_count", return_value=2), \
                patch.object(torch.cuda, "get_device_properties", side_effect=lambda i: props[i]):
            self.assertEqual(str(rt.select_device("auto")), "cuda:1")
            self.assertEqual(str(rt.select_device("cuda")), "cuda:1")
            self.assertEqual(str(rt.select_device("cuda:0")), "cuda:0")
            with self.assertRaises(ValueError):
                rt.select_device("cuda:2")

    def test_cv_rejects_cpu_and_unavailable_backend_instead_of_falling_back(self):
        rt = self.runtime()
        with patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(torch.backends.mps, "is_available", return_value=False):
            for request in ("auto", "cuda", "cuda:0", "mps", "cpu"):
                with self.subTest(request=request), self.assertRaises((ValueError, RuntimeError)):
                    rt.select_device(request, require_accelerator=True)
            self.assertEqual(str(rt.select_device("cpu")), "cpu")
        with self.assertRaises(ValueError):
            rt.select_device("xpu")
        with self.assertRaises(ValueError):
            rt.select_device("cpu:1")

    def test_auto_retains_mps_support(self):
        rt = self.runtime()
        with patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(torch.backends.mps, "is_available", return_value=True):
            self.assertEqual(str(rt.select_device("auto", require_accelerator=True)), "mps")

    def test_provenance_distinguishes_rocm_hardware_and_torch(self):
        rt = self.runtime()
        props = SimpleNamespace(name="AMD Radeon RX 7900 XTX", total_memory=24 * 1024 ** 3,
                                gcnArchName="gfx1100", major=11, minor=0)
        with patch.object(torch.cuda, "get_device_properties", return_value=props), \
                patch.object(torch.version, "hip", "10.0.0"), \
                patch.object(torch.cuda, "get_device_capability", return_value=(11, 0)):
            info = rt.backend_provenance(torch.device("cuda:0"))
        self.assertEqual(info["backend"], "rocm")
        self.assertEqual(info["torch"], torch.__version__)
        self.assertEqual(info["hip"], "10.0.0")
        self.assertEqual(info["device"], "cuda:0")
        self.assertEqual(info["hardware"]["name"], "AMD Radeon RX 7900 XTX")
        self.assertEqual(info["hardware"]["total_memory_bytes"], 25769803776)
        self.assertEqual(info["hardware"]["gcn_arch"], "gfx1100")
        cpu = rt.backend_provenance(torch.device("cpu"))
        self.assertNotEqual(info, cpu)
        self.assertEqual(cpu["backend"], "cpu")

    def test_memory_guard_rejects_over_limit_before_work(self):
        rt = self.runtime()
        divisor = 1024 ** 3 if sys.platform == "darwin" else 1024 ** 2
        with patch.object(rt.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=17 * divisor)):
            with self.assertRaisesRegex(MemoryError, "preprocessing"):
                rt.memory_guard(torch.device("cpu"), max_rss_gib=16, max_device_gib=8, phase="before preprocessing")
        with patch.object(rt.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=divisor)), \
                patch.object(torch.cuda, "memory_allocated", return_value=9 * 1024 ** 3), \
                patch.object(torch.cuda, "memory_reserved", return_value=9 * 1024 ** 3):
            with self.assertRaises(MemoryError):
                rt.memory_guard(torch.device("cuda:0"), max_rss_gib=16, max_device_gib=8, phase="training")

    def test_resume_rejects_finite_member_with_wrong_shape(self):
        import e186_nn_cv as cv
        self.assertTrue(hasattr(cv, "validate_saved_fold"), "Missing complete fold resume validation")
        ids, truth = np.array([3, 4]), np.array([10.0, 20.0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fold.npz"
            np.savez(path, mvt_id=ids, y=truth, reg2=np.array([12.0]), spec_sha256=np.array("same"))
            with self.assertRaises(RuntimeError):
                cv.validate_saved_fold(path, "same", ids, truth)
            np.savez(path, mvt_id=ids, y=truth, reg2=np.array([12.0, 21.0]), spec_sha256=np.array("same"))
            cv.validate_saved_fold(path, "same", ids, truth)

    def test_manifest_is_checked_after_writer_lock_acquisition(self):
        import e186_nn_cv as cv
        self.assertTrue(hasattr(cv, "locked_manifest"), "Missing locked manifest validation")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_manifest.json"
            # Reproduce a competing writer finishing before this writer acquires its lock.
            real_open = os.open
            def open_after_other_writer(lock, flags, mode):
                if str(lock).endswith(".writer.lock"):
                    path.write_text('{"spec_sha256": "other"}')
                return real_open(lock, flags, mode)
            with patch.object(cv.os, "open", side_effect=open_after_other_writer):
                with self.assertRaises(RuntimeError):
                    with cv.locked_manifest(path, {"spec_sha256": "ours"}):
                        self.fail("Incompatible writer manifest was accepted")
            self.assertFalse((Path(directory) / ".writer.lock").exists())


    def test_prediction_uses_model_device_and_averages_individual_heads(self):
        import e186_nn_cv as cv
        from e186_batch_ensemble import BatchEnsembleNet
        torch.manual_seed(21)
        model = BatchEnsembleNet(3, [5], k=4, widths=(8, 8, 4)).eval()
        xn = np.arange(21, dtype=np.float32).reshape(7, 3) / 20
        xc = np.arange(7, dtype=np.int64).reshape(7, 1) % 5
        with torch.no_grad():
            expected = model(torch.from_numpy(xn), torch.from_numpy(xc)).mean(dim=1).numpy()
        with patch.dict(cv.CONFIG, {"predict_batch_size": 3}):
            actual = cv.predict(model, xn, xc)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()

"""Resource isolation checks; no PRC fits and no process signals."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class ResourceCVTests(unittest.TestCase):
    def implementation(self):
        self.assertIsNotNone(importlib.util.find_spec('e184_resource_cv'), 'resource-only launcher missing')
        import e184_resource_cv
        return e184_resource_cv

    def test_environment_removes_ambient_model_changes_and_pins_real_thread_limits(self):
        r = self.implementation()
        with patch.dict(os.environ, {'REG_OVERRIDE': '{"num_leaves":2}', 'FOLDS': '1',
                                    'OMP_NUM_THREADS': '24', 'POLARS_MAX_THREADS': '10'}):
            env = r.worker_environment('/features.parquet', '/new-oof')
        self.assertNotIn('REG_OVERRIDE', env)
        self.assertNotIn('FOLDS', env)
        self.assertEqual(env['FEATURES'], '/features.parquet')
        self.assertEqual(env['CV_OUT'], '/new-oof')
        for name in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'POLARS_MAX_THREADS'):
            self.assertEqual(env[name], '6')
        self.assertEqual(env['OMP_WAIT_POLICY'], 'PASSIVE')
        self.assertEqual(env['GOMP_SPINCOUNT'], '0')
        self.assertEqual(env['CLF_COST_WEIGHT'], '1')
        self.assertEqual(env['CLF_COST_MIN_SPREAD'], '3600')

    def test_worker_patch_changes_only_thread_counts_and_keeps_runner_body(self):
        r = self.implementation()
        original = 'import train as T\nresult = (T.REG_PARAMS.copy(), T.CLF_PARAMS.copy())\n'
        module = SimpleNamespace(REG_PARAMS={'num_threads': 10, 'seed': 42, 'num_leaves': 511},
                                 CLF_PARAMS={'num_threads': 10, 'num_class': 6, 'num_leaves': 127})
        program = r.resource_program(original)
        with patch.dict(sys.modules, {'train': module}):
            namespace = {}
            exec(program, namespace)
        self.assertEqual(namespace['result'], ({'num_threads': 6, 'seed': 42, 'num_leaves': 511},
                                               {'num_threads': 6, 'num_class': 6, 'num_leaves': 127}))
        self.assertTrue(program.endswith(original.split('\n', 1)[1]))
        for wrong in ('pass\n', original + 'import train as T\n'):
            with self.assertRaises(ValueError):
                r.resource_program(wrong)

    def test_nested_runner_is_extracted_without_executing_controller(self):
        r = self.implementation()
        source = 'raise RuntimeError("controller must not execute")\nRUNNER = "import train as T\\npass\\n"\n'
        self.assertEqual(r.extract_runner(source), 'import train as T\npass\n')

    def test_snapshot_copies_exact_sources_and_rejects_reuse(self):
        r = self.implementation()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'root'
            (root / 'src').mkdir(parents=True)
            (root / 'data').mkdir()
            for name in r.SOURCE_NAMES:
                (root / 'src' / name).write_text('original ' + name)
            feature = root / 'data/features_fin1b_e184.parquet'
            feature.write_bytes(b'feature fixture')
            snapshot = root / 'data/isolated'
            out = root / 'data/new_oof'
            manifest = r.prepare_snapshot(root, snapshot, out, 'a' * 40)
            (root / 'src/train.py').write_text('changed later')
            self.assertEqual((snapshot / 'src/train.py').read_text(), 'original train.py')
            self.assertEqual((snapshot / 'data/features_fin1b_e184.parquet').read_bytes(), b'feature fixture')
            self.assertEqual(json.loads((snapshot / 'manifest.json').read_text()), manifest)
            with self.assertRaises(FileExistsError):
                r.prepare_snapshot(root, snapshot, out, 'a' * 40)
            self.assertFalse((root / 'data/cv_preds_e184_hz').exists())

    def test_six_cpu_affinity_required_and_old_process_must_be_gone(self):
        r = self.implementation()
        with patch.object(r.os, 'sched_getaffinity', return_value={0, 2, 4, 6, 8, 10}, create=True):
            r.validate_launch([], proc_root=Path('/does-not-exist'))
        with patch.object(r.os, 'sched_getaffinity', return_value=set(range(12)), create=True):
            with self.assertRaises(RuntimeError):
                r.validate_launch([], proc_root=Path('/does-not-exist'))
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            (proc / '42').mkdir()
            with patch.object(r.os, 'sched_getaffinity', return_value={0, 2, 4, 6, 8, 10}, create=True):
                with self.assertRaises(RuntimeError):
                    r.validate_launch([42], proc_root=proc)


if __name__ == '__main__':
    unittest.main()

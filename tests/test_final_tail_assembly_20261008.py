"""Synthetic contracts for the final tail assembly adapter. No competition rows are opened here.

Run from the repo root:
    .venv/bin/python -m unittest discover -s tests -p 'test_final_tail_assembly_20261008.py'
"""
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR', 'load_as_storage')
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / 'scripts/final_tail_assembly_20261008.py'
spec = importlib.util.spec_from_file_location('finaltail', FILE)
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
ARCHIVE_SRC = ROOT / A.ARCHIVE / 'src'
ID, Y = A.ID, A.Y
UTC = pl.Datetime('us', 'UTC')


def frozen_available(*names):
    return all((ARCHIVE_SRC / n).is_file() for n in names)


class PlanContracts(unittest.TestCase):
    def test_literal_plan_passes_and_forbidden_components_fail(self):
        plan = A.check_plan(A.stage_plan())
        self.assertEqual([s['name'] for s in plan][3], 'e156')
        self.assertNotIn('E81', json.dumps(plan))
        bad = copy.deepcopy(plan)
        bad.insert(9, dict(name='e81', kind='script', args=['blend_submissions.py', 1], env={'BLEND_PARTS': '44:0.1214,60:0.0270'}))
        with self.assertRaises(ValueError):
            A.check_plan(bad)
        bad = copy.deepcopy(plan)
        bad[0]['weights']['cat'] = .7
        with self.assertRaises(ValueError):
            A.check_plan(bad)
        bad = copy.deepcopy(plan)
        bad[0]['weights']['members'] += ',submissions/nn_rank.npz:0.25'
        with self.assertRaises(ValueError):
            A.check_plan(bad)
        bad = copy.deepcopy(plan)
        del bad[3]
        with self.assertRaises(ValueError):
            A.check_plan(bad)
        bad = copy.deepcopy(plan)
        bad[3]['env']['TABPFN_SEEDS'] = '0,1,2,3,4'
        with self.assertRaises(ValueError):
            A.check_plan(bad)


class MemberContracts(unittest.TestCase):
    def npz(self, tmp, name, ids, per_seed, reg=None, **extra):
        p = Path(tmp) / name
        np.savez(p, mvt_id=ids, reg=per_seed.mean(axis=0) if reg is None else reg, per_seed=per_seed, **extra)
        return p

    def test_member_table_rejects_partial_duplicate_or_inconsistent_seed_families(self):
        ids = np.array([3., 1., 2.])
        with tempfile.TemporaryDirectory() as tmp:
            ok = self.npz(tmp, 'ok.npz', ids, np.arange(15.).reshape(5, 3))
            got_ids, reg, receipt = A.member_table(ok, 5)
            self.assertEqual(receipt['seeds'], 5)
            np.testing.assert_array_equal(got_ids, ids)
            with self.assertRaises(ValueError):
                A.member_table(self.npz(tmp, 'partial.npz', ids, np.arange(12.).reshape(4, 3)), 5)
            with self.assertRaises(ValueError):
                A.member_table(self.npz(tmp, 'mixed.npz', ids, np.arange(15.).reshape(5, 3), reg=np.array([0., 0., 0.])), 5)
            with self.assertRaises(ValueError):
                A.member_table(self.npz(tmp, 'dup.npz', np.array([1., 1., 2.]), np.arange(15.).reshape(5, 3)), 5)
            with self.assertRaises(ValueError):
                A.member_table(self.npz(tmp, 'incomplete.npz', ids, np.arange(15.).reshape(5, 3), complete=np.array(False)), 5)
            p = Path(tmp) / 'noseeds.npz'
            np.savez(p, mvt_id=ids, reg=np.ones(3))
            with self.assertRaises(ValueError):
                A.member_table(p, 5)
            A.member_table(p, 3, require_per_seed=False)

    def test_align_follows_prediction_order_and_requires_full_coverage(self):
        ids = np.array([10., 20., 30.])
        member_ids, values = np.array([30., 10., 20.]), np.array([3., 1., 2.])
        np.testing.assert_array_equal(A.align(ids, member_ids, values), [1., 2., 3.])
        with self.assertRaises(ValueError):
            A.align(ids, np.array([30., 10.]), np.array([3., 1.]))
        with self.assertRaises(ValueError):
            A.align(np.array([10., 20., 40.]), member_ids, values)


class CompletionReceipts(unittest.TestCase):
    def test_nn_ten_seed_receipt_and_sha(self):
        ten = dict(seeds=[3, 4, 5, 6, 7, 8, 9], all_seeds=list(range(10)), files={'prediction_10seed.npz': 'a' * 64})
        A.check_completion('nn', ten, 10, 'prediction_10seed.npz', 'a' * 64)
        with self.assertRaises(ValueError):
            A.check_completion('nn', ten, 3, 'prediction_10seed.npz', 'a' * 64)
        with self.assertRaises(ValueError):
            A.check_completion('nn', ten, 10, 'prediction_10seed.npz', 'b' * 64)
        with self.assertRaises(ValueError):
            A.check_completion('nn', dict(seeds=[3, 4, 5, 6, 7, 8, 9]), 10, 'p.npz', 'a' * 64)
        A.check_completion('nn', dict(seeds=[0, 1, 2]), 3, 'prediction.npz', 'c' * 64)
        A.check_completion('cat', dict(fits=[dict(seed=i) for i in range(15)]), 15, 'predictions.npz', 'd' * 64)


class PremixArithmetic(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        n = 40
        self.ids = np.arange(n, dtype=float) + 1
        self.proba = rng.dirichlet(np.ones(6), size=n)
        self.cat, self.lgb = rng.uniform(300, 1500, n), rng.uniform(300, 1500, n)
        self.members = [(w, rng.uniform(300, 1500, n)) for _, w in A.ANCHORS] + [(A.NN_W, rng.uniform(300, 1500, n))]
        cand = rng.uniform(200, 2000, (n, 4))
        cand[rng.random((n, 4)) < .3] = np.nan
        self.cands = [cand[:, j] for j in range(4)]
        self.d_lobt = cand[:, 2].copy()
        self.d_lobt[5] = 87000.  # LOBT a day before take-off: the r + DAY candidate sits at the window edge

    def reference(self, proba, r_cat, r_lgb):
        """Literal blend.py expressions with the frozen train helpers."""
        sys.path.insert(0, str(ARCHIVE_SRC))
        import train as T
        r = .65 * r_cat + (1 - .65) * r_lgb
        mem = [(p, float(w)) for p, (w, _) in zip(range(6), self.members)]
        regs = [g for _, g in self.members]
        r = (1 - sum(w_ for _, w_ in mem)) * r + sum(w_ * g for (_, w_), g in zip(mem, regs))
        values = np.column_stack([r] + self.cands + [r + T.DAY])
        frame = pl.DataFrame({'d_mvt_lobt': self.d_lobt})
        p = np.where(np.isnan(values), 0.0, proba)
        p = T.mask_infeasible(p, values, frame)
        p = p / p.sum(axis=1, keepdims=True)
        pred = np.clip(T.lobt_clip(np.sum(p * np.nan_to_num(values), axis=1), frame), 0, None)
        return dict(reg=r, proba=p, values=values, pred=pred)

    @unittest.skipUnless(frozen_available('train.py', 'features.py', 'target_encoding.py'), 'frozen archive absent')
    def test_all_parts_match_frozen_blend_expressions_exactly(self):
        got = A.premix_arrays(self.proba, self.cat, self.lgb, self.members, self.cands, self.d_lobt)
        want = self.reference(self.proba, self.cat, self.lgb)
        for key in ('reg', 'proba', 'values', 'pred'):
            np.testing.assert_array_equal(got[key], want[key], err_msg=key)
        np.testing.assert_array_equal(np.round(got['pred']), np.round(want['pred']))

    def test_masked_base_probabilities_cannot_replace_raw_classifier_output(self):
        # Base candidate r0 = 5000 masks the day-shift column on row 5 (|91400 - 87000| > 3606) and that zero is permanent;
        # the blended r = 500 puts r + DAY inside the window, so only the raw classifier output keeps the mass there.
        n = len(self.ids)
        cands = [c.copy() for c in self.cands]
        base = A.premix_arrays(self.proba, np.full(n, 5000.), np.full(n, 5000.), [(w, np.full(n, 5000.)) for w, _ in self.members], cands, self.d_lobt)
        self.assertEqual(base['proba'][5, 5], 0.)
        members = [(w, np.full(n, 500.)) for w, _ in self.members]
        raw = A.premix_arrays(self.proba, np.full(n, 500.), np.full(n, 500.), members, cands, self.d_lobt)
        reused = A.premix_arrays(base['proba'], np.full(n, 500.), np.full(n, 500.), members, cands, self.d_lobt)
        self.assertGreater(raw['proba'][5, 5], 0.)
        self.assertEqual(reused['proba'][5, 5], 0.)
        self.assertFalse(np.array_equal(raw['proba'][5], reused['proba'][5]))
        self.assertNotEqual(np.sum(raw['proba'][5] * np.nan_to_num(raw['values'][5])), np.sum(reused['proba'][5] * np.nan_to_num(reused['values'][5])))


class ProtectionContracts(unittest.TestCase):
    def test_protect_restores_served_rows_even_when_tab_equaled_its_input(self):
        schema = {ID: pl.Float64, Y: pl.Int32}
        tab = pl.DataFrame({ID: [1., 2., 3., 4.], Y: [100, 200, 300, 400]}, schema=schema)
        final = pl.DataFrame({ID: [4., 3., 2., 1.], Y: [999, 300, 250, 100]}, schema=schema)  # row 2 and 4 moved later
        out = A.protect(final, tab, np.array([2., 4.]))
        self.assertEqual(out.columns, final.columns)
        self.assertEqual(dict(zip(out[ID].to_list(), out[Y].to_list())), {1.: 100, 2.: 200, 3.: 300, 4.: 400})
        self.assertEqual(out.schema[Y], pl.Int32)
        with self.assertRaises(ValueError):
            A.protect(final, tab, np.array([2., 2.]))
        with self.assertRaises(ValueError):
            A.protect(final, tab, np.array([2., 9.]))
        with self.assertRaises(ValueError):
            A.protect(final, tab, np.array([]))


class SourceTransforms(unittest.TestCase):
    @unittest.skipUnless(frozen_available('e159b_apply.py'), 'frozen archive absent')
    def test_e159b_row_count_transformation_has_single_match_scope(self):
        source = (ARCHIVE_SRC / 'e159b_apply.py').read_text()
        out = A.transform_e159b(source, 670790)
        self.assertEqual(out.count('670790'), 1)
        self.assertNotIn('344841', out)
        self.assertEqual(len(out.splitlines()), len(source.splitlines()))
        compile(out, 'e159b', 'exec')
        with self.assertRaises(ValueError):
            A.transform_e159b(source + '\nX = 344841\n', 670790)
        with self.assertRaises(ValueError):
            A.transform_e159b('print(1)\n', 670790)

    def test_e188_dynamic_rows_and_registration_bridge(self):
        source = (ROOT / A.E188_SCRIPT).read_text()
        out = A.transform_e188(source, 670790)
        self.assertIn('ROWS = 670790', out)
        self.assertEqual(out.count('def validate_registration'), 1)
        self.assertEqual(out.count('E188_BRIDGE'), 1)
        self.assertNotIn('git', out.split('def load_lgb')[0].split('def validate_registration')[1])
        compile(out, 'e188', 'exec')
        with self.assertRaises(ValueError):
            A.transform_e188(source.replace('ROWS = 344841', 'ROWS = 1'), 670790)
        with self.assertRaises(ValueError):
            A.transform_e188(source + '\ndef validate_registration(root, commit):\n    return {}\n', 670790)


class CachedStageWorker(unittest.TestCase):
    """A fake frozen stage with lgb.train; replay must reuse exact models and refuse any drift."""

    STAGE = ('import os, sys\nimport lightgbm as lgb\nimport numpy as np\n'
             'FEATS = ["ap", "x1"]\n'
             'data = np.load("data/train.npz")\n'
             'models = [lgb.train({"objective": "regression", "verbose": -1, "seed": s, "num_threads": 1, "deterministic": True, '
             '"min_data_in_leaf": 1, "min_data_in_bin": 1}, lgb.Dataset(data["x"], data["y"], categorical_feature=[0]), 3) for s in (1, 2)]\n'
             'open("submissions/out.txt", "w").write(str(sum(float(m.predict(data["x"]).sum()) for m in models)))\n')

    def make_work(self, tmp, y_shift=0.0):
        work = Path(tmp) / 'work'
        for d in ('src', 'data', 'submissions', 'jobs', 'logs'):
            (work / d).mkdir(parents=True)
        (work / 'src/fake_stage.py').write_text(self.STAGE)
        rng = np.random.default_rng(0)
        x = np.column_stack([rng.integers(0, 3, 60).astype(float), rng.normal(size=60)])
        np.savez(work / 'data/train.npz', x=x, y=x[:, 1] * 2 + y_shift)
        return work

    def run_worker(self, work, cache, mode):
        job = work / 'jobs' / f'{mode}.json'
        A.save(job, dict(mode=mode, name='fake', args=['fake_stage.py'], expected_calls=2, feature_global='FEATS', cache=str(cache),
                         refit_cache=str(work / 'models/fake'), source_sha256=A.sha(work / 'src/fake_stage.py'), adapter_sha256=A.sha(FILE)))
        env = {**A.environment(1), 'PYTHONPATH': ''}
        return subprocess.run([sys.executable, str(FILE), '_stage', '--work', str(work), '--job', str(job)], cwd=work, env=env,
                              capture_output=True, text=True), job

    def test_replay_exact_reuse_then_signature_drift_refusal_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = self.make_work(tmp)
            # Build the reference cache once through the explicit refit path (needs a stub complete.json with two calls).
            stub = Path(tmp) / 'stub'
            stub.mkdir()
            A.save(stub / 'complete.json', dict(calls=[dict(signature=dict(feature_names=['ap', 'x1']), sha256='0' * 64)] * 2))
            result, job = self.run_worker(work, stub, 'refit')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            refit = A.read(job.with_suffix('.complete.json'))
            self.assertEqual((refit['fits_performed'], refit['cached_calls']), (2, 0))
            cache = work / 'models/fake'
            stored = A.read(cache / 'complete.json')
            self.assertEqual(len(stored['calls']), 2)
            self.assertFalse(stored['calls'][0]['cached_signature_match'])
            out_refit = (work / 'submissions/out.txt').read_text()
            (work / 'submissions/out.txt').unlink()
            # Exact replay: identical arrays, zero fits, identical stage output.
            work2 = self.make_work(Path(tmp) / 'two')
            result, job = self.run_worker(work2, cache, 'replay')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            replay = A.read(job.with_suffix('.complete.json'))
            self.assertEqual((replay['fits_performed'], replay['cached_calls']), (0, 2))
            self.assertEqual((work2 / 'submissions/out.txt').read_text(), out_refit)
            # Drifted labels: replay must fail before any model is trained or output written.
            work3 = self.make_work(Path(tmp) / 'three', y_shift=1.0)
            result, job = self.run_worker(work3, cache, 'replay')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('training signature differs', result.stderr)
            self.assertIn('label', result.stderr)
            self.assertFalse((work3 / 'submissions/out.txt').exists())
            self.assertFalse(job.with_suffix('.complete.json').exists())
            # Tampered cached model file: refused.
            work4 = self.make_work(Path(tmp) / 'four')
            (cache / 'model_0.txt').write_text((cache / 'model_0.txt').read_text() + '\n')
            result, job = self.run_worker(work4, cache, 'replay')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('cached stage model changed', result.stderr)

    def test_unknown_mode_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                A.cached_stage(tmp, 4, dict(name='e141', cache='e141', args=['e141_apply.py', 1, 2], env={}), 'capture')


class FinalizeContracts(unittest.TestCase):
    def make_work(self, tmp, rows=6):
        work = Path(tmp) / 'work'
        for d in ('submissions', 'data/prc-2026-datasets', 'e156/run_0', 'receipts'):
            (work / d).mkdir(parents=True)
        ids = np.arange(rows, dtype=float) + 100
        pl.DataFrame({ID: ids[::-1], Y: pl.Series([None] * rows, dtype=pl.Int32)}).write_parquet(work / 'data/prc-2026-datasets/submitting.parquet')
        A.save(work / 'sealed.json', dict(rows=rows, output_name='gentle-igloo_vtest.parquet'))
        A.save(work / 'e156/run_0/served.json', dict(ids=[101., 103.], values=[5000, 7000], ids_sha256='x'))
        return work, ids

    def test_schema_ids_nonnegative_integers_and_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, ids = self.make_work(tmp)
            good = pl.DataFrame({ID: ids, Y: pl.Series([1, 5000, 2, 7000, 3, 4], dtype=pl.Int32)})
            good.write_parquet(A.sub(work, 1))
            out = A.finalize(work, 1)
            got = pl.read_parquet(out)
            self.assertEqual(got[ID].to_list(), ids[::-1].tolist())  # template order
            self.assertEqual(got.schema, {ID: pl.Float64, Y: pl.Int32})
            with self.assertRaises(FileExistsError):
                A.finalize(work, 1)
            bad = good.with_columns(pl.col(Y).cast(pl.Int64))
            bad.write_parquet(A.sub(work, 2))
            with self.assertRaises(ValueError):
                A.finalize(work, 2)
            good.with_columns(pl.Series(Y, [1, 5000, -2, 7000, 3, 4], dtype=pl.Int32)).write_parquet(A.sub(work, 3))
            with self.assertRaises(ValueError):
                A.finalize(work, 3)
            good.with_columns(pl.Series(Y, [1, 5001, 2, 7000, 3, 4], dtype=pl.Int32)).write_parquet(A.sub(work, 4))
            with self.assertRaises(ValueError):
                A.finalize(work, 4)  # protected E156 value overwritten downstream
            good.head(5).write_parquet(A.sub(work, 5))
            with self.assertRaises(ValueError):
                A.finalize(work, 5)

    def test_assemble_refuses_executed_or_missing_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, _ = self.make_work(tmp)
            A.save(work / 'receipts/00_premix.json', dict(stage='premix'))
            with self.assertRaises(FileExistsError):
                A.assemble(work)


class ServedPredicate(unittest.TestCase):
    @unittest.skipUnless(frozen_available('lirf_table.py', 'train.py', 'features.py', 'target_encoding.py'), 'frozen archive absent')
    def test_bins_and_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / 'src').mkdir()
            (work / 'data').mkdir()
            for n in ('lirf_table.py', 'train.py', 'features.py', 'target_encoding.py'):
                shutil.copy2(ARCHIVE_SRC / n, work / 'src' / n)
            hours = [3, 5, 10, 20, 40, 5, 5, 5]
            pl.DataFrame({ID: [1., 2., 3., 4., 5., 6., 7., 8.], 'split': ['rank'] * 7 + ['train'],
                          'ADEP_mvt': ['LIRF'] * 6 + ['EDDF', 'LIRF'], 'FLIGHT_ID_mvt': [None] * 5 + ['NM', None, None],
                          'd_mvt_sched': [h * 3600. for h in hours]}).write_parquet(work / 'data/features_e106.parquet')
            ids, bins = A.served_predicate(work)
            self.assertEqual(ids.tolist(), [2., 3., 4.])
            self.assertEqual(bins, ['4-8h', '8-12h', '12-30h'])


class OpdiInventory(unittest.TestCase):
    def test_extended_set_is_pinned_by_unique_basename_and_default_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'data/opdi').mkdir(parents=True)
            (root / 'data/opdi_final').mkdir()
            for n in ('data/opdi/ev_20260130_20260209.parquet', 'data/opdi/ev_20260629_20260709.parquet', 'data/opdi/flight_list_202601.parquet',
                      'data/opdi_final/ev_lszh_20260209_20260219.parquet', 'data/opdi_final/ev_lszh_20260130_20260209.parquet', 'data/opdi_final/manifest.json'):
                (root / n).write_bytes(b'x')
            # default: only the held files, flight lists are not event files
            self.assertEqual(A.opdi_inventory(root), ['data/opdi/ev_20260130_20260209.parquet', 'data/opdi/ev_20260629_20260709.parquet'])
            # extended: the two directories together, same window under a different basename is allowed (opdi_wide takes min t per flight/type)
            names = A.opdi_inventory(root, 'data/opdi_final')
            self.assertEqual(names[2:], ['data/opdi_final/ev_lszh_20260130_20260209.parquet', 'data/opdi_final/ev_lszh_20260209_20260219.parquet'])
            self.assertEqual(len({Path(n).name for n in names}), 4)
            # an extended file whose basename shadows a held file would be a silent replacement in <work>/data/opdi: refused
            (root / 'data/opdi_final/ev_20260130_20260209.parquet').write_bytes(b'y')
            with self.assertRaises(ValueError):
                A.opdi_inventory(root, 'data/opdi_final')
            (root / 'data/opdi_final/ev_20260130_20260209.parquet').unlink()
            # an empty extended directory is a configuration error, not "no extra files"
            (root / 'data/opdi_empty').mkdir()
            with self.assertRaises(ValueError):
                A.opdi_inventory(root, 'data/opdi_empty')


class FrozenStageIdentity(unittest.TestCase):
    """Unchanged E185/E167 sources on synthetic rows: only their declared subsets move."""

    def frozen_work(self, tmp, names):
        work = Path(tmp) / 'work'
        for d in ('src', 'data/prc-2026-datasets', 'submissions'):
            (work / d).mkdir(parents=True)
        for n in names:
            shutil.copy2(ARCHIVE_SRC / n, work / 'src' / n)
        return work

    def parts(self, work, version, ids, cands, d_lobt, r):
        proba = np.full((len(ids), 6), 1 / 6)
        parts = A.premix_arrays(proba, r, r, [(w, r) for _, w in A.ANCHORS] + [(A.NN_W, r)], cands, d_lobt)
        np.savez(work / f'submissions/parts_v{version}.npz', mvt_id=ids, **parts)
        pl.DataFrame({ID: ids, Y: pl.Series(np.round(parts['pred']), dtype=pl.Float64).cast(pl.Int32)}).write_parquet(A.sub(work, version))
        return parts

    def run_stage(self, work, args, env):
        return subprocess.run([sys.executable, 'src/' + args[0], *map(str, args[1:])], cwd=work, env={**A.environment(1), **env},
                              capture_output=True, text=True)

    @unittest.skipUnless(frozen_available('e185_apply.py', 'train.py', 'features.py', 'target_encoding.py'), 'frozen archive absent')
    def test_e185_moves_only_lobt_null_rows_outside_lirf(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = self.frozen_work(tmp, ['e185_apply.py', 'train.py', 'features.py', 'target_encoding.py'])
            ids = np.array([1., 2., 3., 4.])
            d_lobt = np.array([np.nan, 900., np.nan, np.nan])
            cands = [np.array([800., 850., 700., 900.]), np.full(4, np.nan), d_lobt, np.full(4, np.nan)]
            r = np.array([600., 650., 500., 700.])
            before = self.parts(work, 1, ids, cands, d_lobt, r)
            pl.DataFrame({ID: ids, 'split': ['rank'] * 4, 'ADEP_mvt': ['EDDF', 'EDDF', 'LIRF', 'EHAM'],
                          'LOBT_flt': pl.Series([None, datetime(2026, 1, 1, tzinfo=timezone.utc), None, None], dtype=UTC),
                          'd_mvt_lobt': d_lobt}).write_parquet(work / 'data/features_e106.parquet')
            np.savez(work / 'data/motion.npz', mvt_id=ids, reg=np.array([1000., 1000., 1000., 1000.]))
            result = self.run_stage(work, ['e185_apply.py', 1, 2], {'E185_W': str(A.E185_W), 'E185_RANK': 'data/motion.npz'})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            after = np.load(work / 'submissions/parts_v2.npz')
            subset = np.array([True, False, False, True])  # LOBT null and not LIRF (months/airports routed identically)
            np.testing.assert_array_equal(after['pred'][~subset], before['pred'][~subset])
            np.testing.assert_array_equal(after['reg'][subset], (1 - A.E185_W) * r[subset] + A.E185_W * 1000.)
            self.assertTrue((after['pred'][subset] != before['pred'][subset]).all())
            out = pl.read_parquet(A.sub(work, 2))
            self.assertEqual(out.schema[Y], pl.Int32)
            self.assertEqual(out[Y].to_list()[1:3], pl.read_parquet(A.sub(work, 1))[Y].to_list()[1:3])

    @unittest.skipUnless(frozen_available('e167_nm_window_rule.py', 'train.py', 'features.py', 'target_encoding.py'), 'frozen archive absent')
    def test_e167_links_departures_through_provided_arrivals_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = self.frozen_work(tmp, ['e167_nm_window_rule.py', 'train.py', 'features.py', 'target_encoding.py'])
            t0 = datetime(2026, 2, 10, 12, tzinfo=timezone.utc)
            ids = np.array([1., 2., 3.])
            # row 1: no-NM departure EDDF->EGLL; matching NM arrival 2 h later with LOBT inside the schedule window and |D| > 3606
            # row 2: no-NM departure whose flight has no provided arrival; row 3: NM departure (never linked)
            dep = pl.DataFrame({ID: ids, 'PHASE_mvt': ['DEP'] * 3, 'FLIGHT_ID_mvt': [None, None, 'NM3'], 'FLIGHT_mvt': ['F1', 'F2', 'F3'],
                                'ADEP_mvt': ['EDDF'] * 3, 'ADES_mvt': ['EGLL'] * 3,
                                'MVT_TIME_UTC_mvt': pl.Series([t0] * 3, dtype=UTC),
                                'SCHED_TIME_UTC_mvt': pl.Series([t0 - timedelta(hours=3)] * 3, dtype=UTC),
                                'LOBT_flt': pl.Series([None] * 3, dtype=UTC), 'AOBT_3_flt': pl.Series([None] * 3, dtype=UTC)})
            arr = pl.DataFrame({ID: [9.], 'PHASE_mvt': ['ARR'], 'FLIGHT_ID_mvt': ['NM9'], 'FLIGHT_mvt': ['F1'], 'ADEP_mvt': ['EDDF'], 'ADES_mvt': ['EGLL'],
                                'MVT_TIME_UTC_mvt': pl.Series([t0 + timedelta(hours=2)], dtype=UTC),
                                'SCHED_TIME_UTC_mvt': pl.Series([t0 - timedelta(hours=3)], dtype=UTC),
                                'LOBT_flt': pl.Series([t0 - timedelta(hours=3, minutes=10)], dtype=UTC),
                                'AOBT_3_flt': pl.Series([t0 - timedelta(hours=3)], dtype=UTC)})
            pl.concat([dep, arr]).write_parquet(work / 'data/prc-2026-datasets/ranking.parquet')
            d_lobt = np.full(3, np.nan)
            cands = [np.full(3, 3 * 3600.), np.full(3, np.nan), d_lobt, np.full(3, np.nan)]
            before = self.parts(work, 1, ids, cands, d_lobt, np.array([600., 600., 600.]))
            pl.DataFrame({ID: ids, 'split': ['rank'] * 3, 'd_mvt_lobt': d_lobt}).write_parquet(work / 'data/features_e106.parquet')
            result = self.run_stage(work, ['e167_nm_window_rule.py', 1, 2], {})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            after = np.load(work / 'submissions/parts_v2.npz')
            self.assertEqual(after['proba'][0, 1], 0.)
            self.assertEqual(after['proba'][0, 5], 0.)
            np.testing.assert_array_equal(after['proba'][1:], before['proba'][1:])
            np.testing.assert_array_equal(after['pred'][1:], before['pred'][1:])
            self.assertNotEqual(after['pred'][0], before['pred'][0])
            self.assertIn('rule A rows 1', result.stdout)


if __name__ == '__main__':
    unittest.main()

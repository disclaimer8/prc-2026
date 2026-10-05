"""Fixed E182 recipe and fail-closed input boundaries; no fits or network writes."""
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import e182_submission as E
except ModuleNotFoundError:
    E = None


class E182Tests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(E, 'E182 guarded builder is missing')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def rank(self, **changes):
        provenance = copy.deepcopy(E.RANK_PROVENANCE)
        provenance['params']['num_threads'] = 12
        provenance['config']['OMP_NUM_THREADS'] = '12'
        seeds = np.arange(15, dtype=float).reshape(5, 3)
        payload = dict(mvt_id=np.array([3., 1., 2.]), reg=seeds.mean(axis=0), per_seed=seeds,
                       complete=True, status='complete', expected_seeds=5, completed_seeds=5,
                       expected_rows=3, mode='eobt', rounds=2736,
                       features_path='/host/data/features_fin1b.parquet',
                       features_sha256=E.FEATURE_SHA, provenance_json=json.dumps(provenance))
        payload.update(changes)
        path = self.root/'rank.npz'
        np.savez(path, **payload)
        return path

    def validate(self, path):
        return E.validate_member(path, np.array([1, 2, 3]), 'eobt', 2736, E.FEATURE_SHA, rows=3)

    def test_complete_rank_accepts_safe_float_ids_and_rejects_partial_wrong_metadata(self):
        self.validate(self.rank())
        for changes in [dict(complete=False), dict(status='checkpoint'), dict(expected_seeds=4),
                        dict(completed_seeds=4), dict(mode='aobt'), dict(rounds=2735),
                        dict(features_sha256='0'*64), dict(mvt_id=[1, 1, 3]),
                        dict(reg=[1., 2., 3.]), dict(per_seed=np.zeros((4, 3)))]:
            with self.subTest(changes=changes), self.assertRaises(E.GuardError):
                self.validate(self.rank(**changes))

    def test_rank_rejects_unregistered_source_params_and_feature_lineage(self):
        for key, value in [('source_sha256', {'train.py': 'f'*64}),
                           ('params', {'objective': 'regression'}),
                           ('feature_columns_sha256', '0'*64),
                           ('config', {'REG_OVERRIDE': '{}'}),
                           ('seed_schedule', {'seed': 42})]:
            provenance = copy.deepcopy(E.RANK_PROVENANCE)
            provenance['params']['num_threads'] = 24
            provenance['config']['OMP_NUM_THREADS'] = '24'
            provenance[key] = value
            with self.subTest(key=key), self.assertRaises(E.GuardError):
                self.validate(self.rank(provenance_json=json.dumps(provenance)))

    def test_fixed_recipe_uses_five_new_members_full_mlp_and_post_rule_parts(self):
        plan = E.stage_commands()
        self.assertEqual(plan[0][0], ['blend.py', '870', '12020', '0.65'])
        self.assertEqual(plan[0][1]['E130_MEMBERS'],
                         'submissions/e130_rawf_rank.npz:0.0333333333333333,'
                         'submissions/e130_lobtf_rank.npz:0.10,submissions/e130_schedf_rank.npz:0.10,'
                         'submissions/e130_eobtf_rank.npz:0.10,submissions/e130_aobtf_rank.npz:0.10,'
                         'submissions/e130_mlpa_rank.npz:0.25')
        for args, env in plan:
            if args[0] in ('e141_apply.py', 'e64_apply.py'):
                self.assertEqual(env['PARTS'], 'submissions/parts_v12021.npz')
        self.assertEqual(plan[-1][0], ['protect_tabpfn_rows.py', '12027', '12022', '12028'])
        self.assertFalse(any('e185_apply' in args[0] for args, _ in plan))

    def test_cat_alias_and_resource_edits_are_only_inside_scratch(self):
        for name, text in [('src/train.py', 'PARAMS = dict(num_threads=10, seed=42)\n'),
                           ('src/e36_adsb_stand_geometry.py', 'PARAMS = dict(num_threads=8)\n'),
                           ('src/e64_visit_stage.py', 'PARAMS = dict(num_threads=8)\n'),
                           (E.CAT, 'fifteen seeds'), ('submissions/cat_reg_rank.npz', 'global old')]:
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        build = self.root/'build'
        E.prepare_workspace(self.root, build, [E.CAT], [], list(E.RESOURCE_PATCHES))
        E.prepare_resources(build)
        self.assertEqual((self.root/'src/train.py').read_text(), 'PARAMS = dict(num_threads=10, seed=42)\n')
        self.assertEqual((build/'src/train.py').read_text(), 'PARAMS = dict(num_threads=6, seed=42)\n')
        self.assertEqual((build/'submissions/cat_reg_rank.npz').read_text(), 'fifteen seeds')
        self.assertEqual((self.root/'submissions/cat_reg_rank.npz').read_text(), 'global old')

    def test_clean_environment_cannot_inherit_experimental_knobs_or_credentials(self):
        with patch.dict(os.environ, {'REG_OVERRIDE': '{}', 'PRC_ACCESS_KEY': 'test', 'CLF_ONLY_PREFIX': 'x'}):
            env = E.clean_environment()
        for key in ('REG_OVERRIDE', 'PRC_ACCESS_KEY', 'CLF_ONLY_PREFIX'):
            self.assertNotIn(key, env)
        self.assertEqual(env['OMP_THREAD_LIMIT'], '6')
        self.assertEqual(env['OMP_WAIT_POLICY'], 'PASSIVE')
        self.assertEqual(env['GOMP_SPINCOUNT'], '0')

    def test_gate_keeps_original_four_folds_trimmed_rule_and_exact_lineage(self):
        good = dict(valid=True, audit_only=True, status='audit_valid', fold_count=6,
                    rows=2085047, target_verified_against_features=True,
                    control='cv_preds_e172_raw_lobt_sched_eobt_aobt_mlpa_f1b',
                    candidate='cv_preds_e172_rawf_lobtf_schedf_eobtf_aobtf_mlpa_f1b',
                    features='features_fin1b.parquet',
                    metrics=dict(lower_decided=5, decided=-.226814, body=-.213973, trimmed=-.209718))
        E.validate_gate(good)
        for changes in [dict(fold_count=5), dict(candidate='wrong'),
                        dict(metrics={**good['metrics'], 'trimmed': 0}),
                        dict(metrics={**good['metrics'], 'lower_decided': 3})]:
            with self.subTest(changes=changes), self.assertRaises(E.GuardError):
                E.validate_gate({**good, **changes})

    def test_version_block_is_exclusive_even_for_intermediate_parts(self):
        (self.root/'submissions').mkdir()
        E.reserve_versions(self.root, 'a'*40)
        with self.assertRaises((E.GuardError, FileExistsError)):
            E.reserve_versions(self.root, 'a'*40)

    def test_cv_service_must_be_frozen_or_finished(self):
        for state in ['LoadState=loaded\nActiveState=active\nFreezerState=frozen\n',
                      'LoadState=loaded\nActiveState=inactive\nFreezerState=running\n']:
            E.validate_cv_state(state)
        for state in ['LoadState=not-found\nActiveState=inactive\n',
                      'LoadState=loaded\nActiveState=active\nFreezerState=running\n',
                      'LoadState=loaded\nActiveState=failed\nFreezerState=running\n']:
            with self.assertRaises(E.GuardError):
                E.validate_cv_state(state)

    def test_upload_uses_shared_intent_ledger_and_never_retries_failed_put(self):
        path = self.root/E.FINAL
        path.parent.mkdir()
        (self.root/'data').mkdir()
        path.write_bytes(b'synthetic bounded candidate')
        client, http = Mock(), Mock()
        client.fput_object.side_effect = TimeoutError('synthetic lost response')
        manifest = {'outputs': {E.FINAL: E.sha256(path)}}
        with patch.object(E, 'preflight', return_value=manifest), \
             patch.object(E.E, 'load_upload_credentials'), \
             patch.object(E.E, 'fetch_leaderboard', return_value=[]), \
             patch.object(E.E, 'bucket_inventory', return_value=[]), \
             patch.object(E.E, 'utcnow', return_value=datetime(2026, 10, 5, 12, tzinfo=timezone.utc)), \
             patch.dict(os.environ, {'PRC_ACCESS_KEY': 'synthetic', 'PRC_SECRET_KEY': 'synthetic'}), \
             patch('e185_single_put_upload.OnePutHTTP', return_value=http), \
             patch('minio.Minio', return_value=client):
            with self.assertRaisesRegex(E.GuardError, 'outcome ambiguous'):
                E.upload_submission(self.root)
            with self.assertRaises(E.GuardError):
                E.upload_submission(self.root)
        self.assertEqual(client.fput_object.call_count, 1)
        self.assertEqual(client.fput_object.call_args.kwargs, {'num_parallel_uploads': 1})
        events = [json.loads(line)['event'] for line in
                  (self.root/'data/submission_upload_ledger.jsonl').read_text().splitlines()]
        self.assertEqual(events, ['intent', 'ambiguous'])


if __name__ == '__main__':
    unittest.main()

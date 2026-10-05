"""Fixed combination boundaries; never fits a model or opens a network socket."""
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
try:
    import e182_e185_submission as C
except ModuleNotFoundError:
    C = None


class ComboTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(C, 'separate E182+E185 builder is missing')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def evidence(self):
        folds = [f'fold_{a}_{a+6}.npz' for a in range(1, 7)]
        flags = {'four_of_six': True, 'trimmed_lower': True, 'body_lower': True}
        manifest = {'registration_commit': C.ADDITIVITY_REGISTRATION, 'weight': .35,
                    'ranking_weight': .35*2/3, 'subset': 'LOBT_flt IS NULL AND ADEP_mvt != LIRF',
                    'control_source': '/root/data/cv_preds_e172_rawf_lobtf_schedf_eobtf_aobtf_mlpa_f1b',
                    'member_source': '/root/data/cv_e130motion', 'gate': {'passed': True, 'flags': flags},
                    'input_sha256': {'data/features_fin1b.parquet': C.B.FEATURE_SHA},
                    'output_sha256': {'verdict.json': C.EVIDENCE_HASHES[C.VERDICT]},
                    'folds': [{'name': n, 'max_abs_rebuild': {'reg': 0., 'pred': 0.}} for n in folds]}
        verdict = {'status': 'accepted', 'valid': True, 'gate_passed': True,
                   'audit_only': False, 'require_body': True, 'fold_count': 6, 'rows': 2085047,
                   'target_verified_against_features': True, 'features': '/data/features_fin1b.parquet',
                   'control': '/data/e185_on_e182_w035_20261005/control',
                   'candidate': '/data/e185_on_e182_w035_20261005/candidate', 'gate_flags': flags,
                   'metrics': {'lower_decided': 5, 'decided': -.081171, 'body': -.066048, 'trimmed': -.060440},
                   'values_consistency': {kind: {n: {'checked': True, 'consistent': True,
                                                      'max_abs_seconds': 0.} for n in folds}
                                          for kind in ('control', 'candidate')}}
        return manifest, verdict

    def test_additivity_requires_exact_registered_lineage_and_original_body_gate(self):
        manifest, verdict = self.evidence()
        C.validate_additivity(manifest, verdict)
        for update in [{'registration_commit': '0'*40}, {'weight': .30},
                       {'ranking_weight': .35}, {'member_source': '/data/other'}, {'folds': []}]:
            with self.subTest(update=update), self.assertRaises(C.GuardError):
                C.validate_additivity({**manifest, **update}, verdict)
        for update in [{'fold_count': 5}, {'control': '/data/old/control'}, {'require_body': False},
                       {'metrics': {**verdict['metrics'], 'body': 0}},
                       {'metrics': {**verdict['metrics'], 'lower_decided': 3}},
                       {'values_consistency': {}}]:
            with self.subTest(update=update), self.assertRaises(C.GuardError):
                C.validate_additivity(manifest, {**verdict, **update})

    def test_evidence_bytes_are_pinned_before_metadata_is_trusted(self):
        for name, evidence in zip((C.ADDITIVITY_MANIFEST, C.VERDICT), self.evidence()):
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(evidence))
        with self.assertRaisesRegex(C.GuardError, 'hash mismatch'):
            C.read_additivity(self.root)

    def test_fixed_recipe_inserts_e185_before_e167_and_keeps_all_tail_parts(self):
        expected = [
            ['e185_apply.py', '12020', '12040'], ['e167_nm_window_rule.py', '12040', '12041'],
            ['@tabpfn', '12041', '12042'], ['e141_apply.py', '12042', '12043'],
            ['e64_apply.py', '12043', '12044'], ['e152_stage_gate.py', '12042', '12044', '12041', '12045'],
            ['blend_submissions.py', '12046'], ['lobt_edge_lift.py', '12046', '12047'],
            ['protect_tabpfn_rows.py', '12047', '12042', '12048']]
        plan = C.stage_commands()
        self.assertEqual([args for args, _ in plan], expected)
        self.assertEqual(float(plan[0][1]['E185_W']), .35*2/3)
        for _, env in plan[3:5]:
            self.assertEqual(env['PARTS'], 'submissions/parts_v12041.npz')
        self.assertEqual(plan[3][1]['E36_BASE_DIR'], 'data/cv_preds_e116prod')
        self.assertEqual(plan[6][1]['BLEND_PARTS'], '44:0.1214,60:0.0270')
        self.assertEqual(C.B.VERSION, 12028)
        self.assertEqual(C.E.VERSION, 12008)

    def test_parent_preflight_failure_precedes_any_reservation_or_build(self):
        with patch.object(C.B, 'preflight', side_effect=C.GuardError('parent changed')) as guard:
            with self.assertRaisesRegex(C.GuardError, 'parent changed'):
                C.build_submission(self.root, 'a'*40)
        guard.assert_called_once_with(self.root.resolve())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_parent_inputs_require_accepted_registration_and_pre_rule_parts(self):
        parent = {'registration_commit': C.PARENT_REGISTRATION,
                  'inputs': {'data/features_fin1b.parquet': 'a'*64},
                  'outputs': {C.BASE: 'b'*64, C.PARTS: 'c'*64, C.B.FINAL: 'd'*64}}
        inputs = C.parent_inputs(parent)
        self.assertEqual(inputs[C.PARTS], 'c'*64)
        for key in (C.BASE, C.PARTS, C.B.FINAL):
            wrong = copy.deepcopy(parent)
            del wrong['outputs'][key]
            with self.assertRaises(C.GuardError):
                C.parent_inputs(wrong)
        with self.assertRaises(C.GuardError):
            C.parent_inputs({**parent, 'registration_commit': '0'*40})

    def test_exclusive_combination_block_does_not_touch_parent_reservation(self):
        subs = self.root/'submissions'
        subs.mkdir()
        old = subs/'gentle-igloo_v12028.parquet'
        old.write_text('parent final')
        C.reserve_versions(self.root, 'a'*40)
        with self.assertRaises((C.GuardError, FileExistsError)):
            C.reserve_versions(self.root, 'a'*40)
        self.assertEqual(old.read_text(), 'parent final')
        record = json.loads((subs/'e182_e185_v12040_12059.reservation.json').read_text())
        self.assertEqual(record['versions'], list(range(12040, 12060)))
        self.assertEqual(record['forecast']['reference'], 'submissions/gentle-igloo_v12008.parquet')

    def test_failed_combination_upload_retains_shared_intent_without_retry(self):
        path = self.root/C.FINAL
        path.parent.mkdir()
        (self.root/'data').mkdir()
        path.write_bytes(b'bounded synthetic combination')
        client, http = Mock(), Mock()
        client.fput_object.side_effect = TimeoutError('synthetic response lost')
        manifest = {'outputs': {C.FINAL: C.sha256(path)}}
        with patch.object(C, 'preflight', return_value=manifest), \
             patch.object(C.E, 'load_upload_credentials'), \
             patch.object(C.E, 'fetch_leaderboard', return_value=[]), \
             patch.object(C.E, 'bucket_inventory', return_value=[]), \
             patch.object(C.E, 'utcnow', return_value=datetime(2026, 10, 5, 12, tzinfo=timezone.utc)), \
             patch.dict(os.environ, {'PRC_ACCESS_KEY': 'synthetic', 'PRC_SECRET_KEY': 'synthetic'}), \
             patch('e185_single_put_upload.OnePutHTTP', return_value=http), \
             patch('minio.Minio', return_value=client):
            with self.assertRaisesRegex(C.GuardError, 'outcome ambiguous'):
                C.upload_submission(self.root)
            with self.assertRaises(C.GuardError):
                C.upload_submission(self.root)
        self.assertEqual(client.fput_object.call_count, 1)
        self.assertEqual(client.fput_object.call_args.args[1], 'gentle-igloo_v12048.parquet')
        self.assertEqual(client.fput_object.call_args.kwargs, {'num_parallel_uploads': 1})
        records = [json.loads(line) for line in
                   (self.root/'data/submission_upload_ledger.jsonl').read_text().splitlines()]
        self.assertEqual([r['event'] for r in records], ['intent', 'ambiguous'])
        self.assertTrue(all(r['filename'] == 'gentle-igloo_v12048.parquet' for r in records))


if __name__ == '__main__':
    unittest.main()

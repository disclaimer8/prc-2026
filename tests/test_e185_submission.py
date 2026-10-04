"""E185 recipe boundary checks; never starts a model fit or an upload."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import e185_submission as E
except ModuleNotFoundError:
    E = None


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(E, 'E185 isolated builder is not implemented')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, value):
        p = self.root/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value)
        return p

    def test_old_e64_oof_outputs_are_isolated_from_source_tree(self):
        self.write('data/cv_preds_e64/fold_1_7.npz', 'old historical OOF')
        self.write('data/cv_preds_nomonth/fold_1_7.npz', 'required source')
        self.write('submissions/gentle-igloo_v1121.parquet', 'donor')
        self.write('src/example.py', 'pass')
        build = self.root/'build'
        E.prepare_workspace(self.root, build,
                            ['data/cv_preds_nomonth/fold_1_7.npz'],
                            ['submissions/gentle-igloo_v1121.parquet'], ['src/example.py'])
        target = build/'data/cv_preds_e64/fold_1_7.npz'
        target.parent.mkdir(parents=True)
        target.write_text('new historical CV output from import')
        self.assertEqual((self.root/'data/cv_preds_e64/fold_1_7.npz').read_text(), 'old historical OOF')
        self.assertEqual((build/'data/cv_preds_nomonth/fold_1_7.npz').read_text(), 'required source')
        self.assertFalse((build/'data').is_symlink())
        (self.root/'src/example.py').write_text('changed source')
        self.assertEqual((build/'src/example.py').read_text(), 'pass')

    def test_registration_gate_rejects_partial_wrong_lineage_or_body_failure(self):
        good = {'valid': True, 'audit_only': True, 'status': 'audit_valid', 'fold_count': 6,
                'rows': 2085047, 'target_verified_against_features': True,
                'control': '/a/data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlpa_f1b',
                'candidate': '/a/data/e185_cv_sub', 'features': '/a/data/features_fin1b.parquet',
                'metrics': {'lower_decided': 5, 'decided': -.047346, 'body': -.041495, 'trimmed': -.032365}}
        E.validate_gate(good)
        for edit in [{'fold_count': 5}, {'rows': 100}, {'candidate': '/a/e182'},
                     {'metrics': {**good['metrics'], 'body': .001}},
                     {'metrics': {**good['metrics'], 'lower_decided': 3}}]:
            with self.subTest(edit=edit), self.assertRaises(E.GuardError):
                E.validate_gate({**good, **edit})

    def test_version_reservation_rejects_any_existing_intermediate(self):
        self.write('submissions/parts_v12001.npz', 'earlier experiment')
        with self.assertRaises(E.GuardError):
            E.reserve_versions(self.root, 'a'*40)
        self.assertEqual((self.root/'submissions/parts_v12001.npz').read_text(), 'earlier experiment')

    def test_exclusive_reservation_cannot_be_reused(self):
        (self.root/'submissions').mkdir()
        E.reserve_versions(self.root, 'a'*40)
        with self.assertRaises((E.GuardError, FileExistsError)):
            E.reserve_versions(self.root, 'a'*40)

    def test_upload_credential_file_is_private_and_limited_to_competition_keys(self):
        keys = {'PRC_ACCESS_KEY': 'synthetic-access', 'PRC_SECRET_KEY': 'synthetic-secret'}
        path = self.write('.prc-upload-credentials.json', json.dumps(keys))
        path.chmod(0o644)
        with self.assertRaises(E.GuardError):
            E.load_upload_credentials(self.root)
        path.chmod(0o600)
        with patch.dict(os.environ, {}, clear=True):
            E.load_upload_credentials(self.root)
            self.assertEqual({k: os.environ[k] for k in keys}, keys)
            self.assertEqual(set(os.environ), set(keys))
        path.write_text(json.dumps({**keys, 'UNRELATED_KEY': 'unexpected'}))
        with self.assertRaises(E.GuardError):
            E.load_upload_credentials(self.root)

    def test_stage_plan_uses_post_e167_parts_for_both_adsb_stages(self):
        plan = E.stage_commands()
        # Independently fixed lineage: E185 v12000, E167 v12001,
        # copied TabPFN v12002, E141 v12003, E64 v12004, gate v12005.
        e141 = next(x for x in plan if x[0][0] == 'e141_apply.py')
        e64 = next(x for x in plan if x[0][0] == 'e64_apply.py')
        self.assertEqual(e141[0], ['e141_apply.py', '12002', '12003'])
        self.assertEqual(e64[0], ['e64_apply.py', '12003', '12004'])
        for _, env in (e141, e64):
            self.assertEqual(env['PARTS'], 'submissions/parts_v12001.npz')
        self.assertEqual(e141[1]['E36_BASE_DIR'], 'data/cv_preds_e116prod')
        self.assertEqual(plan[-1][0], ['protect_tabpfn_rows.py', '12007', '12002', '12008'])


if __name__ == '__main__':
    unittest.main()

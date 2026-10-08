"""Synthetic contracts only; never open real OOF, targets, or parent outputs."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
FILE = ROOT / 'scripts/e207_current_chain_20261008.py'
if FILE.exists():
    spec = importlib.util.spec_from_file_location('child', FILE)
    X = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(X)
else:
    X = None


def reports():
    # Four decided wins, only one body win; no selected 1/4-vs-rest subgroup.
    return [dict(fold=i, metrics={
        'body': {'delta_sse': -20.0 if i == 1 else 1.0},
        'decided': {'delta_sse': -5.0 if i <= 4 else 1.0,
                    'trim3_sse': -2.0, 'trim20_sse': -1.0}}) for i in range(1, 7)]


class Contracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(X, 'CV-only adapter must implement the registered contracts')

    def test_fixed_gate_does_not_inherit_e204_extra_restrictions(self):
        self.assertTrue(X.chain_gate(reports()))

    def test_gate_requires_four_decided_wins_and_all_four_negative_sums(self):
        r = reports(); r[3]['metrics']['decided']['delta_sse'] = 1.0
        self.assertFalse(X.chain_gate(r))
        for pop, key in [('body', 'delta_sse'), ('decided', 'delta_sse'),
                         ('decided', 'trim3_sse'), ('decided', 'trim20_sse')]:
            r = reports(); r[0]['metrics'][pop][key] = 100.0
            self.assertFalse(X.chain_gate(r), (pop, key))

    def test_gate_rejects_duplicate_missing_fold_and_nan(self):
        for r in (reports()[:-1], reports() + [reports()[0]]):
            with self.assertRaises(ValueError): X.chain_gate(r)
        r = reports(); r[5]['fold'] = 1
        with self.assertRaises(ValueError): X.chain_gate(r)
        r = reports(); r[0]['metrics']['decided']['trim20_sse'] = float('nan')
        with self.assertRaises(ValueError): X.chain_gate(r)

    def test_neural_slot_is_added_once_at_quarter_weight(self):
        z = np.zeros(2)
        np.testing.assert_array_equal(X.regular_member(z, z, [z]*5, np.array([4., 8.])), [1., 2.])
        # Same real mixture, replace only neural member; hand-derived delta=.25*4.
        a = X.regular_member(z+100, z+200, [z+v for v in (10,20,30,40,50)], z+80)
        b = X.regular_member(z+100, z+200, [z+v for v in (10,20,30,40,50)], z+84)
        np.testing.assert_allclose(b-a, [1., 1.], rtol=0, atol=1e-13)
        with self.assertRaises(ValueError): X.regular_member(z,z,[z]*5,np.array([np.nan,0.]))

    def test_feasibility_cannot_add_scored_exclusions(self):
        a = np.zeros((2,3), dtype=bool); b = a.copy(); b[0,2] = True
        with self.assertRaises(ValueError): X.fixed_feasibility(a,b,np.array([False,True]))
        X.fixed_feasibility(a,b,np.array([True,False]))

    def test_controller_terminal_requires_matching_successful_owned_gpu(self):
        rows = [dict(event='controller_started', registration_commit=X.PARENT_REG),
                dict(event='started', unit=X.GPU_UNIT, properties={'InvocationID':'owned'}),
                dict(event='exited', unit=X.GPU_UNIT, properties={'InvocationID':'owned',
                     'ActiveState':'active','SubState':'exited','MainPID':'0','Result':'success','ExecMainStatus':'0'}),
                dict(event='complete_no_ranking',gate=True,evaluation_sha256='a'*64)]
        X.terminal_gate(rows,'a'*64)
        for mutate in ('replacement','failure','missing','late_failure','wrong_sha'):
            r=copy.deepcopy(rows); sha='a'*64
            if mutate=='replacement': r[2]['properties']['InvocationID']='foreign'
            if mutate=='failure': r[2]['properties']['ExecMainStatus']='7'
            if mutate=='missing': r.pop(2)
            if mutate=='late_failure': r.append(dict(event='failed_no_retry'))
            if mutate=='wrong_sha': sha='b'*64
            with self.assertRaises(ValueError, msg=mutate): X.terminal_gate(r,sha)

    def test_parent_gate_recomputed_not_trusted_boolean(self):
        r=[{'fold':i,'metrics':{k:-1.0 for k in X.PARENT_KEYS}} for i in range(1,7)]
        e=dict(protocol=X.PARENT_PROTOCOL,registration_commit=X.PARENT_REG,reports=r,
               pooled={k:-6.0 for k in X.PARENT_KEYS},decided_pairs_improved=6,
               six_pair_gate_pass=True,accepted_for_submission=False)
        X.parent_gate(e)
        for key in ('six_pair_gate_pass','accepted_for_submission'):
            q=copy.deepcopy(e);q[key]=not q[key]
            with self.assertRaises(ValueError): X.parent_gate(q)
        q=copy.deepcopy(e);q['reports'][0]['metrics']['body_sse']=100.0
        with self.assertRaises(ValueError): X.parent_gate(q)

    def test_prepared_receipts_reject_old_namespace_or_changed_population(self):
        self.assertTrue(callable(getattr(X,'prepared_gate',None)), 'prepared provenance admission required')
        rows=[dict(fold=f,protocol=X.PARENT_PROTOCOL,registration_commit=X.PARENT_REG,
                   input_spec_sha256=X.MATERIALIZED_SPEC,source_sha256='a'*64,rows=2085047)
              for f in range(1,7)]
        X.prepared_gate(rows,'a'*64)
        for key,value in [('rows',2085046),('registration_commit','b'*40),('input_spec_sha256','b'*64),('source_sha256','b'*64)]:
            r=copy.deepcopy(rows);r[0][key]=value
            with self.assertRaises(ValueError): X.prepared_gate(r,'a'*64)

    def test_file_set_verifier_rejects_changed_and_unsafe_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'a').write_bytes(b'good'); expected={'a':X.A.sha(root/'a')}
            X.check_files(root,expected)
            (root/'a').write_bytes(b'bad')
            with self.assertRaises(ValueError): X.check_files(root,expected)
            with self.assertRaises(ValueError): X.check_files(root,{'../escape':'0'*64})

    def test_remote_registration_receipt_binds_actual_frozen_files(self):
        self.assertTrue(callable(getattr(X,'runtime_registration',None)), 'remote source admission is required')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); reg='b'*40
            names=(X.SELF,X.TEST,X.DOC,X.SPEC)
            for n in names:
                p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(n)
            receipt=dict(protocol=X.PROTOCOL,registration_commit=reg,journal_protocol_verified=True,
                         files={n:X.A.sha(root/n) for n in names})
            (root/X.ATTEST).write_text(json.dumps(receipt))
            X.runtime_registration(root,reg)
            (root/X.SELF).write_text('changed')
            with self.assertRaises(ValueError): X.runtime_registration(root,reg)

    def test_cli_refuses_ranking_and_unregistered_cv_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            for args in (['rank'],['cv','--registration-commit','not-a-commit']):
                r=subprocess.run([sys.executable,'-B',str(FILE),*args,'--root',tmp],capture_output=True,text=True)
                self.assertNotEqual(r.returncode,0)
                self.assertFalse((Path(tmp)/X.AREA/'cv').exists())


if __name__ == '__main__': unittest.main()

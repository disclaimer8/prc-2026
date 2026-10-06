import sys
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    import cat_extra175 as E
except ModuleNotFoundError:
    E=None

class ExtraNodeTests(unittest.TestCase):
    def test_transfer_preserves_exact_90_and_disjoint_ownership(self):
        self.assertIsNotNone(E,'extra-node transfer is not implemented')
        a=E.assignments();E.validate_assignments(a)
        self.assertEqual(a['laptop175'],[[14,4],[14,5],[14,6]])
        self.assertEqual(len(a['laptop']),7)
        bad={k:list(v) for k,v in a.items()};bad['laptop'].append([14,4])
        with self.assertRaises(ValueError):E.validate_assignments(bad)
    def test_drain_needs_every_retained_job_and_rejects_taken_jobs(self):
        self.assertIsNotNone(E,'extra-node transfer is not implemented')
        jobs=E.assignments()['laptop']
        self.assertFalse(E.drained(jobs[:-1],jobs))
        self.assertTrue(E.drained(jobs,jobs))
        with self.assertRaises(ValueError):E.drained(jobs+[[14,4]],jobs)
        with self.assertRaises(ValueError):E.drained(jobs+[jobs[0]],jobs)
    def test_qualification_is_exact_with_no_runtime_substitution(self):
        self.assertIsNotNone(E,'extra-node transfer is not implemented')
        runtime={'python':'3.14','packages':{'catboost':'1.2.10'}}
        report=dict(exact=True,max_abs=0.,rows=344419,runtime=runtime,
                    registration_commit=E.QUAL_REG,prediction_sha256='abc')
        E.require_qualification(report,runtime,'abc')
        for change in ({'exact':False},{'max_abs':1e-9},{'rows':344418},
                       {'prediction_sha256':'changed'},{'runtime':{'python':'other'}},
                       {'registration_commit':'other'}):
            with self.assertRaises(ValueError):E.require_qualification({**report,**change},runtime,'abc')
    def test_failed_activation_write_cannot_publish_armed_marker(self):
        self.assertTrue(hasattr(E,'publish_armed'),'atomic activation publication missing')
        with tempfile.TemporaryDirectory() as folder:
            dest=Path(folder)/'armed.json'
            with patch.object(E.C,'write_json',side_effect=OSError('simulated disk failure')):
                with self.assertRaises(OSError):E.publish_armed(dest,{'lineage':'test'})
            self.assertFalse(dest.exists())
            E.publish_armed(dest,{'lineage':'test'})
            self.assertEqual(E.C.read_json(dest),{'lineage':'test'})
    def test_preflight_detects_started_child_before_attempt_directory(self):
        self.assertTrue(hasattr(E,'reserved_started'),'early subprocess preflight missing')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);jobs=[[14,4],[14,5],[14,6]]
            self.assertFalse(E.reserved_started(root,jobs))
            (root/'fold_4_10-s14.log').touch()
            self.assertTrue(E.reserved_started(root,jobs))

if __name__=='__main__':unittest.main()

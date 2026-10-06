import sys
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cat_night_queue as Q

class QueueTests(unittest.TestCase):
    def test_assignment_covers_exactly_90_once(self):
        a=Q.assignments();Q.validate_assignments(a)
        self.assertEqual([len(a[n]) for n in Q.NODES],[44,20,16,10])
        bad={k:list(v) for k,v in a.items()};bad['mac'][0]=bad['home'][0]
        with self.assertRaises(ValueError):Q.validate_assignments(bad)
    def test_qualification_never_relaxes_laptop_or_ids(self):
        ids=np.array([1.,2.]);p=np.array([100.,200.])
        self.assertTrue(Q.qualification(ids,p,ids,p,'laptop')['passed'])
        self.assertFalse(Q.qualification(ids,p,ids,p+5e-7,'laptop')['passed'])
        self.assertTrue(Q.qualification(ids,p,ids,p+5e-7,'mac')['passed'])
        self.assertFalse(Q.qualification(ids,p,ids,p+2e-6,'mac')['passed'])
        with self.assertRaises(ValueError):Q.qualification(ids,p,ids[::-1],p,'mac')
        with self.assertRaises(ValueError):Q.qualification(ids,p,ids,np.array([100.,np.nan]),'mac')
    def test_node_cannot_claim_anothers_task(self):
        m={'assignments':Q.assignments()}
        Q.require_job(m,'home',0,1)
        with self.assertRaises(ValueError):Q.require_job(m,'hetzner',0,1)
    def test_publication_rejects_partial_or_changed_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'raw.npz';dest=root/'home/fold_1_7/seed_00'
            ids=np.array([1,2]);np.savez(raw,mvt_id=ids,reg=np.array([10.,20.]))
            m={'assignments':Q.assignments(),'lineage':'test'}
            with patch.object(Q.C,'packages',return_value={'python':'test','packages':{}}):
                Q.publish(raw,dest,m,'home',0,1,ids,{'kind':'test'})
                with self.assertRaises(ValueError):Q.publish(raw,dest,m,'home',0,1,ids,{})
            np.testing.assert_array_equal(Q.check_result(dest,m,'home',0,1,ids),[10.,20.])
            np.savez(dest/'prediction.npz',mvt_id=ids,reg=np.array([11.,20.]))
            with self.assertRaises(ValueError):Q.check_result(dest,m,'home',0,1,ids)
    def test_failed_publish_never_exposes_final(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'raw.npz';dest=root/'results/seed_00'
            ids=np.array([1,2]);np.savez(raw,mvt_id=ids,reg=np.array([10.,20.]))
            m={'assignments':Q.assignments(),'lineage':'test'}
            with patch.object(Q.C,'write_json',side_effect=OSError('simulated full disk')):
                with self.assertRaises(OSError):Q.publish(raw,dest,m,'home',0,1,ids,{})
            self.assertFalse(dest.exists())

if __name__=='__main__':unittest.main()

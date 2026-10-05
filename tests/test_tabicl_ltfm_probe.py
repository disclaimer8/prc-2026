"""Contracts for a bounded TabICLv2 member screen; no model downloads."""
import importlib.util
from pathlib import Path
import sys
import unittest
import tempfile
from types import SimpleNamespace
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


class TabICLProbeTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('tabicl_ltfm_probe'),'probe module missing')
        import tabicl_ltfm_probe as T
        return T

    def test_sampling_is_deterministic_temporally_disjoint_and_target_blind(self):
        T=self.module();months=np.repeat(np.arange(1,13),20);ids=np.arange(240)
        fit,test=T.sample_indices(ids,months,np.ones(240,dtype=bool),1,train_per_month=7,test_per_month=5)
        self.assertEqual(len(fit),70);self.assertEqual(len(test),10)
        self.assertTrue(set(months[fit]).isdisjoint({1,7}))
        self.assertEqual(set(months[test]),{1,7});self.assertFalse(set(ids[fit])&set(ids[test]))
        f2,t2=T.sample_indices(ids,months,np.ones(240,dtype=bool),1,train_per_month=7,test_per_month=5)
        np.testing.assert_array_equal(fit,f2);np.testing.assert_array_equal(test,t2)

    def test_member_updates_both_regular_and_dayshift_and_preserves_correction(self):
        T=self.module()
        reg=np.array([100.,200.]);proba=np.array([[.5,0,0,0,0,.5],[0,1,0,0,0,0.]])
        values=np.array([[100.,0,0,0,0,86500.],[200.,800,0,0,0,86600.]])
        z=dict(reg=reg,proba=proba,values=values,pred=(values*proba).sum(axis=1)+13)
        lobt=np.array([np.nan,np.nan]);p=T.blend_member(z,np.array([300.,400.]),lobt)
        np.testing.assert_allclose(p,z['pred']+[30,0])
        np.testing.assert_array_equal(z['values'],values)

    def test_feature_allowlist_excludes_labels_identifiers_month_and_target_encodings(self):
        T=self.module()
        self.assertLessEqual(len(T.FEATURES),100)
        self.assertFalse(set(T.FEATURES)&{'TAXITIME_SEC_mvt','MVT_ID_mvt','BLOCK_TIME_UTC_mvt','month','t_block','split'})
        self.assertFalse(any(c.startswith(('te_','ter_')) for c in T.FEATURES))
        self.assertEqual(len(T.FEATURES),len(set(T.FEATURES)))

    def test_checkpoint_hash_uses_exact_loaded_model_path(self):
        T=self.module()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/T.CHECKPOINT;path.write_bytes(b'fixed-model')
            self.assertEqual(T.checkpoint_files(SimpleNamespace(model_path_=str(path))),{str(path):T.sha(path)})


if __name__=='__main__':unittest.main()

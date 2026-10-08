"""Extra-seed (3..9) adapter: arithmetic identity with the frozen three-seed mean; no data or fits."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
FILE=ROOT/'scripts/final_ordered_trajectory_seeds_20261008.py'
if FILE.exists():
    spec=importlib.util.spec_from_file_location('final_nn_seeds',FILE)
    X=importlib.util.module_from_spec(spec);spec.loader.exec_module(X);F=X.F
else:X=F=None


class ExtraSeedTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(X,'extra-seed adapter contract required')

    def members(self,n,rows=50):
        rng=np.random.default_rng(7);base=rng.normal(900,50,rows)
        return [rng.normal(size=rows).astype(np.float32) for _ in range(n)],base

    def test_constants_keep_recipe_and_extend_seeds_only(self):
        self.assertEqual(X.NEW_SEEDS,(3,4,5,6,7,8,9));self.assertEqual(X.ALL_SEEDS,tuple(range(10)))
        self.assertEqual(F.CONFIG['epochs'],8);self.assertEqual(F.CONFIG['seeds'],[0,1,2]) # frozen parent config unchanged
        self.assertNotEqual(X.PROTOCOL,F.PROTOCOL);self.assertNotEqual(X.SELF,F.SELF)

    def test_ten_seed_mean_uses_frozen_arithmetic_and_first_three_replay(self):
        seeds,base=self.members(10);sd=123.5
        ten=X.seed_mean_n(seeds,sd,base,10)
        np.testing.assert_array_equal(X.seed_mean_n(seeds[:3],sd,base,3),F.seed_mean(seeds[:3],sd,base))
        np.testing.assert_array_equal(ten,np.mean(seeds,axis=0)*sd+base)
        self.assertEqual(ten.shape,base.shape);self.assertTrue(np.isfinite(ten).all())
        # Averaging more seeds moves the member toward the population mean of the seeds, not toward zero residual.
        self.assertLess(X.rms(ten,base),X.rms(F.seed_mean(seeds[:3],sd,base),base))

    def test_seed_mean_rejects_wrong_count_dtype_and_nonfinite(self):
        seeds,base=self.members(10);sd=2.
        with self.assertRaises(ValueError):X.seed_mean_n(seeds[:9],sd,base,10)
        with self.assertRaises(ValueError):X.seed_mean_n([*seeds[:9],seeds[9].astype(np.float64)],sd,base,10)
        bad=seeds[9].copy();bad[0]=np.nan
        with self.assertRaises(ValueError):X.seed_mean_n([*seeds[:9],bad],sd,base,10)
        with self.assertRaises(ValueError):X.seed_mean_n(seeds,0.,base,10)
        with self.assertRaises(ValueError):X.seed_mean_n(seeds,sd,base[:-1],10)

    def test_pairwise_rms_symmetric_zero_diagonal_in_standard_units(self):
        seeds,_=self.members(4);pw=X.pairwise_rms(seeds)
        self.assertEqual(pw.shape,(4,4));np.testing.assert_array_equal(np.diag(pw),0)
        np.testing.assert_array_equal(pw,pw.T)
        self.assertAlmostEqual(pw[0,1],float(np.sqrt(np.mean((seeds[0].astype(np.float64)-seeds[1])**2))))

    def test_statistics_report_per_month_rms_and_global_shift(self):
        seeds,base=self.members(10,rows=60);sd=100.
        reg3=F.seed_mean(seeds[:3],sd,base);reg10=X.seed_mean_n(seeds,sd,base,10)
        months=np.array([1]*20+[2]*20+[7]*20)
        s=X.statistics(seeds,reg3,reg10,sd,months)
        self.assertEqual(set(s['per_month']),{'1','2','7'});self.assertEqual(s['per_month']['7']['rows'],20)
        self.assertAlmostEqual(s['rms_3seed_vs_10seed_sec'],X.rms(reg3,reg10))
        self.assertAlmostEqual(s['pairwise_seed_rms_sec_mean'],s['pairwise_seed_rms_std_units_mean']*sd)
        json.dumps(s,allow_nan=False)

    def test_admit_rejects_foreign_parent_receipt_before_touching_frozen_admission(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);parent=d/'parent.json';parent.write_text(json.dumps(dict(protocol=F.PROTOCOL,config=F.CONFIG)))
            good=dict(protocol=X.PROTOCOL,config=F.CONFIG,seeds=list(X.NEW_SEEDS),all_seeds=list(X.ALL_SEEDS),epochs=8,
                      registration_commit='a'*40,parent_protocol=F.PROTOCOL,parent_registration_commit='b'*40,
                      parent_receipt_sha256=F.C.sha(parent),source_sha256={X.SELF:'0'*64,X.TEST:'0'*64})
            for key,value in [('parent_receipt_sha256','f'*64),('seeds',[0,1,2]),('epochs',9),('config',{**F.CONFIG,'epochs':9}),
                              ('all_seeds',list(range(9))),('protocol',F.PROTOCOL)]:
                bad=d/'bad.json';bad.write_text(json.dumps({**good,key:value}))
                with self.assertRaises(ValueError,msg=key):X.admit(d,bad,parent)
            for name in (X.SELF,X.TEST):(d/name).parent.mkdir(parents=True,exist_ok=True);(d/name).write_text(name)
            drift=d/'drift.json';drift.write_text(json.dumps({**good,'source_sha256':{X.SELF:F.C.sha(d/X.SELF),X.TEST:'0'*64}}))
            with self.assertRaisesRegex(ValueError,'source drift'):X.admit(d,drift,parent)


if __name__=='__main__':unittest.main()

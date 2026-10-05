import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cat_laptop_pools as L

class LaptopPoolTests(unittest.TestCase):
    def test_original_double_clipped_residual_is_preserved(self):
        y=np.array([-100.,200.,8000.,90000.]);base=np.array([0.,300.,-1000.,9000.])
        np.testing.assert_array_equal(L.residual(y,base),[0.,-100.,7200.,-1800.])
    def test_feature_order_must_match_original(self):
        L.check_columns(['a','b'],['b'],['a','b'],['b'])
        with self.assertRaises(ValueError):L.check_columns(['b','a'],['b'],['a','b'],['b'])
        with self.assertRaises(ValueError):L.check_columns(['a','b'],[],['a','b'],['b'])

if __name__=='__main__':unittest.main()

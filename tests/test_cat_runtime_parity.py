import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cat_runtime_parity as P

class ParityTests(unittest.TestCase):
    def test_ids_and_predictions_must_match_exactly(self):
        ids=np.array([1.,2.]);ref=np.array([123.,456.])
        self.assertTrue(P.compare(ids,ref,ids,ref)['exact'])
        self.assertFalse(P.compare(ids,ref,ids,ref+1e-10)['exact'])
        for other in [ids[::-1],np.array([1.,3.])]:
            with self.assertRaises(ValueError):P.compare(ids,ref,other,ref)
        with self.assertRaises(ValueError):P.compare(ids,ref,ids,np.array([123.,np.nan]))

if __name__=='__main__':unittest.main()

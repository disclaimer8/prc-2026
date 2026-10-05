import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import polars as pl
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

class FixedSlotsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('e159_fixed_submission'), 'fixed submission guard is missing')
        import e159_fixed_submission as S
        self.S = S
        self.ids = np.arange(44,dtype=float)
        self.base = self.frame(np.full(44,100))
        self.before = self.frame(np.r_[np.full(42,90),100,100])
        self.full = self.frame(np.r_[np.full(42,100),103,102])
        self.half = self.frame(np.r_[np.full(42,100),102,101])
        self.context = pl.DataFrame({S.ID:self.ids,'ADEP_mvt':['LIRF']*42+['LSZH']*2,'d_mvt_lobt':np.full(44,np.nan)})
    def frame(self,y):
        return pl.DataFrame({'MVT_ID_mvt':self.ids,'TAXITIME_SEC_mvt':np.asarray(y,dtype=np.int32)})
    def run_check(self, full=None, half=None, context=None):
        return self.S.validate_pair(self.full if full is None else full,self.half if half is None else half,self.base,self.base,self.before,self.base,self.context if context is None else context,rows=44)
    def test_accepts_tie_to_even_rounding_and_shuffled_ids(self):
        self.assertEqual(self.run_check(full=self.full.reverse(),half=self.half.reverse())['12060']['changed'],2)
    def test_rejects_fractional_ids(self):
        with self.assertRaises(self.S.GuardError):
            self.run_check(full=self.full.with_columns((pl.col(self.S.ID)+.1).alias(self.S.ID)))
    def test_rejects_duplicate_ids(self):
        with self.assertRaises(self.S.GuardError):
            self.run_check(full=self.full.with_columns(pl.lit(1.).alias(self.S.ID)))
    def test_rejects_non_lszh_change(self):
        ctx=self.context.with_columns(pl.when(pl.col(self.S.ID)==43).then(pl.lit('EDDF')).otherwise(pl.col('ADEP_mvt')).alias('ADEP_mvt'))
        with self.assertRaises(self.S.GuardError): self.run_check(context=ctx)
    def test_rejects_donor_drift(self):
        z=self.full.with_columns(pl.when(pl.col(self.S.ID)==0).then(101).otherwise(pl.col(self.S.TARGET)).cast(pl.Int32).alias(self.S.TARGET))
        with self.assertRaises(self.S.GuardError): self.run_check(full=z)
    def test_rejects_wrong_half_rounding(self):
        z=self.frame(np.r_[np.full(42,100),101,101])
        with self.assertRaises(self.S.GuardError): self.run_check(half=z)
    def test_rejects_lobt_violation(self):
        ctx=self.context.with_columns(pl.when(pl.col(self.S.ID)==43).then(4000.).otherwise(pl.col('d_mvt_lobt')).alias('d_mvt_lobt'))
        with self.assertRaises(self.S.GuardError): self.run_check(context=ctx)
    def test_rejects_null_prediction(self):
        z=self.full.with_columns(pl.when(pl.col(self.S.ID)==43).then(None).otherwise(pl.col(self.S.TARGET)).alias(self.S.TARGET))
        with self.assertRaises(self.S.GuardError): self.run_check(full=z)
    def test_rejects_wrong_schema(self):
        z=self.full.with_columns(pl.col(self.S.TARGET).cast(pl.Float64))
        with self.assertRaises(self.S.GuardError): self.run_check(full=z)
    def test_rejects_archive_added_after_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'data/opdi').mkdir(parents=True)
            (root/'data/opdi/ev_extra.parquet').write_bytes(b'extra')
            with self.assertRaises(self.S.GuardError): self.S.verify_inventory(root,{'inputs':{}})
    def test_reservation_blocks_existing_version_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'submissions').mkdir()
            artifact=root/'submissions/gentle-igloo_v12060.parquet';artifact.write_bytes(b'prior')
            with self.assertRaises(self.S.GuardError): self.S.reserve(root,'a'*40)
            self.assertEqual(artifact.read_bytes(),b'prior')
    def test_rejects_wrong_lineage_inventory(self):
        with self.assertRaises(self.S.GuardError): self.S.validate_lineage({'inputs':{},'sources':{}})

if __name__=='__main__': unittest.main()

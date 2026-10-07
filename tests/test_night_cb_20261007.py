"""Synthetic CB parsing, temporal isolation, support and preregistration contracts."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/night_cb_20261007.py'


class CBContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('cb', SOURCE)
        cls.cb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cb)

    def test_cloud_group_excludes_tcu_text_and_tracks_base_amount(self):
        self.assertEqual(self.cb.parse_cb('EHAM SCT020CB BKN035CB', ''), (1, 1, 2000., 3.))
        self.assertEqual(self.cb.parse_cb('EHAM SCT020CB', 'TSRA'), (1, 0, 2000., 2.))
        no_cb=self.cb.parse_cb('EHAM SCT020TCU RMK CB', '')
        self.assertEqual(no_cb[:2],(0,0))
        self.assertTrue(np.isnan(no_cb[2:]).all())
        self.assertTrue(np.isnan(self.cb.parse_cb(None,'')).all())

    def test_asof_cannot_select_future_or_stale_weather(self):
        index = self.cb.backward_indices(np.array([10,20]), np.array([9,10,19,20,10820,10821]))
        np.testing.assert_array_equal(index, [-1,0,0,1,1,-1])

    def test_duplicate_weather_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            self.cb.backward_indices(np.array([10,10]), np.array([10]))

    def test_id_alignment_rejects_missing_or_duplicate_rows(self):
        np.testing.assert_array_equal(self.cb.align_ids(np.array([9,3]), np.array([3,9])), [1,0])
        for source in [np.array([3,3]),np.array([3,8])]:
            with self.assertRaises(ValueError):
                self.cb.align_ids(np.array([9,3]), source)

    def test_correction_preserves_excluded_and_undecided_exactly(self):
        pred = self.cb.apply_correction(np.array([100.,200.,300.,400.]),
            np.array([900.,-900.,80.,-80.]), np.array([True,False,True,True]),
            np.array([True,True,False,True]))
        np.testing.assert_array_equal(pred,[160.,200.,300.,360.])

    def test_fit_indices_exclude_entire_heldout_month_pair(self):
        month=np.array([1,7,2,8,3,9]);y=np.array([100.,100.,100.,8000.,100.,100.])
        fit,valid=self.cb.fold_masks(month,y,np.ones(6,bool),np.array([1,1,1,1,0,1],bool),1)
        np.testing.assert_array_equal(np.flatnonzero(fit),[2,5])
        np.testing.assert_array_equal(np.flatnonzero(valid),[0,1])

    def test_body_harm_or_trim_harm_cannot_pass_gate(self):
        def row(body=-1,trim=-1,sign=-1):
            return {'body':{'delta_sse':body},'decided':{'delta_sse':sign,'trim3_sse':trim,'trim20_sse':trim}}
        self.assertTrue(self.cb.gate([row() for _ in range(6)]))
        self.assertFalse(self.cb.gate([row() for _ in range(5)]))
        self.assertFalse(self.cb.gate([row(body=1) for _ in range(6)]))
        self.assertFalse(self.cb.gate([row(trim=1) for _ in range(6)]))
        self.assertFalse(self.cb.gate([row(sign=1) for _ in range(3)]+[row() for _ in range(3)]))
        self.assertFalse(self.cb.gate([row(sign=-1) for _ in range(4)]+[row(sign=100) for _ in range(2)]))

    def test_missing_registration_blocks_pack_before_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaises(ValueError):
                self.cb.pack(root,root/'output', '')
            self.assertFalse((root/'output').exists())

    def test_empty_metric_masks_fail_and_zero_sse_is_finite(self):
        zeros=np.zeros(50)
        with self.assertRaises(ValueError):
            self.cb.paired(zeros,zeros,zeros,np.zeros(50,bool))
        result=self.cb.paired(zeros,zeros,zeros,np.ones(50,bool))
        self.assertEqual(result['delta_sse'],0.)
        self.assertEqual(result['delta_rmse'],0.)
        self.assertFalse(self.cb.gate([{'body':result,'decided':result}]*6))

    def test_nan_prediction_cannot_enter_metrics_or_correction(self):
        x=np.zeros(50);bad=x.copy();bad[0]=np.nan
        with self.assertRaises(ValueError):
            self.cb.paired(x,x,bad,np.ones(50,bool))
        with self.assertRaises(ValueError):
            self.cb.apply_correction(x,bad,np.ones(50,bool),np.ones(50,bool))


if __name__ == '__main__':
    unittest.main()

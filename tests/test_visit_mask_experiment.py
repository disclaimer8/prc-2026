"""Temporal and population contracts for the ground-mask augmentation probe."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class VisitMaskTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('visit_mask_experiment'), 'experiment module missing')
        import visit_mask_experiment as E
        return E

    def frame(self):
        # First two qualify; held-out July must never enter January+July fit.
        d = dict(MVT_ID_mvt=[1., 2., 3., 4., 5., 6.], ADEP_mvt=['LIRF']*5+['LTFM'],
                 has_nm=[True, True, True, True, False, True], month=[2,7,3,4,5,6],
                 coverage=[2,3,2,2,2,2], base_icao=['a','a','a',None,'a','a'],
                 air_icao=['a','a','b','b','a','a'], air_vi_has_prev=[1]*6)
        for c in ['vi_first_seen','vi_turnaround','vi_gap','vi_gs_first']:
            d['base_'+c] = [10.]*6; d['air_'+c] = [20.]*6
        d['base_vi_has_prev'] = [1]*6
        return pl.DataFrame(d)

    def test_training_synthetic_population_is_identity_stable_nm_lirf_only(self):
        E = self.module()
        self.assertEqual(E.synthetic_mask(self.frame()).tolist(), [True,True,False,False,False,False])

    def test_heldout_months_and_ids_cannot_enter_augmented_fit(self):
        E = self.module(); d = self.frame()
        natural = d.with_columns(pl.lit(0).alias('coverage'))
        fit, extra, test = E.split_frames(natural, d, 1)
        self.assertNotIn(2., fit['MVT_ID_mvt'].to_list())
        self.assertEqual(extra['MVT_ID_mvt'].to_list(), [1.])
        self.assertEqual(test['MVT_ID_mvt'].to_list(), [2.])

    def test_changed_dropped_fields_alone_do_not_create_synthetic_rows(self):
        E = self.module(); d = self.frame()
        for c in E.A.ACTIVE:
            d = d.with_columns(pl.col('base_'+c).alias('air_'+c))
        d = d.with_columns(pl.lit(999.).alias('air_vi_alt_first'))
        self.assertFalse(E.synthetic_mask(d).any())

    def test_metrics_body_and_trim_cannot_be_rescued_by_one_tail(self):
        E = self.module()
        y = np.array([100.]*25+[10000.]); c = y + 10; n = y + 11; c[-1] += 1000
        m = E.metrics(y,c,n)
        self.assertLess(m['delta_rmse'],0)
        self.assertGreater(m['body_delta_rmse'],0)
        self.assertGreater(m['trimmed_delta_rmse'],0)

    def test_aggregate_trims_each_fold_before_pooling(self):
        E=self.module();rows=[]
        for _ in range(6):
            y=np.full(25,100.);c=y+10;n=y+11;c[-1]+=1000
            rows.append((y,c,n))
        self.assertTrue(hasattr(E,'aggregate_metrics'),'per-fold aggregate missing')
        result=E.aggregate_metrics(rows)
        self.assertGreater(result['trimmed_delta_rmse'],0)
        self.assertEqual(result['trimmed_n'],6*19)

    def test_calendar_is_exact_2025_and_known_airborne_track_has_coverage_one(self):
        E = self.module()
        self.assertEqual(len(E.DAYS),365)
        self.assertEqual((E.DAYS[0],E.DAYS[-1]),('2025-01-01','2025-12-31'))
        d = self.frame(); s = E.synthetic_features(d)
        # adsb_offblock_v2.main: known callsign, no surface candidate -> 1.
        self.assertEqual(s['coverage'].to_list(),[1]*6)
        self.assertEqual(s['vi_first_seen'].to_list(),[20.]*6)
        self.assertEqual(d['coverage'].to_list(),[2,3,2,2,2,2])

    def test_missing_source_day_keeps_ids_but_cannot_create_sensor_rows(self):
        E=self.module()
        self.assertTrue(hasattr(E,'missing_observations'),'explicit missing-source representation missing')
        meta=pl.DataFrame({'MVT_ID_mvt':[1.,2.],'ADEP_mvt':['LIRF','LTFM'],'has_nm':[True,True],'month':[12,12]})
        missing=E.missing_observations(meta)
        self.assertEqual(missing['MVT_ID_mvt'].to_list(),[1.,2.])
        self.assertEqual(missing['coverage'].null_count(),2)
        self.assertEqual(missing['base_vi_has_prev'].to_list(),[0,0])
        self.assertFalse(E.synthetic_mask(missing).any())


if __name__ == '__main__': unittest.main()

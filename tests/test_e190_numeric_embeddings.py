import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import e190_numeric_embeddings as P


class E190Contracts(unittest.TestCase):
    def test_bins_use_training_only_preserve_binary_and_constant(self):
        x = np.column_stack([np.arange(100), np.arange(100) % 2, np.ones(100)]).astype('float32')
        continuous, scalar, bins = P.numeric_bins(x, n_bins=4)
        self.assertEqual(continuous, [0])
        self.assertEqual(scalar, [1, 2])
        np.testing.assert_allclose(bins[0], [0, 24.75, 49.5, 74.25, 99])
        self.assertRaises(ValueError, P.numeric_bins, np.array([[0.], [np.nan]], dtype='float32'))

    def test_alignment_rejects_missing_duplicate_ids_and_target_drift(self):
        expected = dict(mvt_id=np.array([1., 2.]), y=np.array([10., 20.]), month=np.array([1, 7]))
        z = dict(mvt_id=np.array([2., 1.]), y=np.array([20., 10.]), month=np.array([7, 1]), reg2=np.array([200., 100.]))
        np.testing.assert_array_equal(P.align(z, expected), [100., 200.])
        for key, value in [('mvt_id', [1., 1.]), ('mvt_id', [1., 3.]), ('y', [21., 10.]), ('month', [8, 1]), ('reg2', [np.nan, 1.])]:
            bad = dict(z); bad[key] = np.array(value)
            self.assertRaises(ValueError, P.align, bad, expected)

    def test_literal_member_replacement_and_motion_attenuation(self):
        anchors = {k: np.full(3, 100.) for k in P.ANCHORS}
        base = np.full(3, 100.); old = np.full(3, 100.); new = np.array([140., 140., -100.])
        motion = np.full(3, 800.); subset = np.array([False, True, False])
        r0 = P.regular(base, anchors, old, motion, subset)
        r1 = P.regular(base, anchors, new, motion, subset)
        np.testing.assert_allclose(r1-r0, [10., 40.*23/120, -50.])
        np.testing.assert_allclose(r0, [100., 100.*23/30+800.*7/30, 100.])

    def test_mixture_updates_day_endpoint_and_preserves_correction(self):
        z = dict(reg=np.array([100.,100.]), pred=np.array([101.,86502.]),
                 values=np.array([[100.,np.nan,0.,0.,0.,86500.], [100.,0.,0.,0.,0.,86500.]]),
                 proba=np.array([[1.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,1.]]))
        out=P.assemble(z,np.array([150.,150.]),np.full(2,np.nan))
        np.testing.assert_allclose(out['pred'],[151.,86552.])
        np.testing.assert_array_equal(out['values'][:,0],out['reg'])
        np.testing.assert_array_equal(out['values'][:,-1],out['reg']+86400.)
        neutral=P.assemble(z,z['reg'],np.full(2,np.nan))
        np.testing.assert_array_equal(neutral['pred'],z['pred'])

    def test_clipping_is_applied_after_mixture_not_each_member(self):
        z=dict(reg=np.array([100.]),pred=np.array([100.]),values=np.zeros((1,6)),proba=np.array([[1.,0.,0.,0.,0.,0.]]))
        self.assertEqual(P.assemble(z,np.array([-100.]),np.array([np.nan]))['pred'][0],0.)
        self.assertEqual(P.assemble(z,np.array([10000.]),np.array([5000.]))['pred'][0],100.+8606.-1394.)


if __name__=='__main__':unittest.main()

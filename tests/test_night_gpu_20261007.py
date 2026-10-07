import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def worker():
    spec = importlib.util.find_spec('night_gpu_20261007')
    assert spec is not None, 'E204 worker has not been implemented'
    import night_gpu_20261007
    return night_gpu_20261007


class E204Contracts(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('rtdl_num_embeddings'), 'qualified PLE package is on Home')
    def test_network_preserves_capacity_initialization_and_mask_semantics(self):
        p = worker()
        root = Path(__file__).resolve().parents[1]
        p.S.fixed_env()
        n = p.E.trainer(root/'data/mlp_split_20261005/inputs/fold_1_7/src')
        x = np.random.default_rng(204).normal(size=(64,20)).astype(np.float32)
        cont,scalar,bins = p.E.numeric_bins(x)
        args = (n,[3,4],cont,scalar,bins,20)
        torch.manual_seed(17);control=p.ORIGINAL_NETWORK(*args,'ordered').eval()
        torch.manual_seed(17);identity=p.network(*args,'multiscale',dilations=(1,1)).eval()
        torch.manual_seed(17);candidate=p.network(*args,'multiscale').eval()
        self.assertEqual(sum(a.numel() for a in control.parameters()),sum(a.numel() for a in candidate.parameters()))
        self.assertEqual(list(control.state_dict()),list(candidate.state_dict()))
        for key in control.state_dict(): self.assertTrue(torch.equal(control.state_dict()[key],candidate.state_dict()[key]))
        xn=torch.from_numpy(x[:8]);xc=torch.zeros((8,2),dtype=torch.int64)
        v=torch.randn((8,2,64,12));k=torch.ones_like(v,dtype=torch.uint8);present=torch.ones((8,2,64),dtype=torch.uint8)
        present[0]=0;present[1,1]=0;present[2,:,30:]=0;k[3,:,:,4]=0
        inp=(xn,xc,v,k,present)
        with torch.no_grad():
            old=control(*inp);new=candidate(*inp)
            self.assertTrue(torch.equal(old,identity(*inp)))
            self.assertFalse(torch.allclose(old[1:],new[1:],atol=1e-7,rtol=1e-7))
            changed=v.clone();changed[k==0]=999;changed[present==0]=999
            self.assertTrue(torch.equal(new,candidate(xn,xc,changed,k,present)))
        prediction=candidate(*inp);prediction.square().mean().backward()
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(all(a.grad is None or torch.isfinite(a.grad).all() for a in candidate.parameters()))

    def test_dilation_uses_physical_stream_boundaries_and_masks(self):
        p = worker()
        h = torch.tensor([[[[1.], [2.], [300.], [4.], [5.]],
                           [[9.], [8.], [7.], [6.], [5.]]]])
        valid = torch.tensor([[[True, True, False, True, True], [True]*5]])
        got = p.dilated_triplets(h, valid, 2)
        self.assertEqual(got[0, 0].tolist(), [[0., 1., 0.], [0., 2., 4.],
                         [1., 0., 5.], [2., 4., 0.], [0., 5., 0.]])
        self.assertEqual(got[0, 1, 0].tolist(), [0., 9., 7.])
        self.assertRaises(ValueError, p.dilated_triplets, h, valid, 0)

    def test_two_scales_see_nine_offsets_with_span_nineteen(self):
        p = worker()
        h = torch.zeros((1, 2, 32, 1))
        h[0, 0, 15, 0] = 1.
        valid = torch.ones((1, 2, 32), dtype=torch.bool)
        first = p.dilated_triplets(h, valid, 1).sum(-1, keepdim=True)
        second = p.dilated_triplets(first, valid, 8).sum(-1)
        self.assertEqual(torch.nonzero(second[0, 0]).flatten().tolist(), [6,7,8,14,15,16,22,23,24])
        self.assertFalse(bool(second[0, 1].any()))

    @staticmethod
    def report(fold, delta=-1.):
        return dict(fold=fold, metrics=dict(body_sse=delta, decided_sse=delta,
                    decided_trim3_sse=delta, decided_trim20_sse=delta))

    def test_screen_requires_both_pairs_and_all_primary_populations(self):
        p = worker()
        self.assertTrue(p.gate([self.report(1), self.report(4)], 'screen'))
        self.assertFalse(p.gate([self.report(1), self.report(4, .01)], 'screen'))
        r = self.report(4); r['metrics']['decided_trim20_sse'] = 0
        self.assertFalse(p.gate([self.report(1), r], 'screen'))
        self.assertRaises(ValueError, p.gate, [self.report(1)], 'screen')
        self.assertRaises(ValueError, p.gate, [self.report(1),self.report(1)], 'screen')

    def test_confirmation_cannot_be_rescued_by_selected_pairs(self):
        p = worker()
        reports = [self.report(f) for f in (1,4,2,3,5,6)]
        self.assertTrue(p.gate(reports, 'six'))
        reports = [self.report(1,-100),self.report(4,-100),self.report(2,-1),
                   self.report(3,-1),self.report(5,2),self.report(6,2)]
        self.assertFalse(p.gate(reports, 'six'))
        reports[-1]['metrics']['body_sse'] = float('nan')
        self.assertRaises(ValueError,p.gate,reports,'six')

    def test_exact_prediction_metadata_and_finite_member_are_required(self):
        p = worker()
        meta = dict(mvt_id=np.array([1.,2.]), y=np.array([30.,40.]), month=np.array([1,7]))
        good = dict(**meta, pred=np.array([31.,39.]), member=np.array([30.,40.]))
        p.check_prediction(good,meta)
        for key,val in [('mvt_id',[2.,1.]),('mvt_id',[1.,1.]),('y',[30.,41.]),
                        ('month',[1,8]),('member',[30.,np.nan]),('pred',[31.,np.inf])]:
            bad = dict(good);bad[key] = np.array(val)
            self.assertRaises(ValueError,p.check_prediction,bad,meta)

    def test_resource_fit_receipt_rejects_incomplete_training_and_excess_vram(self):
        p = worker()
        fit = dict(seed=0, losses=[1.]*8, peak_gpu_bytes=12*1024**3, parameters=123, peak_rss_gib=8.)
        p.check_fit(fit,0,123)
        for key,val in [('losses',[1.]*7),('seed',1),('parameters',124),
                        ('peak_gpu_bytes',21*1024**3),('peak_rss_gib',21.)]:
            bad=dict(fit);bad[key]=val
            self.assertRaises(ValueError,p.check_fit,bad,0,123)


if __name__ == '__main__': unittest.main()

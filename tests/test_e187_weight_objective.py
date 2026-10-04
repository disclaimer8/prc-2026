import importlib.util
import contextlib
import io
import itertools
import json
import subprocess
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

SPEC = importlib.util.spec_from_file_location('e187', Path(__file__).parents[1] / 'src/e187_weight_objective.py')
E = importlib.util.module_from_spec(SPEC)
if SPEC.loader and Path(SPEC.origin).exists():
    SPEC.loader.exec_module(E)


def fixture(root):
    (root/'src').mkdir()
    (root/'src/train.py').write_text('DAY = 86400.0\nLOBT_WINDOW = 3606\n')
    for name in ('e172_members_cv.py','fold_metrics.py'):
        (root/'src'/name).write_text((Path(__file__).parents[1]/'src'/name).read_text())
    rows = []
    for k in range(1, 7):
        name = f'fold_{k}_{k+6}.npz'
        ids = np.arange(k * 100, k * 100 + 8, dtype=float)
        months = np.array([k]*4 + [k+6]*4)
        z = dict(mvt_id=ids, y=np.full(8, 1100.), pred=np.full(8, 1000.),
                 reg=np.full(8, 1000.), proba=np.tile([1., 0., 0., 0., 0., 0.], (8, 1)),
                 values=np.tile([999., 2000., 0., 0., 0., 99999.], (8, 1)))
        for dirname in ['data/cv_preds_fin1b_hz', 'data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlpa_f1b']:
            p = root / dirname
            p.mkdir(parents=True, exist_ok=True)
            np.savez(p / name, **z)
        for member in ('raw', 'lobt', 'sched', 'eobt', 'aobt', 'mlpa'):
            p = root / f'data/cv_e130{member}'
            p.mkdir(parents=True, exist_ok=True)
            np.savez(p / name, mvt_id=ids[::-1], reg2=np.full(8, 2000.))
        rows.extend(dict(MVT_ID_mvt=x, month=int(m), split='train', ADEP_mvt='EDDF',
                         FLIGHT_ID_mvt=1., d_mvt_sched=100., d_mvt_lobt=None,
                         TAXITIME_SEC_mvt=1100.) for x, m in zip(ids, months))
    pl.DataFrame(rows, schema_overrides={'d_mvt_lobt': pl.Float64}).write_parquet(root / 'data/features_fin1b.parquet')


class ObjectiveTests(unittest.TestCase):
    def test_mislabeled_npy_and_duplicate_archive_names_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.npz'
            with path.open('wb') as f: np.save(f,np.array([1.]))
            with self.assertRaises(E.InvalidInput): E.read_npz(path)
            buffer=io.BytesIO(); np.save(buffer,np.array([1.]))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',UserWarning)
                with zipfile.ZipFile(path,'w') as z:
                    z.writestr('pred.npy',buffer.getvalue())
                    z.writestr('pred.npy',buffer.getvalue())
            with self.assertRaises(E.InvalidInput): E.read_npz(path)

    def test_malformed_prediction_and_corrupt_zip_cli_return_invalid_status(self):
        for mutation in ('string_prediction', 'broken_zip'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); fixture(root)
                path = root/'data/cv_preds_fin1b_hz/fold_1_7.npz'
                if mutation == 'broken_zip':
                    path.write_bytes(b'PK\x03\x04broken')
                else:
                    with np.load(path) as saved: z = {key:saved[key] for key in saved.files}
                    z['pred'] = np.full(8, 'bad')
                    np.savez(path, **z)
                child = subprocess.run([str(Path(__file__).parents[1]/'.venv/bin/python'), str(Path(SPEC.origin)),
                                        '--repo-root',str(root),'--output',str(root/'run')],
                                       text=True,capture_output=True,timeout=15)
                self.assertEqual(child.returncode, 2, child.stderr)
                self.assertIn('INVALID:',child.stderr)
                self.assertNotIn('Traceback',child.stderr)
                self.assertFalse((root/'run/manifest.json').exists())

    def test_all_numeric_prediction_member_and_metadata_arrays_reject_nonreal_dtypes(self):
        for key in ('pred','reg','y','proba','values','mvt_id','reg2','feature_y','feature_lobt','feature_delay','feature_month'):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); fixture(root)
                if key.startswith('feature_'):
                    path=root/'data/features_fin1b.parquet'
                    column={'feature_y':'TAXITIME_SEC_mvt','feature_lobt':'d_mvt_lobt',
                            'feature_delay':'d_mvt_sched','feature_month':'month'}[key]
                    pl.read_parquet(path).with_columns(pl.col(column).cast(pl.String)).write_parquet(path)
                else:
                    dirname='data/cv_e130raw' if key=='reg2' else 'data/cv_preds_fin1b_hz'
                    path=root/dirname/'fold_1_7.npz'
                    with np.load(path) as saved: z={name:saved[name] for name in saved.files}
                    z[key]=np.full(z[key].shape, 'bad')
                    np.savez(path, **z)
                with self.assertRaises(E.InvalidInput): E.validate_inputs(root,E.load_features(root/'data/features_fin1b.parquet'))

    def test_fixed_grid_order_constraints_and_zero_tie(self):
        grid = E.fixed_grid()
        self.assertEqual(tuple(grid[0]), (0.,)*6)
        self.assertEqual(tuple(grid[1]), (0., .05, .05, .05, .10, .15))
        self.assertTrue(np.all(grid.sum(axis=1) <= 1.+1e-9))
        box = [(0,.1),(.05,.2),(.05,.15),(.05,.15),(.1,.2),(.15,.35)]
        axes = [np.round(np.arange(a,b+.01,.05),2) for a,b in box]
        want = [w for w in itertools.product(*axes)
                if sum(w)<=1.+1e-9 and all(a-1e-9<=v<=b+1e-9 for v,(a,b) in zip(w,box))]
        np.testing.assert_array_equal(grid[1:], want)
        all_scores = np.zeros((6,len(grid)))
        self.assertEqual(E.select_indices(all_scores)[0], 0)

    def test_scalar_clipped_mixture_preserves_delta_and_rebuilds_values(self):
        z = dict(reg=np.array([0.,100.,200.]), pred=np.array([17.,25.,300.]),
                 proba=np.array([[.3,.2,.1,.1,.1,.2],[1.,0,0,0,0,0],[1.,0,0,0,0,0]]),
                 values=np.array([[9.,np.nan,30.,40.,50.,9.],[8.,0,0,0,0,8.],[8.,0,0,0,0,8.]]))
        lobt = np.array([100.,np.nan,-4000.])
        members = np.tile([1000.,2000.,3000.,4000.,5000.,6000.],(3,1))
        w = np.array([0,.05,.05,.05,.1,.15])
        result = E.build_prediction(z,members,lobt,w)
        reg = (1.-w.sum())*z['reg'] + members@w
        expected = []
        for i in range(3):
            def scalar(r):
                values = z['values'][i].astype(float).copy()
                values[0],values[-1] = r,r+86400.
                value = sum(float(p)*float(v) for p,v in zip(z['proba'][i],np.nan_to_num(values)))
                if not np.isnan(lobt[i]): value = min(max(value,lobt[i]-3606),lobt[i]+3606)
                return max(value,0.)
            expected.append(z['pred'][i]+scalar(reg[i])-scalar(z['reg'][i]))
        np.testing.assert_allclose(result['pred'],expected,atol=1e-10,rtol=0)
        np.testing.assert_array_equal(result['values'][:,0],reg)
        np.testing.assert_array_equal(result['values'][:,-1],reg+86400.)
        self.assertEqual(z['values'][0,0],9.)

    def test_overwritten_tail_changes_all_selection_and_heldout_labels_cannot_select(self):
        z = dict(y=np.array([100.,10000.]), pred=np.array([0.,0.]), reg=np.array([0.,0.]),
                 proba=np.tile([1.,0,0,0,0,0],(2,1)), values=np.zeros((2,6)))
        members = np.full((2,6),10000.)
        grid = np.array([[0.]*6,[1.,0,0,0,0,0]])
        all_sse,body_sse = E.score_grid(z,members,np.full(2,np.nan),np.array([True,False]),grid,row_chunk=1,grid_chunk=1)
        self.assertEqual(np.argmin(all_sse),1)
        self.assertEqual(np.argmin(body_sse),0)
        scores = np.tile(all_sse,(6,1))
        selected = E.select_indices(scores)
        scores[2] = [1e30,0]
        self.assertEqual(E.select_indices(scores)[2],selected[2])

    def test_chunked_exact_scores_equal_scalar_clipped_scores(self):
        rng = np.random.default_rng(91)
        z = dict(y=rng.uniform(0,20000,31), pred=rng.uniform(0,5000,31), reg=rng.uniform(0,7200,31),
                 proba=rng.dirichlet(np.ones(6),31).astype(np.float32), values=rng.uniform(0,86400,(31,6)).astype(np.float32))
        z['values'][0,2] = np.nan
        lobt = rng.uniform(-4000,90000,31); lobt[::3]=np.nan
        members = rng.uniform(0,7200,(31,6))
        grid = E.fixed_grid()[::77]
        mask = np.arange(31)%3==0
        a,b = E.score_grid(z,members,lobt,mask,grid,row_chunk=7,grid_chunk=4)
        for j,w in enumerate(grid):
            error = (E.build_prediction(z,members,lobt,w)['pred']-z['y'])**2
            self.assertAlmostEqual(a[j]/1e10,error.sum()/1e10,places=12)
            self.assertAlmostEqual(b[j]/1e10,error[mask].sum()/1e10,places=12)

    def test_input_alignment_and_all_sixfold_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); fixture(root)
            features = E.load_features(root/'data/features_fin1b.parquet')
            folds = E.validate_inputs(root,features)
            self.assertEqual(len(folds),6)
            z,r,meta = E.load_fold(root,folds[0],features)
            np.testing.assert_array_equal(r,np.full((8,6),2000.))
            np.testing.assert_array_equal(meta['month'],[1]*4+[7]*4)
            (root/'data/cv_e130raw/fold_6_12.npz').unlink()
            with self.assertRaises(E.InvalidInput): E.validate_inputs(root,features)

    def test_duplicate_mismatched_nonfinite_inputs_reject(self):
        for mutation in ('duplicate','target','nonfinite','month','coverage'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); fixture(root)
                features_path=root/'data/features_fin1b.parquet'
                if mutation in ('month','coverage'):
                    df=pl.read_parquet(features_path)
                    if mutation=='month': df=df.with_columns(pl.when(pl.col('MVT_ID_mvt')==100).then(2).otherwise(pl.col('month')).alias('month'))
                    else: df=df.filter(pl.col('MVT_ID_mvt')!=100)
                    df.write_parquet(features_path)
                else:
                    p=root/'data/cv_preds_fin1b_hz/fold_1_7.npz'
                    with np.load(p) as s: z={k:s[k].copy() for k in s.files}
                    if mutation=='duplicate': z['mvt_id'][1]=z['mvt_id'][0]
                    elif mutation=='target': z['y'][0]=999.
                    else: z['pred'][0]=np.inf
                    np.savez(p,**z)
                with self.assertRaises(E.InvalidInput): E.validate_inputs(root,E.load_features(features_path))

    def test_exact_protection_mask_and_metrics_gate(self):
        meta=dict(month=np.array([1,7,1,7,1,7,1,7]), airport=np.array(['LIRF']*8),
                  nm_missing=np.ones(8,bool), delay=np.array([14400.,14401.,108000.,108001.,0.,np.nan,1.,15000.]))
        masks=E.feature_masks(meta)
        np.testing.assert_array_equal(masks['decided'],[False,False,False,False,True,True,True,False])
        np.testing.assert_array_equal(masks['protected'],[False,True,True,False,False,False,False,True])
        metrics=E.compare_fold(np.zeros(8),np.ones(8),np.full(8,2.),meta)
        self.assertEqual(metrics['sse']['all'],-24.)
        self.assertEqual(metrics['sse']['trimmed'],-6.)
        self.assertEqual(metrics['sse']['protected'],-9.)
        self.assertEqual(metrics['sse']['protected_tail'],0.)
        self.assertEqual(metrics['sse']['nonprotected_tail'],0.)

    def test_reference_parity_fails_before_completed_publication_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); fixture(root); out=root/'run'
            path=root/'data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlpa_f1b/fold_1_7.npz'
            with np.load(path) as saved: z={key:saved[key] for key in saved.files}
            z['pred']=z['pred']+1.
            np.savez(path,**z)
            with self.assertRaises(E.InvalidInput):
                E.run(root,out,row_chunk=4,grid_chunk=64,progress=False)
            self.assertFalse((out/'manifest.json').exists())
            with self.assertRaises(FileExistsError): E.run(root,out,progress=False)

    def test_complete_synthetic_run_rejects_gate_and_outputs_synchronized_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); fixture(root); out=root/'run'
            result=E.run(root,out,row_chunk=5,grid_chunk=23,progress=False)
            self.assertTrue(result['complete'])
            self.assertFalse(result['primary']['gate_pass'])
            self.assertTrue(result['primary']['gate_flags']['six_complete'])
            self.assertEqual(len(result['fold_results']),6)
            self.assertEqual(result['primary']['sse']['all'],0.)
            self.assertEqual(json.loads((out/'manifest.json').read_text())['complete'],True)
            for dirname in ('all_selected','body_selected','fixed_production_pre_tail'):
                self.assertEqual(len(list((out/dirname).glob('fold_*.npz'))),6)
                with np.load(out/dirname/'fold_1_7.npz') as z:
                    np.testing.assert_array_equal(z['values'][:,0],z['reg'])
                    np.testing.assert_array_equal(z['values'][:,-1],z['reg']+86400.)
                    self.assertTrue(z['complete'])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(E.main(['--repo-root',str(root),'--output',str(root/'second')]),1)

    def test_extra_fold_and_member_nonfinite_and_shuffled_reference_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); fixture(root)
            path=root/'data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlpa_f1b/fold_1_7.npz'
            with np.load(path) as saved: z={key:saved[key][::-1] for key in saved.files}
            np.savez(path,**z)
            E.validate_inputs(root,E.load_features(root/'data/features_fin1b.parquet'))
            extra=root/'data/cv_e130raw/fold_7_1.npz'
            np.savez(extra,mvt_id=np.array([1.]),reg2=np.array([1.]))
            with self.assertRaises(E.InvalidInput): E.validate_inputs(root,E.load_features(root/'data/features_fin1b.parquet'))
            extra.unlink()
            path=root/'data/cv_e130raw/fold_1_7.npz'
            with np.load(path) as saved: z={key:saved[key] for key in saved.files}
            z['reg2'][0]=np.nan
            np.savez(path,**z)
            with self.assertRaises(E.InvalidInput): E.validate_inputs(root,E.load_features(root/'data/features_fin1b.parquet'))


if __name__=='__main__': unittest.main()

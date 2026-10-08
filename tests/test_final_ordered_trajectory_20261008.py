"""Synthetic final-query transformations; no challenge data or fits."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import polars as pl

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
FILE=ROOT/'scripts/final_ordered_trajectory_20261008.py'
if FILE.exists():
    spec=importlib.util.spec_from_file_location('final_nn',FILE)
    F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
else:F=None


class FinalQueryTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(F,'final ordered adapter contract required')

    def frames(self):
        fit=pl.DataFrame({'x':np.arange(140,dtype=float),'missing':[None]+list(map(float,range(139))),
                          'constant':[3.]*140,'sparse':[1.]+[None]*139,
                          'cat':['a']*60+['b']*40+[None]*20+['rare']*19+['last']})
        pred=pl.DataFrame({'x':[-10.,70.,1000.],'missing':[None,10.,200.],'constant':[3.]*3,
                           'sparse':[None]*3,'cat':['never','a',None]},schema_overrides={'sparse':pl.Float64})
        return fit,pred

    def test_numeric_category_saved_state_replays_original_arithmetic(self):
        fit,pred=self.frames();cols=fit.columns
        arrays,state=F.prepare_tabular(fit,pred,cols,['cat'])
        xn,xc=F.apply_tabular(pred,state)
        np.testing.assert_array_equal(xn,arrays['xnp']);np.testing.assert_array_equal(xc,arrays['xcp'])
        self.assertEqual(xn.shape,(3,3));self.assertEqual(xn.dtype,np.float32)
        self.assertEqual(xc[0,0],0);self.assertNotEqual(xc[2,0],0)
        self.assertEqual(xn[0].tolist(),[-2.,0.,1.])
        self.assertEqual(state['numeric'][0]['column'],'x')
        self.assertEqual([p['column'] for p in state['numeric']],['x','missing'])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'transforms.json';F.write_json(path,state)
            xx,cc=F.apply_tabular(pred,F.read_json(path))
            np.testing.assert_array_equal(xx,xn);np.testing.assert_array_equal(cc,xc)

    def test_category_codes_preserve_captured_original_order_and_frequency_boundary(self):
        fit,pred=self.frames();a,s=F.prepare_tabular(fit,pred,fit.columns,['cat'])
        c=F.apply_tabular(fit,s)[1][:,0]
        self.assertTrue(np.all(c[120:]==0)) # 19 rare + one singleton never enter vocabulary.
        self.assertTrue(np.all(c[100:120]>0)) # Exactly20 null values are a learned category.
        np.testing.assert_array_equal(c,a['xcf'][:,0])

    def test_transform_rejects_missing_feature_and_tampered_cardinality(self):
        fit,pred=self.frames();_,s=F.prepare_tabular(fit,pred,fit.columns,['cat'])
        with self.assertRaises(ValueError):F.apply_tabular(pred.drop('x'),s)
        s['card'][0]=1
        with self.assertRaises(ValueError):F.apply_tabular(pred,s)

    def test_final_ids_use_admitted_dynamic_template_without_old_rank_limit(self):
        ids=np.arange(670790,dtype=np.int64)
        F.validate_ids(ids,ids[::-1],670790)
        with self.assertRaises(ValueError):F.validate_ids(ids[:344841],ids,670790)
        wrong=ids.copy();wrong[-1]=wrong[0]
        with self.assertRaises(ValueError):F.validate_ids(wrong,ids,670790)

    def test_seed_mean_requires_complete_three_finite_float32_members(self):
        seeds=[np.array([1.,2.],np.float32),np.array([2.,3.],np.float32),np.array([3.,4.],np.float32)]
        np.testing.assert_array_equal(F.seed_mean(seeds,2.,np.array([10.,20.])),[14.,26.])
        for bad in (seeds[:2],[seeds[0],seeds[1],np.array([np.nan,0],np.float32)]):
            with self.assertRaises(ValueError):F.seed_mean(bad,2.,np.array([10.,20.]))

    def test_original_unfilled_sampler_remains_original_on_duplicate_times(self):
        t=np.r_[np.zeros(80),np.ones(80)*100]
        chosen=F.R.sample_time_indices(t)
        self.assertLess(len(chosen),64)
        np.testing.assert_array_equal(chosen,F.original_indices(t))

    def test_raw_contract_rejects_future_and_unknown_padding(self):
        import datetime as dt
        dep=pl.DataFrame({'MVT_ID_mvt':[5],'PHASE_mvt':['DEP'],'ADEP_mvt':['EDDF'],
            'CALLSIGN_flt':['AAA'],'FLIGHT_mvt':['AAA'],'FLIGHT_ID_mvt':['f'],
            'MVT_TIME_UTC_mvt':[dt.datetime(2026,2,1,12)]})
        arrays,_=F.R.encode_day(None,dep,'2026-02-01')
        self.assertTrue(callable(getattr(F,'validate_raw',None)))
        F.validate_raw(arrays,dep)
        arrays['point_times'][0,0,0]=arrays['takeoff'][0]+1
        with self.assertRaises(ValueError):F.validate_raw(arrays,dep)
        arrays,_=F.R.encode_day(None,dep,'2026-02-01');arrays['values'][0,0,0,0]=1
        with self.assertRaises(ValueError):F.validate_raw(arrays,dep)

    def test_train_only_normalization_and_safe_artifact_paths(self):
        z=np.zeros((4,2,64,12),np.float32);z[:2]=3;z[2:]=1e8;k=np.ones_like(z,np.uint8)
        count,total,sq=F.S.norm_stats(z,k,np.array([True,True,False,False]))
        mean,std=F.S.normalization(count,total,sq)
        np.testing.assert_array_equal(mean,np.full(6,3.));np.testing.assert_array_equal(std,np.ones(6))
        self.assertTrue(callable(getattr(F,'safe',None)))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for bad in ('../escape','/absolute','nested/../../escape'):
                with self.assertRaises(ValueError):F.safe(root,bad)
            self.assertEqual(F.safe(root,'data/valid'),(root/'data/valid').resolve())

    def test_artifacts_reject_content_drift_extra_files_and_wrong_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);r=d/'receipt.json';out=d/'out';out.mkdir()
            F.write_json(r,{'registration_commit':'a'*40})
            (out/'one.txt').write_text('bound')
            F.finish(out,'synthetic',r,synthetic_only=True)
            F.checked(out,'synthetic',r)
            (out/'one.txt').write_text('changed')
            with self.assertRaises(ValueError):F.checked(out,'synthetic',r)
            (out/'one.txt').write_text('bound');(out/'unexpected').write_text('extra')
            with self.assertRaises(ValueError):F.checked(out,'synthetic',r)
            (out/'unexpected').unlink()
            F.write_json(d/'another.json',{'registration_commit':'b'*40})
            with self.assertRaises(ValueError):F.checked(out,'synthetic',d/'another.json')

    def test_staging_failure_is_marked_without_publishing_complete_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'target'
            with self.assertRaisesRegex(RuntimeError,'synthetic'):
                with F.staging(out) as stage:
                    (stage/'partial').write_text('small');raise RuntimeError('synthetic failure')
            self.assertFalse(out.exists());failed=list(Path(tmp).glob('.target.partial-*'))
            self.assertEqual(len(failed),1)
            self.assertEqual(F.read_json(failed[0]/'failure.json')['error'],'RuntimeError')

    def test_ple_bins_save_load_exact_and_have_no_heldout_input(self):
        train=np.column_stack([np.arange(150,dtype=np.float32),np.zeros(150,np.float32)])
        c,s,b=F.E.numeric_bins(train)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bins.npz';F.save_bins(path,c,s,b);cc,ss,bb=F.load_bins(path)
            self.assertEqual(c,cc);self.assertEqual(s,ss);self.assertEqual(c,[0]);self.assertEqual(s,[1])
            for left,right in zip(b,bb):np.testing.assert_array_equal(left,right)

    def test_reuse_accepts_only_bound_original_training_recipe(self):
        import datetime as dt
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);source=d/'input';out=d/'original';source.mkdir();out.mkdir()
            (source/'departures').mkdir();day='2025-01-01';(out/day).mkdir()
            dep=pl.DataFrame({'MVT_ID_mvt':[5],'PHASE_mvt':['DEP'],'ADEP_mvt':['EDDF'],
                'CALLSIGN_flt':['AAA'],'FLIGHT_mvt':['AAA'],'FLIGHT_ID_mvt':['f'],
                'MVT_TIME_UTC_mvt':[dt.datetime(2025,1,1,12)]})
            dp=source/'departures'/f'{day}.parquet';dep.write_parquet(dp)
            arrays,stats=F.R.encode_day(None,dep,day);npz=out/day/'sequence.npz';np.savez_compressed(npz,**arrays)
            stats.update(source_sha256=F.C.sha(F.R.__file__),npz_sha256=F.C.sha(npz))
            F.write_json(out/day/'receipt.json',stats)
            manifest=dict(protocol=F.R.PROTOCOL,registration_commit=F.S.RAW_REG,departure_columns=F.R.DEP_COLS,
                numeric_columns=F.R.NUM_NAMES,days=[day],raw_input_sha256={'data/training_one.parquet':'bound'},
                point_inputs={day:{'exists':False}},files={'scripts/e194_trajectory_sequence.py':F.C.sha(F.R.__file__),
                'departures/'+day+'.parquet':F.C.sha(dp)})
            F.write_json(source/'manifest.json',manifest)
            F.write_json(out/'complete.json',dict(protocol=F.R.PROTOCOL,registration_commit=F.S.RAW_REG,
                input_manifest_sha256=F.C.sha(source/'manifest.json'),reports=[stats]))
            index=F.reusable_training([(source,out)],{'training_one.parquet':'bound'})
            self.assertEqual(index[day]['sha256'],F.C.sha(npz))
            with self.assertRaises(ValueError):F.reusable_training([(source,out)],{'training_one.parquet':'changed'})
            manifest['protocol']='E207-filled'
            (source/'manifest.json').unlink();F.write_json(source/'manifest.json',manifest)
            with self.assertRaises(ValueError):F.reusable_training([(source,out)],{'training_one.parquet':'bound'})

    def test_target_encoding_capture_preserves_month_out_and_full_stats(self):
        self.assertTrue(callable(getattr(F,'capture_encodings',None)),'used TE/TER state must be persisted')
        spec=importlib.util.spec_from_file_location('original_te',ROOT/'src/target_encoding.py')
        te=importlib.util.module_from_spec(spec);spec.loader.exec_module(te)
        n=240
        fit=pl.DataFrame({'MVT_ID_mvt':np.arange(n),'month':[1]*120+[2]*120,
             'TAXITIME_SEC_mvt':[400.]*120+[800.]*120,'d_mvt_aobt':[200.]*n,
             'ADEP_mvt':['EDDF']*n,'STAND_mvt':['A1']*n,'RUNWAY_mvt':['09L']*n,
             'local_hour':[12.5]*n,'AIRCRAFT_OPERATOR_flt':['OP']*n,
             'AIRCRAFT_TYPE_mvt':['A320']*n,'ADES_mvt':['LFPG']*n})
        pred=fit.head(2).with_columns(pl.Series('MVT_ID_mvt',[1000,1001]),pl.lit(None,dtype=pl.Float64).alias('TAXITIME_SEC_mvt'))
        a,b,state=F.capture_encodings(fit,pred,np.ones(n,bool),te)
        self.assertEqual(a['te_stand'][0],800.);self.assertEqual(a['te_stand'][120],400.)
        self.assertEqual(b['te_stand'].to_list(),[600.,600.]);self.assertEqual(b['ter_stand_rwy'][0],400.)
        replay=F.apply_encodings(pred,state,'full')
        self.assertEqual(replay['te_stand_n'].to_list(),[240,240])
        self.assertTrue(replay.select(state['columns']).equals(b.select(state['columns'])))
        with self.assertRaises(ValueError):F.capture_encodings(fit,pred.with_columns(pl.lit(999.).alias('TAXITIME_SEC_mvt')),np.ones(n,bool),te)


if __name__=='__main__':unittest.main()

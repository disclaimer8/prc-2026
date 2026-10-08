import datetime as dt
import importlib.util
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import patch
from collections import Counter

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def worker():
    assert importlib.util.find_spec('e207_full_token_sampler_v3') is not None, 'E207 worker missing'
    import e207_full_token_sampler_v3 as w
    return w


class E207Contracts(unittest.TestCase):
    def test_systemd_retained_success_failure_and_identity(self):
        import e207_full_token_queue_v3_20261008 as q
        starting=dict(ActiveState='activating',SubState='start',MainPID='0',InvocationID='owned',Result='success',ExecMainStatus='0')
        running=dict(starting,ActiveState='active',SubState='running',MainPID='123')
        exited=dict(starting,ActiveState='active',SubState='exited')
        failed=dict(starting,ActiveState='failed',SubState='failed',Result='exit-code',ExecMainStatus='7')
        self.assertFalse(q.active(exited))
        for state in (starting,running,dict(starting,ActiveState='deactivating',SubState='stop')):
            self.assertTrue(q.active(state));self.assertFalse(q.unit_finished(state,starting))
        self.assertTrue(q.unit_finished(exited,starting))
        for state in (failed,dict(exited,ExecMainStatus='7'),dict(running,InvocationID='replacement'),dict(exited,InvocationID=''),dict(exited,Result='timeout')):
            with self.subTest(state=state),self.assertRaises(ValueError):q.unit_finished(state,starting)

    def test_real_compressed_archive_materialized_once_independent_of_rows(self):
        w=worker();original=np.lib.npyio.NpzFile.__getitem__
        for rows in (1,7):
            points,departure=self.fixture()
            deps=pl.concat([departure.with_columns(pl.lit(float(i+1)).alias('MVT_ID_mvt')) for i in range(rows)])
            old,_=w.R.encode_day(points,deps,'2025-01-01');expected,_=w.encode_day(points,deps,'2025-01-01')
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);points.write_parquet(root/'surface.parquet');deps.write_parquet(root/'dep.parquet')
                np.savez_compressed(root/'old.npz',**old)
                record=dict(day='2025-01-01',shard=0,dep='dep.parquet',surface='surface.parquet',old='old.npz')
                for key in ('dep','surface','old'):record[key+'_sha256']=w.C.sha(root/record[key])
                calls=Counter()
                def counted(archive,key):
                    calls[key]+=1
                    return original(archive,key)
                with patch.object(np.lib.npyio.NpzFile,'__getitem__',counted):
                    receipt=w.extract_day((str(root),record))
                self.assertEqual(calls,Counter({key:1 for key in old}),f'rows={rows}: {calls}')
                with np.load(root/w.RAW/'s0/2025-01-01/sequence.npz',allow_pickle=False) as actual:
                    for key,value in expected.items():
                        self.assertTrue(np.array_equal(actual[key],value,equal_nan=True) if value.dtype.kind in 'fc' else np.array_equal(actual[key],value),key)
                self.assertEqual(receipt['stats']['original_selected_points'],int(old['present'].sum()))

    def test_irregular_duplicate_times_fill_slots_without_losing_raw_indices(self):
        w = worker()
        for n in [0, 1, 2, 63, 64, 65, 100, 501]:
            t = np.sort(np.random.default_rng(n).integers(0, 17, n)).astype(float)
            old = w.R.sample_time_indices(t)
            got = w.sample_full_indices(t)
            self.assertEqual(len(got), min(64, n))
            self.assertEqual(len(np.unique(got)), len(got))
            self.assertTrue(np.all(np.diff(got) > 0))
            self.assertTrue(set(old).issubset(got))
            self.assertTrue(np.array_equal(got, w.sample_full_indices(t)))
            self.assertTrue(np.all((got >= 0) & (got < n)))
            if n <= 64:
                self.assertEqual(got.tolist(), list(range(n)))
        # Raw indices remain distinct even when timestamps are identical.
        got = w.sample_full_indices(np.zeros(100))
        self.assertEqual(len(got), 64)
        self.assertEqual((got[0], got[-1]), (0, 99))

    def test_nonfinite_or_unsorted_times_are_rejected(self):
        w = worker()
        for t in [[1., 0.], [0., np.nan], [0., np.inf]]:
            with self.assertRaises(ValueError): w.sample_full_indices(np.array(t))

    @staticmethod
    def fixture():
        anchor = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
        times = np.r_[np.arange(100)/10, 100., 200., 310.]
        deps = pl.DataFrame({'MVT_ID_mvt':[1.], 'PHASE_mvt':['DEP'], 'ADEP_mvt':['LIRF'],
            'CALLSIGN_flt':['TEST'], 'FLIGHT_mvt':['TEST'], 'FLIGHT_ID_mvt':[1.],
            'MVT_TIME_UTC_mvt':[anchor+dt.timedelta(seconds=300)]})
        n=len(times)
        points=pl.DataFrame({'icao':['abc123']*n,'airport':['LIRF']*n,
            't':anchor.timestamp()+times,'lat':[41.8]*n,'lon':[12.25]*n,
            'alt':[None]*n,'gs':np.arange(n).astype(float),'track':[0.]*n,
            'on_ground':[True]*n,'callsign':['TEST']*n})
        return points,deps

    def test_encode_preserves_metadata_old_values_and_future_cutoff(self):
        w=worker();points,deps=self.fixture()
        old,_=w.R.encode_day(points,deps,'2025-01-01')
        new,_=w.encode_day(points,deps,'2025-01-01')
        w.validate_arrays(old,new)
        self.assertEqual(new['present'][0,0].sum(),64)
        self.assertEqual(new['raw_counts'][0,0],102)
        self.assertTrue(np.isnan(new['point_times'][0,1]).all())
        self.assertTrue((new['source_rows'][0,1]==-1).all())
        self.assertTrue((new['point_times'][0,0]<=new['takeoff'][0]).all())
        selected=new['source_rows'][0,0]
        self.assertTrue(np.array_equal(new['point_times'][0,0],points['t'].to_numpy()[selected]))
        self.assertFalse(np.array_equal(old['present'],new['present']))

    def test_validation_rejects_old_value_metadata_or_pad_mutation(self):
        w=worker();points,deps=self.fixture()
        old,_=w.R.encode_day(points,deps,'2025-01-01');new,_=w.encode_day(points,deps,'2025-01-01')
        for kind in ['values','ids','future','pad','count','nan_time','negative_source']:
            bad={k:v.copy() for k,v in new.items()}
            if kind=='values':bad['values'][0,0,0,5]+=1
            elif kind=='ids':bad['ids'][0]+=1
            elif kind=='future':bad['point_times'][0,0,-1]=bad['takeoff'][0]+1
            elif kind=='pad':bad['source_rows'][0,1,0]=0
            elif kind=='nan_time':bad['point_times'][0,0,-1]=np.nan
            elif kind=='negative_source':bad['source_rows'][0,0,-1]=-1
            else:bad['present'][0,0,-1]=0
            with self.subTest(kind=kind),self.assertRaises(ValueError):w.validate_arrays(old,bad)

    def test_sixpair_gate_requires_decided_body_trims_and_four_pairs(self):
        w=worker();reports=[dict(fold=i,metrics=dict(body_sse=-1.,decided_sse=-1.,decided_trim3_sse=-1.,decided_trim20_sse=-1.)) for i in range(1,7)]
        self.assertTrue(w.gate(reports))
        for key in ['body_sse','decided_sse','decided_trim3_sse','decided_trim20_sse']:
            bad=[dict(fold=r['fold'],metrics=dict(r['metrics'])) for r in reports]
            bad[-1]['metrics'][key]=6.
            self.assertFalse(w.gate(bad))
        bad=[dict(fold=r['fold'],metrics=dict(r['metrics'])) for r in reports]
        for r in bad[3:]:r['metrics']['decided_sse']=.1
        self.assertFalse(w.gate(bad))
        with self.assertRaises(ValueError):w.gate(reports[:-1])
        with self.assertRaises(ValueError):w.gate(reports[:-1]+[reports[0]])

    def test_training_uses_exact_frozen_model_and_fit(self):
        w=worker()
        self.assertIs(w.network,w.S.network)
        self.assertIs(w.fit,w.S.fit)
        self.assertIs(w.normalize,w.S.normalize)
        self.assertIs(w.norm_stats,w.S.norm_stats)

    def test_preparation_rebinds_provenance_without_mutating_frozen_globals(self):
        w=worker()
        self.assertIs(w.prepare_fold.__code__,w.S.prepare_fold.__code__)
        self.assertIs(w.verify_prepared.__code__,w.S.verify_prepared.__code__)
        self.assertEqual(w.S.FOLDS,(1,4))
        self.assertEqual(w.prepare_fold.__globals__['FOLDS'],(1,2,3,4,5,6))
        self.assertIs(w.prepare_fold.__globals__['verify_bundle'],w.verify_bundle)
        self.assertIs(w.R.encode_day.__globals__['sample_time_indices'],w.R.sample_time_indices)

    def test_normalizer_cannot_see_changed_holdout_trajectories(self):
        w=worker();v=np.arange(8*2*64*12,dtype=float).reshape(8,2,64,12);known=np.ones(v.shape,np.uint8)
        mask=np.array([True,False]*4);original=w.norm_stats(v,known,mask)
        v[~mask]=1e9;known[~mask]=0
        for before,after in zip(original,w.norm_stats(v,known,mask)):self.assertTrue(np.array_equal(before,after))

    def test_derived_raw_rejects_duplicate_days_and_wrong_registration(self):
        w=worker();reg='a'*40
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);records=[]
            for shard in (0,1):
                day=f'2025-01-0{shard+1}';records.append(dict(day=day,rows=1,shard=shard))
                folder=root/w.RAW/f's{shard}';folder.mkdir(parents=True)
                w.C.write_json(folder/'complete.json',dict(protocol=w.PROTOCOL,registration_commit=reg,input_spec_sha256='parent',shard=shard,
                    reports=[dict(day=day,rows=1,sha256='b'*64,stats=dict(selected_points=64))]))
            m=dict(spec_sha256='parent',spec=dict(days=records))
            got=w.derived_raw(root,m,reg)
            self.assertEqual([r['day'] for r in got['raw_days']],['2025-01-01','2025-01-02'])
            with self.assertRaises(ValueError):w.derived_raw(root,m,'c'*40)
            path=root/w.RAW/'s0/complete.json';r=w.C.read_json(path);r['reports']*=2;path.write_text(__import__('json').dumps(r))
            with self.assertRaises(ValueError):w.derived_raw(root,m,reg)

    def test_signal_cleanup_targets_only_the_owned_process_group(self):
        import e207_full_token_queue_v3_20261008 as q
        owned=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
        other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
        try:
            q.terminate_owned_process(owned)
            self.assertIsNotNone(owned.poll())
            self.assertIsNone(other.poll())
            with self.assertRaises(InterruptedError):q.raise_stop(15,None)
        finally:
            q.terminate_owned_process(owned);q.terminate_owned_process(other)

    def test_actual_signal_while_shared_lock_held_finishes_cleanup(self):
        code='''
import signal,sys,threading
sys.path.insert(0,'scripts')
import e207_full_token_queue_v3_20261008 as q
lock=threading.RLock();stopped=threading.Event();events=[];started=[False]
def cleanup():
    started[0]=True
    with lock:
        events.append('cleaned')
        signal.raise_signal(signal.SIGTERM)  # Repeated signal is ignored.
handler=q.stopping_handler(cleanup,lambda:started[0],stopped)
signal.signal(signal.SIGTERM,handler)
try:
    with lock:signal.raise_signal(signal.SIGTERM)
except InterruptedError:events.append('interrupted')
assert events==['cleaned','interrupted'] and stopped.is_set()
# First signal arriving during an existing cleanup must not abort that cleanup.
signal.signal(signal.SIGTERM,handler)
with stopped._cond:signal.raise_signal(signal.SIGTERM)
assert events==['cleaned','interrupted']
'''
        subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],check=True,timeout=10)


if __name__=='__main__':unittest.main()

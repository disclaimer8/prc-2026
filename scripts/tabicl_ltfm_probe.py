"""Fixed TabICLv2 LTFM screen on two month pairs; no ranking or weight search.

Small regular-regime training contexts, raw clipped target. Frozen .15 member
weight against E182+E185 pre-ADS-B OOF; this is NOT full v12061 validation.
"""
from __future__ import annotations
import argparse
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','POLARS_MAX_THREADS'):
    os.environ.setdefault(key,'2')
os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR','load_as_storage')
import numpy as np
import polars as pl

PROTOCOL='TabICLv2-LTFM-2pairs-1000permonth-2048test-raw-w015-v1'
ID,Y='MVT_ID_mvt','TAXITIME_SEC_mvt'
CHECKPOINT='tabicl-regressor-v2-20260212.ckpt'
WEIGHT=.15
FEATURES=[
    'AIRCRAFT_TYPE_mvt','AIRCRAFT_OPERATOR_flt','MARKET_SEGMENT_flt','WK_TBL_CAT_flt',
    'FLIGHT_TYPE_flt','ades_region','airline_prefix','stand_prefix',
    'local_hour','local_dow','d_mvt_aobt','d_mvt_eobt','d_mvt_lobt','d_mvt_iobt',
    'd_mvt_sched','d_aobt_eobt','d_eobt_iobt','d_aobt_sched','flight_time_plan',
    'aobt_minute_exact','aobt_infeasible','diverted',
    'dep_before_10','dep_after_10','arr_before_10','dep_before_30','dep_after_30',
    'arr_before_30','dep_before_60','dep_after_60','arr_before_60',
    'arr_around_ob_10','arr_around_ob_30','offblocks_around_ob_10','offblocks_around_ob_30',
    'taxiing_at_ob','takeoffs_between_ob_mvt','taxi_in_mean_30','taxi_in_mean_60',
    'rwy_dep_before_10','rwy_dep_before_30','rwy_arr_before_10','rwy_since_last_arr','rwy_to_next_arr',
    'nb_ap_30_0_n','nb_ap_30_0_mean','nb_ap_30_0_std','nb_ap_last','nb_ap_last_age',
    'nb_rwy_60_0_mean','nb_rwy_last','nb_rwy_last_age',
    'fw_ap_15_mean','fw_ap_60_mean','fw_ap_next_gap','nbf_ap_30_0_mean','nbs_ap_30_0_mean',
    'wx_temp','wx_spread','wx_wind','wx_gust','wx_vis_mi','wx_ceiling_ft',
    'wx_snow','wx_freezing','wx_rain','wx_thunder','wx_fog','wx_icing',
    'wx_snow_recent','wx_icing_recent','metar_age_min','wxr_headwind','wxr_crosswind',
    'geo_dist_m','route_m','plan_taxi_res','plan_nm_taxi']


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def sample_indices(ids,months,regular,a,train_per_month=1000,test_per_month=2048):
    ids=np.asarray(ids);months=np.asarray(months);regular=np.asarray(regular,dtype=bool)
    if len(np.unique(ids))!=len(ids) or a not in (1,2):raise ValueError('invalid IDs/fold')
    rng=np.random.default_rng(18900+a);fit=[];test=[]
    for month in range(1,13):
        held=month in (a,a+6)
        eligible=np.flatnonzero((months==month)&(True if held else regular))
        eligible=eligible[np.argsort(ids[eligible],kind='stable')]
        count=min(len(eligible),test_per_month if held else train_per_month)
        selected=np.sort(rng.choice(eligible,size=count,replace=False))
        (test if held else fit).extend(selected.tolist())
    fit,test=np.asarray(fit,dtype=int),np.asarray(test,dtype=int)
    if set(ids[fit])&set(ids[test]) or np.isin(months[fit],[a,a+6]).any():raise RuntimeError('temporal leakage')
    return fit,test


def mix(z,reg,lobt):
    values=z['values'].astype(float).copy();values[:,0]=reg;values[:,-1]=reg+86400.
    pred=np.sum(z['proba'].astype(float)*np.nan_to_num(values),axis=1)
    return np.maximum(np.where(np.isnan(lobt),pred,np.clip(pred,lobt-3606.,lobt+3606.)),0.)


def blend_member(z,member,lobt):
    member=np.asarray(member,dtype=float)
    if member.shape!=z['reg'].shape or not np.isfinite(member).all():raise ValueError('invalid member')
    reg=(1-WEIGHT)*z['reg']+WEIGHT*member
    return z['pred']+mix(z,reg,lobt)-mix(z,z['reg'],lobt)


def prepare(root,bundle,registration):
    sys.path.insert(0,str(root/'src'))
    import train as T
    import e185_on_e182_fixed as E
    if T.TOL!=6 or T.REG_CLIP!=(0,7200) or os.environ.get('SCHED_TOL_LONG','0')!='0':raise RuntimeError('regime settings drift')
    bundle.mkdir(parents=True,exist_ok=False)
    parent=root/'data/e185_on_e182_w035_20261005/manifest.json'
    if sha(parent)!='e73a34324dd937bea1f1f6a8392079db675fba3024311376df87684855510e21':raise RuntimeError('wrong current member parent')
    feature_path=root/'data/features_fin1b.parquet'
    columns=list(dict.fromkeys([ID,Y,'split','ADEP_mvt','FLIGHT_ID_mvt','month',*FEATURES]))
    frame=pl.read_parquet(feature_path,columns=columns).filter((pl.col('split')=='train')&(pl.col('ADEP_mvt')=='LTFM')&pl.col('FLIGHT_ID_mvt').is_not_null()).sort(ID)
    if frame[ID].n_unique()!=frame.height:raise RuntimeError('duplicate feature IDs')
    regular=T.regime_label(frame)==0
    inputs={str(feature_path.relative_to(root)):sha(feature_path),str(parent.relative_to(root)):sha(parent),
            'src/train.py':sha(root/'src/train.py'),'src/e185_on_e182_fixed.py':sha(root/'src/e185_on_e182_fixed.py')}
    folds=[]
    for a in (1,2):
        fi,ti=sample_indices(frame[ID].to_numpy(),frame['month'].to_numpy(),regular,a)
        fit,test=frame[fi],frame[ti]
        fit.select([ID,'month',Y,*FEATURES]).write_parquet(bundle/f'fit_{a}.parquet')
        test.select([ID,'month',Y,*FEATURES]).write_parquet(bundle/f'test_{a}.parquet')
        path=root/f'data/e185_on_e182_w035_20261005/candidate/fold_{a}_{a+6}.npz';inputs[str(path.relative_to(root))]=sha(path)
        with np.load(path,allow_pickle=False) as loaded:z={k:loaded[k] for k in loaded.files}
        E.validate_arrays(z,path)
        order=np.argsort(z['mvt_id']);idx=order[np.searchsorted(z['mvt_id'][order],test[ID].to_numpy())]
        if not np.array_equal(z['mvt_id'][idx],test[ID]) or not np.array_equal(z['y'][idx],test[Y]):raise RuntimeError('parent ID/target mismatch')
        base={k:z[k][idx] for k in ['mvt_id','y','reg','pred','proba','values']}
        if not np.allclose(base['values'][:,0],base['reg'],rtol=0,atol=1e-6) or not np.allclose(base['values'][:,-1],base['reg']+86400,rtol=0,atol=1e-6):raise RuntimeError('stale mixture endpoints')
        np.savez_compressed(bundle/f'parent_{a}.npz',**base)
        folds.append(dict(fold=a,train_rows=fit.height,test_rows=test.height,train_months=sorted(fit['month'].unique().to_list()),test_months=sorted(test['month'].unique().to_list())))
    outputs={p.name:sha(p) for p in bundle.glob('*') if p.is_file()}
    for p,h in inputs.items():
        if sha(root/p)!=h:raise RuntimeError(f'input drift {p}')
    save(bundle/'manifest.json',dict(protocol=PROTOCOL,registration_commit=registration,features=FEATURES,folds=folds,
        input_sha256=inputs,files=outputs,checkpoint=CHECKPOINT,tabicl_version='2.2.0',weight=WEIGHT,
        code_sha256=sha(Path(__file__)),target='clip(y,0,7200), regular regime fit only',
        screen='both paired-month mixture body and trimmed deltas <0; no submission acceptance'))
    print(json.dumps(dict(prepared=str(bundle),folds=folds,features=len(FEATURES))),flush=True)


def model(checkpoint_path=None):
    # No LightGBM import in this process; ROCm torch loaded before other ML backends.
    import torch
    from tabicl import TabICLRegressor
    if importlib.metadata.version('tabicl')!='2.2.0' or not torch.cuda.is_available():raise RuntimeError('TabICL/ROCm unavailable')
    if torch.version.hip is None or '7900 XTX' not in torch.cuda.get_device_name(0):raise RuntimeError('wrong GPU backend')
    torch.set_num_threads(2);torch.manual_seed(42)
    return TabICLRegressor(n_estimators=4,batch_size=1,kv_cache='repr',device='cuda:0',
        checkpoint_version=CHECKPOINT,model_path=checkpoint_path,allow_auto_download=checkpoint_path is None,
        use_amp=True,use_fa3=False,offload_mode=False,
        random_state=42,n_jobs=2,verbose=True)


def runtime():
    import torch
    return dict(packages={n:importlib.metadata.version(n) for n in ['tabicl','torch','numpy','scikit-learn','polars','pandas','huggingface-hub']},
                gpu=torch.cuda.get_device_name(0),hip=torch.version.hip,
                max_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                max_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)


def checkpoint_files(fitted):
    # Pin the exact file the installed implementation loaded, not a guessed
    # HuggingFace organization/cache directory. Run reuses this path offline.
    path=Path(fitted.model_path_)
    if path.name!=CHECKPOINT or not path.is_file():raise RuntimeError('unexpected loaded checkpoint path')
    return {str(path):sha(path)}


def smoke(output):
    import pandas as pd
    rng=np.random.default_rng(42);x=pd.DataFrame(rng.normal(size=(288,8)),columns=[f'x{i}' for i in range(8)])
    x['cat']=np.where(x.x0>0,'A','B');x.loc[::13,'cat']=None;x.loc[::17,'x3']=np.nan
    y=1000+100*x.x0.to_numpy()+20*x.x1.to_numpy()
    t=time.monotonic();m=model();m.fit(x.iloc[:256],y[:256]);pred=np.asarray(m.predict(x.iloc[256:]))
    if pred.shape!=(32,) or not np.isfinite(pred).all():raise RuntimeError('smoke nonfinite/shape')
    # Forces completed GPU execution and captures the exact downloaded checkpoint.
    import torch
    torch.cuda.synchronize()
    result=dict(synthetic_only=True,passed=True,rows=32,seconds=time.monotonic()-t,
        runtime=runtime(),checkpoint_sha256=checkpoint_files(m),code_sha256=sha(Path(__file__)))
    save(output,result);print(json.dumps(result),flush=True)


def run(bundle,output,registration,smoke_report):
    import torch
    from visit_mask_experiment import metrics,aggregate_metrics
    m=json.loads((bundle/'manifest.json').read_text());s=json.loads(smoke_report.read_text())
    if (m['protocol']!=PROTOCOL or m['registration_commit']!=registration or m['features']!=FEATURES or
        m['code_sha256']!=sha(Path(__file__)) or not s['passed'] or s['code_sha256']!=sha(Path(__file__))):raise RuntimeError('protocol/code/smoke mismatch')
    for p,h in m['files'].items():
        if sha(bundle/p)!=h:raise RuntimeError(f'prepared input changed {p}')
    for p,h in s['checkpoint_sha256'].items():
        if sha(p)!=h:raise RuntimeError('checkpoint changed after smoke')
    if runtime()['packages']!=s['runtime']['packages']:raise RuntimeError('runtime changed after smoke')
    output.mkdir(parents=True,exist_ok=False);results=[];rows=[]
    for a in (1,2):
        start=time.monotonic();fit=pl.read_parquet(bundle/f'fit_{a}.parquet');test=pl.read_parquet(bundle/f'test_{a}.parquet')
        if set(fit[ID])&set(test[ID]) or fit.filter(pl.col('month').is_in([a,a+6])).height:raise RuntimeError('held-out month in fit')
        with np.load(bundle/f'parent_{a}.npz',allow_pickle=False) as p:z={k:p[k] for k in p.files}
        if not np.array_equal(z['mvt_id'],test[ID]) or not np.array_equal(z['y'],test[Y]):raise RuntimeError('ID order/target drift')
        if len(s['checkpoint_sha256'])!=1:raise RuntimeError('one immutable checkpoint required')
        reg=model(next(iter(s['checkpoint_sha256'])));reg.fit(fit.select(FEATURES).to_pandas(),np.clip(fit[Y].to_numpy(),0,7200))
        values=[];pdf=test.select(FEATURES).to_pandas()
        for begin in range(0,len(pdf),512):
            values.append(np.asarray(reg.predict(pdf.iloc[begin:begin+512])))
            print(json.dumps(dict(fold=a,predicted=min(begin+512,len(pdf)),total=len(pdf))),flush=True)
        member=np.maximum(np.concatenate(values),0.)
        candidate=blend_member(z,member,test['d_mvt_lobt'].to_numpy())
        result=metrics(z['y'],z['pred'],candidate);result.update(fold=a,seconds=time.monotonic()-start,runtime=runtime())
        rows.append((z['y'],z['pred'],candidate));results.append(result)
        np.savez_compressed(output/f'fold_{a}_{a+6}.npz',mvt_id=z['mvt_id'],y=z['y'],control=z['pred'],candidate=candidate,member=member)
        save(output/f'fold_{a}_{a+6}.json',result);print(json.dumps(result),flush=True)
        del reg;gc.collect();torch.cuda.empty_cache()
    total=aggregate_metrics(rows)
    passed=all(r['delta_rmse']<0 and r['body_delta_rmse']<0 and r['trimmed_delta_rmse']<0 for r in results)
    save(output/'result.json',dict(protocol=PROTOCOL,registration_commit=registration,folds=results,total=total,
        proceed_to_full_cv=passed,accepted_for_submission=False,input_manifest_sha=sha(bundle/'manifest.json'),
        smoke_sha=sha(smoke_report),limitation='Sampled LTFM NM pre-ADS-B E182+E185 component screen, not full current v12061 CV.',
        files={p.name:sha(p) for p in output.glob('*') if p.is_file()}))
    print(json.dumps(dict(proceed_to_full_cv=passed,accepted_for_submission=False)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','smoke','run'])
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--bundle',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke-report',type=Path)
    p.add_argument('--registration-commit');a=p.parse_args()
    if a.command!='smoke' and not re.fullmatch('[a-f0-9]{40}',a.registration_commit or ''):p.error('registration required')
    root=a.root.resolve();out=(root/a.output).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    with (out.parent/'.tabicl_probe.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.command=='prepare':prepare(root,out,a.registration_commit)
        elif a.command=='smoke':smoke(out)
        else:
            if not a.bundle or not a.smoke_report:p.error('bundle and smoke report required')
            run((root/a.bundle).resolve(),out,a.registration_commit,(root/a.smoke_report).resolve())


if __name__=='__main__':main()

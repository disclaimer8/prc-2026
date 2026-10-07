"""Bounded ADS-B diagnosis on frozen E195 joint/pre-E81 OOF; no ML fits.

Observable cross-half lag estimates and explicitly label-aware oracle diagnostics
are separate. No ranking predictions, pipeline changes or deployment decision.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import ctypes
import shutil
import subprocess
import tempfile
import time

for k,v in {'POLARS_MAX_THREADS':'2','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1',
            'MKL_NUM_THREADS':'1','POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR':'load_as_storage'}.items():os.environ[k]=v
import numpy as np
import polars as pl

PROTOCOL='ADSBAUDIT-E195joint-firstmove-crosshalf-lag-oracle-airportcoverage-v1'
SELF='scripts/adsb_stage_audit_20261007.py'
BASE='analysis/adsb_stage_audit_20261007'
DOC=BASE+'/protocol.md'
RECEIPT=BASE+'/input_receipt.json'
OOF='data/e195_pre_e81_20261007_v2'
STAGE='data/e195_stage_inputs_20261007'
ID,Y='MVT_ID_mvt','TAXITIME_SEC_mvt'
OOF_SHA='271b01811a624dc1eb01aa8c98d249d97a7eb2d3814062860beb54b3e3787ebb'
NATURAL_SHA='a82c14fc733b6f7355b9c7cca65a8cf223d13d2a05d7dba5adeb73420e952a16'
STAGE_SPEC='5c4a0281a6669d856331317128917097b2a87df55a5c44f3cb0f0d26602441b2'
FIELDS=('mvt_first_move','sw_mvt_first_move')
MIN_TRAIN=100
MAX_DURATION=43200
SOURCES=('scripts/cat_ally_windows.py','scripts/e195_current_chain.py','src/e36_adsb_stand_geometry.py','scripts/e195_pre_e81_comparison.py','src/adsb_offblock_v2.py','src/e30_adsb_stage_v2.py','src/e141_apply.py')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,x):
    with Path(path).open('x') as f:json.dump(x,f,sort_keys=True,indent=2,allow_nan=False)


def rss():
    if sys.platform=='win32':
        import psutil
        return int(psutil.Process().memory_info().peak_wset)
    import resource
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)


MEMORY_LIMIT=6*1024**3

def windows_limit(memory_limit=MEMORY_LIMIT):
    """Kernel-enforced committed-memory limit for this process, no global edits."""
    if sys.platform != 'win32':
        raise RuntimeError('Windows worker required')
    from ctypes import wintypes as W

    class Basic(ctypes.Structure):
        _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                    ('flags', W.DWORD), ('min_ws', ctypes.c_size_t),
                    ('max_ws', ctypes.c_size_t), ('active', W.DWORD),
                    ('affinity', ctypes.c_size_t), ('priority', W.DWORD),
                    ('scheduling', W.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ('read_ops', 'write_ops', 'other_ops', 'read', 'write', 'other')]

    class Extended(ctypes.Structure):
        _fields_ = [('basic', Basic), ('io', IO),
                    ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                    ('peak_process', ctypes.c_size_t), ('peak_job', ctypes.c_size_t)]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, W.LPCWSTR]
    kernel.CreateJobObjectW.restype = W.HANDLE
    kernel.SetInformationJobObject.argtypes = [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD]
    kernel.SetInformationJobObject.restype = W.BOOL
    kernel.QueryInformationJobObject.argtypes = [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD, ctypes.c_void_p]
    kernel.QueryInformationJobObject.restype = W.BOOL
    kernel.GetCurrentProcess.restype = W.HANDLE
    kernel.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
    kernel.AssignProcessToJobObject.restype = W.BOOL
    kernel.CloseHandle.argtypes = [W.HANDLE]
    kernel.SetThreadExecutionState.argtypes = [W.DWORD]
    kernel.SetThreadExecutionState.restype = W.DWORD
    handle = kernel.CreateJobObjectW(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    info = Extended()
    info.basic.flags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.process_memory = memory_limit
    if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())
    observed = Extended()
    if not kernel.QueryInformationJobObject(handle, 9, ctypes.byref(observed), ctypes.sizeof(observed), None):
        raise ctypes.WinError(ctypes.get_last_error())
    if observed.process_memory != memory_limit or not observed.basic.flags & 0x100:
        raise RuntimeError('Windows memory limit was not applied')
    return kernel, handle


def guard():
    if rss()>2*1024**3:raise MemoryError('audit exceeded2GiB measured RSS; stop')


def seal(root):
    dest=root/RECEIPT
    if dest.exists():raise ValueError('fresh sealed receipt required')
    stage=read(root/STAGE/'manifest.json');result=read(root/OOF/'result.json')
    if stage['spec_sha256']!=STAGE_SPEC or digest(stage['spec'])!=STAGE_SPEC or stage['spec']['files']['natural_features.parquet']!=NATURAL_SHA:raise ValueError('stage lineage differs')
    if result['registration']!='bc19847d23d4485717bafa84744a14b49af0da19' or result['outputs']['predictions.parquet']!=OOF_SHA or result['rows']!=2085047 or result['excluded']!=126:raise ValueError('frozen OOF lineage differs')
    if stage['spec']['worker_sha256']!=sha(root/'scripts/e195_current_chain.py'):raise ValueError('frozen stage builder source differs')
    for directory in ('data/adsblol_kt1','data/adsblol_sw_kt1'):
        if directory+'/offblock2_2025-01-01.parquet' not in stage['spec']['inputs']:raise ValueError('natural feature sensor lineage differs')
    paths=[SELF,DOC,*SOURCES,STAGE+'/manifest.json',STAGE+'/natural_features.parquet',OOF+'/result.json',OOF+'/predictions.parquet','data/features_fin1b.parquet']
    raw={}
    for folder in ('data/adsblol_kt1','data/adsblol_sw_kt1'):
        files=sorted(p for p in (root/folder).glob('offblock2_2026-*.parquet') if p.stem.split('_')[-1][5:7] in ('01','07'))
        expected={datetime.date(2026,mo,d).isoformat() for mo in (1,7) for d in range(1,32)}
        if {p.stem.split('_')[-1] for p in files}!=expected:raise ValueError('ranking raw day inventory differs')
        raw[folder]=[str(p.relative_to(root)) for p in files];paths.extend(raw[folder])
    inputs={p:sha(root/p) for p in paths}
    if inputs[OOF+'/predictions.parquet']!=OOF_SHA or inputs[STAGE+'/natural_features.parquet']!=NATURAL_SHA:raise ValueError('parent data hash differs')
    write(dest,dict(protocol=PROTOCOL,inputs=inputs,ranking_raw_files=raw,fields=list(FIELDS),min_train=MIN_TRAIN,max_duration=MAX_DURATION,comparison_computed=False))
    print('SEALED',len(inputs),'files; no comparative calculation',flush=True)


def verify(root,reg):
    if not re.fullmatch('[0-9a-f]{40}',reg or ''):raise ValueError('full registration commit required')
    for p in (SELF,DOC,RECEIPT):
        raw=subprocess.check_output(['git','show',reg+':'+p],cwd=root)
        if hashlib.sha256(raw).hexdigest()!=sha(root/p):raise ValueError('registered artifact drift')
    if PROTOCOL.encode() not in subprocess.check_output(['git','show',reg+':HYPOTHESES_LOG.md'],cwd=root):raise ValueError('registered journal entry missing')
    r=read(root/RECEIPT)
    if r['protocol']!=PROTOCOL or r['fields']!=list(FIELDS):raise ValueError('receipt protocol differs')
    for p,h in r['inputs'].items():
        if sha(root/p)!=h:raise ValueError('sealed input changed: '+p)
    return r


def valid(x,cov):return (cov>=2)&np.isfinite(x)&(x>=0)&(x<=MAX_DURATION)


def calibration(x,y,ap,cov,train,test,ids):
    """Read y only through train indices; test features determine lookup only."""
    tr=np.flatnonzero(train);te=np.flatnonzero(test)
    if not len(tr):raise ValueError('no training-half eligible observations')
    lag=x[tr]-y[tr];global_lag=float(np.median(lag));output=np.full(len(x),np.nan);records=[]
    for airport,coverage in sorted(set(zip(ap[te],cov[te]))):
        ix=te[(ap[te]==airport)&(cov[te]==coverage)]
        group=(ap[tr]==airport)&(cov[tr]==coverage);source='airport_coverage'
        if group.sum()<MIN_TRAIN:group=cov[tr]==coverage;source='coverage_fallback'
        if group.sum()<MIN_TRAIN:group=np.ones(len(tr),bool);source='global_fallback'
        estimate=float(np.median(lag[group])) if group.any() else global_lag
        output[ix]=estimate
        records.append(dict(airport=str(airport),coverage=int(coverage),source=source,train_rows=int(group.sum()),test_rows=len(ix),lag_seconds=estimate,
            train_id_sha256=hashlib.sha256(ids[tr[group]].astype('<i8').tobytes()).hexdigest()))
    if not np.isfinite(output[te]).all():raise ValueError('lag lookup incomplete')
    return output,records


def crosshalf(x,y,ap,cov,month,eligible,ids):
    lag=np.full(len(x),np.nan);records=[]
    for train_half in (1,2):
        train=eligible&((month<=6) if train_half==1 else (month>=7));test=eligible&~((month<=6) if train_half==1 else (month>=7))
        result,table=calibration(x,y,ap,cov,train,test,ids)
        lag[test]=result[test];records.extend([dict(train_half=train_half,test_half=3-train_half,**r) for r in table])
        if np.intersect1d(ids[train],ids[test]).size:raise ValueError('calibration/heldout IDs overlap')
    return lag,records


def metric(y,control,pred,x,eligible,indices):
    ix=np.asarray(indices,dtype=np.int64);n=len(ix)
    if not n:return None
    yy=y[ix];ec=(control[ix]-yy)**2;ep=(pred[ix]-yy)**2;delta=ep-ec;order=np.sort(delta)
    use=eligible[ix];nuse=int(use.sum());record=dict(rows=n,eligible_rows=nuse,rmse_control=float(np.sqrt(ec.mean())),rmse_proxy=float(np.sqrt(ep.mean())),
        delta_rmse=float(np.sqrt(ep.mean())-np.sqrt(ec.mean())),delta_sse=float(delta.sum()),mean_bias_control=float((control[ix]-yy).mean()),mean_bias_proxy=float((pred[ix]-yy).mean()),
        trim3_sse=float(order[3:-3].sum()) if n>6 else None,trim20_sse=float(order[20:-20].sum()) if n>40 else None)
    if nuse:
        residual=x[ix][use]-yy[use];oracle_lag=float(residual.mean());oracle_error=residual-oracle_lag
        total=float(ec[~use].sum()+np.square(oracle_error).sum())
        record['oracle_same_evaluation_cell_constant_lag']=dict(label_leaking=True,lag_seconds=oracle_lag,eligible_rows=nuse,
            family='unclipped unit-slope movement duration minus one scalar fitted on this exact evaluated cell',
            rmse=float(np.sqrt(total/n)),delta_sse=float(total-ec.sum()))
        record['oracle_per_row_joint_or_fixed_proxy']=dict(label_leaking=True,delta_sse=float(np.minimum(delta,0).sum()),
            rmse=float(np.sqrt(np.minimum(ec,ep).mean())),proxy_chosen_rows=int((delta<0).sum()))
    return record


def grouped_indices(frame,keys):
    if not keys:return [(dict(),np.arange(frame.height))]
    return [(dict(zip(keys,values[:-1])),np.asarray(values[-1],dtype=np.int64)) for values in frame.with_row_index('index').group_by(keys).agg(pl.col('index')).sort(keys).iter_rows()]


def ranking(root,receipt):
    rk=pl.scan_parquet(root/'data/features_fin1b.parquet').filter(pl.col('split')=='rank').select(ID,'ADEP_mvt','month','t_mvt',pl.col('FLIGHT_ID_mvt').is_not_null().alias('has_nm')).collect()
    if rk.height!=344841 or rk[ID].n_unique()!=rk.height:raise ValueError('ranking identity population differs')
    for folder,prefix in [('data/adsblol_kt1',''),('data/adsblol_sw_kt1','sw_')]:
        raw=pl.concat([pl.read_parquet(root/p,columns=[ID,'coverage','t_first_move']) for p in receipt['ranking_raw_files'][folder]])
        if raw[ID].n_unique()!=raw.height:raise ValueError('duplicate raw ranking ID; no arbitrary deduplication')
        raw=raw.rename({'coverage':prefix+'coverage','t_first_move':prefix+'t_first_move'})
        rk=rk.join(raw,on=ID,how='left',validate='1:1',maintain_order='left')
        rk=rk.with_columns((pl.col('t_mvt')-pl.col(prefix+'t_first_move')).alias(prefix+'mvt_first_move'))
    rk=rk.with_columns(pl.col('coverage').fill_null(-1),pl.col('sw_coverage').fill_null(-1))
    return rk


def training(root):
    oof=pl.read_parquet(root/OOF/'predictions.parquet').sort(ID)
    cols=[ID,Y,'month','ADEP_mvt','has_nm','coverage','sw_coverage','vi_has_prev','mvt_first_move','sw_mvt_first_move','mvt_move']
    natural=pl.read_parquet(root/STAGE/'natural_features.parquet',columns=cols)
    if oof.height!=2085047 or oof[ID].n_unique()!=oof.height or natural.height!=oof.height or natural[ID].n_unique()!=natural.height:raise ValueError('full training identity population differs')
    d=oof.join(natural,on=ID,how='left',validate='1:1',maintain_order='left',suffix='_natural');del natural,oof
    for k in (Y,'month','ADEP_mvt','has_nm'):
        if not np.array_equal(d[k].to_numpy(),d[k+'_natural'].to_numpy()):raise ValueError('natural/control metadata mismatch '+k)
    if int(d['excluded_tabpfn'].sum())!=126 or int(d['excluded_feasibility'].sum())!=0 or int(d['evaluation_mask'].sum())!=2084921:raise ValueError('E195 evaluation mask changed')
    return d


def pack(root,reg):
    receipt=verify(root,reg);dest=root/BASE/'portable_inputs'
    if dest.exists():raise ValueError('fresh portable input required')
    stage=Path(tempfile.mkdtemp(prefix='.portable.partial-',dir=root/BASE))
    d=training(root);d.write_parquet(stage/'training.parquet');del d
    rk=ranking(root,receipt);rk.write_parquet(stage/'ranking.parquet');del rk
    for src,name in [(SELF,Path(SELF).name),(DOC,'protocol.md'),(RECEIPT,'input_receipt.json')]:shutil.copy2(root/src,stage/name)
    guard();spec=dict(protocol=PROTOCOL,registration_commit=reg,input_receipt_sha256=sha(root/RECEIPT),files={p.name:sha(p) for p in stage.iterdir() if p.is_file()},rows=2085047,rank_rows=344841)
    write(stage/'manifest.json',dict(spec=spec,spec_sha256=digest(spec)));stage.rename(dest);print('PACKED NO COMPARISONS',sha(dest/'manifest.json'),flush=True)


def portable(bundle,reg,manifest_sha):
    if sha(bundle/'manifest.json')!=manifest_sha:raise ValueError('transport manifest differs')
    m=read(bundle/'manifest.json');s=m['spec']
    if s['protocol']!=PROTOCOL or s['registration_commit']!=reg or m['spec_sha256']!=digest(s):raise ValueError('portable lineage differs')
    if sha(__file__)!=s['files'][Path(SELF).name]:raise ValueError('active worker differs')
    for n,h in s['files'].items():
        if sha(bundle/n)!=h:raise ValueError('portable file differs: '+n)
    return m


def run(bundle,destination,reg,manifest_sha):
    manifest=portable(bundle,reg,manifest_sha);resource_receipt={}
    if sys.platform=='win32':
        import psutil
        kernel,handle=windows_limit();proc=psutil.Process();proc.cpu_affinity([0,2]);proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        battery=psutil.sensors_battery()
        if not battery or not battery.power_plugged or psutil.virtual_memory().available<3*1024**3:raise RuntimeError('ASUS needs AC and3GiBfree')
        resource_receipt=dict(affinity=proc.cpu_affinity(),committed_memory_limit=MEMORY_LIMIT,priority=proc.nice())
    if destination.exists():raise ValueError('fresh diagnostic output required')
    start=time.monotonic();tmp=Path(tempfile.mkdtemp(prefix='.results.partial-',dir=destination.parent))
    write(tmp/'invocation.json',dict(pid=os.getpid(),registration_commit=reg,input_manifest_sha256=manifest_sha,started=datetime.datetime.now(datetime.timezone.utc).isoformat(),resource=resource_receipt,python=sys.version,numpy=np.__version__,polars=pl.__version__))
    d=pl.read_parquet(bundle/'training.parquet')
    d=d.with_columns(pl.col('coverage').fill_null(-1),pl.col('sw_coverage').fill_null(-1),(pl.col('month')>6).cast(pl.Int8).add(1).alias('half'))
    ids=d[ID].to_numpy();y=d[Y].to_numpy().astype(float);current=d['joint'].to_numpy().astype(float);included=d['evaluation_mask'].to_numpy()
    if not np.isfinite(ids).all() or not np.equal(ids,np.rint(ids)).all() or not np.isfinite(y).all() or not np.isfinite(current).all():raise ValueError('IDs/targets/predictions invalid')
    ap=d['ADEP_mvt'].to_numpy();cov=d['coverage'].to_numpy();month=d['month'].to_numpy();nm=d['has_nm'].to_numpy()
    if not set(np.unique(cov)).issubset({-1,0,1,2,3}):raise ValueError('unknown coverage class')
    body=included&(y<=7200);decided=included&~((ap=='LIRF')&~nm&(d['D'].to_numpy()>=14400));guard()
    rank=pl.read_parquet(bundle/'ranking.parquet');guard();tables=[];records=[];support=[];projections=[];decisions={}
    audit=d.select(ID,Y,'ADEP_mvt','month','coverage','has_nm','evaluation_mask','joint')
    groups={name:grouped_indices(d,keys) for name,keys in [('all',[]),('airport_coverage',['ADEP_mvt','coverage']),('airport_coverage_nm',['ADEP_mvt','coverage','has_nm']),('month',['month']),('half',['half'])]}
    for field in FIELDS:
        x=d[field].to_numpy();eligible=included&valid(x,cov);lag,table=crosshalf(x,y,ap,cov,month,eligible,ids)
        tables.extend([dict(field=field,**r) for r in table]);pred=current.copy();pred[eligible]=np.maximum(x[eligible]-lag[eligible],0);zero=current.copy();zero[eligible]=x[eligible]
        if not np.isfinite(pred).all() or not np.array_equal(pred[~eligible],current[~eligible]):raise ValueError('fallback/prediction contract differs')
        audit=audit.with_columns(pl.Series(field+'_eligible',eligible),pl.Series(field+'_lag',lag),pl.Series(field+'_proxy',pred))
        for family,gg in groups.items():
            for group,ix in gg:
                for population,mask in [('included',included),('body',body),('decided',decided)]:
                    subset=ix[mask[ix]]
                    if not len(subset):continue
                    for arm,p in [('zero_lag',zero),('cross_half_median_lag',pred)]:
                        records.append(dict(field=field,arm=arm,grouping=family,group=group,population=population,**metric(y,current,p,x,eligible,subset)))
        # Ranking is used only for observed support counts, never targets or predictions.
        for population,frame,xx,cc,ok in [('training',d,x,cov,included),('ranking',rank,rank[field].to_numpy(),rank['coverage'].to_numpy(),np.ones(rank.height,bool))]:
            vv=valid(xx,cc)&ok
            for group,ix in grouped_indices(frame,['ADEP_mvt','coverage','has_nm']):
                support.append(dict(field=field,population=population,**group,rows=len(ix),evaluation_rows=int(ok[ix].sum()),eligible_rows=int(vv[ix].sum())))
        # Reweight fixed 2025 cell MSE deltas by 2026 label-free cell shares;
        # unsupported cells remain unknown, not zero-error extrapolations.
        pooled={tuple(r['group'][k] for k in ('ADEP_mvt','coverage','has_nm')):r for r in records if r['field']==field and r['arm']=='cross_half_median_lag' and r['grouping']=='airport_coverage_nm' and r['population']=='body'}
        value=0.;covered=0
        for cell in support:
            if cell['field']!=field or cell['population']!='ranking':continue
            r=pooled.get(tuple(cell[k] for k in ('ADEP_mvt','coverage','has_nm')))
            if r is not None:value+=cell['rows']/344841*r['delta_sse']/r['rows'];covered+=cell['rows']
        projections.append(dict(field=field,ranking_rows=344841,matched_cell_rows=covered,unmatched_cell_rows=344841-covered,
            support_weighted_2025_body_delta_mse=value,interpretation='Descriptive source-year cell reweighting, not a leaderboard RMSE or transfer forecast'))
        summary=[r for r in records if r['field']==field and r['arm']=='cross_half_median_lag']
        halves=[r for r in summary if r['grouping']=='half' and r['population']=='body'];dec=next(r for r in summary if r['grouping']=='all' and r['population']=='decided')
        decisions[field]=dict(observable_signal_for_separate_followup=len(halves)==2 and all(r['delta_sse']<0 for r in halves) and dec['delta_sse']<0,
            rule='Both cross-half body SSE deltas and pooled decided SSE delta negative; diagnostic only, never deploy')
        guard();print('COMPARED',field,'eligible',int(eligible.sum()),flush=True)
    audit.write_parquet(tmp/'rows.parquet')
    write(tmp/'lag_tables.json',tables);write(tmp/'comparisons.json',records);write(tmp/'support.json',support)
    result=dict(protocol=PROTOCOL,registration_commit=reg,input_receipt_sha256=manifest['spec']['input_receipt_sha256'],input_manifest_sha256=manifest_sha,resource=resource_receipt,rows=len(y),included=int(included.sum()),excluded=126,
        primary_field=FIELDS[0],secondary_field=FIELDS[1],decisions=decisions,projections=projections,
        limitation='Frozen E195joint afterE152 beforeE81/LOBT/finalTabPFN/LSZH; historical upstream not fullynested. No MLtraining or pipeline change.',
        coverage_definition={'-1':'missing original feature','0':'no uniquely matched track','1':'track without accepted surface segment','2':'accepted surface segment; no qualifying final stop','3':'at least90s stationary interval before final movement; not guaranteed stand'},
        moment_definition={'mvt_first_move':'MVT minus first movement after initial stationary run (original1kt)','sw_mvt_first_move':'MVT minus1kt stand-walk firstmovement','mvt_move':'MVT minus end of last qualifying stop; NOT firstmovement'},
        timing=time.monotonic()-start,peak_rss_bytes=rss(),accepted_for_submission=False,models_fitted=0,ranking_predictions_created=False)
    portable(bundle,reg,manifest_sha);result['outputs']={p.name:sha(p) for p in tmp.iterdir() if p.is_file()};write(tmp/'result.json',result)
    tmp.rename(destination);print('COMPLETE DIAGNOSIS ONLY',json.dumps(result),flush=True)


def selftest():
    x=np.array([100.,110.,130.,150.]);y=np.array([90.,100.,777.,888.]);ap=np.array(['A']*4);cov=np.array([3]*4);ids=np.arange(4)
    tr=np.array([1,1,0,0],bool);te=~tr
    a,_=calibration(x,y,ap,cov,tr,te,ids);changed=y.copy();changed[te]=-999999;b,_=calibration(x,changed,ap,cov,tr,te,ids)
    assert np.array_equal(a[te],b[te]) and np.array_equal(a[te],[10,10])
    assert np.array_equal(valid(np.array([np.nan,-1,0,43200,43201]),np.array([3]*5)),[False,False,True,True,False])
    r=metric(np.array([2.,5.]),np.array([3.,8.]),np.array([2.,7.]),np.array([10.,20.]),np.ones(2,bool),np.arange(2))
    assert r['delta_sse']==-6 and r['oracle_per_row_joint_or_fixed_proxy']['delta_sse']==-6
    print('PASS: heldout labels cannot affect observable lag; masks/sign/selector arithmetic')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['seal','selftest','pack','run']);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--registration-commit');p.add_argument('--bundle',type=Path);p.add_argument('--output',type=Path);p.add_argument('--input-manifest-sha256');a=p.parse_args();root=a.root.resolve()
    if a.action=='seal':seal(root)
    elif a.action=='selftest':selftest()
    elif a.action=='pack':pack(root,a.registration_commit)
    else:run(a.bundle.resolve(),a.output.resolve(),a.registration_commit,a.input_manifest_sha256)


if __name__=='__main__':main()

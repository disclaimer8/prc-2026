"""E199: observed OTHER-transponder geometry, fixed retrospective CPU screen.

No inferred occupancy from intervals, no interpolation, no neighbor targets.
The full-day E194 identity makes this offline MVT-relative, not online causal.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
for k,v in {'POLARS_MAX_THREADS':'2','OPENBLAS_NUM_THREADS':'1','OMP_WAIT_POLICY':'PASSIVE','GOMP_SPINCOUNT':'0','POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR':'load_as_storage'}.items():
    os.environ.setdefault(k,v)
import numpy as np
import polars as pl
import e194_trajectory_sequence as Q
HELPER_SHA='c2cf0a747f25de4219680799e1dac0a162007258908306f9670e9c3474c2e0eb'
if hashlib.sha256(Path(Q.__file__).read_bytes()).hexdigest()!=HELPER_SHA:raise ValueError('frozen E194 helper drift')

PROTOCOL='E199-other-ground-spatial-state-v-counts-retrospective-2pair-3seed-v1'
SELF='scripts/e199_airport_surface_state.py'
DOC='analysis/e199_protocol_20261007.md'
OFFSETS=(600,120)
MAXAGE=60.
SHARED=('n_recent','n_ground','n_valid','n_stopped','n_moving','n_speed_unknown','n_stale_unknown','n_stale_rejected','age_mean','age_max','speed_mean','speed_std','heading_known','heading_coherence')
SPATIAL=('cells250','cells500','maxcell250','maxcell500','hhi250','hhi500','radius_rms','close250_fraction','close500_fraction','stopped_nn_mean','stopped_close250_fraction')
CONTEXT=('taxiing_at_ob','dep_before_10','dep_before_30','arr_before_10','arr_before_30','rwy_dep_before_10','rwy_dep_before_30','local_hour','d_mvt_aobt','d_mvt_eobt','d_mvt_lobt','d_mvt_iobt','d_mvt_sched','plan_nm_taxi')
FEATURE_SHA='f061a441f6c97c232d9a403d612471e45020eb4ad56f501c2acfd7220a7d3052'
PARENT='data/e195_pre_e81_20261007_v2/predictions.parquet'
PARENT_RESULT_SHA='94ba00d1522b7d598bc9d3b42d3361ad68af9dd41ca19b120c5c07f023e05f9e'
FOLDS=(1,4)
REMAINING=(2,3,5,6)
SEEDS=(42,1,2)

def sha(p):return Q.sha(p)
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Q.save(p,x)
def registration(root,commit):
    for path in (SELF,DOC):
        if hashlib.sha256(subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=root)).hexdigest()!=sha(root/path):raise ValueError('registered source drift '+path)
    if PROTOCOL.encode() not in subprocess.check_output(['git','show',f'{commit}:HYPOTHESES_LOG.md'],cwd=root):raise ValueError('journal registration absent')

def summaries(rows,cutoff,lat0,lon0):
    """rows are one LATEST observation per nonself ICAO, before any filters.
    columns: t,lat,lon,ground,speed,heading,stale. Unknown is NaN.
    """
    out=np.full(len(SHARED)+len(SPATIAL),np.nan,np.float32)
    if len(rows)==0:
        out[:8]=0;out[12]=0
        return out
    age=cutoff-rows[:,0]
    if np.any(age<0) or np.any(age>MAXAGE):raise ValueError('invalid latest-point age')
    ground=rows[:,3]==1
    valid=ground & np.isfinite(rows[:,1:3]).all(1) & (np.abs(rows[:,1])<=90) & (np.abs(rows[:,2])<=180) & (rows[:,6]!=1)
    z=rows[valid];n=len(z)
    speed=np.where((z[:,4]>=0)&(z[:,4]<=100),z[:,4],np.nan)
    stopped=speed<=3;moving=speed>3;known=np.isfinite(speed)
    heading=np.where((z[:,5]>=0)&(z[:,5]<360),z[:,5],np.nan)
    hd=heading[np.isfinite(heading)]
    coherence=float(np.abs(np.exp(1j*np.deg2rad(hd)).mean())) if len(hd)>=2 else np.nan
    out[:len(SHARED)]=[len(rows),ground.sum(),n,stopped.sum(),moving.sum(),(~known).sum(),np.isnan(z[:,6]).sum(),(ground&(rows[:,6]==1)).sum(),float((cutoff-z[:,0]).mean()) if n else np.nan,float((cutoff-z[:,0]).max()) if n else np.nan,float(speed[known].mean()) if known.any() else np.nan,float(speed[known].std()) if known.any() else np.nan,np.isfinite(heading).sum(),coherence]
    if n<2:return out
    xy=np.column_stack([(z[:,2]-lon0)*111320*np.cos(np.deg2rad(lat0)),(z[:,1]-lat0)*111320])
    geom=[]
    cells=[];mx=[];hh=[]
    for size in (250,500):
        _,counts=np.unique(np.floor(xy/size).astype(np.int64),axis=0,return_counts=True)
        cells.append(len(counts));mx.append(counts.max());hh.append(np.sum((counts/n)**2))
    # Blocked pair reductions: no quadratic full matrix allocation and no cap.
    close=np.zeros(2,dtype=np.int64);stopclose=0;nearest=[];ns=int(stopped.sum())
    for lo in range(0,n,128):
        hi=min(n,lo+128);dist=np.sqrt(np.sum((xy[lo:hi,None,:]-xy[None,:,:])**2,axis=2))
        ii=np.arange(lo,hi);dist[np.arange(hi-lo),ii]=np.inf
        close += [(dist<r).sum() for r in (250,500)]
        for j in np.flatnonzero(stopped[lo:hi]):
            nearest.append(np.min(dist[j,stopped]) if ns>1 else np.nan)
            stopclose+=int((dist[j,stopped]<250).sum())
    rad=float(np.sqrt(np.mean(np.sum((xy-xy.mean(0))**2,axis=1))))
    geom=[*cells,*mx,*hh,rad,close[0]/(n*(n-1)),close[1]/(n*(n-1)),float(np.mean(nearest)) if ns>1 else np.nan,stopclose/(ns*(ns-1)) if ns>1 else np.nan]
    out[len(SHARED):]=geom
    return out

def encode(points,ids,identity,airport,takeoff,day):
    n=len(ids);values=np.full((n,len(OFFSETS),len(SHARED)+len(SPATIAL)),np.nan,np.float32)
    known_identity=identity!='';safe=np.zeros((n,len(OFFSETS)),bool)
    if points is None:return values,safe
    if 'stale' not in points.columns:points=points.with_columns(pl.lit(None,dtype=pl.Boolean).alias('stale'))
    if points['t'].null_count() or not points['t'].is_finite().all():raise ValueError('invalid raw timestamps')
    start=dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp()
    # Never borrow the last ground point after an airborne or stale update.
    points=points.with_row_index('_source').sort(['airport','icao','t','_source'])
    for (ap,),g in points.group_by('airport',maintain_order=True):
        ix=np.flatnonzero((airport==ap)&known_identity)
        if not len(ix):continue
        cut=(takeoff[ix,None]-np.array(OFFSETS)[None,:]).reshape(-1)
        owners=np.repeat(identity[ix],len(OFFSETS))
        safe_ap=(cut>=start+MAXAGE)&(cut<start+86400)
        records=[[] for _ in range(len(cut))]
        for (icao,),h in g.group_by('icao',maintain_order=True):
            if not isinstance(icao,str) or not icao:continue
            a=np.column_stack([h[c].cast(pl.Float64).to_numpy() for c in ('t','lat','lon','on_ground','gs','track','stale')])
            at=np.searchsorted(a[:,0],cut,side='right')-1;ok=(at>=0)&safe_ap&(owners!=icao)
            at=np.maximum(at,0);ok &= (cut-a[at,0]<=MAXAGE)&(cut>=a[at,0])
            for j in np.flatnonzero(ok):records[j].append(a[at[j]])
        for j,rows in enumerate(records):
            k,o=divmod(j,len(OFFSETS))
            if safe_ap[j]:values[ix[k],o]=summaries(np.asarray(rows,dtype=np.float64).reshape(-1,7),cut[j],*Q.AIRPORTS[ap]);safe[ix[k],o]=True
    return values,safe

def day_job(args):
    root,bundle,out,day=args;root=Path(root);bundle=Path(bundle);out=Path(out)
    spec=read(bundle/'manifest.json');entry=spec['days'][day]
    for name,h in entry['files'].items():
        if sha(root/name)!=h:raise ValueError('raw/identity input drift '+name)
    with np.load(root/entry['identity']) as z:
        ids=z['ids'];identity=z['identity'];airport=z['airport'];takeoff=z['takeoff']
    deps=pl.read_parquet(root/entry['departures'])
    if any(len(x)!=len(ids) for x in (identity,airport,takeoff)) or len(np.unique(ids))!=len(ids):raise ValueError('day identity shape/duplicates')
    if not np.array_equal(ids,Q.ids_checked(deps)) or not np.array_equal(airport,deps['ADEP_mvt'].to_numpy()) or not np.array_equal(takeoff,deps['MVT_TIME_UTC_mvt'].dt.epoch('s').to_numpy()):raise ValueError('original departure mapping differs')
    if not deps['MVT_TIME_UTC_mvt'].dt.strftime('%Y-%m-%d').eq(day).all():raise ValueError('departure day mismatch')
    raw=root/entry['raw'] if entry['raw'] else None
    points=pl.read_parquet(raw) if raw else None
    t=time.monotonic();v,safe=encode(points,ids,identity,airport,takeoff,day)
    eligible=(identity!='')&np.any((v[:,:,SHARED.index('n_valid')]>=2)&safe,axis=1)
    columns={'MVT_ID_mvt':ids,'day':np.repeat(day,len(ids)),'airport':airport,'identity_known':identity!='','eligible':eligible}
    for i,off in enumerate(OFFSETS):
        columns[f'safe_{off}']=safe[:,i]
        for j,name in enumerate(SHARED+SPATIAL):columns[f's{off}_{name}']=v[:,i,j]
    dest=out/(day+'.parquet');pl.DataFrame(columns).write_parquet(dest,compression='zstd')
    stats=[]
    for ap in Q.AIRPORTS:
        m=airport==ap
        if m.any():stats.append(dict(airport=ap,rows=int(m.sum()),identity=int(((identity!='')&m).sum()),eligible=int((eligible&m).sum()),snapshot_n_valid_mean=[float(np.nan_to_num(v[m,i,2]).mean()) for i in range(len(OFFSETS))]))
    receipt=dict(day=day,rows=len(ids),eligible=int(eligible.sum()),identity=int((identity!='').sum()),elapsed_s=time.monotonic()-t,sha256=sha(dest),airports=stats)
    save(out/(day+'.json'),receipt);print(json.dumps(receipt),flush=True);return receipt

def pack(root,out,commit):
    registration(root,commit)
    if out.exists():raise FileExistsError(out)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent));(stage/'scripts').mkdir()
    for name in (Path(SELF).name,'e194_trajectory_sequence.py'):shutil.copy2(root/'scripts'/name,stage/'scripts'/name)
    days={}
    for shard in (0,1):
        source=root/f'data/e194_inputs_20261007_s{shard}';m=read(source/'manifest.json')
        done=read(root/f'data/e194_sequence_20261007_s{shard}/complete.json')
        if done['input_manifest_sha256']!=sha(source/'manifest.json') or done['registration_commit']!='fc7107de145a87a5a1ff93314f318b440b1de825':raise ValueError('E194 complete/input lineage differs')
        if m['registration_commit']!='fc7107de145a87a5a1ff93314f318b440b1de825':raise ValueError('original E194 registration differs')
        for day in m['days']:
            identity=f'data/e194_sequence_20261007_s{shard}/{day}/sequence.npz'
            raw=f'data/e194_inputs_20261007_s{shard}/surface/{day}.parquet'
            departures=f'data/e194_inputs_20261007_s{shard}/departures/{day}.parquet'
            receipt=read(root/f'data/e194_sequence_20261007_s{shard}/{day}/receipt.json')
            if receipt['source_sha256']!=HELPER_SHA or receipt['day']!=day or receipt not in done['reports']:raise ValueError('original day receipt differs')
            files={identity:receipt['npz_sha256'],departures:m['files'][f'departures/{day}.parquet']}
            exists=m['point_inputs'][day]['exists']
            if exists:files[raw]=m['point_inputs'][day]['sha256']
            elif day=='2025-12-31' and not (root/raw).exists():raw=None
            else:raise ValueError('unexpected absent raw day')
            for name,h in files.items():
                if sha(root/name)!=h:raise ValueError('original E194 source mismatch '+name)
            days[day]=dict(shard=shard,identity=identity,raw=raw,departures=departures,files=files)
    if sorted(days)!=sorted(Q.DAYS):raise ValueError('427-day inventory differs')
    save(stage/'manifest.json',dict(protocol=PROTOCOL,registration_commit=commit,days=days,source_sha256=sha(root/SELF),offsets=OFFSETS,max_age=MAXAGE,label_free=True))
    stage.rename(out);print(json.dumps(dict(packed=str(out),days=len(days),manifest_sha256=sha(out/'manifest.json'))),flush=True)

def extract(root,bundle,out,commit,shard,workers,probe):
    m=read(bundle/'manifest.json')
    if m['protocol']!=PROTOCOL or m['registration_commit']!=commit or m['source_sha256']!=sha(__file__):raise ValueError('bundle source/protocol differs')
    if out.exists():raise FileExistsError(out)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent))
    days=sorted(d for d,x in m['days'].items() if x['shard']==shard)
    if probe:days=[d for d in days if d in probe.split(',')]
    if not days:raise ValueError('no assigned days')
    with ProcessPoolExecutor(max_workers=workers) as pool:receipts=list(pool.map(day_job,[(str(root),str(bundle),str(stage),d) for d in days]))
    save(stage/'complete.json',dict(protocol=PROTOCOL,registration_commit=commit,input_manifest_sha256=sha(bundle/'manifest.json'),shard=shard,days=days,rows=sum(x['rows'] for x in receipts),receipts=receipts))
    stage.rename(out)

def merge(root,bundle,parts,out,commit):
    registration(root,commit)
    m=read(bundle/'manifest.json');frames=[];days=[];inputs={}
    if out.exists():raise FileExistsError(out)
    for part in parts:
        part=Path(part);r=read(part/'complete.json')
        if r['input_manifest_sha256']!=sha(bundle/'manifest.json') or r['registration_commit']!=commit:raise ValueError('shard lineage differs')
        for d in r['receipts']:
            p=part/(d['day']+'.parquet')
            if sha(p)!=d['sha256']:raise ValueError('day feature drift')
            frames.append(pl.read_parquet(p));days.append(d['day']);inputs[str(p)]=d['sha256']
    if len(days)!=427 or sorted(days)!=sorted(m['days']):raise ValueError('not all427 unique days')
    allf=pl.concat(frames);del frames
    if len(allf)!=2429888 or allf['MVT_ID_mvt'].n_unique()!=len(allf):raise ValueError('not all2429888 unique IDs')
    if sha(root/'data/features_fin1b.parquet')!=FEATURE_SHA:raise ValueError('FIN1b drift')
    if sha(root/'data/e195_pre_e81_20261007_v2/result.json')!=PARENT_RESULT_SHA:raise ValueError('parent result drift')
    if sha(root/PARENT)!=read(root/'data/e195_pre_e81_20261007_v2/result.json')['outputs']['predictions.parquet']:raise ValueError('parent predictions drift')
    parent=pl.read_parquet(root/PARENT).with_columns(pl.col('MVT_ID_mvt').cast(pl.Int64))
    if len(parent)!=2085047 or parent['MVT_ID_mvt'].n_unique()!=len(parent):raise ValueError('parent IDs differ')
    f=pl.read_parquet(root/'data/features_fin1b.parquet',columns=['MVT_ID_mvt','split',*CONTEXT]).with_columns(pl.col('MVT_ID_mvt').cast(pl.Int64))
    if not set(f['MVT_ID_mvt'])==set(allf['MVT_ID_mvt']):raise ValueError('rank/train all-ID set differs')
    table=parent.join(allf,on='MVT_ID_mvt',how='left',validate='1:1').join(f.drop('split'),on='MVT_ID_mvt',how='left',validate='1:1')
    if len(table)!=len(parent) or table['eligible'].null_count():raise ValueError('feature alignment failed')
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent));table.write_parquet(stage/'training.parquet',compression='zstd')
    allf.filter(pl.col('day').str.starts_with('2026')).write_parquet(stage/'ranking_features.parquet',compression='zstd')
    cov=allf.with_columns(pl.col('day').str.slice(0,7).alias('period')).group_by(['period','airport']).agg(pl.len().alias('rows'),pl.col('identity_known').sum().alias('identity'),pl.col('eligible').sum().alias('eligible')).sort(['period','airport'])
    cov.write_parquet(stage/'coverage.parquet');cov.write_csv(stage/'coverage.csv')
    # Fixed observable admission: useful Jan AND Jul ranking, plus both2025 halves.
    def count(period):return int(allf.filter(pl.col('day').str.starts_with(period))['eligible'].sum())
    admission=all(count(period)>=10000 for period in ('2025-01','2025-07','2026-01','2026-07'))
    save(stage/'manifest.json',dict(protocol=PROTOCOL,registration_commit=commit,source_sha256=sha(root/SELF),raw_bundle_sha256=sha(bundle/'manifest.json'),feature_sha256=FEATURE_SHA,parent_sha256=sha(root/PARENT),parent_result_sha256=PARENT_RESULT_SHA,raw_feature_inputs=inputs,files={p.name:sha(p) for p in stage.iterdir()},rows=len(table),rank_rows=int(allf['day'].str.starts_with('2026').sum()),coverage_admission=admission,eligible_train=int(table['eligible'].sum()),label_free_admission=True))
    stage.rename(out);print(json.dumps(dict(merged=str(out),coverage_admission=admission,eligible_train=int(table['eligible'].sum()))),flush=True)

def load_screen(bundle,commit):
    m=read(bundle/'manifest.json')
    if m['protocol']!=PROTOCOL or m['registration_commit']!=commit or m['source_sha256']!=sha(__file__) or not m['coverage_admission']:raise ValueError('screen admission/source failure')
    for name,h in m['files'].items():
        if sha(bundle/name)!=h:raise ValueError('screen bundle drift')
    d=pl.read_parquet(bundle/'training.parquet')
    if len(d)!=2085047 or d['MVT_ID_mvt'].n_unique()!=len(d):raise ValueError('training IDs differ')
    return m,d

def fit(bundle,out,commit,fold,threads,screen_evaluation=None):
    # LGB-only fresh process: current E188 runtime audit reproduced libomp abort
    # even with Torch-first. This worker has no tensor dependency.
    if 'torch' in sys.modules:raise RuntimeError('fresh LGB-only process required')
    import lightgbm as lgb
    if out.exists():raise FileExistsError(out)
    m,d=load_screen(bundle,commit);start=time.monotonic()
    screen_sha=None
    if fold in REMAINING:
        if screen_evaluation is None:raise ValueError('remaining pairs require passed screen receipt')
        sc=read(screen_evaluation)
        strict=all(all(r[c][k]<0 for k in ('body_sse','decided_sse','trim3_sse','trim20_sse')) for r in sc['reports'] for c in ('spatial_vs_counts','spatial_vs_parent'))
        if sc['protocol']!=PROTOCOL or sc['registration_commit']!=commit or sc['input_manifest_sha256']!=sha(bundle/'manifest.json') or not sc['spatial_screen_pass'] or not strict or sorted(r['fold'] for r in sc['reports'])!=list(FOLDS):raise ValueError('invalid screen admission')
        screen_sha=sha(screen_evaluation)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent))
    month=d['month'].to_numpy();parent=d['joint'].to_numpy().astype(float);y=d['TAXITIME_SEC_mvt'].to_numpy().astype(float)
    eligible=d['eligible'].to_numpy()&d['evaluation_mask'].to_numpy();te=np.isin(month,[fold,fold+6]);tr=~te&eligible;apply=te&eligible
    if tr.sum()<10000 or apply.sum()<1000:raise ValueError('fixed support guard failed')
    if not np.isfinite(y).all() or not np.isfinite(parent).all():raise ValueError('nonfinite labels/parent')
    common=[*CONTEXT,*[f's{off}_{c}' for off in OFFSETS for c in SHARED],*[f'safe_{off}' for off in OFFSETS]]
    geometry=[f's{off}_{c}' for off in OFFSETS for c in SPATIAL]
    x=np.column_stack([d.select(common).to_numpy(),parent,np.array([list(Q.AIRPORTS).index(a) for a in d['ADEP_mvt']]),d.select(geometry).to_numpy()]).astype(np.float32)
    if np.isinf(x).any():raise ValueError('infinite feature')
    colnames=[*common,'parent_joint','airport_code',*geometry];cg=len(common)+1
    params=dict(objective='regression',metric='l2',learning_rate=.05,num_leaves=31,min_data_in_leaf=500,max_depth=-1,feature_fraction=1.,bagging_fraction=.8,bagging_freq=1,lambda_l2=10.,num_threads=threads,verbosity=-1,deterministic=True,force_col_wise=True,max_bin=255)
    arms=[];predictions={}
    for arm in ('counts','spatial'):
        xx=x.copy()
        if arm=='counts':xx[:,-len(geometry):]=0 # Same tree budget and shape; no spatial input.
        xt=xx[tr];xp=xx[apply];del xx
        target=(y-parent)[tr];fits=[];corrections=[]
        for seed in SEEDS:
            begin=time.monotonic();p={**params,'seed':seed,'bagging_seed':seed,'feature_fraction_seed':seed,'data_random_seed':seed}
            model=lgb.train(p,lgb.Dataset(xt,label=target,feature_name=colnames,categorical_feature=[cg]),num_boost_round=300)
            correction=model.predict(xp,num_threads=threads);mp=stage/f'{arm}_seed{seed}.txt';npth=stage/f'{arm}_seed{seed}.npy'
            if not np.isfinite(correction).all():raise ValueError('nonfinite correction')
            model.save_model(str(mp));np.save(npth,correction,allow_pickle=False);corrections.append(correction)
            r=dict(seed=seed,rounds=300,seconds=time.monotonic()-begin,model_sha256=sha(mp),prediction_sha256=sha(npth));fits.append(r);print(json.dumps(dict(fold=fold,arm=arm,**r)),flush=True)
        pred=parent[te].copy();pred[eligible[te]]=np.maximum(0,parent[apply]+np.mean(corrections,axis=0));predictions[arm]=pred
        arms.append(dict(arm=arm,fits=fits,seeds=list(SEEDS),features=colnames))
    np.savez_compressed(stage/'prediction.npz',ids=d['MVT_ID_mvt'].to_numpy()[te],y=y[te],month=month[te],parent=parent[te],eligible=eligible[te],evaluation_mask=d['evaluation_mask'].to_numpy()[te],airport=d['ADEP_mvt'].to_numpy()[te].astype('U4'),has_nm=d['has_nm'].to_numpy()[te],D=d['D'].to_numpy()[te],counts=predictions['counts'],spatial=predictions['spatial'])
    import platform
    save(stage/'complete.json',dict(protocol=PROTOCOL,registration_commit=commit,input_manifest_sha256=sha(bundle/'manifest.json'),screen_evaluation_sha256=screen_sha,fold=fold,heldout_months=[fold,fold+6],train_rows=int(tr.sum()),apply_rows=int(apply.sum()),arms=arms,threads=threads,runtime=dict(python=platform.python_version(),numpy=np.__version__,lightgbm=lgb.__version__),seconds=time.monotonic()-start,files={p.name:sha(p) for p in stage.iterdir()}))
    stage.rename(out)

def metrics(base,pred,y,mask,decided):
    delta=((pred-y)**2-(base-y)**2)[mask];ys=y[mask];dec=decided[mask];ordered=np.sort(delta)
    return dict(rows=int(mask.sum()),all_sse=float(delta.sum()),body_sse=float(delta[ys<=7200].sum()),decided_sse=float(delta[dec].sum()),trim3_sse=float(ordered[3:-3].sum()),trim20_sse=float(ordered[20:-20].sum()),delta_rmse=float(np.sqrt(np.mean((pred[mask]-y[mask])**2))-np.sqrt(np.mean((base[mask]-y[mask])**2))))

def evaluate(bundle,parts,out,commit):
    m,d=load_screen(bundle,commit);reports=[];seen=[]
    for part in parts:
        part=Path(part);r=read(part/'complete.json');fold=r['fold'];seen.append(fold)
        if r['registration_commit']!=commit or r['input_manifest_sha256']!=sha(bundle/'manifest.json'):raise ValueError('fit lineage differs')
        if len(r['arms'])!=2 or any(a['seeds']!=list(SEEDS) or len(a['fits'])!=3 or any(f['rounds']!=300 for f in a['fits']) for a in r['arms']):raise ValueError('fit inventory differs')
        for name,h in r['files'].items():
            if sha(part/name)!=h:raise ValueError('fit artifact drift')
        with np.load(part/'prediction.npz') as z:
            mask=np.isin(d['month'].to_numpy(),[fold,fold+6]);ref=d.filter(pl.Series(mask))
            for field,col in [('ids','MVT_ID_mvt'),('y','TAXITIME_SEC_mvt'),('month','month'),('parent','joint')]:
                if not np.array_equal(z[field],ref[col].to_numpy()):raise ValueError('prediction order/label/parent mismatch')
            for arm in ('counts','spatial'):
                mean=np.mean([np.load(part/f'{arm}_seed{s}.npy') for s in SEEDS],axis=0);expected=z['parent'].copy();expected[z['eligible']]=np.maximum(0,expected[z['eligible']]+mean)
                if not np.array_equal(expected,z[arm]):raise ValueError('mean/fallback arithmetic mismatch')
            decided=~((z['airport']=='LIRF')&~z['has_nm']&(z['D']>=14400));ok=z['evaluation_mask']
            report=dict(fold=fold,spatial_vs_counts=metrics(z['counts'],z['spatial'],z['y'],ok,decided),spatial_vs_parent=metrics(z['parent'],z['spatial'],z['y'],ok,decided),counts_vs_parent=metrics(z['parent'],z['counts'],z['y'],ok,decided))
            report['months']={str(k):metrics(z['counts'],z['spatial'],z['y'],ok&(z['month']==k),decided) for k in [fold,fold+6]};reports.append(report)
    if sorted(seen) not in (list(FOLDS),list(range(1,7))):raise ValueError('both screen pairs or all6 pairs required')
    screen_reports=[r for r in reports if r['fold'] in FOLDS]
    passed=all(all(r[c][k]<0 for k in ('body_sse','decided_sse','trim3_sse','trim20_sse')) for r in screen_reports for c in ('spatial_vs_counts','spatial_vs_parent'))
    fullpass=passed and len(reports)==6 and all(sum(r[c]['decided_sse']<0 for r in reports)>=4 and all(sum(r[c][k] for r in reports)<0 for k in ('body_sse','trim3_sse','trim20_sse')) for c in ('spatial_vs_counts','spatial_vs_parent'))
    grouped={}
    for group,folds in [('selected_screen',FOLDS),('untouched_confirmation',REMAINING)]:
        rr=[r for r in reports if r['fold'] in folds]
        if rr:grouped[group]={c:{k:sum(r[c][k] for r in rr) for k in ('rows','all_sse','body_sse','decided_sse','trim3_sse','trim20_sse')} for c in ('spatial_vs_counts','spatial_vs_parent','counts_vs_parent')}
    result=dict(protocol=PROTOCOL,registration_commit=commit,input_manifest_sha256=sha(bundle/'manifest.json'),reports=reports,groups=grouped,screen_folds=list(FOLDS),confirmation_folds=list(REMAINING),spatial_screen_pass=passed,six_pair_gate_pass=fullpass if len(reports)==6 else None,accepted_for_submission=False,limitation='Retrospective residual screen on historical joint E195 upstream OOF, not nested CV or full chain. Own fold labels excluded from stage fitting but not guaranteed absent from other-fold parent construction. A PASS requires independent base-member confirmation before production.')
    save(out,result);print(json.dumps(result),flush=True)

def selftest():
    rows=np.array([[100,50,8,1,0,0,0],[100,50.001,8,1,0,0,0],[100,50.1,8,0,5,20,0]],float)
    a=summaries(rows,110,50,8);assert a[0]==3 and a[2]==2 and a[len(SHARED)+7]==1
    b=rows.copy();b[:,2]+=.1;assert np.allclose(a,summaries(b,110,50,8),equal_nan=True,atol=1e-4) # all spatial outputs translation invariant except grid edges here.
    day='2025-01-01';t=dt.datetime(2025,1,1,tzinfo=dt.timezone.utc).timestamp()+1000
    d=pl.DataFrame({'icao':['own','x','x','y','z'],'airport':['EDDF']*5,'t':[t-120,t-130,t-119,t-125,t-125],'lat':[50]*5,'lon':[8]*5,'on_ground':[True,True,False,True,True],'gs':[0,0,5,None,0],'track':[0]*5,'stale':[False,False,False,None,True]})
    v,s=encode(d,np.array([1,2]),np.array(['own','']),np.array(['EDDF','EDDF']),np.array([t,t]),day)
    assert v[0,1,0]==3 and v[0,1,2]==2 # x latest before cutoff is t-130 ground; t-119 future excluded.
    assert v[0,1,5]==1 and v[0,1,6]==1 and np.isnan(v[1]).all()
    d=d.with_columns(pl.when(pl.col('t')==t-119).then(t-120).otherwise(pl.col('t')).alias('t'))
    v,_=encode(d,np.array([1]),np.array(['own']),np.array(['EDDF']),np.array([t]),day)
    assert v[0,1,2]==1 # latest airborne update prevents ghost ground occupancy.
    print('PASS latest-before-cutoff, own exclusion, airborne/stale rejection, unknown masks, spatial summaries')

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['selftest','pack','extract','merge','fit','evaluate']);p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--input',type=Path);p.add_argument('--output',type=Path);p.add_argument('--registration-commit');p.add_argument('--shard',type=int,choices=[0,1]);p.add_argument('--workers',type=int,default=3);p.add_argument('--probe-days',default='');p.add_argument('--parts',nargs='+',type=Path);p.add_argument('--fold',type=int,choices=range(1,7));p.add_argument('--threads',type=int,default=6);p.add_argument('--screen-evaluation',type=Path);a=p.parse_args();root=a.root.resolve()
    if a.action=='selftest':return selftest()
    if a.action=='pack':return pack(root,a.output.resolve(),a.registration_commit)
    if a.action=='merge':return merge(root,a.input.resolve(),a.parts,a.output.resolve(),a.registration_commit)
    if a.action=='fit':
        if a.threads not in (6,8,12):raise ValueError('fixed6/8/12 CPU cap')
        return fit(a.input.resolve(),a.output.resolve(),a.registration_commit,a.fold,a.threads,a.screen_evaluation)
    if a.action=='evaluate':return evaluate(a.input.resolve(),a.parts,a.output.resolve(),a.registration_commit)
    if a.workers not in (1,2,3):raise ValueError('1..3 workers allowed')
    return extract(root,a.input.resolve(),a.output.resolve(),a.registration_commit,a.shard,a.workers,a.probe_days)

if __name__=='__main__':main()

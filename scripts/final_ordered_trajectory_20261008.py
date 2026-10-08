"""Fresh final-view ordered E203 member; original sampler and frozen fitter.

All real-data actions require separate root preregistration. No E207 sampler,
CV/weight selection, tail assembly, submission, or environment modification.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

for k,v in {'POLARS_MAX_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
            'OMP_NUM_THREADS':'2','OMP_THREAD_LIMIT':'2','OMP_WAIT_POLICY':'PASSIVE',
            'GOMP_SPINCOUNT':'0','POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR':'load_as_storage'}.items():
    os.environ.setdefault(k,v)
import numpy as np
import polars as pl
import e194_trajectory_sequence as R
import e197_trajectory_encoder as S
import mlp_lowmem_prepare as L

E,M,C=S.E,S.M,S.C
PROTOCOL='final-original-ordered-E203-fresh2025-three-seed-eight-epoch-v1'
SELF='scripts/final_ordered_trajectory_20261008.py'
TEST='tests/test_final_ordered_trajectory_20261008.py'
AREA='analysis/final_ordered_trajectory_20261008'
DOC=AREA+'/PROTOCOL.md'
TRAIN_ROWS,FINAL_ROWS=2085047,670790
ID,Y='MVT_ID_mvt','TAXITIME_SEC_mvt'
original_indices=R.sample_time_indices
FEATURE_PROTOCOL='final-fresh-FIN1b-E113-motion-20261008-v1'
CONFIG={**S.CONFIG,'folds':[],'arms':['ordered'],'prep_workers':1,'full_2025':True,
        'sampler':'original-E194-unfilled','train_rows':TRAIN_ROWS,'final_rows':FINAL_ROWS}
ARRAYS=('xnf','xcf','xnp','xcp','y_scaled','base_p')
HELPERS=tuple(sorted({*('scripts/'+n for n in S.FILES),
    'scripts/e194_trajectory_sequence.py','scripts/final_tabular_models_20261008.py',
    *('src/'+n for n in M.SOURCES)}))


def write_json(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def read_json(path):return json.loads(Path(path).read_text())


def validate_ids(ids,template,rows):
    for a in (ids,template):
        if (a.shape!=(rows,) or not np.isfinite(a).all() or np.any(np.abs(a)>2**53) or not np.array_equal(a,np.rint(a))
            or len(np.unique(a))!=rows):raise ValueError('incomplete/duplicate/nonintegral admitted IDs')
    if not np.array_equal(np.sort(ids),np.sort(template)):raise ValueError('final ID set differs from template')


def seed_mean(seeds,sd,base):
    if (len(seeds)!=3 or not np.isfinite(sd) or sd<=0 or np.asarray(base).ndim!=1
        or not np.isfinite(base).all() or any(s.shape!=base.shape or s.dtype!=np.float32
        or not np.isfinite(s).all() for s in seeds)):raise ValueError('three finite float32 seeds and residual scale required')
    return np.mean(seeds,axis=0)*sd+base


def apply_tabular(frame,state):
    nums=state['numeric'];cats=state['categorical'];card=state['card']
    if state['version']!=1 or len(card)!=len(cats):raise ValueError('transform schema differs')
    if not {x['column'] for x in [*nums,*cats]}.issubset(frame.columns):raise ValueError('missing model column')
    width=sum(1+int(p['missing']) for p in nums)
    xn=np.empty((len(frame),width),np.float32);xc=np.empty((len(frame),len(cats)),np.int64)
    j=0
    for p in nums:
        q=np.asarray(p['quantiles'],np.float64)
        if q.ndim!=1 or len(q)<2 or not np.isfinite(q).all() or np.any(np.diff(q)<=0):raise ValueError('invalid numeric knots')
        a=frame[p['column']].cast(pl.Float64).to_numpy()
        xn[:,j]=np.where(np.isfinite(a),np.interp(a,q,np.linspace(-2,2,len(q))),0.0);j+=1
        if p['missing']:xn[:,j]=~np.isfinite(a);j+=1
    for j,p in enumerate(cats):
        mp={x['value']:x['code'] for x in p['mapping']}
        if len(mp)!=len(p['mapping']) or sorted(mp.values())!=list(range(1,card[j])):
            raise ValueError('category map/cardinality differs')
        xc[:,j]=np.fromiter((mp.get(v,0) for v in frame[p['column']].cast(pl.String).to_list()),dtype=np.int64,count=len(frame))
    return xn,xc


def prepare_tabular(fit,pred,cols,categorical):
    """Call the original preparer once; capture its actual category codes.

    Reconstruct deterministic numeric plans, then prove chunkwise exact replay.
    Never rerun value_counts to invent a potentially different category order.
    """
    xnf,xcf,xnp,xcp,card=L.prepare(fit,pred,cols,categorical=categorical)
    cats=[c for c in categorical if c in cols];nums=[c for c in cols if c not in cats]
    state=dict(version=1,columns=list(cols),numeric=[],categorical=[],card=list(card))
    for c in nums:
        a=fit[c].cast(pl.Float64).to_numpy();ok=np.isfinite(a)
        if ok.sum()<100:continue
        q=np.unique(np.quantile(a[ok],np.linspace(0,1,257)))
        if len(q)<2:continue
        state['numeric'].append(dict(column=c,quantiles=q.tolist(),missing=bool((~ok).any())))
    for j,c in enumerate(cats):
        pairs=pl.DataFrame({'value':fit[c].cast(pl.String),'code':xcf[:,j]}).unique()
        if pairs['value'].n_unique()!=pairs.height:raise ValueError('one category mapped to multiple original codes')
        mapping=pairs.filter(pl.col('code')>0).sort('code').to_dicts()
        state['categorical'].append(dict(column=c,mapping=mapping))
    for frame,xn,xc in ((fit,xnf,xcf),(pred,xnp,xcp)):
        for start in range(0,len(frame),65536):
            a,b=apply_tabular(frame.slice(start,65536),state)
            if not np.array_equal(a,xn[start:start+len(a)]) or not np.array_equal(b,xc[start:start+len(b)]):
                raise ValueError('persisted preprocessing does not replay original arrays exactly')
    return dict(xnf=xnf,xcf=xcf,xnp=xnp,xcp=xcp),state


def apply_encodings(frame,state,scope='full'):
    rows=frame.with_columns(local_hour_int=pl.col('local_hour').floor().cast(pl.Int8))
    encoded=rows.select(ID)
    for plan in state['scopes'][scope]:
        keys=plan['keys'];schema={k:rows.schema[k] for k in keys}
        stats=pl.from_dicts(plan['stats'],schema={**schema,'_s':pl.Float64,'_n':pl.UInt32})
        prior=pl.from_dicts(plan['prior'],schema={'ADEP_mvt':rows.schema['ADEP_mvt'],'_prior':pl.Float64})
        part=(rows.select(ID,*keys).join(stats,on=keys,how='left',nulls_equal=True)
              .join(prior,on='ADEP_mvt',how='left').with_columns(
                  ((pl.col('_s').fill_null(0)+20.*pl.col('_prior'))/(pl.col('_n').fill_null(0)+20.)).alias(plan['name']),
                  pl.col('_n').fill_null(0).alias(plan['name']+'_n')).select(ID,plan['name'],plan['name']+'_n'))
        encoded=encoded.join(part,on=ID)
    return frame.join(encoded,on=ID,how='left')


def capture_encodings(fit,pred,regular,te):
    """Persist only used TE/TER, preserving the unchanged original preparation."""
    if pred[Y].is_not_null().any():raise ValueError('final targets must be null before encodings')
    if os.environ.get('REGIME_RATES','0')!='0' or te.SMOOTH!=20.:raise ValueError('encoding recipe differs')
    actual_fit,actual_pred=te.add_encodings(fit,pred,regular)
    src=te._prep(fit).filter(pl.Series(regular))
    specs=[(n,k,'_y') for n,k in te.KEYS.items()]+[(n,k,'_r') for n,k in te.RESID_KEYS.items()]
    state=dict(version=1,columns=[c for n,_,_ in specs for c in (n,n+'_n')],scopes={})
    months=fit['month'].unique().sort().to_list()
    for scope in ['full',*[str(m) for m in months]]:
        data=src if scope=='full' else src.filter(pl.col('month')!=int(scope))
        state['scopes'][scope]=[]
        for name,keys,col in specs:
            prior=data.group_by('ADEP_mvt').agg(_prior=pl.col(col).mean())
            stats=data.drop_nulls(col).group_by(keys).agg(_s=pl.col(col).sum(),_n=pl.len())
            state['scopes'][scope].append(dict(name=name,keys=keys,prior=prior.to_dicts(),stats=stats.to_dicts()))
        rows=pred if scope=='full' else fit.filter(pl.col('month')==int(scope))
        expected=actual_pred if scope=='full' else actual_fit.filter(pl.col('month')==int(scope))
        actual=apply_encodings(rows,state,scope)
        if not actual.select(ID,*state['columns']).equals(expected.select(ID,*state['columns'])):
            raise ValueError('persisted TE/TER does not replay original encodings exactly')
    return actual_fit,actual_pred,state


def id_sha(ids):return hashlib.sha256(np.asarray(ids,dtype='<f8').tobytes()).hexdigest()


def safe(root,name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:raise ValueError('unsafe artifact path')
    result=(root/p).resolve()
    if not result.is_relative_to(root.resolve()):raise ValueError('artifact leaves workspace')
    return result


def validate_raw(a,deps):
    n=len(deps);shape=(n,2,64,12);p=a['present'];k=a['known'];v=a['values']
    if (v.shape!=shape or v.dtype!=np.float32 or k.shape!=shape or k.dtype!=np.uint8
        or p.shape!=shape[:3] or p.dtype!=np.uint8 or np.any(p>1) or np.any(k>p[...,None])
        or not np.isfinite(v).all() or np.any(v[k==0]!=0)):
        raise ValueError('raw tensor shape/mask/unknown-value contract differs')
    if (not np.array_equal(a['ids'],R.ids_checked(deps))
        or not np.array_equal(a['takeoff'],deps['MVT_TIME_UTC_mvt'].dt.epoch('s').to_numpy().astype(float))
        or not np.array_equal(a['airport'],deps['ADEP_mvt'].to_numpy())):
        raise ValueError('raw ID/time/airport differs from fresh movement')
    pad=p==0
    if (a['point_times'].shape!=p.shape or a['source_rows'].shape!=p.shape
        or np.any(a['point_times']>a['takeoff'][:,None,None])
        or not np.isnan(a['point_times'][pad]).all() or np.any(a['source_rows'][pad]!=-1)
        or not np.isfinite(a['point_times'][~pad]).all() or np.any(a['source_rows'][~pad]<0)
        or np.any(p.sum(-1)>np.minimum(a['raw_counts'],64))):
        raise ValueError('raw future/PAD/source/count contract differs')


def attest(root,commit,feature_manifest,out):
    """Small local Git attestation; no dataset preparation or target access."""
    if not re.fullmatch('[a-f0-9]{40}',commit or ''):raise ValueError('full registration required')
    pins={}
    for name in (SELF,TEST,DOC,*HELPERS):
        data=subprocess.check_output(['git','show',commit+':'+name],cwd=root)
        pins[name]=hashlib.sha256(data).hexdigest()
        if C.sha(root/name)!=pins[name]:raise ValueError('registered source drift '+name)
    if PROTOCOL.encode() not in subprocess.check_output(['git','show',commit+':HYPOTHESES_LOG.md'],cwd=root):
        raise ValueError('registered journal protocol absent')
    fm=read_json(feature_manifest)
    if fm['protocol']!=FEATURE_PROTOCOL or (fm['contract']['train_rows'],fm['contract']['rank_rows'])!=(TRAIN_ROWS,FINAL_ROWS):
        raise ValueError('fresh final feature manifest required')
    write_json(out,dict(protocol=PROTOCOL,registration_commit=commit,config=CONFIG,source_sha256=pins,
                        feature_manifest_sha256=C.sha(feature_manifest)))


def admit(root,receipt):
    r=read_json(receipt)
    if (r['protocol']!=PROTOCOL or r['config']!=CONFIG
        or not re.fullmatch('[a-f0-9]{40}',r['registration_commit'])
        or set(r['source_sha256'])!={SELF,TEST,DOC,*HELPERS}):raise ValueError('registration contract differs')
    for name,h in r['source_sha256'].items():
        if C.sha(safe(root,name))!=h:raise ValueError('active registered source drift '+name)
    if any(r['source_sha256']['src/'+n]!=h for n,h in M.SOURCES.items()):
        raise ValueError('original E203 trainer closure changed')
    for module in (R,S,E,M,L,C,S.D):
        name='scripts/'+Path(module.__file__).name
        if Path(module.__file__).resolve()!=safe(root,name):raise ValueError('active helper loaded from another source tree')
    return r


def feature_inputs(work,r,complete=True):
    mp=work/'manifest.json'
    if C.sha(mp)!=r['feature_manifest_sha256']:raise ValueError('feature manifest admission differs')
    m=read_json(mp)
    if m['protocol']!=FEATURE_PROTOCOL or (m['contract']['train_rows'],m['contract']['rank_rows'])!=(TRAIN_ROWS,FINAL_ROWS):
        raise ValueError('feature population differs')
    files={f['path']:f['sha256'] for f in m['files']}
    raw=sorted(n for n in files if re.fullmatch(r'data/prc-2026-datasets/training_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}\.parquet',n))
    if len(raw)!=12:raise ValueError('12 canonical training movement files required')
    rank='data/prc-2026-datasets/ranking.parquet';template='data/prc-2026-datasets/submitting.parquet'
    for n in (*raw,rank,template):
        if n not in files or C.sha(safe(work,n))!=files[n]:raise ValueError('canonical challenge input drift '+n)
    if not complete:return m,[safe(work,n) for n in (*raw,rank)]
    c=read_json(work/'feature_complete.json');feature=work/'data/features_fin1b.parquet'
    if (c['protocol']!=FEATURE_PROTOCOL or c['manifest_sha256']!=C.sha(mp)
        or c['registration_commit']!=m['registration_commit'] or c['fits_performed']!=0
        or (c['train_rows'],c['rank_rows'])!=(TRAIN_ROWS,FINAL_ROWS)
        or c['feature_path']!='data/features_fin1b.parquet' or C.sha(feature)!=c['feature_sha256']
        or c['template_sha256']!=files[template]):raise ValueError('feature completion proof differs')
    feature_output=c['outputs'].get('data/features_fin1b.parquet',{})
    if (feature_output.get('sha256')!=c['feature_sha256']
        or c['source_sha256']!={n:h for n,h in files.items() if n.endswith('.py')}
        or feature_output.get('train_ordered_id_sha256')!=c['train_ordered_id_sha256']
        or feature_output.get('rank_ordered_id_sha256')!=c['rank_ordered_id_sha256']):
        raise ValueError('feature output/source receipt binding differs')
    return m,c,feature,safe(work,template)


def capacity(cpu,gib,free,output):
    if sys.platform!='linux':raise ValueError('real neural data work is Linux-only; no Mac bulk')
    rel=next(x[3:] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    p=Path('/sys/fs/cgroup')/rel.lstrip('/');mem=(p/'memory.max').read_text().strip()
    quota,period=(p/'cpu.max').read_text().split();swap=(p/'memory.swap.max').read_text().strip()
    if mem=='max' or int(mem)>gib*1024**3 or quota=='max' or int(quota)>cpu*int(period) or swap!='0':
        raise ValueError(f'require bounded <={cpu}CPU/<={gib}GiB/swap0 cgroup')
    parent=output
    while not parent.exists():parent=parent.parent
    if shutil.disk_usage(parent).free<free*1024**3:raise ValueError(f'require {free}GiB free disk')
    return dict(cgroup=rel,cpu_max=[int(quota),int(period)],memory_max=int(mem),swap_max=0)


@contextmanager
def staging(out):
    if out.exists():raise ValueError('fresh output required; no automatic retry/resume')
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent))
    try:
        yield stage
        stage.rename(out)
    except BaseException as error:
        write_json(stage/'failure.json',dict(error=type(error).__name__,message=str(error),time=time.time()))
        raise


def finish(stage,kind,receipt,**evidence):
    r=read_json(receipt)
    write_json(stage/'complete.json',dict(protocol=PROTOCOL,kind=kind,registration_commit=r['registration_commit'],
        receipt_sha256=C.sha(receipt),config=CONFIG,files={str(p.relative_to(stage)):C.sha(p)
        for p in sorted(stage.rglob('*')) if p.is_file()},**evidence))


def checked(path,kind,receipt):
    r=read_json(receipt);c=read_json(path/'complete.json')
    if (c['protocol']!=PROTOCOL or c['kind']!=kind or c['config']!=CONFIG
        or c['registration_commit']!=r['registration_commit'] or c['receipt_sha256']!=C.sha(receipt)):
        raise ValueError('artifact registration/lineage differs')
    actual={str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}-{'complete.json'}
    if actual!=set(c['files']):raise ValueError('artifact file inventory differs')
    for name,h in c['files'].items():
        if C.sha(safe(path,name))!=h:raise ValueError('artifact hash drift '+name)
    return c


def reusable_training(pairs,movement_hashes):
    """Admit only original E194 products tied to unchanged canonical 2025 files."""
    index={}
    for source,output in pairs:
        m=read_json(source/'manifest.json');c=read_json(output/'complete.json')
        if (m['protocol']!=R.PROTOCOL or m['registration_commit']!=S.RAW_REG
            or c['registration_commit']!=S.RAW_REG or c['protocol']!=R.PROTOCOL
            or c['input_manifest_sha256']!=C.sha(source/'manifest.json')
            or m['files'].get('scripts/e194_trajectory_sequence.py')!=C.sha(R.__file__)
            or m['departure_columns']!=R.DEP_COLS or m['numeric_columns']!=R.NUM_NAMES):
            raise ValueError('reuse must prove original unfilled E194 lineage')
        old={Path(n).name:h for n,h in m['raw_input_sha256'].items() if Path(n).name.startswith('training_')}
        if old!=movement_hashes:raise ValueError('reuse canonical 2025 movement hashes differ')
        reports={x['day']:x for x in c['reports']}
        for day in m['days']:
            if not day.startswith('2025-'):continue
            if day in index or day not in reports:raise ValueError('overlap/missing reused training day')
            dp=source/'departures'/f'{day}.parquet';npz=output/day/'sequence.npz'
            report=read_json(output/day/'receipt.json')
            if (report!=reports[day] or report['source_sha256']!=C.sha(R.__file__)
                or C.sha(dp)!=m['files']['departures/'+day+'.parquet'] or C.sha(npz)!=report['npz_sha256']):
                raise ValueError('reuse departure/sequence receipt drift')
            index[day]=dict(departures=str(dp),npz=str(npz),sha256=report['npz_sha256'],
                point_sha256=m['point_inputs'][day].get('sha256'),input_manifest_sha256=C.sha(source/'manifest.json'),
                complete_sha256=C.sha(output/'complete.json'),receipt_sha256=C.sha(output/day/'receipt.json'))
    return index


def raw(root,work,receipt,out,surface_dirs,reuse_pairs):
    r=admit(root,receipt);guard=capacity(3,10,8,out);m,movements=feature_inputs(work,r,complete=False)
    hashes={p.name:C.sha(p) for p in movements if p.name.startswith('training_')}
    reuse=reusable_training(reuse_pairs,hashes)
    deps=pl.scan_parquet(movements).select(R.DEP_COLS).filter(pl.col('PHASE_mvt')=='DEP').collect()
    ids=R.ids_checked(deps)
    if len(ids)!=TRAIN_ROWS+FINAL_ROWS or id_sha(np.sort(ids))!=m['contract']['all_sorted_id_sha256']:
        raise ValueError('raw query population differs from admitted final view')
    deps=deps.with_columns(pl.col('MVT_TIME_UTC_mvt').dt.strftime('%Y-%m-%d').alias('_day'))
    if deps['_day'].null_count():raise ValueError('null movement date')
    days=deps['_day'].unique().sort().to_list();start=time.monotonic()
    with staging(out) as stage:
        (stage/'departures').mkdir();plan=[]
        # Capture all input hashes before any encoding, then verify again before completion.
        for day in days:
            dep=deps.filter(pl.col('_day')==day).drop('_day').sort(ID)
            dep.write_parquet(stage/'departures'/f'{day}.parquet')
            candidates=[p/f'surface_{day}.parquet' for p in surface_dirs if (p/f'surface_{day}.parquet').is_file()]
            ph={str(p.resolve()):C.sha(p) for p in candidates}
            if not candidates and day!='2025-12-31':raise ValueError('missing canonical surface day '+day)
            if len(set(ph.values()))>1:raise ValueError('conflicting same-day surface inputs '+day)
            point_sha=next(iter(ph.values()),None);old=reuse.get(day)
            if old is not None and (old['point_sha256']!=point_sha or not pl.read_parquet(old['departures'],columns=R.DEP_COLS).equals(dep)):
                raise ValueError('reuse raw surface or complete movement fields differ '+day)
            plan.append(dict(day=day,rows=len(dep),surface=next(iter(ph),None),surface_sha256=point_sha,reuse=old))
        write_json(stage/'raw_plan.json',dict(days=plan,movement_sha256={str(p):C.sha(p) for p in movements},
            feature_manifest_sha256=r['feature_manifest_sha256'],same_day_only=True,source_sha256=C.sha(R.__file__)))
        del deps;gc.collect()
        for item in plan:
            day=item['day'];dest=stage/day;dest.mkdir();dep=pl.read_parquet(stage/'departures'/f'{day}.parquet')
            if item['reuse']:
                original=Path(item['reuse']['npz'])
                if C.sha(original)!=item['reuse']['sha256']:raise ValueError('reused NPZ changed')
                shutil.copyfile(original,dest/'sequence.npz')
                with np.load(dest/'sequence.npz',allow_pickle=False) as data:
                    arrays={k:data[k] for k in data.files} # each compressed member exactly once
            else:
                points=None
                if item['surface']:
                    sp=Path(item['surface'])
                    if C.sha(sp)!=item['surface_sha256']:raise ValueError('surface changed before encode')
                    available=pl.scan_parquet(sp).collect_schema()
                    if not (set(R.POINT_COLS)-{'stale'}).issubset(available):raise ValueError('surface schema differs')
                    points=pl.read_parquet(sp,columns=[c for c in R.POINT_COLS if c in available])
                arrays,_=R.encode_day(points,dep,day);del points
                np.savez_compressed(dest/'sequence.npz',**arrays)
            validate_raw(arrays,dep);del arrays,dep
            if item['surface'] and C.sha(item['surface'])!=item['surface_sha256']:raise ValueError('surface changed during encode')
            print(json.dumps(dict(raw_day=day,rows=item['rows'],reused_original_2025=bool(item['reuse']))),flush=True)
        feature_inputs(work,r,complete=False);admit(root,receipt)
        finish(stage,'raw',receipt,feature_manifest_sha256=r['feature_manifest_sha256'],days=days,
               rows=len(ids),reused_days=sum(bool(x['reuse']) for x in plan),seconds=time.monotonic()-start,cgroup=guard)


def fixed_environment():
    keep={k:os.environ[k] for k in ('HOME','PATH','TMPDIR','LANG') if k in os.environ}
    os.environ.clear();os.environ.update({**keep,**M.FIXED_ENV,'VECLIB_MAXIMUM_THREADS':'1','FEATURES':'__immutable_final_inputs_only__'})


def trainer(root):
    N=E.trainer(root/'src')
    for name in ('e100_nn_partner','train','target_encoding','features'):
        module=sys.modules[name]
        if Path(module.__file__).resolve()!=(root/'src'/(name+'.py')).resolve():raise ValueError('cached trainer from another source tree')
    if str(N.DEV)!='cuda:0' or N.EPOCHS!=8:raise ValueError('unchanged trainer configuration required')
    return N


def validate_queries(a,s):
    n=s['ordinary_rows'];p=FINAL_ROWS
    if n<4096 or not np.isfinite(s['sd']) or s['sd']<=0:raise ValueError('ordinary population/scale differs')
    for name,rows,dtype in [('xnf',n,np.float32),('xnp',p,np.float32),('xcf',n,np.int64),('xcp',p,np.int64)]:
        x=a[name]
        if x.ndim!=2 or len(x)!=rows or x.dtype!=dtype:raise ValueError('query matrix shape/dtype differs')
        for start in range(0,rows,65536):
            if not np.isfinite(x[start:start+65536]).all():raise ValueError('nonfinite queries')
    if a['xnf'].shape[1]!=a['xnp'].shape[1]:raise ValueError('numeric widths differ')
    for name in ('xcf','xcp'):
        if a[name].shape[1]!=len(s['card']):raise ValueError('categorical width differs')
        for j,card in enumerate(s['card']):
            if card<1 or a[name][:,j].min()<0 or a[name][:,j].max()>=card:raise ValueError('category code differs')
    for name,rows in [('y_scaled',n),('base_p',p)]:
        if a[name].shape!=(rows,) or a[name].dtype!=np.float64 or not np.isfinite(a[name]).all():raise ValueError('target/proxy tensor differs')


def queries(root,work,receipt,out):
    r=admit(root,receipt);guard=capacity(6,40,40,out);m,c,feature,template=feature_inputs(work,r)
    fixed_environment();producer=M.runtime();N=trainer(root)
    import target_encoding as te
    import final_tabular_models_20261008 as proxy
    frame=pl.read_parquet(feature);fit=frame.filter(pl.col('split')=='train');pred=frame.filter(pl.col('split')=='rank');del frame
    ids=pred[ID].to_numpy().astype(np.float64);train_ids=fit[ID].to_numpy().astype(np.float64)
    validate_ids(ids,pl.read_parquet(template,columns=[ID])[ID].to_numpy(),FINAL_ROWS)
    validate_ids(train_ids,train_ids,TRAIN_ROWS)
    if (np.intersect1d(ids,train_ids).size or pred[Y].is_not_null().any() or pred['BLOCK_TIME_UTC_mvt'].is_not_null().any()
        or set(fit['month'].unique().to_list())!=set(range(1,13))
        or id_sha(train_ids)!=c['train_ordered_id_sha256'] or id_sha(ids)!=c['rank_ordered_id_sha256']):
        raise ValueError('feature rows/target boundary/months differ')
    lab=N.T.regime_label(fit);tr=np.flatnonzero(lab==0);pi=np.arange(TRAIN_ROWS,TRAIN_ROWS+FINAL_ROWS,dtype=np.int64)
    fit,pred,encoding=capture_encodings(fit,pred,lab==0,te)
    if not np.array_equal(fit[ID].to_numpy(),train_ids) or not np.array_equal(pred[ID].to_numpy(),ids):
        raise ValueError('encoding reordered training/final rows relative to regime labels')
    cols=N.T.feature_columns(fit)
    if (Y in cols or ID in cols or any(x.startswith(('tq_','rr_')) for x in cols)
        or not set(encoding['columns']).issubset(cols)):raise ValueError('actual E203 feature encoding contract differs')
    ordinary=fit.filter(pl.Series(lab==0));base_f,base_p=N.T.proxy_base(ordinary,pred)
    proxy_state=proxy.fit_proxy(ordinary)
    for data,expected in ((ordinary,base_f),(pred,base_p)):
        if not np.array_equal(proxy.apply_proxy(data,proxy_state),expected,equal_nan=True):
            raise ValueError('serialized shared proxy differs from original E46 arithmetic')
    y=np.clip(np.clip(ordinary[Y].to_numpy(),*N.T.REG_CLIP)-base_f,-7200,7200);sd=float(y.std())
    a,tabular=prepare_tabular(ordinary,pred,cols,N.T.CATEGORICAL);a.update(y_scaled=y/sd,base_p=base_p)
    s=dict(card=tabular['card'],sd=sd,ordinary_rows=len(ordinary),columns=cols,train_rows=TRAIN_ROWS,final_rows=FINAL_ROWS,
           train_ordered_id_sha256=id_sha(train_ids),final_ordered_id_sha256=id_sha(ids),ordinary_ordered_id_sha256=id_sha(train_ids[tr]))
    validate_queries(a,s)
    with staging(out) as stage:
        for name,value in a.items():np.save(stage/(name+'.npy'),value,allow_pickle=False)
        for name,value in [('query_ids',np.r_[train_ids,ids]),('train_indices',tr),('pred_indices',pi),('mvt_id',ids)]:
            np.save(stage/(name+'.npy'),value,allow_pickle=False)
        write_json(stage/'transforms.json',dict(tabular=tabular,encodings=encoding,proxy=proxy_state,sd=sd))
        write_json(stage/'spec.json',s)
        feature_inputs(work,r);admit(root,receipt)
        finish(stage,'queries',receipt,spec=s,feature_manifest_sha256=r['feature_manifest_sha256'],
               feature_complete_sha256=C.sha(work/'feature_complete.json'),feature_sha256=c['feature_sha256'],producer_runtime=producer,cgroup=guard)


def query_arrays(bundle):
    s=read_json(bundle/'spec.json');a={n:np.load(bundle/(n+'.npy'),allow_pickle=False) for n in ARRAYS}
    validate_queries(a,s)
    ids=np.load(bundle/'query_ids.npy',allow_pickle=False);tr=np.load(bundle/'train_indices.npy',allow_pickle=False)
    pi=np.load(bundle/'pred_indices.npy',allow_pickle=False);final=np.load(bundle/'mvt_id.npy',allow_pickle=False)
    validate_ids(ids,ids,TRAIN_ROWS+FINAL_ROWS);validate_ids(final,ids[TRAIN_ROWS:],FINAL_ROWS)
    if (tr.dtype!=np.int64 or len(tr)!=s['ordinary_rows'] or tr.ndim!=1 or np.any(np.diff(tr)<=0)
        or tr.min()<0 or tr.max()>=TRAIN_ROWS or not np.array_equal(pi,np.arange(TRAIN_ROWS,len(ids)))
        or not np.array_equal(final,ids[pi]) or id_sha(ids[tr])!=s['ordinary_ordered_id_sha256']):
        raise ValueError('ordinary/context/prediction index contract differs')
    return a,s,ids,tr,pi


def context(root,receipt,qb,rawdir,out):
    admit(root,receipt);guard=capacity(3,10,20,out)
    q=checked(qb,'queries',receipt);r=checked(rawdir,'raw',receipt)
    if q['feature_manifest_sha256']!=r['feature_manifest_sha256']:raise ValueError('raw/query final view differs')
    ids=np.load(qb/'query_ids.npy',allow_pickle=False);tr=np.load(qb/'train_indices.npy',allow_pickle=False)
    order=np.argsort(ids);sorted_ids=ids[order];fit=np.zeros(len(ids),bool);fit[tr]=True;seen=np.zeros(len(ids),bool)
    count=np.zeros(6,np.int64);total=np.zeros(6);squares=np.zeros(6)
    for day in r['days']:
        with np.load(rawdir/day/'sequence.npz',allow_pickle=False) as z:
            ix=S.map_day(z['ids'],sorted_ids,order);values,known=z['values'],z['known']
        if seen[ix].any():raise ValueError('raw query duplicates')
        seen[ix]=True;n,s,ss=S.norm_stats(values,known,fit[ix]);count+=n;total+=s;squares+=ss
    if not seen.all():raise ValueError('raw queries incomplete')
    mean,std=S.normalization(count,total,squares)
    with staging(out) as stage:
        shape=(len(ids),2,64,12)
        arrays={'values':np.lib.format.open_memmap(stage/'values.npy',mode='w+',dtype=np.float16,shape=shape),
            'known':np.lib.format.open_memmap(stage/'known.npy',mode='w+',dtype=np.uint8,shape=shape),
            'present':np.lib.format.open_memmap(stage/'present.npy',mode='w+',dtype=np.uint8,shape=shape[:3])}
        for day in r['days']:
            with np.load(rawdir/day/'sequence.npz',allow_pickle=False) as data:
                a={k:data[k] for k in data.files}
            validate_raw(a,pl.read_parquet(rawdir/'departures'/f'{day}.parquet'))
            ix=S.map_day(a['ids'],sorted_ids,order)
            arrays['values'][ix]=S.normalize(a['values'],a['known'],mean,std)
            arrays['known'][ix]=a['known'];arrays['present'][ix]=a['present']
            for target in arrays.values():S.release_pages(target,write=True)
            del a
        del arrays;gc.collect();np.save(stage/'query_ids.npy',ids,allow_pickle=False)
        write_json(stage/'normalization.json',dict(count=count.tolist(),mean=mean.tolist(),std=std.tolist(),channels=list(range(6)),
            fitted_query_rows=len(tr),fitted_query_ids_sha256=id_sha(ids[tr]),ordinary_2025_only=True))
        admit(root,receipt)
        finish(stage,'context',receipt,query_complete_sha256=C.sha(qb/'complete.json'),raw_complete_sha256=C.sha(rawdir/'complete.json'),
               rows=len(ids),cgroup=guard)


def gpu_tables(prepared,rows):
    import torch
    tables={}
    for name,dtype in [('values',torch.float16),('known',torch.uint8),('present',torch.uint8)]:
        a=np.load(prepared/(name+'.npy'),mmap_mode='r',allow_pickle=False)
        shape=(rows,2,64,12) if name!='present' else (rows,2,64)
        if a.shape!=shape or a.dtype!=np.dtype('float16' if name=='values' else 'uint8'):raise ValueError('dynamic context shape/dtype differs')
        t=torch.empty(shape,dtype=dtype,device='cuda:0')
        for lo in range(0,rows,8192):
            chunk=np.array(a[lo:lo+8192],copy=True);t[lo:lo+len(chunk)].copy_(torch.from_numpy(chunk));S.release_pages(a)
        tables[name]=t;del a,chunk
    return tables


def checked_context(prepared,qb,receipt,ids,tr):
    c=checked(prepared,'context',receipt);norm=read_json(prepared/'normalization.json')
    if (c['query_complete_sha256']!=C.sha(qb/'complete.json') or c['rows']!=len(ids)
        or not np.array_equal(np.load(prepared/'query_ids.npy',allow_pickle=False),ids)
        or norm['fitted_query_rows']!=len(tr) or norm['fitted_query_ids_sha256']!=id_sha(ids[tr])
        or norm['ordinary_2025_only'] is not True):raise ValueError('context/query/normalization lineage differs')
    return c


def save_bins(path,continuous,scalar,bins):
    np.savez(path,continuous=continuous,scalar=scalar,**{f'bin_{j}':b for j,b in enumerate(bins)})


def load_bins(path):
    with np.load(path,allow_pickle=False) as z:
        continuous=z['continuous'].tolist();scalar=z['scalar'].tolist()
        if set(z.files)!={'continuous','scalar',*(f'bin_{j}' for j in range(len(continuous)))}:raise ValueError('PLE bins incomplete')
        return continuous,scalar,[z[f'bin_{j}'] for j in range(len(continuous))]


def predict_checkpoint(root,qb,model,arrays,spec,bins,tables,pi):
    import torch
    N=trainer(root);cont,scalar,knots=bins
    net=S.network(N,spec['card'],cont,scalar,knots,arrays['xnp'].shape[1],'ordered').to('cuda:0')
    net.load_state_dict(torch.load(model,map_location='cuda:0',weights_only=True));net.eval();parts=[]
    with torch.no_grad():
        for lo in range(0,len(pi),4096):
            parts.append(net(torch.from_numpy(arrays['xnp'][lo:lo+4096]).to('cuda:0'),
                torch.from_numpy(arrays['xcp'][lo:lo+4096]).to('cuda:0'),*S.context_batch(tables,pi[lo:lo+4096])).cpu().numpy())
    result=np.concatenate(parts);del net,parts;gc.collect();torch.cuda.empty_cache()
    if result.shape!=(len(pi),) or result.dtype!=np.float32 or not np.isfinite(result).all():raise ValueError('checkpoint prediction differs')
    return result


def smoke(root,receipt,qb,out):
    """Post-registration synthetic-only capacity admission, using actual query dimensions."""
    r=admit(root,receipt);guard=capacity(2,12,25,out);checked(qb,'queries',receipt);fixed_environment()
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);runtime=E.package_runtime();N=trainer(root)
    s=read_json(qb/'spec.json');x=np.load(qb/'xnf.npy',mmap_mode='r',allow_pickle=False)
    width=x.shape[1];rng=np.random.default_rng(203);synthetic=rng.normal(size=(4096,width)).astype(np.float32)
    cont,scalar,bins=E.numeric_bins(synthetic);del x
    xn=torch.from_numpy(synthetic).to('cuda:0');xc=torch.zeros((4096,len(s['card'])),dtype=torch.int64,device='cuda:0')
    v=torch.randn((4096,2,64,12),dtype=torch.float16,device='cuda:0');k=torch.ones_like(v,dtype=torch.uint8)
    p=torch.ones((4096,2,64),dtype=torch.uint8,device='cuda:0');p[:8]=0;p[8:16,1]=0;k[16:24,:,:,4]=0
    reserve=torch.empty(((TRAIN_ROWS+FINAL_ROWS)*2*64*37,),dtype=torch.uint8,device='cuda:0')
    torch.manual_seed(203);net=S.network(N,s['card'],cont,scalar,bins,width,'ordered').to('cuda:0')
    opt=torch.optim.AdamW(net.parameters(),lr=.002);torch.cuda.reset_peak_memory_stats()
    for _ in range(6):
        y=net(xn,xc,v,k,p);loss=y.square().mean();opt.zero_grad();loss.backward();opt.step()
        if not torch.isfinite(y).all() or any(x.grad is not None and not torch.isfinite(x.grad).all() for x in net.parameters()):
            raise ValueError('synthetic finite/gradient failure')
    net.eval()
    with torch.no_grad():
        base=net(xn[:32],xc[:32],v[:32],k[:32],p[:32]);changed=v[:32].clone();changed[k[:32]==0]=1234;changed[p[:32]==0]=1234
        if not torch.allclose(base,net(xn[:32],xc[:32],changed,k[:32],p[:32]),atol=1e-6,rtol=1e-5):raise ValueError('unknown/PAD affects network')
    with staging(out) as stage:
        torch.save(net.state_dict(),stage/'synthetic.pt')
        clone=S.network(N,s['card'],cont,scalar,bins,width,'ordered').to('cuda:0')
        clone.load_state_dict(torch.load(stage/'synthetic.pt',weights_only=True,map_location='cuda:0'));clone.eval()
        with torch.no_grad():
            if not torch.equal(base,clone(xn[:32],xc[:32],v[:32],k[:32],p[:32])):raise ValueError('checkpoint reload differs')
        # Retain the tiny receipt, not a synthetic checkpoint mistaken for a target model.
        (stage/'synthetic.pt').unlink()
        finish(stage,'smoke',receipt,synthetic_only=True,passed=True,query_complete_sha256=C.sha(qb/'complete.json'),
            runtime=runtime,parameters=sum(x.numel() for x in net.parameters()),peak_gpu_bytes=torch.cuda.max_memory_allocated(),cgroup=guard)
    del reserve,net,clone,opt


def run(root,receipt,qb,prepared,smoke_dir,out):
    admit(root,receipt);guard=capacity(2,12,25,out);fixed_environment()
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    q=checked(qb,'queries',receipt);sm=checked(smoke_dir,'smoke',receipt);runtime=E.package_runtime()
    if (sm['passed'] is not True or sm['synthetic_only'] is not True or sm['runtime']!=runtime
        or sm['query_complete_sha256']!=C.sha(qb/'complete.json')):raise ValueError('full-shape runtime smoke not admitted')
    arrays,spec,ids,tr,pi=query_arrays(qb);checked_context(prepared,qb,receipt,ids,tr)
    cont,scalar,bins=E.numeric_bins(arrays['xnf']);N=trainer(root);tables=gpu_tables(prepared,len(ids))
    with staging(out) as stage:
        save_bins(stage/'bins.npz',cont,scalar,bins);fits=[]
        for seed in (0,1,2):
            fits.append(S.fit(arrays,spec,N,cont,scalar,bins,tables,tr,pi,stage,'ordered',seed))
            saved=np.load(stage/f'seed_{seed}.npy',allow_pickle=False)
            replay=predict_checkpoint(root,qb,stage/f'model_{seed}.pt',arrays,spec,(cont,scalar,bins),tables,pi)
            if not np.array_equal(replay,saved):raise ValueError('persisted target checkpoint does not replay exact predictions')
        if len({f['parameters'] for f in fits})!=1 or any(len(f['losses'])!=8 or not np.isfinite(f['losses']).all() for f in fits):
            raise ValueError('three complete same-capacity eight-epoch fits required')
        reg=seed_mean([np.load(stage/f'seed_{i}.npy',allow_pickle=False) for i in range(3)],spec['sd'],arrays['base_p'])
        np.savez_compressed(stage/'prediction.npz',mvt_id=ids[pi],reg=reg,reg2=reg)
        shutil.copyfile(qb/'transforms.json',stage/'transforms.json');shutil.copyfile(prepared/'normalization.json',stage/'normalization.json')
        write_json(stage/'spec.json',spec);admit(root,receipt)
        finish(stage,'models',receipt,query_complete_sha256=C.sha(qb/'complete.json'),context_complete_sha256=C.sha(prepared/'complete.json'),
            smoke_complete_sha256=C.sha(smoke_dir/'complete.json'),runtime=runtime,seeds=[0,1,2],fits=fits,
            checkpoint_replay_exact=True,rows=FINAL_ROWS,raw_member_only=True,cgroup=guard)


def replay(root,work,receipt,qb,prepared,models,out):
    """Zero-fit replay from saved transformations and checkpoints on the admitted final view."""
    r=admit(root,receipt);guard=capacity(2,12,25,out);fixed_environment()
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    import final_tabular_models_20261008 as proxy
    q=checked(qb,'queries',receipt);m=checked(models,'models',receipt);runtime=E.package_runtime()
    if (m['query_complete_sha256']!=C.sha(qb/'complete.json') or m['context_complete_sha256']!=C.sha(prepared/'complete.json')
        or m['runtime']!=runtime or m['seeds']!=[0,1,2] or m['checkpoint_replay_exact'] is not True):raise ValueError('model artifact lineage differs')
    _,fc,feature,template=feature_inputs(work,r)
    if fc['feature_sha256']!=q['feature_sha256']:raise ValueError('replay feature differs')
    state=read_json(models/'transforms.json');spec=read_json(models/'spec.json')
    pred=pl.scan_parquet(feature).filter(pl.col('split')=='rank').collect()
    if pred[Y].is_not_null().any():raise ValueError('final target boundary violated')
    pred=apply_encodings(pred,state['encodings']);xn,xc=apply_tabular(pred,state['tabular'])
    base=proxy.apply_proxy(pred,state['proxy']);ids=np.load(qb/'query_ids.npy',allow_pickle=False)
    tr=np.load(qb/'train_indices.npy',allow_pickle=False);pi=np.load(qb/'pred_indices.npy',allow_pickle=False)
    checked_context(prepared,qb,receipt,ids,tr)
    if not np.array_equal(pred[ID].to_numpy(),ids[pi]):raise ValueError('replay row order differs')
    for name,value in [('xnp',xn),('xcp',xc),('base_p',base)]:
        if not np.array_equal(value,np.load(qb/(name+'.npy'),allow_pickle=False)):raise ValueError('serialized query replay differs '+name)
    del pred;gc.collect();tables=gpu_tables(prepared,len(ids));bins=load_bins(models/'bins.npz')
    a=dict(xnp=xn,xcp=xc);seeds=[]
    with staging(out) as stage:
        for seed in (0,1,2):
            value=predict_checkpoint(root,qb,models/f'model_{seed}.pt',a,spec,bins,tables,pi)
            if not np.array_equal(value,np.load(models/f'seed_{seed}.npy',allow_pickle=False)):raise ValueError('replay prediction differs')
            seeds.append(value)
        reg=seed_mean(seeds,state['sd'],base)
        with np.load(models/'prediction.npz',allow_pickle=False) as z:
            if not np.array_equal(reg,z['reg']) or not np.array_equal(ids[pi],z['mvt_id']):raise ValueError('replay mean differs')
        np.savez_compressed(stage/'prediction.npz',mvt_id=ids[pi],reg=reg,reg2=reg)
        finish(stage,'replay',receipt,models_complete_sha256=C.sha(models/'complete.json'),runtime=runtime,rows=len(pi),fits_performed=0,
               saved_transforms_exact=True,saved_checkpoints_exact=True,cgroup=guard)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['attest','raw','queries','context','smoke','run','replay'])
    for name in ('root','work','receipt','output','feature-manifest','queries','raw','prepared','smoke','models'):
        p.add_argument('--'+name,type=Path,default=Path.cwd() if name=='root' else None)
    p.add_argument('--registration-commit');p.add_argument('--surface-dir',type=Path,action='append',default=[])
    p.add_argument('--reuse',type=Path,nargs=2,action='append',default=[],metavar=('ORIGINAL_INPUT','ORIGINAL_OUTPUT'))
    a=p.parse_args();root=a.root.resolve()
    if a.output is None:p.error('--output is required')
    out=a.output.resolve()
    if a.action=='attest':attest(root,a.registration_commit,a.feature_manifest,out);return
    if a.receipt is None:p.error('--receipt is required')
    def stopped(signum,frame):raise SystemExit('bounded worker stopped by signal '+str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,stopped)
    failure=out.with_name(out.name+'.failure.json')
    if out.exists() or failure.exists():raise ValueError('fresh output namespace required; no automatic retry')
    signal.alarm(6*3600)
    try:
        if a.action=='raw':raw(root,a.work,a.receipt,out,a.surface_dir,a.reuse)
        elif a.action=='queries':queries(root,a.work,a.receipt,out)
        elif a.action=='context':context(root,a.receipt,a.queries,a.raw,out)
        elif a.action=='smoke':smoke(root,a.receipt,a.queries,out)
        elif a.action=='run':run(root,a.receipt,a.queries,a.prepared,a.smoke,out)
        else:replay(root,a.work,a.receipt,a.queries,a.prepared,a.models,out)
    except BaseException as error:
        out.parent.mkdir(parents=True,exist_ok=True)
        write_json(failure,dict(protocol=PROTOCOL,action=a.action,error=type(error).__name__,message=str(error),time=time.time()))
        raise


if __name__=='__main__':main()

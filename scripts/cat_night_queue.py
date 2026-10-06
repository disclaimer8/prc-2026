"""Fixed, provenance-preserving distribution of the existing 90 Cat fits.

No quality selection, new hyperparameters, reduced seed count or submission.
Home hands off after the next completed fit; failed partials remain untouched.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

os.environ.setdefault('POLARS_MAX_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import polars as pl
import cat_fin1b_oof as C
import cat_runtime_parity as R
import cat_laptop_pools as L

PROTOCOL='cat-fin1b-distributed-90fit-night-v1'
NODES=('home','hetzner','mac','laptop')
THREADS=dict(home=24,hetzner=6,mac=14,laptop=6)
OLD_UNIT='prc-cat-fin1b-oof-20261005.service'


def assignments():
    jobs=[list(j) for j in C.schedule()]
    return dict(home=jobs[:44],hetzner=jobs[44:64],mac=jobs[64:80],laptop=jobs[80:])


def validate_assignments(a):
    if set(a)!=set(NODES):raise ValueError('node inventory differs')
    jobs=[tuple(j) for n in NODES for j in a[n]]
    if len(jobs)!=90 or len(set(jobs))!=90 or set(jobs)!=set(C.schedule()):
        raise ValueError('assignment must cover all 90 jobs exactly once')


def require_job(m,node,seed,fold):
    if [seed,fold] not in m['assignments'][node]:raise ValueError('job belongs to another node')


def qualification(ids,ref,actual_ids,pred,node):
    r=R.compare(ids,ref,actual_ids,pred)
    tolerance=1e-6 if node=='mac' else 0.
    return {**r,'absolute_tolerance':tolerance,'passed':r['max_abs']<=tolerance}


def name(fold):return f'fold_{fold}_{fold+6}'


def helpers():
    return {Path(p).name:C.sha(p) for p in (__file__,C.__file__,R.__file__,L.__file__)}


def original(snapshot,fold=None):
    m=C.read_json(snapshot/'manifest.json')
    if C.digest({k:v for k,v in m.items() if k!='lineage'})!=R.HOME_LINEAGE or m['lineage']!=R.HOME_LINEAGE:
        raise ValueError('original lineage differs')
    if m['environment']!=C.SCIENTIFIC or m['source_sha256']!=C.SOURCES or C.sha(snapshot/'runner.py')!=m['runner_sha256'] or C.sha(C.__file__)!=m['runner_sha256']:
        raise ValueError('original source/environment differs')
    C.verify_sources(snapshot/'sources')
    if fold is not None:
        for file in ('features.parquet','metadata.npz'):
            if C.sha(snapshot/'inputs'/name(fold)/file)!=m['inputs'][name(fold)][file]:raise ValueError('original fold input changed')
    return m


def register(snapshot,bundle,registration):
    m=original(snapshot)
    for fold in range(1,7):original(snapshot,fold)
    _,ids,ref=R.snapshot(snapshot)
    bundle.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(snapshot/'fold_1_7/seed_00/prediction.npz',bundle/'qualification_reference.npz')
    spec=dict(protocol=PROTOCOL,registration_commit=registration,assignments=assignments(),threads=THREADS,
        home_lineage=m['lineage'],original_manifest_sha256=C.sha(snapshot/'manifest.json'),helpers=helpers(),
        inputs=m['inputs'],runtime_packages=m['runtime']['packages'],home_runtime=m['runtime'],columns=m['columns'],
        reference_sha256=C.sha(bundle/'qualification_reference.npz'),
        qualification=dict(hetzner='previous exact, c982056fbe7ac50cd347cd79c7310ac12e2acd8f',
                           laptop='previous exact, 55304792ee0bc0e39868c9138904723522f5fee9',mac='new full fold1 seed0 absolute max<=1e-6, rtol0'),
        home_drain='next completed fit after worker starts; preserve all completed and any follow-on partial',
        accepted_for_submission=False)
    validate_assignments(spec['assignments']);spec['lineage']=C.digest(spec)
    C.write_json(bundle/'manifest.json',spec)


def verify(bundle,registration):
    m=C.read_json(bundle/'manifest.json')
    if m['lineage']!=C.digest({k:v for k,v in m.items() if k!='lineage'}) or m['protocol']!=PROTOCOL or m['registration_commit']!=registration:
        raise ValueError('queue manifest drift')
    if m['assignments']!=assignments() or m['threads']!=THREADS or m['helpers']!=helpers():raise ValueError('assignment/source drift')
    validate_assignments(m['assignments'])
    if C.sha(bundle/'qualification_reference.npz')!=m['reference_sha256']:raise ValueError('qualification reference changed')
    return m


def metadata(snapshot,fold):
    with np.load(snapshot/'inputs'/name(fold)/'metadata.npz',allow_pickle=False) as z:return dict(z)


def prepare_pool(snapshot,bundle,registration,fold):
    m=verify(bundle,registration);original(snapshot,fold)
    dest=bundle/'pools'/name(fold);dest.mkdir(parents=True,exist_ok=False)
    clean=C.child_environment(snapshot/'inputs'/name(fold)/'features.parquet',dest)
    clean.update(OMP_NUM_THREADS='6',OMP_THREAD_LIMIT='6',CATBOOST_THREADS='6')
    os.environ.clear();os.environ.update(clean)
    if np.__version__!=m['runtime_packages']['numpy'] or pl.__version__!=m['runtime_packages']['polars']:raise ValueError('preparation library drift')
    import torch  # before LightGBM on Mac
    sys.path.insert(0,str(snapshot/'sources'));import train as T
    from target_encoding import add_encodings
    df=pl.read_parquet(snapshot/'inputs'/name(fold)/'features.parquet')
    train=df.filter(pl.col('split')=='train');rank=df.filter(pl.col('split')=='rank')
    labels=T.regime_label(train);train,rank=add_encodings(train,rank,labels==0,fit_labels=labels)
    cols=T.feature_columns(train);cats=[c for c in T.CATEGORICAL if c in cols]
    L.check_columns(cols,cats,m['columns']['features'],m['columns']['categoricals'])
    fit=train.filter(pl.Series(labels==0));base_f,base_p=T.proxy_base(fit,rank)
    target=L.residual(fit[T.TARGET].to_numpy(),base_f);meta=metadata(snapshot,fold)
    if T.REG_CLIP!=(0,7200) or not np.array_equal(rank['MVT_ID_mvt'].to_numpy(),meta['mvt_id']):raise ValueError('pool target/order drift')
    if not np.isfinite(target).all() or not np.isfinite(base_p).all():raise ValueError('nonfinite prepared inputs')
    for n,d in [('train',fit),('rank',rank)]:
        d.select(cols).write_parquet(dest/f'{n}.parquet')
        if not d.select(cols).equals(pl.read_parquet(dest/f'{n}.parquet')):raise ValueError('Parquet roundtrip differs')
    np.savez(dest/'arrays.npz',mvt_id=meta['mvt_id'],target=target,base_p=base_p,train_ids=fit['MVT_ID_mvt'].to_numpy())
    original(snapshot,fold)
    proof=dict(lineage=m['lineage'],fold=fold,source_input_sha256=m['inputs'][name(fold)]['features.parquet'],
        metadata_sha256=m['inputs'][name(fold)]['metadata.npz'],train_rows=fit.height,rank_rows=rank.height,
        files={p.name:C.sha(p) for p in dest.iterdir() if p.is_file()})
    C.write_json(dest/'manifest.json',proof);print(f'PREPARED {name(fold)}: {fit.height}/{rank.height}',flush=True)


def runtime_check(node,m):
    runtime=C.packages()
    if runtime['packages']!=m['runtime_packages']:raise ValueError('runtime package drift')
    if node=='home' and runtime!=m['home_runtime']:raise ValueError('Home runtime changed')
    if os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError('nice>=10 required')
    if node=='hetzner':R.resources()
    elif node=='laptop':L.limits()
    elif node=='home':
        cg=next(s[3:] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
        p=Path('/sys/fs/cgroup')/cg.lstrip('/');q,period=(p/'cpu.max').read_text().split()
        if q=='max' or int(q)>24*int(period) or int((p/'memory.max').read_text())>24*1024**3 or (p/'memory.swap.max').read_text().strip()!='0':raise ValueError('Home CPU/RAM/swap bounds differ')
    elif node=='mac' and sys.platform!='darwin':raise ValueError('Mac worker requires Darwin')
    return runtime


def fit_source(snapshot,attempt,node,seed,fold):
    original(snapshot,fold);attempt.mkdir(parents=True,exist_ok=False)
    env=C.child_environment(snapshot/'inputs'/name(fold)/'features.parquet',attempt)
    env.update(CATBOOST_THREADS=str(THREADS[node]),OMP_NUM_THREADS=str(THREADS[node]),OMP_THREAD_LIMIT=str(THREADS[node]))
    command=[sys.executable,str(snapshot/'sources/catboost_seeds.py'),'5600',str(seed)]
    if node=='mac':
        command=[sys.executable,'-c','import torch, runpy, sys; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")',*command[1:]]
    C.write_json(attempt/'invocation.json',dict(command=command,environment=env,node=node,seed=seed,fold=fold))
    start=time.monotonic()
    with (attempt/'fit.log').open('x') as log:
        try:r=subprocess.run(command,cwd=attempt,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=6*3600)
        except subprocess.TimeoutExpired:C.write_json(attempt/'exit.json',dict(timeout=True,seconds=time.monotonic()-start));raise
    C.write_json(attempt/'exit.json',dict(returncode=r.returncode,seconds=time.monotonic()-start))
    if r.returncode:raise ValueError(f'fit failed {node}/{fold}/{seed}')
    original(snapshot,fold)
    return attempt/f'raw_seed{seed}.npz'


def fit_pool(bundle,attempt,m,seed,fold):
    dest=bundle/'pools'/name(fold);p=C.read_json(dest/'manifest.json')
    seal=C.read_json(bundle/'pools_complete.json')
    def check():
        if seal['lineage']!=m['lineage'] or C.sha(dest/'manifest.json')!=seal['manifests'][name(fold)]:raise ValueError('pool seal differs')
        if p['lineage']!=m['lineage'] or p['fold']!=fold or p['source_input_sha256']!=m['inputs'][name(fold)]['features.parquet'] or p['metadata_sha256']!=m['inputs'][name(fold)]['metadata.npz']:raise ValueError('pool provenance differs')
        if set(p['files'])!={'train.parquet','rank.parquet','arrays.npz'}:raise ValueError('pool file inventory differs')
        for n,h in p['files'].items():
            if C.sha(dest/n)!=h:raise ValueError('pool bytes changed')
    check();attempt.mkdir(parents=True,exist_ok=False);os.chdir(attempt)
    from catboost import Pool,CatBoostRegressor
    with np.load(dest/'arrays.npz',allow_pickle=False) as z:a=dict(z)
    if len(a['target'])!=p['train_rows'] or len(a['mvt_id'])!=p['rank_rows']:raise ValueError('pool row count differs')
    def pool(n,y=None):
        df=pl.read_parquet(dest/f'{n}.parquet')
        if df.columns!=m['columns']['features']:raise ValueError('pool columns differ')
        pdf=df.to_pandas();del df
        for c in m['columns']['categoricals']:pdf[c]=pdf[c].fillna('NA').astype(str)
        result=Pool(pdf,y,cat_features=m['columns']['categoricals']);del pdf;gc.collect();return result
    parameters={**L.PARAMETERS,'random_seed':seed}
    model=CatBoostRegressor(**parameters);train_pool=pool('train',a['target']);model.fit(train_pool)
    del train_pool;gc.collect();reg=model.predict(pool('rank'))+a['base_p']
    path=attempt/f'raw_seed{seed}.npz';np.savez(path,mvt_id=a['mvt_id'],reg=reg);check()
    return path


def seal_pools(snapshot,bundle,registration):
    m=verify(bundle,registration);manifests={}
    for fold in range(1,7):
        original(snapshot,fold);dest=bundle/'pools'/name(fold);p=C.read_json(dest/'manifest.json')
        if p['lineage']!=m['lineage'] or p['fold']!=fold or p['source_input_sha256']!=m['inputs'][name(fold)]['features.parquet'] or p['metadata_sha256']!=m['inputs'][name(fold)]['metadata.npz']:raise ValueError('pool provenance differs')
        if set(p['files'])!={'train.parquet','rank.parquet','arrays.npz'}:raise ValueError('pool inventory differs')
        for n,h in p['files'].items():
            if C.sha(dest/n)!=h:raise ValueError('pool drift')
        with np.load(dest/'arrays.npz',allow_pickle=False) as z:
            if not np.array_equal(z['mvt_id'],metadata(snapshot,fold)['mvt_id']):raise ValueError('pool metadata differs')
        manifests[name(fold)]=C.sha(dest/'manifest.json')
    C.write_json(bundle/'pools_complete.json',dict(lineage=m['lineage'],manifests=manifests))


def qualify(node,snapshot,bundle,work,m,previous):
    with np.load(bundle/'qualification_reference.npz',allow_pickle=False) as z:ref=dict(z)
    if node=='mac':raw=fit_source(snapshot,work/'qualification','mac',0,1)
    else:
        limit=time.monotonic()+6*3600
        while not (previous/'report.json').exists():
            if time.monotonic()>limit:raise ValueError('previous qualification timeout')
            if node!='laptop':raise ValueError('server qualification is absent')
            state=subprocess.run(['systemctl','--user','is-active','--quiet','prc-cat-laptop-parity-20261005.service'])
            if state.returncode:raise ValueError('laptop qualification stopped without a report')
            time.sleep(20)
        old=C.read_json(previous/'report.json')
        raw=previous/('attempt/raw_seed0.npz' if node=='hetzner' else 'prediction.npz')
        expected='c982056fbe7ac50cd347cd79c7310ac12e2acd8f' if node=='hetzner' else '55304792ee0bc0e39868c9138904723522f5fee9'
        if old['registration_commit']!=expected or not old['exact'] or old['max_abs']!=0 or old['runtime']!=C.packages() or C.sha(raw)!=old['prediction_sha256']:raise ValueError('previous qualification failed or drifted')
        if node=='laptop':
            while subprocess.run(['systemctl','--user','is-active','--quiet','prc-cat-laptop-parity-20261005.service']).returncode==0:
                if time.monotonic()>limit:raise ValueError('previous qualification did not exit')
                time.sleep(5)
    with np.load(raw,allow_pickle=False) as z:r=qualification(ref['mvt_id'],ref['reg'],z['mvt_id'],z['reg'],node)
    C.write_json(work/'qualification.json',{**r,'prediction_sha256':C.sha(raw),'runtime':C.packages(),'lineage':m['lineage'],
        'previous_report_sha256':C.sha(previous/'report.json') if previous else None})
    if not r['passed']:raise ValueError(f'{node} backend qualification failed')


def drain_home(snapshot,work,m):
    def completed():return sorted(snapshot.glob('fold_*/seed_*/manifest.json'))
    start=completed();target=len(start)+1;deadline=time.monotonic()+2*3600
    C.write_json(work/'drain_start.json',dict(existing=len(start),target=target,mode='wait for next completed fit'))
    while subprocess.run(['systemctl','--user','is-active','--quiet',OLD_UNIT]).returncode==0 and len(completed())<target:
        if time.monotonic()>deadline:raise ValueError('Home drain timed out; existing service untouched')
        time.sleep(5)
    subprocess.run(['systemctl','--user','stop',OLD_UNIT],check=True,timeout=60)
    if subprocess.run(['systemctl','--user','is-active','--quiet',OLD_UNIT]).returncode==0:raise ValueError('Home original still active')
    mainpid=subprocess.check_output(['systemctl','--user','show',OLD_UNIT,'-p','MainPID','--value'],text=True).strip()
    if mainpid!='0':raise ValueError('Home original process remains')
    if any([int(p.parent.name[5:]),int(p.parent.parent.name.split('_')[1])] not in m['assignments']['home'] for p in completed()):raise ValueError('original completed work exceeds Home assignment')
    C.write_json(work/'drain_complete.json',dict(completed=len(completed()),original_unit=OLD_UNIT,partial_attempts_preserved=True))


def publish(raw,dest,m,node,seed,fold,ids,origin):
    require_job(m,node,seed,fold);C.prediction(raw,ids)
    if dest.exists():raise ValueError('refusing result overwrite')
    stage=dest.parent/('.publish-'+uuid.uuid4().hex);stage.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(raw,stage/'prediction.npz');C.prediction(stage/'prediction.npz',ids)
    C.write_json(stage/'receipt.json',dict(lineage=m['lineage'],node=node,seed=seed,fold=fold,rows=len(ids),
        sha256=C.sha(stage/'prediction.npz'),origin=origin,runtime=C.packages()))
    os.rename(stage,dest)


def check_result(dest,m,node,seed,fold,ids):
    require_job(m,node,seed,fold);r=C.read_json(dest/'receipt.json')
    if any(r[k]!=v for k,v in dict(lineage=m['lineage'],node=node,seed=seed,fold=fold,rows=len(ids)).items()) or C.sha(dest/'prediction.npz')!=r['sha256']:raise ValueError('result receipt drift')
    return C.prediction(dest/'prediction.npz',ids)


def run_node(snapshot,bundle,work,registration,node,previous):
    m=verify(bundle,registration);base=original(snapshot)
    if C.sha(snapshot/'manifest.json')!=m['original_manifest_sha256']:raise ValueError('original manifest differs')
    runtime=runtime_check(node,m);work.mkdir(parents=True,exist_ok=False)
    C.write_json(work/'invocation.json',dict(node=node,registration=registration,lineage=m['lineage'],runtime=runtime,tasks=m['assignments'][node]))
    if node=='home':drain_home(snapshot,work,m)
    else:qualify(node,snapshot,bundle,work,m,previous)
    if node=='laptop':
        seal_sha=C.sha(bundle/'pools_complete.json')
        C.write_json(work/'pool_seal.json',dict(sha256=seal_sha,lineage=m['lineage']))
    with C.writer_lock(work/'writer'):
        for index,(seed,fold) in enumerate(m['assignments'][node]):
            ids=metadata(snapshot,fold)['mvt_id'];dest=work/name(fold)/f'seed_{seed:02}'
            old=snapshot/name(fold)/f'seed_{seed:02}'
            if node=='home' and old.exists():
                C.verify_seed(old,ids,dict(seed=seed,fold=fold,lineage=R.HOME_LINEAGE))
                publish(old/'prediction.npz',dest,m,node,seed,fold,ids,dict(kind='imported_original',manifest_sha256=C.sha(old/'manifest.json')))
                print(f'IMPORTED {node} {name(fold)} seed={seed}',flush=True);continue
            print(f'START {node} {index+1}/{len(m["assignments"][node])} {name(fold)} seed={seed}',flush=True)
            attempt=work/'attempts'/f'{name(fold)}-s{seed:02}';started=time.monotonic()
            if node=='laptop':
                if C.sha(bundle/'pools_complete.json')!=seal_sha:raise ValueError('pool seal changed during queue')
                env=C.child_environment('__prepared_pools__',attempt);env.update(CATBOOST_THREADS='6',OMP_NUM_THREADS='6',OMP_THREAD_LIMIT='6')
                with (work/f'{name(fold)}-s{seed:02}.log').open('x') as log:
                    subprocess.run([sys.executable,str(Path(__file__).resolve()),'pool-fit','--bundle',str(bundle),'--work',str(attempt),
                        '--registration-commit',registration,'--seed',str(seed),'--fold',str(fold)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=6*3600)
                raw=attempt/f'raw_seed{seed}.npz'
            else:raw=fit_source(snapshot,attempt,node,seed,fold)
            verify(bundle,registration);original(snapshot,fold)
            origin=dict(kind='new_fit',seconds=time.monotonic()-started,
                qualification_sha256=C.sha(work/'qualification.json') if node!='home' else C.sha(work/'drain_complete.json'),
                input_sha256=m['inputs'][name(fold)]['features.parquet'])
            if node=='laptop':origin['pool_manifest_sha256']=C.sha(bundle/'pools'/name(fold)/'manifest.json')
            publish(raw,dest,m,node,seed,fold,ids,origin)
            print(f'COMPLETE {node} {index+1}/{len(m["assignments"][node])} {name(fold)} seed={seed}',flush=True)
    C.write_json(work/'complete.json',dict(lineage=m['lineage'],node=node,tasks=m['assignments'][node]))


def merge(snapshot,bundle,work,registration):
    m=verify(bundle,registration);final=work/'merged'
    if final.exists():raise ValueError('merged output exists')
    merged=work/('.merge-'+uuid.uuid4().hex);merged.mkdir(exist_ok=False);runtimes={};qualifications={}
    for n in NODES:
        r=C.read_json(work/n/'complete.json')
        if r!={'lineage':m['lineage'],'node':n,'tasks':m['assignments'][n]}:raise ValueError('node incomplete')
        inv=C.read_json(work/n/'invocation.json');runtimes[n]=inv['runtime']
        if inv['lineage']!=m['lineage'] or inv['registration']!=registration or inv['tasks']!=m['assignments'][n] or runtimes[n]['packages']!=m['runtime_packages']:raise ValueError('node invocation drift')
        if n!='home':
            q=C.read_json(work/n/'qualification.json');qualifications[n]=q
            if q['lineage']!=m['lineage'] or not q['passed'] or q['runtime']!=runtimes[n] or q['rows']!=344419 or q['absolute_tolerance']!=(1e-6 if n=='mac' else 0.) or q['max_abs']>q['absolute_tolerance']:raise ValueError('node qualification differs')
        elif runtimes[n]!=m['home_runtime']:raise ValueError('Home runtime drift')
    for fold in range(1,7):
        original(snapshot,fold);meta=metadata(snapshot,fold);pred=[];receipts=[]
        for seed in range(15):
            node=next(n for n in NODES if [seed,fold] in m['assignments'][n]);d=work/node/name(fold)/f'seed_{seed:02}'
            receipt=C.read_json(d/'receipt.json')
            if receipt['runtime']!=runtimes[node]:raise ValueError('receipt runtime differs')
            pred.append(check_result(d,m,node,seed,fold,meta['mvt_id']));receipts.append(C.sha(d/'receipt.json'))
        stack=np.stack(pred);np.savez(merged/f'{name(fold)}.npz',**meta,per_seed=stack,reg2=stack.mean(axis=0))
        C.write_json(merged/f'{name(fold)}.json',dict(lineage=m['lineage'],receipt_sha256=receipts,seeds=list(range(15))))
    C.write_json(merged/'complete.json',dict(lineage=m['lineage'],fits=90,folds=6,runtimes=runtimes,qualifications=qualifications,accepted_for_submission=False))
    os.rename(merged,final)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['register','prepare-pool','seal-pools','run','pool-fit','merge'])
    p.add_argument('--snapshot',type=Path);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--work',type=Path)
    p.add_argument('--node',choices=NODES);p.add_argument('--previous',type=Path);p.add_argument('--fold',type=int,choices=range(1,7));p.add_argument('--seed',type=int,choices=range(15))
    p.add_argument('--registration-commit',required=True);a=p.parse_args()
    if not re.fullmatch('[a-f0-9]{40}',a.registration_commit):p.error('full registration SHA required')
    snap=a.snapshot.resolve() if a.snapshot else None;b=a.bundle.resolve();w=a.work.resolve() if a.work else None
    if a.action=='register':register(snap,b,a.registration_commit)
    elif a.action=='prepare-pool':prepare_pool(snap,b,a.registration_commit,a.fold)
    elif a.action=='seal-pools':seal_pools(snap,b,a.registration_commit)
    elif a.action=='pool-fit':
        m=verify(b,a.registration_commit);require_job(m,'laptop',a.seed,a.fold);fit_pool(b,w,m,a.seed,a.fold)
    elif a.action=='run':run_node(snap,b,w,a.registration_commit,a.node,a.previous.resolve() if a.previous else None)
    else:merge(snap,b,w,a.registration_commit)


if __name__=='__main__':main()

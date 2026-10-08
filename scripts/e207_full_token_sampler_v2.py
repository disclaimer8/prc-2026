"""E207: full K64 actual observations, unchanged E197 ordered model.

Scientific worker only. All real-data actions require a registered immutable
bundle. No ranking, event sampler, new features, or automatic retries.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types

import numpy as np
import e194_trajectory_sequence as R
import e197_trajectory_encoder as S
import e202_trajectory_extension as Q
import night_gpu_20261007 as G

E,D,M,C=S.E,S.D,S.M,S.C
PROTOCOL='E207-full-K64-retain-original-raw-index-fill-sixpair-3seed8epoch-v2'
SELF='scripts/e207_full_token_sampler_v2.py'
QUEUE='scripts/e207_full_token_queue_v2_20261008.py'
DOC='analysis/e207_full_token_v2_20261008/PROTOCOL.md'
TEST='tests/test_e207_full_token_sampler_v2.py'
INPUT='data/e207_v2_inputs_20261008'
RAW='data/e207_v2_raw_20261008'
MATERIALIZED='data/e207_v2_materialized_20261008.json'
PREPARED='data/e207_v2_prepared_20261008'
INCOMING='data/e207_v2_incoming_20261008'
OUTPUT='data/e207_v2_trajectory_20261008'
SMOKE='data/e207_v2_smoke_20261008.json'
FOLDS=(1,2,3,4,5,6)
NODE_FOLDS={171:(1,2,5),175:(3,4,6)}
KEYS=('body_sse','decided_sse','decided_trim3_sse','decided_trim20_sse')
CONFIG=dict(folds=list(FOLDS),seeds=[0,1,2],epochs=8,arm='ordered',
    sampler='retain-old-plus-rint-linspace-unselected-raw-indices',
    max_fits=18,screen=False,ranking_authorized=False,model=S.CONFIG,
    prior_body_delta_seconds=0,expert_range_seconds=[-1,1],
    max_gpu_bytes=20*1024**3,deadline_seconds=21600)
FILES=tuple(dict.fromkeys([Path(SELF).name,Path(QUEUE).name,
    'e194_trajectory_sequence.py','e202_trajectory_extension.py','night_gpu_20261007.py',*S.FILES]))
network,fit,normalize,norm_stats=S.network,S.fit,S.normalize,S.norm_stats


def sample_full_indices(t):
    t=np.asarray(t)
    if t.ndim!=1 or not np.isfinite(t).all() or (np.diff(t)<0).any():
        raise ValueError('timestamps must be finite and sorted')
    old=R.sample_time_indices(t)
    need=min(64,len(t))-len(old)
    if not need:return old
    available=np.ones(len(t),bool);available[old]=False
    remaining=np.flatnonzero(available)
    extra=remaining[np.rint(np.linspace(0,len(remaining)-1,need)).astype(np.int64)]
    result=np.sort(np.concatenate([old,extra]))
    if len(np.unique(result))!=min(64,len(t)) or not np.isin(old,result).all():
        raise ValueError('full sampler cardinality/superset failed')
    return result


# New globals belong to this function; the frozen E194 module is never mutated.
encode_day=types.FunctionType(R.encode_day.__code__,
    {**R.encode_day.__globals__,'sample_time_indices':sample_full_indices},'encode_day')


def validate_arrays(old,new):
    for key in ('ids','raw_ranges','raw_counts','status','identity','identity_candidates','airport','takeoff'):
        if not np.array_equal(old[key],new[key]):raise ValueError('metadata changed: '+key)
    p=new['present'];v=new['values'];k=new['known'];t=new['point_times'];sr=new['source_rows']
    if (not np.isfinite(v).all() or (p>1).any() or (k>p[...,None]).any()
        or not np.array_equal(p.sum(2),np.minimum(64,new['raw_counts']))
        or (t>new['takeoff'][:,None,None]).any()
        or not np.isfinite(t[p>0]).all() or (sr[p>0]<0).any()
        or (v[k==0]!=0).any() or not np.isnan(t[p==0]).all() or (sr[p==0]!=-1).any()):
        raise ValueError('token count, cutoff or PAD contract failed')
    added=p.astype(bool).copy()
    for i in range(len(p)):
        for stream in range(2):
            n=int(p[i,stream].sum());m=int(old['present'][i,stream].sum())
            if not (p[i,stream,:n]==1).all() or p[i,stream,n:].any():raise ValueError('non-prefix PAD')
            current=sr[i,stream,:n];before=old['source_rows'][i,stream,:m]
            if len(np.unique(current))!=n or (np.diff(t[i,stream,:n])<0).any():raise ValueError('duplicate raw index/order')
            lookup={int(x):j for j,x in enumerate(current)}
            if any(int(x) not in lookup for x in before):raise ValueError('original observation dropped')
            ix=[lookup[int(x)] for x in before]
            added[i,stream,ix]=False
            for key in ('values','known','point_times'):
                if not np.array_equal(new[key][i,stream,ix],old[key][i,stream,:m],equal_nan=True):
                    raise ValueError('retained observation changed: '+key)
            if new['raw_counts'][i,stream]<=64:
                for key in ('values','known','present','source_rows','point_times'):
                    if not np.array_equal(old[key][i,stream],new[key][i,stream],equal_nan=True):
                        raise ValueError('small stream changed: '+key)
    return added


def gate(reports):
    if len(reports)!=6 or {r['fold'] for r in reports}!=set(FOLDS):raise ValueError('exact six unique pairs required')
    if any(not np.isfinite([r['metrics'][k] for k in KEYS]).all() for r in reports):raise ValueError('nonfinite metrics')
    return sum(r['metrics']['decided_sse']<0 for r in reports)>=4 and all(sum(r['metrics'][k] for r in reports)<0 for k in KEYS)


def registration(root,reg):
    if not re.fullmatch('[0-9a-f]{40}',reg or ''):raise ValueError('full registration SHA required')
    for name in (SELF,QUEUE,DOC,TEST):
        raw=subprocess.check_output(['git','show',reg+':'+name],cwd=root)
        if hashlib.sha256(raw).hexdigest()!=C.sha(root/name):raise ValueError('registered source differs: '+name)
    if PROTOCOL.encode() not in subprocess.check_output(['git','show',reg+':HYPOTHESES_LOG.md'],cwd=root):
        raise ValueError('journal protocol missing')


def pack(root,reg):
    registration(root,reg);out=root/INPUT
    if out.exists():raise ValueError('fresh bundle required')
    out.parent.mkdir(exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix='.e207-input-',dir=out.parent));(tmp/'scripts').mkdir()
    for name in FILES:shutil.copy2(root/'scripts'/name,tmp/'scripts'/name)
    shutil.copy2(root/DOC,tmp/'protocol.md');folds=[];runtime=None
    for fold in FOLDS:
        parent='e197' if fold in (1,4) else 'e202';name=f'fold_{fold}_{fold+6}'
        bundle=root/f'data/{parent}_inputs_20261007';original=root/f'data/{parent}_trajectory_20261007'
        m=C.read_json(bundle/'manifest.json');s=m['spec'];complete=C.read_json(original/'complete.json')
        expected=Q.OLD_SPEC if parent=='e197' else '6d604eaaca830fa83c18ebb78986b028d338869123d5f8c10b2450dde1ec4309'
        expected_reg=Q.OLD_REG if parent=='e197' else '55fe9b68f7cc8405681f6c85d576979e067b55c3'
        if (m['spec_sha256']!=expected or C.digest(s)!=expected or complete['input_spec_sha256']!=expected
            or s['registration_commit']!=expected_reg or complete['registration_commit']!=expected_reg):
            raise ValueError('saved control lineage differs')
        if runtime is not None and runtime!=complete['runtime']:raise ValueError('saved runtime differs')
        runtime=complete['runtime'];fs=next(x for x in s['folds'] if x['fold']==fold)
        Q.checked_arm(root,bundle/name,root/fs['query_bundle'],original/name/'ordered','ordered',complete)
        if M.verify_bundle(root/fs['query_bundle'],E.OLD_REGISTRATION)['spec_sha256']!=fs['query_spec_sha256']:
            raise ValueError('saved query lineage differs')
        for script,h in s['scripts'].items():
            if C.sha(root/'scripts'/script)!=h:raise ValueError('frozen source differs '+script)
        fd=tmp/name;fd.mkdir()
        for f in ('train_indices.npy','pred_indices.npy','control.npz','evaluation.npz'):
            if C.sha(bundle/name/f)!=s['files'][name+'/'+f]:raise ValueError('saved metadata changed')
            shutil.copy2(bundle/name/f,fd/f)
        old_manifest=C.read_json(original/name/'ordered/manifest.json')
        if old_manifest not in complete['arms'] or old_manifest['seeds']!=[0,1,2]:raise ValueError('ordered receipt differs')
        for f,h in old_manifest['files'].items():
            if C.sha(original/name/'ordered'/f)!=h:raise ValueError('saved ordered artifact changed')
        shutil.copy2(original/name/'ordered/prediction.npz',fd/'ordered.npz')
        C.write_json(fd/'ordered_manifest.json',old_manifest)
        if fold==1:shutil.copy2(bundle/'query_ids.npy',tmp/'query_ids.npy')
        elif C.sha(bundle/'query_ids.npy')!=C.sha(tmp/'query_ids.npy'):raise ValueError('query ID order differs')
        folds.append(dict(fold=fold,query_bundle=fs['query_bundle'],query_spec_sha256=fs['query_spec_sha256'],
            parameters=old_manifest['fits'][0]['parameters']))
    days=[]
    for shard in (0,1):
        folder=root/f'data/e194_inputs_20261007_s{shard}';raw=R.read(folder/'manifest.json')
        oldroot=root/f'data/e194_sequence_20261007_s{shard}';old=R.read(oldroot/'complete.json')
        if raw['registration_commit']!=S.RAW_REG or old['registration_commit']!=S.RAW_REG:raise ValueError('raw lineage differs')
        for r in old['reports']:
            day=r['day']
            if not day.startswith('2025-'):continue
            dep=f'departures/{day}.parquet';surf=f'surface/{day}.parquet'
            days.append(dict(day=day,shard=shard,rows=r['rows'],
                dep=str((folder/dep).relative_to(root)),dep_sha256=raw['files'][dep],
                surface=str((folder/surf).relative_to(root)) if surf in raw['files'] else None,
                surface_sha256=raw['files'].get(surf),old=str((oldroot/day/'sequence.npz').relative_to(root)),old_sha256=r['npz_sha256']))
    days.sort(key=lambda d:d['day'])
    if len(days)!=365 or sum(d['rows'] for d in days)!=2085047:raise ValueError('raw day population differs')
    spec=dict(protocol=PROTOCOL,registration_commit=reg,configuration=CONFIG,runtime=runtime,folds=folds,days=days,
        scripts={n:C.sha(tmp/'scripts'/n) for n in FILES},files={str(p.relative_to(tmp)):C.sha(p) for p in tmp.rglob('*') if p.is_file()})
    C.write_json(tmp/'manifest.json',dict(spec=spec,spec_sha256=C.digest(spec)));registration(root,reg);tmp.rename(out)
    print('PACKED',C.digest(spec),flush=True)


def verify(root,reg,queries=False):
    b=root/INPUT;m=C.read_json(b/'manifest.json');s=m['spec']
    if s['protocol']!=PROTOCOL or s['registration_commit']!=reg or s['configuration']!=CONFIG or C.digest(s)!=m['spec_sha256']:
        raise ValueError('registered input specification differs')
    for f,h in s['scripts'].items():
        if C.sha(Path(__file__).parent/f)!=h:raise ValueError('executing helper changed: '+f)
    for f,h in s['files'].items():
        if C.sha(b/f)!=h:raise ValueError('input changed: '+f)
    if queries:
        for fs in s['folds']:
            if M.verify_bundle(root/fs['query_bundle'],E.OLD_REGISTRATION)['spec_sha256']!=fs['query_spec_sha256']:
                raise ValueError('query tensor lineage differs')
    return m


def extract_day(task):
    root,record=task;root=Path(root);day=record['day'];dest=root/RAW/f's{record["shard"]}'/day
    if dest.exists():raise ValueError('raw day exists; no retry')
    for key in ('dep','surface','old'):
        if record[key] and C.sha(root/record[key])!=record[key+'_sha256']:raise ValueError('raw input changed: '+key)
    points=R.pl.read_parquet(root/record['surface']) if record['surface'] else None
    deps=R.pl.read_parquet(root/record['dep'],columns=R.DEP_COLS)
    new,stats=encode_day(points,deps,day)
    # NpzFile is a lazy archive: repeated indexing decompresses whole members.
    # Materialize each member exactly once before row-wise invariant checks.
    with np.load(root/record['old'],allow_pickle=False) as archive:
        old={key:archive[key] for key in archive.files}
    added=validate_arrays(old,new);stats['original_selected_points']=int(old['present'].sum())
    stats.update(added_points=int(added.sum()),
        added_duplicate_timestamp=int((added&(new['values'][...,11]==1)).sum()),
        added_stale=int((added&(new['known'][...,9]==1)&(new['values'][...,9]==1)).sum()),
        added_unknown_stale=int((added&(new['known'][...,9]==0)).sum()))
    for ap,entry in stats['by_airport'].items():
        mask=new['airport']==ap;entry.update(selected_points=int(new['present'][mask].sum()),added_points=int(added[mask].sum()))
    dest.mkdir(parents=True);np.savez_compressed(dest/'sequence.npz',**new)
    receipt=dict(day=day,rows=len(new['ids']),sha256=C.sha(dest/'sequence.npz'),stats=stats)
    C.write_json(dest/'receipt.json',receipt);print('EXTRACTED',day,stats['selected_points'],flush=True)
    return receipt


def extract(root,reg,shard):
    m=verify(root,reg);guard=S.prep_cgroup();out=root/RAW/f's{shard}'
    if out.exists():raise ValueError('fresh raw shard required')
    out.mkdir(parents=True);days=[d for d in m['spec']['days'] if d['shard']==shard]
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        reports=list(pool.map(extract_day,[(str(root),d) for d in days]))
    C.write_json(out/'complete.json',dict(protocol=PROTOCOL,registration_commit=reg,input_spec_sha256=m['spec_sha256'],
        shard=shard,reports=reports,cgroup=guard,label_free=True))


def derived_raw(root,m,reg):
    days=[];complete={}
    for shard in (0,1):
        p=root/RAW/f's{shard}/complete.json';r=C.read_json(p)
        if r['protocol']!=PROTOCOL or r['registration_commit']!=reg or r['input_spec_sha256']!=m['spec_sha256'] or r['shard']!=shard:
            raise ValueError('extraction receipt parent differs')
        expected={d['day']:d['rows'] for d in m['spec']['days'] if d['shard']==shard}
        if len(r['reports'])!=len(expected) or {d['day']:d['rows'] for d in r['reports']}!=expected:raise ValueError('incomplete raw shard')
        complete[str(p.relative_to(root))]=C.sha(p)
        for d in r['reports']:
            days.append(dict(day=d['day'],path=f'{RAW}/s{shard}/{d["day"]}/sequence.npz',sha256=d['sha256'],rows=d['rows']))
    return dict(raw_days=sorted(days,key=lambda d:d['day']),raw_complete_sha256=complete)


def materialize(root,reg):
    m=verify(root,reg)
    spec={**m['spec'],'parent_spec_sha256':m['spec_sha256'],**derived_raw(root,m,reg)}
    C.write_json(root/MATERIALIZED,dict(spec=spec,spec_sha256=C.digest(spec)))


def verify_bundle(root,bundle,reg,queries=False):
    parent=verify(root,reg,queries);m=C.read_json(root/MATERIALIZED);s=m['spec']
    if s['parent_spec_sha256']!=parent['spec_sha256'] or C.digest(s)!=m['spec_sha256']:raise ValueError('materialized parent changed')
    expected={**parent['spec'],'parent_spec_sha256':parent['spec_sha256'],**derived_raw(root,parent,reg)}
    if s!=expected:raise ValueError('materialized configuration drift')
    for name,h in s['raw_complete_sha256'].items():
        if C.sha(root/name)!=h:raise ValueError('raw complete changed')
    expected_days={d['day']:d['rows'] for d in parent['spec']['days']}
    if len(s['raw_days'])!=365 or {d['day']:d['rows'] for d in s['raw_days']}!=expected_days:raise ValueError('materialized population differs')
    return m


# Execute the exact frozen preparation code with only provenance/paths/folds
# rebound. Scientific normalization and writing rules remain the same code.
_prep_globals={**S.prepare_fold.__globals__,'__file__':__file__,'PROTOCOL':PROTOCOL,
    'FOLDS':FOLDS,'verify_bundle':verify_bundle}
prepare_fold=types.FunctionType(S.prepare_fold.__code__,_prep_globals,'prepare_fold')
verify_prepared=types.FunctionType(S.verify_prepared.__code__,_prep_globals,'verify_prepared')


def prepare_node(root,reg,node):
    for fold in NODE_FOLDS[node]:prepare_fold(root,root/INPUT,root/PREPARED/f'fold_{fold}_{fold+6}',reg,fold)
    C.write_json(root/f'data/e207_v2_prep_{node}_complete.json',dict(protocol=PROTOCOL,registration_commit=reg,folds=list(NODE_FOLDS[node])))


def admit(root,reg,fold):
    m=verify_bundle(root,root/INPUT,reg);src=root/INCOMING/f'fold_{fold}_{fold+6}';dest=root/PREPARED/src.name
    if fold not in FOLDS or dest.exists():raise ValueError('fresh valid admission required')
    r=verify_prepared(root/INPUT,src,m,fold);norm=C.read_json(src/'normalization.json')
    if norm['heldout_months']!=[fold,fold+6] or not np.isfinite(norm['mean']).all() or not np.isfinite(norm['std']).all() or min(norm['std'])<=0:
        raise ValueError('normalization invalid')
    shape=(2085047,2,64);arrays={k:np.load(src/(k+'.npy'),mmap_mode='r') for k in ('values','known','present','support')}
    for key,dims,dtype in [('values',shape+(12,),'float16'),('known',shape+(12,),'uint8'),('present',shape,'uint8'),('support',shape[:2],'uint8')]:
        if arrays[key].shape!=dims or arrays[key].dtype!=np.dtype(dtype):raise ValueError('tensor layout differs')
    counts=np.zeros(4,np.int64)
    for lo in range(0,2085047,4096):
        v,k,p,s=[np.asarray(arrays[n][lo:lo+4096]) for n in ('values','known','present','support')]
        if not np.isfinite(v).all() or (k>p[...,None]).any() or (p>1).any() or (s>1).any() or (v[k==0]!=0).any() or (np.abs(v[...,:6])>8).any():raise ValueError('admission numeric masks failed')
        if not np.array_equal(s,(p.sum(2)>0).astype(np.uint8)):raise ValueError('support differs')
        counts+=np.array([p.sum(),s[:,0].sum(),s[:,1].sum(),(s[:,0]&s[:,1]).sum()],np.int64)
        for a in arrays.values():S.release_pages(a)
    expected_points=sum(d['stats']['selected_points'] for shard in (0,1) for d in C.read_json(root/RAW/f's{shard}/complete.json')['reports'])
    if counts.tolist()!=[expected_points,1251417,1345381,1022407]:raise ValueError('coverage differs')
    C.write_json(src/'admission.json',dict(protocol=PROTOCOL,registration_commit=reg,input_spec_sha256=m['spec_sha256'],fold=fold,
        passed=True,prepared_complete_sha256=C.sha(src/'complete.json'),counts=counts.tolist()))
    del arrays;gc.collect();dest.parent.mkdir(parents=True,exist_ok=True);src.rename(dest);print('ADMITTED',fold,flush=True)


def smoke(root,reg):
    m=verify(root,reg);M.verify_cgroup()
    if (root/SMOKE).exists():raise ValueError('fresh smoke required')
    import torch
    torch.cuda.set_per_process_memory_fraction(CONFIG['max_gpu_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
    # Frozen full-shape synthetic test reserves all 9.875GB trajectory bytes,
    # checks both equal-capacity encoders, PAD invariance and backward pass.
    S.smoke(root,root/SMOKE);r=C.read_json(root/SMOKE)
    if not r['passed'] or r['runtime']!=m['spec']['runtime'] or any(c['peak_gpu_bytes']>CONFIG['max_gpu_bytes'] for c in r['checks']):
        raise ValueError('full-shape smoke/runtime/memory failed')


def run(root,reg):
    S.fixed_env();import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);guard=M.verify_cgroup()
    torch.cuda.set_per_process_memory_fraction(CONFIG['max_gpu_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
    m=verify_bundle(root,root/INPUT,reg,queries=True);sm=C.read_json(root/SMOKE)
    if not sm['passed'] or sm['worker_sha256']!=C.sha(S.__file__) or sm['runtime']!=E.package_runtime() or sm['runtime']!=m['spec']['runtime']:
        raise ValueError('qualified smoke/runtime mismatch')
    out=root/OUTPUT
    if out.exists():raise ValueError('fresh GPU output required; no resume')
    out.mkdir();receipts=[];preps=[];deadline=time.monotonic()+CONFIG['deadline_seconds']
    for fs in m['spec']['folds']:
        fold=fs['fold'];name=f'fold_{fold}_{fold+6}';prep=root/PREPARED/name;fd=root/INPUT/name
        while not (prep/'admission.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('6h GPU wait deadline')
            print('WAIT PREPARED',fold,flush=True);time.sleep(15)
        ad=C.read_json(prep/'admission.json')
        if ad['registration_commit']!=reg or ad['input_spec_sha256']!=m['spec_sha256'] or ad['fold']!=fold or not ad['passed'] or ad['prepared_complete_sha256']!=C.sha(prep/'complete.json'):
            raise ValueError('admission proof differs')
        preps.append(verify_prepared(root/INPUT,prep,m,fold));qb=root/fs['query_bundle'];spec=M.verify_bundle(qb,E.OLD_REGISTRATION)['spec'];arrays,_=M.load_arrays(qb,spec)
        tr=np.load(fd/'train_indices.npy');pr=np.load(fd/'pred_indices.npy')
        if len(tr)!=len(arrays['xnf']) or len(pr)!=len(arrays['xnp']):raise ValueError('query/context shape differs')
        cont,scalar,bins=E.numeric_bins(arrays['xnf']);N=E.trainer(qb/'src');tables=S.gpu_tables(prep)
        od=out/name/'ordered';od.mkdir(parents=True);np.savez(od/'bins.npz',continuous=cont,scalar=scalar,**{f'bin_{i}':b for i,b in enumerate(bins)})
        fits=[]
        for seed in (0,1,2):
            if time.monotonic()>deadline:raise TimeoutError('6h GPU fit deadline')
            f=fit(arrays,spec,N,cont,scalar,bins,tables,tr,pr,od,'ordered',seed)
            if f['parameters']!=fs['parameters'] or len(f['losses'])!=8 or f['peak_gpu_bytes']>CONFIG['max_gpu_bytes']:raise ValueError('fit capacity/epochs/memory differs')
            fits.append(f)
        member=(np.mean([np.load(od/f'seed_{i}.npy') for i in (0,1,2)],axis=0)*spec['sd']+arrays['base_p'])[arrays['control_indices']]
        ctrl=E.read(fd/'control.npz');cand=E.assemble(ctrl,ctrl['reg']+ctrl['mlp_coefficient']*(member-ctrl['plain']),ctrl['lobt'])
        np.savez_compressed(od/'prediction.npz',**{k:cand[k] for k in ('mvt_id','y','month','pred','reg','values','proba')},member=member)
        r=dict(fold=fold,arm='ordered',seeds=[0,1,2],fits=fits,files={p.name:C.sha(p) for p in od.iterdir() if p.is_file()})
        C.write_json(od/'manifest.json',r);receipts.append(r);print('COMPLETE FOLD',fold,flush=True)
        del arrays,tables;gc.collect();torch.cuda.empty_cache()
    C.write_json(out/'complete.json',dict(protocol=PROTOCOL,registration_commit=reg,input_spec_sha256=m['spec_sha256'],runtime=E.package_runtime(),
        cgroup=guard,arms=receipts,prepared_receipts=preps,accepted_for_submission=False))
    evaluate(root,reg)


def evaluate(root,reg):
    m=verify_bundle(root,root/INPUT,reg);out=root/OUTPUT;c=C.read_json(out/'complete.json')
    if c['protocol']!=PROTOCOL or c['registration_commit']!=reg or c['input_spec_sha256']!=m['spec_sha256'] or len(c['arms'])!=6:
        raise ValueError('incomplete GPU output')
    reports=[];ids=[]
    for fs in m['spec']['folds']:
        fold=fs['fold'];name=f'fold_{fold}_{fold+6}';fd=root/INPUT/name;meta=E.read(fd/'evaluation.npz')
        cand=Q.checked_arm(root,fd,root/fs['query_bundle'],out/name/'ordered','ordered',c);old=E.read(fd/'ordered.npz')
        G.check_prediction(old,meta);G.check_prediction(cand,meta)
        if any(f['parameters']!=fs['parameters'] for r in c['arms'] if r['fold']==fold for f in r['fits']):raise ValueError('capacity changed')
        pr=np.load(fd/'pred_indices.npy');ix=np.load(root/fs['query_bundle']/'control_indices.npy')
        support=np.load(root/PREPARED/name/'support.npy',mmap_mode='r');sup=np.asarray(support[pr[ix]])
        masks={'own':sup[:,0]>0,'previous_only':(sup[:,0]==0)&(sup[:,1]>0),'no_points':sup.sum(1)==0}
        masks.update({f'airport_{ap}':meta['airport']==ap for ap in np.unique(meta['airport'])})
        masks.update({f'month_{mo}':meta['month']==mo for mo in np.unique(meta['month'])})
        delta=(cand['pred']-meta['y'])**2-(old['pred']-meta['y'])**2
        diagnostics={label:dict(rows=int(mask.sum()),all_sse=float(delta[mask].sum()),
            body_sse=float(delta[mask&(meta['y']<=7200)].sum()),decided_sse=float(delta[mask&meta['decided']].sum())) for label,mask in masks.items()}
        reports.append(dict(fold=fold,metrics=G.measured(old['pred'],cand['pred'],meta),diagnostics=diagnostics));ids.append(meta['mvt_id'])
        S.release_pages(support);del support,sup
    ids=np.concatenate(ids)
    if len(ids)!=2085047 or len(np.unique(ids))!=len(ids):raise ValueError('six-pair population differs')
    result=dict(protocol=PROTOCOL,registration_commit=reg,reports=reports,
        pooled={k:float(sum(r['metrics'][k] for r in reports)) for k in KEYS},
        decided_pairs_improved=sum(r['metrics']['decided_sse']<0 for r in reports),six_pair_gate_pass=gate(reports),
        accepted_for_submission=False,limitation='Historical selected upstream; LGB5/preADSB component only. Current E203 chain and ranking forbidden without separate protocol.')
    C.write_json(out/'evaluation.json',result);print(json.dumps(result),flush=True)


def metadata(root):
    """Only counts from existing E194 arrays; no extraction or target access."""
    groups={};receipts={};paths=sorted(root.glob('data/e194_sequence_20261007_s*/????-??-??/sequence.npz'))
    if len(paths)!=427:raise ValueError('exact existing427days required')
    for path in paths:
        day=path.parent.name;receipts[str(path.with_name('receipt.json').relative_to(root))]=C.sha(path.with_name('receipt.json'))
        with np.load(path,allow_pickle=False) as z:
            raw=z['raw_counts'];present=z['present'].sum(2);aps=z['airport']
            for ap in np.unique(aps):
                mask=aps==ap;key=day[:7]+'/'+str(ap)
                r=groups.setdefault(key,dict(rows=0,own_present=0,previous_present=0,old_points=0,full_points=0,added_points=0,compressed_own=0,compressed_previous=0))
                r['rows']+=int(mask.sum());r['own_present']+=int((raw[mask,0]>0).sum());r['previous_present']+=int((raw[mask,1]>0).sum())
                r['old_points']+=int(present[mask].sum());r['full_points']+=int(np.minimum(64,raw[mask]).sum());r['added_points']=r['full_points']-r['old_points']
                r['compressed_own']+=int((raw[mask,0]>64).sum());r['compressed_previous']+=int((raw[mask,1]>64).sum())
    C.write_json(root/'analysis/e207_full_token_v2_20261008/metadata.json',dict(label_free=True,days=427,groups=groups,receipt_sha256=receipts,
        limitation='Full slot counts from min64raw_count, not new token attributes or independent observations; no labels/OOF read.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['pack','verify','extract','materialize','prepare-node','admit','smoke','run','evaluate','gpu-queue','metadata'])
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--registration-commit')
    p.add_argument('--shard',type=int,choices=[0,1]);p.add_argument('--node',type=int,choices=[171,175]);p.add_argument('--fold',type=int,choices=FOLDS)
    a=p.parse_args();root=a.root.resolve();reg=a.registration_commit
    if a.action=='metadata':metadata(root)
    elif not re.fullmatch('[0-9a-f]{40}',reg or ''):raise ValueError('registered commit required')
    elif a.action=='extract':extract(root,reg,a.shard)
    elif a.action=='gpu-queue':
        for action in ('smoke','run'):
            subprocess.run([sys.executable,'-u',__file__,action,'--root',str(root),'--registration-commit',reg],check=True,timeout=21600)
    elif a.action=='prepare-node':prepare_node(root,reg,a.node)
    elif a.action=='admit':admit(root,reg,a.fold)
    else:globals()[a.action](root,reg)


if __name__=='__main__':main()

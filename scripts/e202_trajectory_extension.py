"""Fixed remaining-four-pair extension of E197, with per-fold prepared admission.

Uses immutable E197 normalization helpers, network, GPU tables and fitter.
No fitting/architecture changes; only provenance, streaming orchestration and
six-pair reporting are new. Ranking is outside this protocol.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import numpy as np
import e197_trajectory_encoder as S
E,D,M,C=S.E,S.D,S.M,S.C
PROTOCOL='E202-trajectory-remaining4pairs-flat-ordered-3seed8epoch-v1'
SELF='scripts/e202_trajectory_extension.py'
DOC='analysis/e202_trajectory_protocol_20261007.md'
FOLDS=(2,3,5,6)
ARMS=S.ARMS
OLD_REG='bfe793c62f7cecf633c7cc31c747f48eb0676f2d'
OLD_SPEC='93d90131cdfa32c04b8f8c3513f3e7f7cc19c9695d8f47d3fe98b4cc8c0a407d'
OLD_INPUT='data/e197_inputs_20261007'
OLD_OUTPUT='data/e197_trajectory_20261007'
OLD_COMPLETE='84842a0734975ae76bc27ebdf0fdc5841bc769a7dc8322255672d0c48c4f75db'
OLD_EVALUATION='e1b6903892865a948d896ecbaf895bca456ad7de2ea23427d13977bd81f3fb3e'
META='data/e193_weather_inputs_20261007'
META_SPEC='2b8f65e7212c91523d87a166ceb955376784223ee1dfb9cf1b403fa2f88a0f70'
INPUT='data/e202_inputs_20261007'
PREPARED='data/e202_prepared_20261007'
INCOMING='data/e202_transfer_20261007'
OUTPUT='data/e202_trajectory_20261007'
FILES=(Path(SELF).name,'e202_execution_queue_20261007.py',*S.FILES)
CONFIG={**S.CONFIG,'folds':list(FOLDS),'selected_screen_folds':[1,4],'per_fold_admission':True,
        'decided_pairs_required':4,'trim_population':'all rows; sum within-fold symmetric trim3/20'}
map_day,norm_stats,normalization,normalize,release_pages,prep_cgroup=(S.map_day,S.norm_stats,S.normalization,S.normalize,S.release_pages,S.prep_cgroup)


def registration(root,reg):
    if not re.fullmatch('[0-9a-f]{40}',reg or ''):raise ValueError('full registration required')
    for name in (SELF,DOC,'scripts/e202_execution_queue_20261007.py'):
        raw=subprocess.check_output(['git','show',reg+':'+name],cwd=root)
        if hashlib.sha256(raw).hexdigest()!=C.sha(root/name):raise ValueError('registered source/protocol changed')
    if PROTOCOL.encode() not in subprocess.check_output(['git','show',reg+':HYPOTHESES_LOG.md'],cwd=root):raise ValueError('journal missing protocol')


def checked_arm(root,fd,qb,od,arm,complete):
    r=C.read_json(od/'manifest.json');meta=E.read(fd/'evaluation.npz');ctrl=E.read(fd/'control.npz')
    if r not in complete['arms'] or r['arm']!=arm or r['seeds']!=[0,1,2] or [f['seed'] for f in r['fits']]!=[0,1,2] or any(len(f['losses'])!=8 or not np.isfinite(f['losses']).all() for f in r['fits']):raise ValueError('arm/seed/epoch receipt')
    for n,h in r['files'].items():
        if C.sha(od/n)!=h:raise ValueError('model/prediction hash differs')
    z=E.read(od/'prediction.npz');s=M.verify_bundle(qb,E.OLD_REGISTRATION)['spec']
    if any(not np.array_equal(z[k],meta[k]) for k in ('mvt_id','y','month')):raise ValueError('prediction metadata differs')
    seeds=[np.load(od/f'seed_{i}.npy') for i in (0,1,2)]
    if any(not np.isfinite(p).all() for p in seeds):raise ValueError('nonfinite seed')
    member=(np.mean(seeds,axis=0)*s['sd']+np.load(qb/'base_p.npy'))[np.load(qb/'control_indices.npy')]
    if not np.array_equal(member,z['member']):raise ValueError('seed mean differs')
    pred=E.assemble(ctrl,ctrl['reg']+ctrl['mlp_coefficient']*(member-ctrl['plain']),ctrl['lobt'])['pred']
    if not np.isfinite(pred).all() or not np.array_equal(pred,z['pred']):raise ValueError('mixture differs')
    return z


def check_screen(root):
    if C.sha(root/OLD_OUTPUT/'complete.json')!=OLD_COMPLETE or C.sha(root/OLD_OUTPUT/'evaluation.json')!=OLD_EVALUATION:raise ValueError('independently verified screen hashes changed')
    m=S.verify_bundle(root,root/OLD_INPUT,OLD_REG,queries=True);c=C.read_json(root/OLD_OUTPUT/'complete.json');ev=C.read_json(root/OLD_OUTPUT/'evaluation.json')
    if m['spec_sha256']!=OLD_SPEC or c['registration_commit']!=OLD_REG or c['input_spec_sha256']!=OLD_SPEC or len(c['arms'])!=4 or not ev['local_order_screen_pass'] or not ev['context_capacity_screen_pass']:raise ValueError('original E197 screen did not pass')
    for fs in m['spec']['folds']:
        fold=fs['fold'];name=f'fold_{fold}_{fold+6}';fd=root/OLD_INPUT/name;meta=E.read(fd/'evaluation.npz')
        pred={a:checked_arm(root,fd,root/fs['query_bundle'],root/OLD_OUTPUT/name/a,a,c)['pred'] for a in ARMS};ple=E.read(fd/'ple.npz')['pred']
        report=next(r for r in ev['reports'] if r['fold']==fold)
        for key,control,candidate in [('ordered_vs_flat',pred['flat'],pred['ordered']),('ordered_vs_ple',ple,pred['ordered']),('flat_vs_ple',ple,pred['flat'])]:
            actual=D.metrics(control,candidate,meta)
            if actual!=report[key] or any(actual[k]>=0 for k in ('body_sse','decided_sse','trim3_sse','trim20_sse')):raise ValueError('screen reconstruction/gate differs')
    return m,c,ev


def pack(root,out,reg):
    registration(root,reg)
    if out.exists():raise ValueError('fresh pack required')
    old,comp,ev=check_screen(root);parent=C.read_json(root/META/'manifest.json');ms=parent['spec']
    if parent['spec_sha256']!=META_SPEC or C.digest(ms)!=META_SPEC or ms['registration_commit']!=S.RAW_REG:raise ValueError('metadata parent changed')
    qpath=root/META/'query_ids.npy'
    if C.sha(qpath)!=ms['files']['query_ids.npy']:raise ValueError('query order changed')
    ids=np.load(qpath).astype(np.int64)
    if not np.array_equal(ids,np.load(root/OLD_INPUT/'query_ids.npy')):raise ValueError('E193/E197 query order differs')
    tmp=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent));(tmp/'scripts').mkdir()
    for n in FILES:shutil.copy2(root/'scripts'/n,tmp/'scripts'/n)
    shutil.copy2(root/DOC,tmp/'protocol.md');np.save(tmp/'query_ids.npy',ids);folds=[]
    for fold in FOLDS:
        name=f'fold_{fold}_{fold+6}';src=root/META/name;fd=tmp/name;fd.mkdir();fs=next(x for x in ms['folds'] if x['fold']==fold)
        qm=M.verify_bundle(root/fs['query_bundle'],E.OLD_REGISTRATION)
        if qm['spec_sha256']!=fs['query_spec_sha256'] or fs['alignment_numeric_columns_checked']!=8:raise ValueError('original tensor alignment proof differs')
        for n in ('train_indices.npy','pred_indices.npy','control.npz','evaluation.npz','ple.npz'):
            if C.sha(src/n)!=ms['files'][name+'/'+n]:raise ValueError('metadata file changed')
            shutil.copy2(src/n,fd/n)
        tr,pr=np.load(fd/'train_indices.npy'),np.load(fd/'pred_indices.npy');meta=E.read(root/fs['query_bundle']/'metadata.npz');ix=np.load(root/fs['query_bundle']/'control_indices.npy')
        if len(tr)!=fs['train_rows'] or len(pr)!=fs['pred_rows'] or len(np.unique(tr))!=len(tr) or len(np.unique(pr))!=len(pr) or np.intersect1d(tr,pr).size:raise ValueError('train/holdout mapping differs')
        if not np.array_equal(ids[pr][ix],meta['mvt_id']):raise ValueError('query holdout IDs differ')
        for n in ('control','evaluation','ple'):
            if any(not np.array_equal(E.read(fd/(n+'.npz'))[k],v) for k,v in meta.items()):raise ValueError('metadata/control mismatch')
        folds.append({**fs,'training_query_ids_sha256':C.digest(ids[tr].tolist())})
    for fold in (1,4):
        name=f'fold_{fold}_{fold+6}';fd=tmp/'selected_screen'/name;fd.mkdir(parents=True)
        for n in ('evaluation.npz','ple.npz'):shutil.copy2(root/OLD_INPUT/name/n,fd/n)
        for arm in ARMS:shutil.copy2(root/OLD_OUTPUT/name/arm/'prediction.npz',fd/(arm+'.npz'))
    C.write_json(tmp/'selected_screen/evaluation.json',ev)
    smoke=C.read_json(root/'data/e197_smoke_20261007.json')
    if not smoke['passed'] or smoke['worker_sha256']!=C.sha(S.__file__) or smoke['runtime']!=comp['runtime']:raise ValueError('qualified unchanged model/runtime mismatch')
    shutil.copy2(root/'data/e197_smoke_20261007.json',tmp/'smoke.json')
    spec=dict(protocol=PROTOCOL,registration_commit=reg,configuration=CONFIG,folds=folds,runtime=comp['runtime'],
        raw_registration=S.RAW_REG,raw_days=old['spec']['raw_days'],raw_complete_sha256=old['spec']['raw_complete_sha256'],
        metadata_parent_spec=META_SPEC,screen_spec=OLD_SPEC,screen_complete_sha256=C.sha(root/OLD_OUTPUT/'complete.json'),screen_evaluation_sha256=C.sha(root/OLD_OUTPUT/'evaluation.json'),
        scripts={n:C.sha(tmp/'scripts'/n) for n in FILES},files={str(p.relative_to(tmp)):C.sha(p) for p in tmp.rglob('*') if p.is_file()})
    C.write_json(tmp/'manifest.json',dict(spec=spec,spec_sha256=C.digest(spec)));tmp.rename(out);print('PACKED fixedremaining4pairs',C.digest(spec),flush=True)


def verify_bundle(root,bundle,registration,queries=False):
    m=C.read_json(bundle/'manifest.json');s=m['spec']
    if s['protocol']!=PROTOCOL or s['registration_commit']!=registration or s['configuration']!=CONFIG or m['spec_sha256']!=C.digest(s) or [f['fold'] for f in s['folds']]!=list(FOLDS):raise ValueError('input spec differs')
    for n,h in s['scripts'].items():
        if C.sha(Path(__file__).parent/n)!=h:raise ValueError('active source changed')
    for n,h in s['files'].items():
        if C.sha(bundle/n)!=h:raise ValueError('packed file changed')
    if queries:
        for fs in s['folds']:
            if M.verify_bundle(root/fs['query_bundle'],E.OLD_REGISTRATION)['spec_sha256']!=fs['query_spec_sha256']:raise ValueError('original tensor changed')
    return m


# CPU preparation and verification copied verbatim from frozen E197.
# Numerical helpers above remain direct references to unchanged S helpers.
def prepare_fold(root,bundle,output,registration,fold):
    m = verify_bundle(root,bundle,registration); guard = prep_cgroup()
    if fold not in FOLDS or output.exists(): raise ValueError('fixed fold/fresh prep destination required')
    spec = m['spec']; fd = bundle/f'fold_{fold}_{fold+6}'
    for name,h in spec['raw_complete_sha256'].items():
        if C.sha(root/name) != h: raise ValueError('raw complete receipt drift')
    for d in spec['raw_days']:
        if C.sha(root/d['path']) != d['sha256']: raise ValueError('raw trajectory input drift')
    ids = np.load(bundle/'query_ids.npy'); order = np.argsort(ids); sorted_ids=ids[order]
    tr = np.load(fd/'train_indices.npy'); pr=np.load(fd/'pred_indices.npy')
    fit = np.zeros(len(ids),bool);fit[tr]=True
    seen = np.zeros(len(ids),bool); count=np.zeros(6,np.int64);total=np.zeros(6);squares=np.zeros(6)
    start=time.monotonic()
    def statistics(d):
        with np.load(root/d['path'],allow_pickle=False) as a:
            ix=map_day(a['ids'],sorted_ids,order)
            if int(d['day'][5:7]) in (fold,fold+6) and fit[ix].any(): raise ValueError('heldout trajectories entered normalization')
            return ix,norm_stats(a['values'],a['known'],fit[ix])
    with ThreadPoolExecutor(max_workers=3) as pool:
        for begin in range(0,len(spec['raw_days']),3):
            for ix,(n,s,ss) in pool.map(statistics,spec['raw_days'][begin:begin+3]):
                if seen[ix].any(): raise ValueError('overlapping raw day IDs')
                seen[ix]=True;count+=n;total+=s;squares+=ss
    if not seen.all(): raise ValueError('missing trajectory rows in full train population')
    mean,std=normalization(count,total,squares)
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+output.name+'.partial-',dir=output.parent))
    shape=(len(ids),2,64,12)
    arrays={'values':np.lib.format.open_memmap(stage/'values.npy',mode='w+',dtype=np.float16,shape=shape),
            'known':np.lib.format.open_memmap(stage/'known.npy',mode='w+',dtype=np.uint8,shape=shape),
            'present':np.lib.format.open_memmap(stage/'present.npy',mode='w+',dtype=np.uint8,shape=shape[:3]),
            'support':np.lib.format.open_memmap(stage/'support.npy',mode='w+',dtype=np.uint8,shape=(len(ids),2))}
    def write_day(d):
        with np.load(root/d['path'],allow_pickle=False) as a:
            ix=map_day(a['ids'],sorted_ids,order);v,k,p=a['values'],a['known'],a['present']
            if np.any(k>p[...,None]) or np.any(a['point_times']>a['takeoff'][:,None,None]): raise ValueError('raw mask/cutoff contract differs')
            arrays['values'][ix]=normalize(v,k,mean,std);arrays['known'][ix]=k;arrays['present'][ix]=p
            arrays['support'][ix]=(a['raw_counts']>0).astype(np.uint8)
        return len(ix)
    done=0
    with ThreadPoolExecutor(max_workers=3) as pool:
        for begin in range(0,len(spec['raw_days']),3):
            done+=sum(pool.map(write_day,spec['raw_days'][begin:begin+3]))
            for a in arrays.values(): release_pages(a,write=True)
            if begin%30==0: print(f'fold{fold} packed {begin+min(3,len(spec["raw_days"])-begin)}/365 days, {done} rows',flush=True)
    if done!=len(ids): raise ValueError('prepared row count differs')
    del arrays;gc.collect()
    np.save(stage/'query_ids.npy',ids,allow_pickle=False)
    C.write_json(stage/'normalization.json',dict(channels=list(range(6)),count=count.tolist(),mean=mean.tolist(),std=std.tolist(),
            fitted_query_rows=len(tr),fitted_query_ids_sha256=C.digest(ids[tr].tolist()),heldout_months=[fold,fold+6]))
    for d in spec['raw_days']:
        if C.sha(root/d['path'])!=d['sha256']: raise ValueError('raw input changed while preparing')
    receipt=dict(protocol=PROTOCOL,registration_commit=registration,input_spec_sha256=m['spec_sha256'],fold=fold,
            rows=len(ids),train_rows=len(tr),pred_rows=len(pr),seconds=time.monotonic()-start,cgroup=guard,
            runtime=dict(numpy=np.__version__),source_sha256=C.sha(__file__),
            files={p.name:C.sha(p) for p in stage.iterdir() if p.is_file()},accepted_for_submission=False)
    C.write_json(stage/'complete.json',receipt);stage.rename(output)
    print(json.dumps(dict(prepared=str(output),rows=len(ids),seconds=receipt['seconds'])),flush=True)


def verify_prepared(bundle,prepared,manifest,fold):
    r=C.read_json(prepared/'complete.json')
    if (r['protocol']!=PROTOCOL or r['registration_commit']!=manifest['spec']['registration_commit']
        or r['input_spec_sha256']!=manifest['spec_sha256'] or r['fold']!=fold
        or r['source_sha256']!=C.sha(__file__) or r['rows']!=2085047): raise ValueError('prepared lineage mismatch')
    if set(r['files'])!={'values.npy','known.npy','present.npy','support.npy','query_ids.npy','normalization.json'}:
        raise ValueError('prepared file inventory mismatch')
    for name,h in r['files'].items():
        if C.sha(prepared/name)!=h: raise ValueError('prepared table hash drift: '+name)
    ids=np.load(prepared/'query_ids.npy')
    if not np.array_equal(ids,np.load(bundle/'query_ids.npy')): raise ValueError('prepared query ordering differs')
    norm=C.read_json(prepared/'normalization.json');tr=np.load(bundle/f'fold_{fold}_{fold+6}'/'train_indices.npy')
    if norm['fitted_query_ids_sha256']!=C.digest(ids[tr].tolist()): raise ValueError('normalization training rows differ')
    return r


def prepare_node(root,bundle,reg,node):
    assigned={171:(2,5),175:(3,6)}[node]
    for fold in assigned:prepare_fold(root,bundle,root/PREPARED/f'fold_{fold}_{fold+6}',reg,fold)
    m=verify_bundle(root,bundle,reg);receipts=[]
    for fold in assigned:
        folder=root/PREPARED/f'fold_{fold}_{fold+6}';r=verify_prepared(bundle,folder,m,fold)
        receipts.append(dict(fold=fold,complete_sha256=C.sha(folder/'complete.json'),files=r['files']))
    marker=root/f'data/e202_handoff_{node}_20261007.json'
    C.write_json(marker,dict(status='two_prepared_complete',protocol=PROTOCOL,registration_commit=reg,
        input_spec_sha256=m['spec_sha256'],node=node,folds=list(assigned),prepared_receipts=receipts,
        release_condition='CPU preparation complete; immutable file transfers and Home admission continue independently'))
    print('COMPLETE TWO PREPARED FOLDS; transfer/admission still required',node,flush=True)


def admit(root,bundle,reg,fold):
    m=verify_bundle(root,bundle,reg);name=f'fold_{fold}_{fold+6}';src=root/INCOMING/name;dest=root/PREPARED/name
    if fold not in FOLDS or dest.exists():raise ValueError('fresh admitted fold required')
    r=verify_prepared(bundle,src,m,fold);norm=C.read_json(src/'normalization.json')
    if norm['heldout_months']!=[fold,fold+6] or not np.isfinite(norm['mean']).all() or not np.isfinite(norm['std']).all() or min(norm['std'])<=0:raise ValueError('normalization receipt differs')
    shapes={'values':((2085047,2,64,12),'float16'),'known':((2085047,2,64,12),'uint8'),'present':((2085047,2,64),'uint8'),'support':((2085047,2),'uint8')}
    arrays={k:np.load(src/(k+'.npy'),mmap_mode='r') for k in shapes}
    for k,a in arrays.items():
        if a.shape!=shapes[k][0] or a.dtype!=np.dtype(shapes[k][1]):raise ValueError('prepared shape/dtype differs')
    counts=np.zeros(4,np.int64)
    for lo in range(0,2085047,4096):
        v,k,p,s=[np.asarray(arrays[n][lo:lo+4096]) for n in ('values','known','present','support')]
        if not np.isfinite(v).all() or (k>1).any() or (p>1).any() or (s>1).any() or (k>p[...,None]).any() or (v[k==0]!=0).any() or (np.abs(v[...,:6])>8).any():raise ValueError('numeric/mask contract differs')
        if not np.array_equal(s,(p.sum(2)>0).astype(np.uint8)):raise ValueError('support/present differs')
        counts+=np.array([int(p.sum()),int(s[:,0].sum()),int(s[:,1].sum()),int((s[:,0]&s[:,1]).sum())])
        for a in arrays.values():release_pages(a)
    if counts.tolist()!=[88770275,1251417,1345381,1022407]:raise ValueError('raw observation coverage differs')
    del arrays;gc.collect();dest.parent.mkdir(parents=True,exist_ok=True)
    C.write_json(src/'admission.json',dict(protocol=PROTOCOL,registration_commit=reg,input_spec_sha256=m['spec_sha256'],fold=fold,
        passed=True,rows=2085047,prepared_complete_sha256=C.sha(src/'complete.json'),counts=counts.tolist(),files=r['files']))
    src.rename(dest);print('ADMITTED',fold,flush=True)


def await_prepared(root,bundle,m,fold):
    path=root/PREPARED/f'fold_{fold}_{fold+6}';deadline=time.monotonic()+7200
    while not (path/'admission.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('prepared fold not admitted within2h')
        print('WAIT ADMITTED FOLD',fold,flush=True);time.sleep(15)
    a=C.read_json(path/'admission.json')
    if not a['passed'] or a['registration_commit']!=m['spec']['registration_commit'] or a['input_spec_sha256']!=m['spec_sha256'] or a['fold']!=fold or a['prepared_complete_sha256']!=C.sha(path/'complete.json'):raise ValueError('admission proof differs')
    return verify_prepared(bundle,path,m,fold)


def run(root,bundle,out,reg):
    S.fixed_env();import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);guard=M.verify_cgroup();m=verify_bundle(root,bundle,reg,queries=True)
    runtime=E.package_runtime();sm=C.read_json(bundle/'smoke.json')
    if runtime!=m['spec']['runtime'] or sm['runtime']!=runtime or sm['worker_sha256']!=C.sha(S.__file__) or not sm['passed']:raise ValueError('unchanged qualified model/runtime differs')
    if out.exists():raise ValueError('fresh output required; no resume')
    tmp=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent));receipts=[];preps=[]
    for fs in m['spec']['folds']:
        fold=fs['fold'];name=f'fold_{fold}_{fold+6}';fd=bundle/name;prepared=root/PREPARED/name
        preps.append(await_prepared(root,bundle,m,fold));qb=root/fs['query_bundle'];s=M.verify_bundle(qb,E.OLD_REGISTRATION)['spec'];arrays,_=M.load_arrays(qb,s)
        tr,pr=np.load(fd/'train_indices.npy'),np.load(fd/'pred_indices.npy')
        if len(tr)!=len(arrays['xnf']) or len(pr)!=len(arrays['xnp']):raise ValueError('query/context shapes differ')
        cont,scalar,bins=E.numeric_bins(arrays['xnf']);N=E.trainer(qb/'src');tables=S.gpu_tables(prepared);ctrl=E.read(fd/'control.npz')
        for arm in ARMS:
            od=tmp/name/arm;od.mkdir(parents=True);np.savez(od/'bins.npz',continuous=cont,scalar=scalar,**{f'bin_{i}':b for i,b in enumerate(bins)})
            print('START',name,arm,flush=True);fits=[S.fit(arrays,s,N,cont,scalar,bins,tables,tr,pr,od,arm,seed) for seed in (0,1,2)]
            member=(np.mean([np.load(od/f'seed_{i}.npy') for i in (0,1,2)],axis=0)*s['sd']+arrays['base_p'])[arrays['control_indices']]
            cand=E.assemble(ctrl,ctrl['reg']+ctrl['mlp_coefficient']*(member-ctrl['plain']),ctrl['lobt'])
            np.savez_compressed(od/'prediction.npz',**{k:cand[k] for k in ('mvt_id','y','month','pred','reg','values','proba')},member=member)
            r=dict(fold=fold,arm=arm,seeds=[0,1,2],fits=fits,files={p.name:C.sha(p) for p in od.iterdir() if p.is_file()});C.write_json(od/'manifest.json',r);receipts.append(r)
        if len({f['parameters'] for r in receipts[-2:] for f in r['fits']})!=1:raise ValueError('matched capacity differs')
        support=np.load(prepared/'support.npy',mmap_mode='r');np.save(tmp/name/'support.npy',np.array(support[pr[arrays['control_indices']]],copy=True));release_pages(support)
        del arrays,tables,support;gc.collect();torch.cuda.empty_cache();print('COMPLETE SIX FITS',name,flush=True)
    verify_bundle(root,bundle,reg,queries=True)
    C.write_json(tmp/'complete.json',dict(protocol=PROTOCOL,registration_commit=reg,input_spec_sha256=m['spec_sha256'],runtime=runtime,cgroup=guard,arms=receipts,prepared_receipts=preps,
        support_sha256={str(p.relative_to(tmp)):C.sha(p) for p in tmp.glob('*/support.npy')},accepted_for_submission=False))
    tmp.rename(out);print('COMPLETE24FITS',flush=True);evaluate(root,bundle,out,reg)


def measured(a,b,meta):
    r=D.metrics(a,b,meta)
    for k,mask in [('body',meta['y']<=7200),('decided',meta['decided'])]:
        r[k+'_rows']=int(mask.sum());r[k+'_rmse_control']=float(np.sqrt(np.mean((a[mask]-meta['y'][mask])**2)));r[k+'_rmse_candidate']=float(np.sqrt(np.mean((b[mask]-meta['y'][mask])**2)))
    return r


def pooled(reports,key):
    return dict(folds=len(reports),decided_folds_improved=sum(r[key]['decided_sse']<0 for r in reports),**{k:float(sum(r[key][k] for r in reports)) for k in ('all_sse','body_sse','decided_sse','trim3_sse','trim20_sse')})


def passed(x):return x['folds']==6 and x['decided_folds_improved']>=4 and all(x[k]<0 for k in ('body_sse','trim3_sse','trim20_sse'))


def evaluate(root,bundle,out,reg):
    m=verify_bundle(root,bundle,reg,queries=True);complete=C.read_json(out/'complete.json')
    if complete['protocol']!=PROTOCOL or complete['registration_commit']!=reg or complete['input_spec_sha256']!=m['spec_sha256'] or len(complete['arms'])!=8 or complete['runtime']!=m['spec']['runtime']:raise ValueError('extension incomplete')
    reports=[];ids=[];old=C.read_json(bundle/'selected_screen/evaluation.json')
    for fold in range(1,7):
        name=f'fold_{fold}_{fold+6}';selected=fold in (1,4);fd=bundle/'selected_screen'/name if selected else bundle/name;meta=E.read(fd/'evaluation.npz');ids.append(meta['mvt_id'])
        if selected:pred={a:E.read(fd/(a+'.npz')) for a in ARMS}
        else:
            fs=next(f for f in m['spec']['folds'] if f['fold']==fold);pred={a:checked_arm(root,fd,root/fs['query_bundle'],out/name/a,a,complete) for a in ARMS}
        for z in pred.values():
            if any(not np.array_equal(z[k],meta[k]) for k in ('mvt_id','y','month')) or not np.isfinite(z['pred']).all():raise ValueError('selected metadata/finite differs')
        ple=E.read(fd/'ple.npz');comparisons={key:measured(a,b,meta) for key,a,b in [('ordered_vs_flat',pred['flat']['pred'],pred['ordered']['pred']),('ordered_vs_ple',ple['pred'],pred['ordered']['pred']),('flat_vs_ple',ple['pred'],pred['flat']['pred'])]}
        if selected:diagnostics=next(r for r in old['reports'] if r['fold']==fold)['support_diagnostics']
        else:
            path=out/name/'support.npy'
            if C.sha(path)!=complete['support_sha256'][name+'/support.npy']:raise ValueError('heldout support hash differs')
            sup=np.load(path);groups={'own':sup[:,0]>0,'previous_only':(sup[:,0]==0)&(sup[:,1]>0),'no_points':sup.sum(1)==0,'LIRF_NM':(meta['airport']=='LIRF')&meta['has_nm'],'LTFM':meta['airport']=='LTFM'}
            groups.update({f'month{mo}':meta['month']==mo for mo in np.unique(meta['month'])});delta=(pred['ordered']['pred']-meta['y'])**2-(pred['flat']['pred']-meta['y'])**2
            diagnostics={k:dict(rows=int(v.sum()),ordered_minus_flat_body_sse=float(delta[v&(meta['y']<=7200)].sum())) for k,v in groups.items()}
        reports.append(dict(fold=fold,role='selected_screen' if selected else 'remaining_confirmation',support_diagnostics=diagnostics,**comparisons))
    flat=np.concatenate(ids)
    if len(flat)!=2085047 or len(np.unique(flat))!=len(flat):raise ValueError('six-pair ID coverage differs')
    keys=('ordered_vs_flat','ordered_vs_ple','flat_vs_ple');pool={k:pooled(reports,k) for k in keys};remaining={k:pooled([r for r in reports if r['role']=='remaining_confirmation'],k) for k in keys}
    result=dict(protocol=PROTOCOL,registration_commit=reg,reports=reports,pooled_six_pairs=pool,remaining_four_pairs=remaining,
        local_order_six_pair_pass=passed(pool['ordered_vs_flat']) and passed(pool['ordered_vs_ple']),context_capacity_six_pair_pass=passed(pool['flat_vs_ple']),
        accepted_for_submission=False,limitation='Selected2screen+remaining4confirmation; all6containsselection. LGB5/pre-ADSB component comparison, no current-chain acceptance or ranking.')
    C.write_json(out/'evaluation.json',result);print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['pack','verify','prepare-node','admit','run','evaluate','selftest']);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--registration-commit');p.add_argument('--fold',type=int);p.add_argument('--node',type=int,choices=[171,175]);a=p.parse_args();root=a.root.resolve();bundle=root/INPUT
    if a.action=='selftest':
        import ast,inspect
        S.selftest();assert set(FOLDS).isdisjoint((1,4))
        for fn in (prepare_fold,verify_prepared):
            assert ast.dump(ast.parse(inspect.getsource(fn)),include_attributes=False)==ast.dump(ast.parse(inspect.getsource(getattr(S,fn.__name__))),include_attributes=False)
        good=dict(folds=6,decided_folds_improved=4,body_sse=-1,trim3_sse=-1,trim20_sse=-1)
        assert passed(good) and passed({**good,'decided_sse':100})
        assert not passed({**good,'decided_folds_improved':3}) and not passed({**good,'folds':4})
        for key in ('body_sse','trim3_sse','trim20_sse'):assert not passed({**good,key:0})
        print('PASS fixed4foldgate boundaries and unchangedE197 preparation/model helpers');return
    if a.action=='pack':pack(root,bundle,a.registration_commit)
    elif a.action=='verify':print(verify_bundle(root,bundle,a.registration_commit)['spec_sha256'])
    elif a.action=='prepare-node':prepare_node(root,bundle,a.registration_commit,a.node)
    elif a.action=='admit':admit(root,bundle,a.registration_commit,a.fold)
    elif a.action=='run':run(root,bundle,root/OUTPUT,a.registration_commit)
    else:evaluate(root,bundle,root/OUTPUT,a.registration_commit)


if __name__=='__main__':main()

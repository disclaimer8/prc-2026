"""E197: matched local-order versus order-free encoders of actual E194 points.

CPU laptops prepare immutable train-only normalizations and float16 mmap tables.
ROCm trains 2 folds x 2 arms x 3 seeds, with unchanged E190 query/target tensors.
No physical pseudo-labels, prefix augmentation, interpolation, or ranking fitting.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import tempfile
import time

for key, value in {'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'1',
        'MKL_NUM_THREADS':'1', 'POLARS_MAX_THREADS':'2',
        'POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR':'load_as_storage'}.items():
    os.environ.setdefault(key, value)
import numpy as np
import e190_numeric_embeddings as E
import e192_event_context as D
M, C = E.M, E.M.C

PROTOCOL = 'E197-own-previous-local-order-vs-flat-K64-2pair-3seed-8epoch-v1'
RAW_REG = 'fc7107de145a87a5a1ff93314f318b440b1de825'
PARENT = 'data/e192_event_inputs_20261006'
FOLDS, ARMS = (1, 4), ('flat', 'ordered')
FILES = ('e197_trajectory_encoder.py', 'e190_numeric_embeddings.py',
         'e192_event_context.py', 'mlp_gpu_component_oof.py',
         'mlp_lowmem_prepare.py', 'cat_fin1b_oof.py')
CONFIG = dict(folds=list(FOLDS), arms=list(ARMS), seeds=[0,1,2], epochs=8,
              batch=4096, width=512, bins=48, d_embedding=12, ple_version='B',
              token_width=64, message_blocks=2, streams=2, tokens=64, channels=12,
              normalization='ordinary-fit-only-mean-std-channels0to5-clip8',
              values_dtype='float16', masks_dtype='uint8', prep_workers=3,
              lr=.002, weight_decay=.00001, pct_start=.1)


def fixed_env():
    os.environ.update(M.FIXED_ENV)
    os.environ['FEATURES'] = '__immutable_query_tensors_only__'


def check_registration(root, registration):
    if not re.fullmatch('[0-9a-f]{40}', registration or ''):
        raise ValueError('full registration commit required')
    for name in ('scripts/e197_trajectory_encoder.py', 'analysis/e197_trajectory_protocol_20261007.md'):
        raw = subprocess.check_output(['git','show',registration+':'+name], cwd=root)
        if hashlib.sha256(raw).hexdigest() != C.sha(root/name):
            raise ValueError('registered source/protocol drift: '+name)
    journal = subprocess.check_output(['git','show',registration+':HYPOTHESES_LOG.md'], cwd=root)
    if PROTOCOL.encode() not in journal:
        raise ValueError('journal must include exact registered protocol')


def pack(root, output, registration):
    check_registration(root, registration)
    if output.exists(): raise ValueError('fresh input bundle required')
    audit = C.read_json(root/'data/e194_verification_20261007/verification.json')
    if audit['status'] != 'PASS' or audit['unique_ids'] != 2429888 or audit['days'] != 427 or audit['registration_commit'] != RAW_REG:
        raise ValueError('E194 extraction not independently admitted')
    parent = C.read_json(root/PARENT/'manifest.json')
    if parent['spec_sha256'] != C.digest(parent['spec']): raise ValueError('parent digest mismatch')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.'+output.name+'.partial-', dir=output.parent))
    (stage/'scripts').mkdir()
    for name in FILES: shutil.copy2(root/'scripts'/name, stage/'scripts'/name)
    shutil.copy2(root/'analysis/e197_trajectory_protocol_20261007.md', stage/'protocol.md')
    qpath = root/PARENT/'query_ids.npy'
    if C.sha(qpath) != parent['spec']['files']['query_ids.npy']: raise ValueError('parent query IDs drift')
    query_ids = np.load(qpath).astype(np.int64)
    if len(query_ids) != 2085047 or len(np.unique(query_ids)) != len(query_ids): raise ValueError('query identity population differs')
    np.save(stage/'query_ids.npy', query_ids, allow_pickle=False)
    folds = []
    for fold in FOLDS:
        name = f'fold_{fold}_{fold+6}'; src = root/PARENT/name; dst = stage/name; dst.mkdir()
        fs = next(f for f in parent['spec']['folds'] if f['fold'] == fold)
        qb = root/fs['query_bundle']; qm = M.verify_bundle(qb,E.OLD_REGISTRATION)
        if fs['query_spec_sha256'] != qm['spec_sha256']: raise ValueError('original query lineage differs')
        for filename in ('train_indices.npy','pred_indices.npy','control.npz','evaluation.npz','ple.npz'):
            if C.sha(src/filename) != parent['spec']['files'][name+'/'+filename]: raise ValueError('parent file drift')
            shutil.copy2(src/filename,dst/filename)
        tr, pr = np.load(dst/'train_indices.npy'), np.load(dst/'pred_indices.npy')
        if len(np.unique(tr)) != len(tr) or len(np.unique(pr)) != len(pr) or np.intersect1d(tr,pr).size:
            raise ValueError('train/prediction indices duplicate or overlap')
        if C.digest(query_ids[tr].astype(float).tolist()) != fs['train_query_id_sha256']:
            raise ValueError('ordinary query training order differs')
        meta = E.read(qb/'metadata.npz'); ix = np.load(qb/'control_indices.npy')
        if not np.array_equal(query_ids[pr][ix],meta['mvt_id']): raise ValueError('heldout query order differs')
        if any(not np.array_equal(E.read(dst/'evaluation.npz')[k],v) for k,v in meta.items()):
            raise ValueError('evaluation targets differ from immutable tensor metadata')
        folds.append(dict(fold=fold, query_bundle=fs['query_bundle'], query_spec_sha256=qm['spec_sha256'],
                          train_rows=len(tr), pred_rows=len(pr), alignment_numeric_columns_checked=fs['alignment_numeric_columns_checked']))
    days, raw_complete = [], {}
    for shard in (0,1):
        folder = root/f'data/e194_sequence_20261007_s{shard}'
        receipt = C.read_json(folder/'complete.json')
        if receipt['registration_commit'] != RAW_REG: raise ValueError('wrong raw extraction registration')
        raw_complete[str((folder/'complete.json').relative_to(root))] = C.sha(folder/'complete.json')
        for report in receipt['reports']:
            if not report['day'].startswith('2025'): continue
            path = folder/report['day']/'sequence.npz'
            if C.sha(path) != report['npz_sha256']: raise ValueError('raw sequence drift')
            days.append(dict(day=report['day'], path=str(path.relative_to(root)), sha256=report['npz_sha256'], rows=report['rows']))
    days.sort(key=lambda z:z['day'])
    if len(days) != 365 or sum(d['rows'] for d in days) != len(query_ids): raise ValueError('training day coverage differs')
    spec = dict(protocol=PROTOCOL, registration_commit=registration, configuration=CONFIG,
                parent_spec_sha256=parent['spec_sha256'], raw_registration=RAW_REG,
                raw_complete_sha256=raw_complete, raw_days=days, folds=folds,
                raw_audit_sha256=C.sha(root/'data/e194_verification_20261007/verification.json'),
                scripts={n:C.sha(stage/'scripts'/n) for n in FILES},
                files={str(p.relative_to(stage)):C.sha(p) for p in stage.rglob('*') if p.is_file()})
    C.write_json(stage/'manifest.json',dict(spec=spec,spec_sha256=C.digest(spec)))
    check_registration(root,registration)
    stage.rename(output)
    print(json.dumps(dict(packed=str(output), raw_days=365, rows=len(query_ids), no_target_fit=True)),flush=True)


def verify_bundle(root, bundle, registration, queries=False):
    m = C.read_json(bundle/'manifest.json'); s = m['spec']
    if (s['protocol'] != PROTOCOL or s['configuration'] != CONFIG or s['registration_commit'] != registration
            or s['raw_registration'] != RAW_REG or m['spec_sha256'] != C.digest(s)
            or [f['fold'] for f in s['folds']] != list(FOLDS)):
        raise ValueError('protocol/registration/configuration drift')
    for name,h in s['scripts'].items():
        if C.sha(Path(__file__).parent/name) != h: raise ValueError('active helper drift: '+name)
    for name,h in s['files'].items():
        if C.sha(bundle/name) != h: raise ValueError('input file drift: '+name)
    if queries:
        for fs in s['folds']:
            if M.verify_bundle(root/fs['query_bundle'],E.OLD_REGISTRATION)['spec_sha256'] != fs['query_spec_sha256']:
                raise ValueError('original query tensor drift')
    return m


def prep_cgroup():
    relative = next(x[3:] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    path = Path('/sys/fs/cgroup')/relative.lstrip('/')
    mem,swap = [(path/n).read_text().strip() for n in ('memory.max','memory.swap.max')]
    quota,period = (path/'cpu.max').read_text().split()
    if mem == 'max' or int(mem)>10*1024**3 or swap != '0' or quota == 'max' or int(quota)>6*int(period):
        raise ValueError('CPU prep requires <=6CPU, <=10GiB RAM, swap0')
    return dict(path=relative,memory_max=int(mem),cpu_max=[int(quota),int(period)],swap_max=0)


def map_day(ids, sorted_ids, order):
    positions = np.searchsorted(sorted_ids,ids)
    if (positions>=len(sorted_ids)).any() or not np.array_equal(sorted_ids[positions],ids):
        raise ValueError('missing/unexpected trajectory query ID')
    if len(np.unique(ids)) != len(ids): raise ValueError('duplicate trajectory day IDs')
    return order[positions]


def norm_stats(values,known,fit):
    z = values[fit,:,:,:6].astype(np.float64); k = known[fit,:,:,:6].astype(bool)
    if not np.isfinite(z).all(): raise ValueError('nonfinite raw numeric values')
    return k.sum((0,1,2),dtype=np.int64), np.where(k,z,0).sum((0,1,2)), np.where(k,z*z,0).sum((0,1,2))


def normalization(count,total,squares):
    mean = total/np.maximum(count,1)
    std = np.sqrt(np.maximum(squares/np.maximum(count,1)-mean*mean,0))
    std = np.where((count>1)&(std>1e-6),std,1.)
    return mean,std


def normalize(values,known,mean,std):
    z = values.copy()
    z[...,:6] = np.clip((z[...,:6]-mean)/std,-8,8)
    z = np.where(known,z,0).astype(np.float16)
    if not np.isfinite(z).all(): raise ValueError('nonfinite normalized tensor')
    return z


def release_pages(a,write=False):
    if write: a.flush()
    if hasattr(a,'_mmap') and hasattr(a._mmap,'madvise'):
        a._mmap.madvise(mmap.MADV_DONTNEED)


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


def pools(h,valid):
    import torch
    mask=valid.unsqueeze(-1)
    mean=(h*mask).sum(-2)/mask.sum(-2).clamp_min(1)
    maximum=h.masked_fill(~mask,float('-inf')).amax(-2)
    maximum=torch.where(valid.any(-1,keepdim=True),maximum,torch.zeros_like(maximum))
    return mean,maximum


def triplets(h,valid,arm):
    import torch
    if arm=='ordered':
        zero=torch.zeros_like(h[...,:1,:])
        left=torch.cat([zero,h[...,:-1,:]],dim=-2)
        right=torch.cat([h[...,1:,:],zero],dim=-2)
        return torch.cat([left,h,right],dim=-1)
    mean,maximum=pools(h,valid)
    return torch.cat([h,mean.unsqueeze(-2).expand_as(h),maximum.unsqueeze(-2).expand_as(h)],dim=-1)


def network(N,card,continuous,scalar,bins,n_num,arm):
    import torch
    from torch import nn
    class TrajectoryNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.query=E.net_factory(N.Net,n_num,card,512,continuous,scalar,bins)
            self.query.body[-1]=nn.Identity()
            self.token=nn.Sequential(nn.Linear(24,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
            self.messages=nn.ModuleList([nn.Sequential(nn.Linear(192,64),nn.SiLU(),nn.Linear(64,64)) for _ in range(2)])
            self.fusion=nn.Sequential(nn.Linear(256+4*64+2,256),nn.SiLU(),nn.Linear(256,1))
        def forward(self,xn,xc,v,k,p):
            valid=p.bool();known=k.float()*valid[...,None]
            h=self.token(torch.cat([v.float()*known,known],dim=-1))*valid[...,None]
            for block in self.messages:
                h=(h+block(triplets(h,valid,arm)))*valid[...,None]
            mean,maximum=pools(h,valid)
            context=torch.cat([mean.flatten(1),maximum.flatten(1),valid.float().mean(-1)],dim=-1)
            return self.fusion(torch.cat([self.query(xn,xc),context],dim=-1)).squeeze(-1)
    return TrajectoryNet()


def gpu_tables(prepared):
    import torch
    tables={}
    for name,dtype in [('values',torch.float16),('known',torch.uint8),('present',torch.uint8)]:
        a=np.load(prepared/(name+'.npy'),mmap_mode='r')
        expected=(2085047,2,64,12) if name!='present' else (2085047,2,64)
        if a.shape!=expected or a.dtype!=np.dtype('float16' if name=='values' else 'uint8'):
            raise ValueError('prepared tensor shape/dtype differs')
        target=torch.empty(a.shape,dtype=dtype,device='cuda:0')
        for lo in range(0,len(a),8192):
            chunk=np.array(a[lo:lo+8192],copy=True)
            target[lo:lo+len(chunk)].copy_(torch.from_numpy(chunk))
            release_pages(a)
        tables[name]=target;del a,chunk
        print(f'{name} table resident on GPU; CPU mmap pages released',flush=True)
    return tables


def context_batch(t,indices):
    import torch
    ix=torch.as_tensor(indices,dtype=torch.int64,device='cuda:0')
    return tuple(t[n][ix] for n in ('values','known','present'))


def fit(arrays,spec,N,cont,scalar,bins,tables,tridx,pidx,out,arm,seed):
    import torch
    torch.manual_seed(seed);np.random.seed(seed)
    net=network(N,spec['card'],cont,scalar,bins,arrays['xnf'].shape[1],arm).to('cuda:0')
    opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.00001)
    bs=4096;n=len(tridx);steps=8*(n//bs)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.002,total_steps=steps,pct_start=.1)
    xn,xc,y=(torch.from_numpy(arrays[k]) for k in ('xnf','xcf','y_scaled'));y=y.float()
    rng=np.random.default_rng(seed);losses=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for ep in range(8):
        net.train();perm=rng.permutation(n);total=0.
        for batch in range(n//bs):
            ix=perm[batch*bs:(batch+1)*bs]
            pred=net(xn[ix].to('cuda:0'),xc[ix].to('cuda:0'),*context_batch(tables,tridx[ix]))
            loss=torch.nn.functional.mse_loss(pred,y[ix].to('cuda:0'))
            if not torch.isfinite(loss): raise ValueError('nonfinite training loss')
            opt.zero_grad();loss.backward();opt.step();sched.step();total+=loss.item()
        losses.append(total/(n//bs));print(f'{arm} seed{seed} epoch{ep+1}/8 mse={losses[-1]:.6f}',flush=True)
    net.eval();parts=[]
    with torch.no_grad():
        for lo in range(0,len(pidx),bs):
            parts.append(net(torch.from_numpy(arrays['xnp'][lo:lo+bs]).to('cuda:0'),
                torch.from_numpy(arrays['xcp'][lo:lo+bs]).to('cuda:0'),*context_batch(tables,pidx[lo:lo+bs])).cpu().numpy())
    prediction=np.concatenate(parts)
    if prediction.shape!=(len(pidx),) or not np.isfinite(prediction).all(): raise ValueError('incomplete/nonfinite prediction')
    np.save(out/f'seed_{seed}.npy',prediction,allow_pickle=False);torch.save(net.state_dict(),out/f'model_{seed}.pt')
    result=dict(seed=seed,seconds=time.monotonic()-start,parameters=sum(p.numel() for p in net.parameters()),
                losses=losses,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    del net,opt,sched,xn,xc,y,parts;gc.collect();torch.cuda.empty_cache()
    return result


def run(root,bundle,prepared,out,registration,smoke_path):
    fixed_env();import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    guard=M.verify_cgroup();manifest=verify_bundle(root,bundle,registration,queries=True)
    sm=C.read_json(smoke_path);runtime=E.package_runtime()
    if not sm['passed'] or sm['worker_sha256']!=C.sha(__file__) or sm['runtime']!=runtime:
        raise ValueError('synthetic smoke/runtime differs')
    if out.exists(): raise ValueError('fresh output required; no automatic resume')
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent));done=[];prep_receipts=[]
    for fs in manifest['spec']['folds']:
        fold=fs['fold'];name=f'fold_{fold}_{fold+6}';fd=bundle/name
        prep_receipts.append(verify_prepared(bundle,prepared/name,manifest,fold))
        qb=root/fs['query_bundle'];spec=M.verify_bundle(qb,E.OLD_REGISTRATION)['spec'];arrays,_=M.load_arrays(qb,spec)
        tridx=np.load(fd/'train_indices.npy');pidx=np.load(fd/'pred_indices.npy')
        if len(tridx)!=len(arrays['xnf']) or len(pidx)!=len(arrays['xnp']): raise ValueError('query/context row mismatch')
        cont,scalar,bins=E.numeric_bins(arrays['xnf']);N=E.trainer(qb/'src');tables=gpu_tables(prepared/name)
        ctrl=E.read(fd/'control.npz')
        for arm in ARMS:
            od=stage/name/arm;od.mkdir(parents=True);fits=[]
            np.savez(od/'bins.npz',continuous=cont,scalar=scalar,**{f'bin_{j}':b for j,b in enumerate(bins)})
            for seed in (0,1,2): fits.append(fit(arrays,spec,N,cont,scalar,bins,tables,tridx,pidx,od,arm,seed))
            member=(np.mean([np.load(od/f'seed_{s}.npy') for s in (0,1,2)],axis=0)*spec['sd']+arrays['base_p'])[arrays['control_indices']]
            cand=E.assemble(ctrl,ctrl['reg']+ctrl['mlp_coefficient']*(member-ctrl['plain']),ctrl['lobt'])
            np.savez_compressed(od/'prediction.npz',**{k:cand[k] for k in ('mvt_id','y','month','pred','reg','values','proba')},member=member)
            receipt=dict(fold=fold,arm=arm,seeds=[0,1,2],fits=fits,files={p.name:C.sha(p) for p in od.iterdir() if p.is_file()})
            C.write_json(od/'manifest.json',receipt);done.append(receipt)
        if len({f['parameters'] for r in done[-2:] for f in r['fits']})!=1: raise ValueError('arms capacity differs')
        del arrays,tables;gc.collect();torch.cuda.empty_cache()
    verify_bundle(root,bundle,registration,queries=True)
    C.write_json(stage/'complete.json',dict(protocol=PROTOCOL,registration_commit=registration,
            input_spec_sha256=manifest['spec_sha256'],runtime=runtime,cgroup=guard,arms=done,
            prepared_receipts=prep_receipts,accepted_for_submission=False))
    stage.rename(out);print('COMPLETE 12 fits; no automatic submission acceptance',flush=True)


def evaluate(root,bundle,prepared,out,registration):
    m=verify_bundle(root,bundle,registration,queries=True);complete=C.read_json(out/'complete.json');reports=[]
    if complete['registration_commit']!=registration or complete['input_spec_sha256']!=m['spec_sha256'] or len(complete['arms'])!=4:
        raise ValueError('experiment incomplete')
    for fs in m['spec']['folds']:
        fold=fs['fold'];name=f'fold_{fold}_{fold+6}';fd=bundle/name;meta=E.read(fd/'evaluation.npz');pred={}
        qb=root/fs['query_bundle'];s=M.verify_bundle(qb,E.OLD_REGISTRATION)['spec'];ctrl=E.read(fd/'control.npz')
        for arm in ARMS:
            od=out/name/arm;receipt=C.read_json(od/'manifest.json')
            if receipt not in complete['arms'] or receipt['seeds']!=[0,1,2]: raise ValueError('arm receipt differs')
            for p,h in receipt['files'].items():
                if C.sha(od/p)!=h: raise ValueError('result hash drift')
            z=E.read(od/'prediction.npz')
            if any(not np.array_equal(z[k],meta[k]) for k in ('mvt_id','y','month')): raise ValueError('prediction metadata differs')
            expected=(np.mean([np.load(od/f'seed_{s}.npy') for s in (0,1,2)],axis=0)*s['sd']+np.load(qb/'base_p.npy'))[np.load(qb/'control_indices.npy')]
            if not np.array_equal(expected,z['member']): raise ValueError('seed reconstruction mismatch')
            expected=E.assemble(ctrl,ctrl['reg']+ctrl['mlp_coefficient']*(expected-ctrl['plain']),ctrl['lobt'])
            if not np.array_equal(expected['pred'],z['pred']): raise ValueError('mixture arithmetic differs')
            pred[arm]=z['pred']
        ple=E.read(fd/'ple.npz')
        if not np.array_equal(ple['mvt_id'],meta['mvt_id']): raise ValueError('PLE control IDs differ')
        comparisons=dict(ordered_vs_flat=D.metrics(pred['flat'],pred['ordered'],meta),
                         ordered_vs_ple=D.metrics(ple['pred'],pred['ordered'],meta),
                         flat_vs_ple=D.metrics(ple['pred'],pred['flat'],meta))
        support=np.load(prepared/name/'support.npy',mmap_mode='r')
        pi=np.load(fd/'pred_indices.npy')[np.load(qb/'control_indices.npy')];sup=np.array(support[pi],copy=True);release_pages(support)
        groups={'own':sup[:,0]>0,'previous_only':(sup[:,0]==0)&(sup[:,1]>0),'no_points':sup.sum(1)==0,
                'LIRF_NM':(meta['airport']=='LIRF')&meta['has_nm'],'LTFM':meta['airport']=='LTFM'}
        groups.update({f'month{month}':meta['month']==month for month in np.unique(meta['month'])})
        diagnostics={}
        for label,mask in groups.items():
            diagnostics[label]=dict(rows=int(mask.sum()),ordered_minus_flat_body_sse=float((((pred['ordered']-meta['y'])**2-(pred['flat']-meta['y'])**2)[mask&(meta['y']<=7200)]).sum()))
        reports.append(dict(fold=fold,**comparisons,support_diagnostics=diagnostics))
    def passed(key): return all(all(r[key][metric]<0 for metric in ('body_sse','decided_sse','trim3_sse','trim20_sse')) for r in reports)
    result=dict(protocol=PROTOCOL,registration_commit=registration,reports=reports,
        local_order_screen_pass=passed('ordered_vs_flat') and passed('ordered_vs_ple'),
        context_capacity_screen_pass=passed('flat_vs_ple'),accepted_for_submission=False,
        limitation='Two fixed pairs, LGB5/pre-ADSB screen. Flat vs saved PLE combines inputs and added capacity. All arms have ages; only local-order inductive bias is isolated.')
    C.write_json(out/'evaluation.json',result);print(json.dumps(result,indent=2),flush=True)


def selftest():
    z=np.arange(4*2*64*12,dtype=np.float32).reshape(4,2,64,12);k=np.ones(z.shape,np.uint8);k[0,:,:,0]=0
    fit=np.array([True,True,False,False]);a=norm_stats(z,k,fit);changed=z.copy();changed[2:]=1e9
    b=norm_stats(changed,k,fit)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    mean,std=normalization(*a);v=normalize(z,k,mean,std)
    assert v.dtype==np.float16 and np.isfinite(v).all() and (v[k==0]==0).all()
    assert np.allclose(mean[0],z[1,:,:,0].mean())
    m,s=normalization(np.zeros(6),np.zeros(6),np.zeros(6));assert not m.any() and (s==1).all()
    assert np.array_equal(map_day(np.array([4,1]),np.array([1,2,4]),np.array([2,0,1])),[1,2])
    for ids in (np.array([3]),np.array([1,1])):
        try: map_day(ids,np.array([1,2]),np.array([0,1]))
        except ValueError: pass
        else: raise AssertionError('invalid IDs admitted')
    print('PASS: train-only moments, missing/zero variance, float16 finite, exact ID mapping and duplicates',flush=True)


def smoke(root,out):
    fixed_env();import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    N=E.trainer(root/'data/mlp_split_20261005/inputs/fold_1_7/src');rng=np.random.default_rng(197)
    x=rng.normal(size=(4096,270)).astype(np.float32);x[:,-20:]=x[:,-20:]>0
    cont,scalar,bins=E.numeric_bins(x);card=[11,49,1509,36,61,2189,155,6,10,7,446,196,586,887]
    xn=torch.from_numpy(x).to('cuda:0');xc=torch.zeros((4096,len(card)),dtype=torch.int64,device='cuda:0')
    v=torch.randn((4096,2,64,12),dtype=torch.float16,device='cuda:0')
    k=torch.ones_like(v,dtype=torch.uint8);p=torch.ones((4096,2,64),dtype=torch.uint8,device='cuda:0')
    p[:8]=0;p[8:16,1]=0;p[16:24,:,32:]=0;k[24:32,:,:,4]=0
    # Reserve exact live-table bytes so the full-batch smoke measures real VRAM demand.
    reserve=torch.empty((2085047*2*64*37,),dtype=torch.uint8,device='cuda:0');checks=[];counts=[]
    for arm in ARMS:
        torch.manual_seed(197);net=network(N,card,cont,scalar,bins,270,arm).to('cuda:0')
        counts.append(sum(x.numel() for x in net.parameters()));opt=torch.optim.AdamW(net.parameters(),lr=.002)
        torch.cuda.reset_peak_memory_stats();start=None
        for step in range(6):
            if step==2:torch.cuda.synchronize();start=time.monotonic()
            prediction=net(xn,xc,v,k,p);loss=prediction.square().mean();opt.zero_grad();loss.backward();opt.step()
            if not torch.isfinite(prediction).all() or any(x.grad is not None and not torch.isfinite(x.grad).all() for x in net.parameters()):
                raise ValueError('synthetic finite/gradient failure')
        torch.cuda.synchronize();seconds=time.monotonic()-start;net.eval()
        with torch.no_grad():
            ix=torch.arange(32,device='cuda:0');base=net(xn[ix],xc[ix],v[ix],k[ix],p[ix])
            vk=v[ix].clone();vk[k[ix]==0]=1234;vk[p[ix]==0]=1234
            if not torch.allclose(base,net(xn[ix],xc[ix],vk,k[ix],p[ix]),atol=1e-6,rtol=1e-5): raise ValueError('unknown/PAD values affect predictions')
            perm=torch.randperm(64,device='cuda:0')
            swapped=net(xn[ix],xc[ix],v[ix][:,:,perm],k[ix][:,:,perm],p[ix][:,:,perm])
            invariant=torch.allclose(base,swapped,atol=1e-5,rtol=1e-4)
            if arm=='flat' and not invariant: raise ValueError('flat permutation invariance failed')
            if arm=='ordered' and invariant: raise ValueError('ordered encoder insensitive to shuffled adjacency')
            h=torch.zeros((1,2,64,64),device='cuda:0');h[:,0,-1]=1
            if triplets(h,torch.ones((1,2,64),dtype=torch.bool,device='cuda:0'),'ordered')[:,1].any():
                raise ValueError('local mixing crossed visit boundary')
        checks.append(dict(arm=arm,parameters=counts[-1],seconds_per_batch=seconds/4,
                           peak_gpu_bytes=torch.cuda.max_memory_allocated(),flat_permutation_invariant=bool(invariant)))
        del net,opt,prediction,loss;gc.collect();torch.cuda.empty_cache()
    if len(set(counts))!=1:raise ValueError('matched capacity differs')
    del reserve
    C.write_json(out,dict(passed=True,synthetic_only=True,worker_sha256=C.sha(__file__),runtime=E.package_runtime(),checks=checks))
    print(json.dumps(checks),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['selftest','smoke','pack','prepare-fold','verify','run','evaluate'])
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--input',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--prepared',type=Path);p.add_argument('--smoke',type=Path)
    p.add_argument('--fold',type=int);p.add_argument('--registration-commit');a=p.parse_args();root=a.root.resolve()
    if a.action=='selftest':selftest();return
    if a.action=='smoke':smoke(root,a.output.resolve());return
    if not re.fullmatch('[0-9a-f]{40}',a.registration_commit or ''):p.error('full registration required')
    if a.action=='pack':pack(root,a.output.resolve(),a.registration_commit)
    elif a.action=='prepare-fold':prepare_fold(root,a.input.resolve(),a.output.resolve(),a.registration_commit,a.fold)
    elif a.action=='verify':
        m=verify_bundle(root,a.input.resolve(),a.registration_commit,queries=True)
        for f in FOLDS:verify_prepared(a.input.resolve(),a.prepared.resolve()/f'fold_{f}_{f+6}',m,f)
        print('PASS bundle + both prepared folds + original query tensors',flush=True)
    elif a.action=='run':run(root,a.input.resolve(),a.prepared.resolve(),a.output.resolve(),a.registration_commit,a.smoke.resolve())
    else:evaluate(root,a.input.resolve(),a.prepared.resolve(),a.output.resolve(),a.registration_commit)


if __name__=='__main__': main()

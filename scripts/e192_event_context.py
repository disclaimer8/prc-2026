"""E192 fixed two-pair/three-seed event graph vs matched flat event context.

Explicit observable-field allowlist; no BLOCK/TAXITIME/neighbor predictions.
Original immutable MLP query tensors are referenced, never copied or modified.
"""
from __future__ import annotations
import argparse
import functools
import gc
import json
import os
from pathlib import Path
import re
import resource
import shutil
import sys
import tempfile
import time

for k,v in {'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
            'POLARS_MAX_THREADS':'2','POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR':'load_as_storage'}.items():
    os.environ.setdefault(k,v)
import numpy as np
import polars as pl
import e190_numeric_embeddings as E
import mlp_gpu_component_oof as M
C=M.C
PROTOCOL='E192-observed-event-relational-vs-flat-K64-2pair-3seed-8epoch-v1'
FOLDS=(1,4)
ARMS=('flat','graph')
K=64
SOURCES=('e192_event_context.py','e190_numeric_embeddings.py','mlp_gpu_component_oof.py',
         'mlp_lowmem_prepare.py','cat_fin1b_oof.py')
RAW_COLS=('MVT_ID_mvt','PHASE_mvt','ADEP_mvt','ADES_mvt','MVT_TIME_UTC_mvt',
          'RUNWAY_mvt','STAND_mvt','AIRCRAFT_TYPE_mvt','WK_TBL_CAT_flt',
          'AIRCRAFT_OPERATOR_flt','AOBT_3_flt','LOBT_flt','SCHED_TIME_UTC_mvt')
CATS=('airport','PHASE_mvt','RUNWAY_mvt','STAND_mvt','AIRCRAFT_TYPE_mvt',
      'WK_TBL_CAT_flt','AIRCRAFT_OPERATOR_flt')
CONFIG=dict(folds=list(FOLDS),arms=list(ARMS),seeds=[0,1,2],epochs=8,batch=4096,
            width=512,bins=48,d_embedding=12,ple_version='B',activation=False,
            context_width=64,category_width=8,neighbors=64,half_window_seconds=3600,
            time_edge_seconds=120,category_min_count=20,lr=.002,weight_decay=.00001,
            optimizer='AdamW',schedule='OneCycleLR',pct_start=.1)


def fixed_env():
    for k,v in M.FIXED_ENV.items():os.environ[k]=v
    os.environ['FEATURES']='__immutable_query_tensors_only__'


def exact_ids(wanted, actual):
    wanted=np.asarray(wanted);actual=np.asarray(actual)
    if not C.integral_ids(wanted) or not C.integral_ids(actual) or len(np.unique(actual))!=len(actual):
        raise ValueError('invalid/duplicate IDs')
    order=np.argsort(actual); ix=np.searchsorted(actual[order],wanted)
    if (ix>=len(order)).any() or not np.array_equal(actual[order[ix]],wanted):raise ValueError('missing IDs')
    return order[ix]


def neighbors(query_positions, times, airports):
    """32 preceding/32 following in stable airport,time,ID order; PAD=0."""
    qp=np.asarray(query_positions,dtype=np.int64)
    offsets=np.r_[np.arange(-32,0),np.arange(1,33)]
    ix=qp[:,None]+offsets
    valid=(ix>=1)&(ix<len(times))
    clipped=np.clip(ix,0,len(times)-1)
    valid &= airports[clipped]==airports[qp,None]
    valid &= np.abs(times[clipped]-times[qp,None])<=3600
    return np.where(valid,clipped,0).astype(np.int32)


def categorical_codes(values, fit_mask):
    """PAD0, unknown/null1, sorted train-only categories from frequency>=20."""
    vc=pl.DataFrame({'v':values}).filter(pl.Series(fit_mask)).drop_nulls().group_by('v').len()
    vocab=vc.filter(pl.col('len')>=20)['v'].sort().to_list()
    mp={v:i+2 for i,v in enumerate(vocab)}
    result=np.fromiter((mp.get(v,1) for v in values),dtype=np.int32,count=len(values))
    return result,vocab


def observable_table(paths):
    # Selecting only this allowlist prevents accidental target access in tokens.
    frames=[pl.read_parquet(p,columns=list(RAW_COLS)) for p in paths]
    d=pl.concat(frames,how='vertical_relaxed')
    d=d.with_columns(pl.when(pl.col('PHASE_mvt')=='DEP').then(pl.col('ADEP_mvt'))
                    .otherwise(pl.col('ADES_mvt')).alias('airport'))
    d=d.filter(pl.col('airport').is_in(
        ['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LTFM','LSZH']))
    if not C.integral_ids(d[C.ID].to_numpy()) or d[C.ID].n_unique()!=d.height or d['MVT_TIME_UTC_mvt'].null_count():raise ValueError('raw identity/time invalid')
    if not d['PHASE_mvt'].is_in(['DEP','ARR']).all():raise ValueError('unexpected movement phase')
    unknown=['','UNKNOWN','UNK','NA','N/A','NONE','NULL','-','?','??','???']
    d=d.with_columns([pl.when(pl.col(c).str.strip_chars().str.to_uppercase().is_in(unknown))
                       .then(None).otherwise(pl.col(c)).alias(c) for c in CATS if c not in ('airport','PHASE_mvt')])
    return d.sort(['airport','MVT_TIME_UTC_mvt',C.ID])


def availability(d):
    rows=[]
    for key,f in d.group_by(['airport','PHASE_mvt'],maintain_order=True):
        rows.append(dict(airport=key[0],phase=key[1],rows=f.height,
                         missing_fraction={c:float(f[c].null_count()/f.height) for c in RAW_COLS if c not in (C.ID,'ADEP_mvt','ADES_mvt')}))
    return rows


def verify_query_order(fit,pred,cols,categorical,xnf,xnp):
    """Reconstruct eight early numeric tensor columns over all rows, no fitting."""
    j=0
    for c in cols:
        if c in categorical:continue
        a=fit[c].cast(pl.Float64).to_numpy();b=pred[c].cast(pl.Float64).to_numpy()
        ok=np.isfinite(a)
        if ok.sum()<100:continue
        q=np.unique(np.quantile(a[ok],np.linspace(0,1,257)))
        if len(q)<2:continue
        for values,expected in ((a,xnf),(b,xnp)):
            v=np.where(np.isfinite(values),np.interp(values,q,np.linspace(-2,2,len(q))),0).astype(np.float32)
            if not np.array_equal(v,expected[:,j]):raise ValueError('query tensor row/order mismatch: '+c)
        j+=1
        if not ok.all():
            if not np.array_equal((~ok).astype(np.float32),xnf[:,j]) or not np.array_equal((~np.isfinite(b)).astype(np.float32),xnp[:,j]):
                raise ValueError('query missing-indicator row/order mismatch: '+c)
            j+=1
        if j>=8:return j
    raise ValueError('need eight independent early numeric query columns for alignment proof')


def pack(root,out,registration):
    if out.exists():raise ValueError('fresh immutable pack destination required')
    if C.sha(root/'data/features_fin1b.parquet')!=M.FEATURE_SHA:raise ValueError('FIN1b SHA differs')
    if shutil.disk_usage(root).free<4*1024**3:raise ValueError('need >=4GiB pack headroom')
    paths=sorted((root/'data/prc-2026-datasets').glob('training_*.parquet'))
    if len(paths)!=12:raise ValueError('all12 training raw files required')
    hashes={str(p.relative_to(root)):C.sha(p) for p in paths}
    rank_path=root/'data/prc-2026-datasets/ranking.parquet'
    hashes[str(rank_path.relative_to(root))]=C.sha(rank_path)
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent));(stage/'scripts').mkdir()
    for n in SOURCES:shutil.copy2(root/'scripts'/n,stage/'scripts'/n)
    print('Reading allowlisted movement observations; labels are not token inputs',flush=True)
    d=observable_table(paths);rank=observable_table([rank_path])
    av={'training':availability(d),'ranking':availability(rank)};del rank;gc.collect()
    n=d.height; ids=np.r_[0,d[C.ID].to_numpy().astype(np.int64)]
    times=np.r_[0,d['MVT_TIME_UTC_mvt'].dt.epoch('s').to_numpy()].astype(np.int64)
    airport_list=d['airport'].unique().sort().to_list(); amap={a:i+1 for i,a in enumerate(airport_list)}
    airports=np.r_[0,np.fromiter((amap[a] for a in d['airport']),dtype=np.int16,count=n)]
    months=d['MVT_TIME_UTC_mvt'].dt.month().to_numpy()
    keys=np.zeros((n+1,3),np.int32)
    # Identity codes used ONLY for exact observed equality, not embeddings/statistics.
    for j,c in enumerate(('RUNWAY_mvt','STAND_mvt','AIRCRAFT_OPERATOR_flt')):
        vals=d[c].to_list(); seen=sorted({v for v in vals if v is not None});mp={v:i+1 for i,v in enumerate(seen)}
        keys[1:,j]=np.fromiter((mp.get(v,0) for v in vals),dtype=np.int32,count=n)
    nums=np.zeros((n+1,6),np.float32)
    for j,c in enumerate(('AOBT_3_flt','LOBT_flt','SCHED_TIME_UTC_mvt')):
        diff=(d['MVT_TIME_UTC_mvt']-d[c]).dt.total_seconds().to_numpy().astype(float)
        ok=np.isfinite(diff);nums[1:,2*j]=np.where(ok,np.clip(diff,-7200,7200)/7200,0)
        nums[1:,2*j+1]=ok
    for name,a in dict(event_ids=ids,event_times=times,event_airports=airports,event_keys=keys,event_nums=nums).items():
        np.save(stage/(name+'.npy'),a,allow_pickle=False)
    fixed_env(); M.runtime()  # torch precedes any trainer/LightGBM import.
    sys.path.insert(0,str(root/'src'));import e100_nn_partner as N
    # Own-row labels only reconstruct the unchanged original query population.
    frame=pl.read_parquet(root/'data/features_fin1b.parquet').filter(pl.col('split')=='train')
    if frame.height!=2085047:raise ValueError('full training coverage differs')
    allids=frame[C.ID].to_numpy();qpos=exact_ids(allids,ids)
    if not (d['PHASE_mvt'].to_numpy()[qpos-1]=='DEP').all():raise ValueError('query is not departure')
    if not np.array_equal(times[qpos],frame['MVT_TIME_UTC_mvt'].dt.epoch('s').to_numpy()):raise ValueError('query event time mismatch')
    np.save(stage/'query_event.npy',qpos.astype(np.int32),allow_pickle=False)
    nb=np.lib.format.open_memmap(stage/'neighbors.npy',mode='w+',dtype=np.int32,shape=(len(allids),K))
    counts=np.zeros(len(allids),np.int16)
    for i in range(0,len(allids),32768):
        z=neighbors(qpos[i:i+32768],times,airports);nb[i:i+len(z)]=z;counts[i:i+len(z)]=(z>0).sum(1)
    nb.flush();del nb
    np.save(stage/'query_ids.npy',allids.astype(np.float64),allow_pickle=False)
    av['train_context']={'queries':len(allids),'empty':int((counts==0).sum()),'full64':int((counts==64).sum()),'median':float(np.median(counts))}
    fold_specs=[]
    for a in FOLDS:
        name=f'fold_{a}_{a+6}';fd=stage/name;fd.mkdir()
        qb=root/E.WORK/'inputs'/name;bm=M.verify_bundle(qb,E.OLD_REGISTRATION);s=bm['spec']
        fit=frame.filter(~pl.col('month').is_in([a,a+6])); pred=frame.filter(pl.col('month').is_in([a,a+6]))
        lab=N.T.regime_label(fit);trainids=fit[C.ID].to_numpy()[lab==0];predids=pred[C.ID].to_numpy()
        xnf=np.load(qb/'xnf.npy',mmap_mode='r');xnp=np.load(qb/'xnp.npy',mmap_mode='r')
        if len(trainids)!=len(xnf) or len(predids)!=len(xnp):raise ValueError('original query row population mismatch')
        order_checked=verify_query_order(fit.filter(pl.Series(lab==0)),pred,N.T.feature_columns(fit),N.T.CATEGORICAL,xnf,xnp)
        tridx=exact_ids(trainids,allids).astype(np.int32);pidx=exact_ids(predids,allids).astype(np.int32)
        np.save(fd/'train_indices.npy',tridx,allow_pickle=False);np.save(fd/'pred_indices.npy',pidx,allow_pickle=False)
        ec=np.zeros((n+1,len(CATS)),np.int32);vocabs={};card=[]
        fitmask=~np.isin(months,[a,a+6])
        for j,c in enumerate(CATS):
            code,vocab=categorical_codes(d[c].to_list(),fitmask);ec[1:,j]=code;vocabs[c]=vocab;card.append(len(vocab)+2)
        np.save(fd/'event_cat.npy',ec,allow_pickle=False);C.write_json(fd/'vocab.json',vocabs)
        idx=np.load(qb/'control_indices.npy',allow_pickle=False)
        with np.load(qb/'metadata.npz',allow_pickle=False) as z:meta=dict(z)
        if not np.array_equal(predids[idx],meta['mvt_id']) or not np.array_equal(pred[C.TARGET].to_numpy()[idx],meta['y']):
            raise ValueError('heldout tensor IDs/targets mismatch')
        tail=(pred['ADEP_mvt']=='LIRF').to_numpy() & pred['FLIGHT_ID_mvt'].is_null().to_numpy()
        decided=~(tail & (pred['d_mvt_sched'].to_numpy()>=14400))
        np.savez(fd/'evaluation.npz',**meta,airport=pred['ADEP_mvt'].to_numpy()[idx].astype('U4'),
                 has_nm=pred['FLIGHT_ID_mvt'].is_not_null().to_numpy()[idx],decided=decided[idx],
                 day=(pred['MVT_TIME_UTC_mvt'].dt.epoch('s').to_numpy()[idx]//86400))
        ctrl=root/'data/e190_inputs_20261005'/f'{name}.npz';ple=root/'data/e190_numemb_20261005'/name/'prediction.npz'
        shutil.copy2(ctrl,fd/'control.npz');shutil.copy2(ple,fd/'ple.npz')
        fold_specs.append(dict(fold=a,query_bundle=str(qb.relative_to(root)),query_spec_sha256=bm['spec_sha256'],
                               train_rows=len(tridx),pred_rows=len(pidx),event_card=card,
                               heldout_unknown_fraction={c:float((ec[1:,j][~fitmask]==1).mean()) for j,c in enumerate(CATS)},
                               train_query_id_sha256=C.digest(trainids.tolist()),alignment_numeric_columns_checked=order_checked))
        print(f'Packed {name}: {len(tridx)} train, {len(pidx)} validation; categories train-only',flush=True)
    C.write_json(stage/'availability.json',av)
    spec=dict(protocol=PROTOCOL,registration_commit=registration,configuration=CONFIG,
              feature_sha256=M.FEATURE_SHA,source_raw_sha256=hashes,
              script_sha256={n:C.sha(stage/'scripts'/n) for n in SOURCES},
              query_registration=E.OLD_REGISTRATION,folds=fold_specs,events=n,
              files={str(p.relative_to(stage)):C.sha(p) for p in stage.rglob('*') if p.is_file()})
    for p,h in hashes.items():
        if C.sha(root/p)!=h:raise ValueError('raw inputs changed while packing')
    C.write_json(stage/'manifest.json',dict(spec=spec,spec_sha256=C.digest(spec)))
    stage.rename(out);print(json.dumps(dict(packed=str(out),events=n,spec_sha256=C.digest(spec))),flush=True)


def verify(root,bundle,registration):
    m=C.read_json(bundle/'manifest.json');s=m['spec']
    if (m['spec_sha256']!=C.digest(s) or s['protocol']!=PROTOCOL or s['registration_commit']!=registration
        or s['configuration']!=CONFIG or s['feature_sha256']!=M.FEATURE_SHA
        or [f['fold'] for f in s['folds']]!=list(FOLDS)):
        raise ValueError('protocol drift')
    for n,h in s['script_sha256'].items():
        if C.sha(Path(__file__).parent/n)!=h:raise ValueError('active worker/helper drift '+n)
    for p,h in s['files'].items():
        if C.sha(bundle/p)!=h:raise ValueError('packed input drift '+p)
    for f in s['folds']:
        qm=M.verify_bundle(root/f['query_bundle'],E.OLD_REGISTRATION)
        if qm['spec_sha256']!=f['query_spec_sha256']:raise ValueError('original query tensors drift')
    return m


def network(N, card, continuous, scalar, bins, n_num, event_card, arm):
    import torch
    from torch import nn
    class ContextNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.query=E.net_factory(N.Net,n_num,card,512,continuous,scalar,bins)
            self.query.body[-1]=nn.Identity()
            self.event_emb=nn.ModuleList([nn.Embedding(k,8,padding_idx=0) for k in event_card])
            # 6 observed proxy/missing numbers; delta+8fixed Fourier time; 3query equality flags.
            self.token=nn.Sequential(nn.Linear(8*len(event_card)+18,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
            self.message=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,64))
            self.fusion=nn.Sequential(nn.Linear(256+64,256),nn.SiLU(),nn.Linear(256,1))
            self.arm=arm
        def forward(self,xn,xc,ec,en,dt,keys,qkeys):
            valid=ec[:,:,0]!=0
            equality=((keys==qkeys[:,None,:])&(keys>0)).float()
            tt=dt.float()/3600
            phase=dt.float().unsqueeze(-1)*(2*np.pi/torch.tensor([60.,300.,900.,3600.],device=dt.device))
            numeric=torch.cat([en,tt.unsqueeze(-1),torch.sin(phase),torch.cos(phase),equality],dim=-1)
            token=torch.cat([e(ec[:,:,j]) for j,e in enumerate(self.event_emb)]+[numeric],dim=-1)
            h=self.token(token)*valid.unsqueeze(-1)
            pair=valid[:,:,None]&valid[:,None,:]
            diagonal=torch.eye(K,device=dt.device,dtype=torch.bool).unsqueeze(0)
            pair=pair&~diagonal
            if self.arm=='graph':
                rel=[]
                for j in (0,1):
                    e=(keys[:,:,j,None]==keys[:,None,:,j])&(keys[:,:,j,None]>0)&pair
                    rel.append(e.float()/e.sum(-1,keepdim=True).clamp_min(1))
                e=((dt[:,:,None]-dt[:,None,:]).abs()<=120)&pair
                rel.append(e.float()/e.sum(-1,keepdim=True).clamp_min(1))
                adjacency=(rel[0]+rel[1]+rel[2])/3
            else:
                adjacency=pair.float()/pair.sum(-1,keepdim=True).clamp_min(1)
            messages=torch.bmm(adjacency,h)
            updated=(h+self.message(torch.cat([h,messages],dim=-1)))*valid.unsqueeze(-1)
            pool=updated.sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
            return self.fusion(torch.cat([self.query(xn,xc),pool],dim=1)).squeeze(-1)
    return ContextNet()


def context_tables(bundle,fd,device):
    import torch
    return dict(ec=torch.as_tensor(np.load(fd/'event_cat.npy'),dtype=torch.int64,device=device),
                en=torch.as_tensor(np.load(bundle/'event_nums.npy'),device=device),
                keys=torch.as_tensor(np.load(bundle/'event_keys.npy'),dtype=torch.int64,device=device),
                times=torch.as_tensor(np.load(bundle/'event_times.npy'),dtype=torch.int64,device=device),
                query=np.load(bundle/'query_event.npy',mmap_mode='r'),
                neighbors=np.load(bundle/'neighbors.npy',mmap_mode='r'))


def context_batch(t,qi,device):
    import torch
    ix=torch.as_tensor(np.asarray(t['neighbors'][qi],dtype=np.int64),device=device)
    q=torch.as_tensor(np.asarray(t['query'][qi],dtype=np.int64),device=device)
    dt=t['times'][ix]-t['times'][q,None]
    dt=torch.where(ix>0,dt,torch.zeros_like(dt))
    return t['ec'][ix],t['en'][ix],dt,t['keys'][ix],t['keys'][q]


def fit_predict(root,bundle,fd,fs,out,arm,seed,runtime):
    import torch
    from torch import nn
    qb=root/fs['query_bundle'];qm=M.verify_bundle(qb,E.OLD_REGISTRATION);s=qm['spec']
    arrays,meta=M.load_arrays(qb,s)
    tridx=np.load(fd/'train_indices.npy');pidx=np.load(fd/'pred_indices.npy')
    cont,scalar,bins=E.numeric_bins(arrays['xnf'])
    t=context_tables(bundle,fd,'cuda:0')
    N=E.trainer(qb/'src');torch.manual_seed(seed);np.random.seed(seed)
    net=network(N,s['card'],cont,scalar,bins,arrays['xnf'].shape[1],fs['event_card'],arm).to('cuda:0')
    opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.00001)
    bs=4096;n=len(tridx);sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.002,total_steps=8*(n//bs),pct_start=.1)
    xn=torch.from_numpy(arrays['xnf']);xc=torch.from_numpy(arrays['xcf']);y=torch.from_numpy(arrays['y_scaled'].astype(np.float32))
    rng=np.random.default_rng(seed);started=time.monotonic(); losses=[]
    for ep in range(8):
        net.train();perm=rng.permutation(n);total=0.
        for batch in range(n//bs):
            ix=perm[batch*bs:(batch+1)*bs];c=context_batch(t,tridx[ix],'cuda:0')
            pred=net(xn[ix].to('cuda:0'),xc[ix].to('cuda:0'),*c)
            loss=nn.functional.mse_loss(pred,y[ix].to('cuda:0'))
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            opt.zero_grad();loss.backward();opt.step();sched.step();total+=loss.item()
        losses.append(total/(n//bs));print(f'fold{fs["fold"]} {arm} seed{seed} epoch{ep+1}/8 mse={losses[-1]:.6f}',flush=True)
    net.eval();chunks=[]
    with torch.no_grad():
        for i in range(0,len(pidx),bs):
            c=context_batch(t,pidx[i:i+bs],'cuda:0')
            chunks.append(net(torch.from_numpy(arrays['xnp'][i:i+bs]).to('cuda:0'),
                              torch.from_numpy(arrays['xcp'][i:i+bs]).to('cuda:0'),*c).cpu().numpy())
    prediction=np.concatenate(chunks)
    if prediction.shape!=(len(pidx),) or not np.isfinite(prediction).all():raise ValueError('prediction incomplete/nonfinite')
    np.save(out/f'seed_{seed}.npy',prediction,allow_pickle=False);torch.save(net.state_dict(),out/f'model_{seed}.pt')
    if seed==0:np.savez(out/'bins.npz',continuous=cont,scalar=scalar,**{f'bin_{j}':b for j,b in enumerate(bins)})
    result=dict(seed=seed,seconds=time.monotonic()-started,parameters=sum(p.numel() for p in net.parameters()),losses=losses,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    del net,opt,sched,arrays,t,xn,xc,y,chunks;gc.collect();torch.cuda.empty_cache()
    return result


def run(root,bundle,out,registration,smoke_path):
    fixed_env();cgroup=M.verify_cgroup()
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if out.exists():raise ValueError('fresh output required; no auto-resume')
    m=verify(root,bundle,registration);sm=C.read_json(smoke_path);runtime=E.package_runtime()
    if not sm['passed'] or sm['worker_sha256']!=C.sha(__file__) or sm['runtime']!=runtime:raise ValueError('synthetic smoke/runtime drift')
    out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent))
    completed=[]
    for fs in m['spec']['folds']:
        name=f'fold_{fs["fold"]}_{fs["fold"]+6}';fd=bundle/name
        qm=M.verify_bundle(root/fs['query_bundle'],E.OLD_REGISTRATION);s=qm['spec']
        for arm in ARMS:
            od=stage/name/arm;od.mkdir(parents=True);fits=[]
            for seed in (0,1,2):fits.append(fit_predict(root,bundle,fd,fs,od,arm,seed,runtime))
            seeds=[np.load(od/f'seed_{i}.npy') for i in (0,1,2)]
            base=np.load(root/fs['query_bundle']/'base_p.npy');idx=np.load(root/fs['query_bundle']/'control_indices.npy')
            member=(np.mean(seeds,axis=0)*s['sd']+base)[idx]
            ctrl=E.read(fd/'control.npz')
            cand=E.assemble(ctrl,ctrl['reg']+ctrl['mlp_coefficient']*(member-ctrl['plain']),ctrl['lobt'])
            np.savez_compressed(od/'prediction.npz',**{k:cand[k] for k in ('mvt_id','y','month','pred','reg','values','proba')},member=member)
            receipt=dict(fold=fs['fold'],arm=arm,seeds=[0,1,2],fits=fits,
                         files={p.name:C.sha(p) for p in od.iterdir() if p.is_file()})
            C.write_json(od/'manifest.json',receipt);completed.append(receipt)
        if len({f['parameters'] for r in completed[-2:] for f in r['fits']})!=1:raise ValueError('arms capacity differs')
    verify(root,bundle,registration)
    C.write_json(stage/'complete.json',dict(protocol=PROTOCOL,registration_commit=registration,input_spec_sha256=m['spec_sha256'],
                                          runtime=runtime,cgroup=cgroup,arms=completed,accepted_for_submission=False))
    stage.rename(out);print('COMPLETE12fits; no quality selection or submission acceptance',flush=True)


def metrics(ctrl,pred,meta):
    y=meta['y'];ec=(ctrl-y)**2;ep=(pred-y)**2;delta=ep-ec;order=np.sort(delta)
    result={'rows':len(y),'rmse_control':float(np.sqrt(ec.mean())),'rmse_candidate':float(np.sqrt(ep.mean())),
            'all_sse':float(delta.sum()),'body_sse':float(delta[y<=7200].sum()),
            'decided_sse':float(delta[meta['decided']].sum()),
            'trim3_sse':float(order[3:-3].sum()),'trim20_sse':float(order[20:-20].sum())}
    lm=(meta['airport']=='LTFM')&meta['has_nm']&(y<=7200)
    result['ltfm_nm_body_sse']=float(delta[lm].sum());result['ltfm_nm_body_rows']=int(lm.sum())
    groups=np.char.add(np.char.add(meta['airport'].astype('U4'),':'),meta['day'].astype('U16'))
    _,ix=np.unique(groups,return_inverse=True);n=np.bincount(ix);a=np.bincount(ix,weights=ec);b=np.bincount(ix,weights=ep)
    rng=np.random.default_rng(192);deltas=[]
    for _ in range(1000):
        sample=rng.integers(0,len(n),len(n));den=n[sample].sum()
        deltas.append(np.sqrt(b[sample].sum()/den)-np.sqrt(a[sample].sum()/den))
    result['airport_day_bootstrap_rmse_95']=np.quantile(deltas,[.025,.975]).tolist()
    result['months']={str(month):float(delta[meta['month']==month].sum()) for month in np.unique(meta['month'])}
    return result


def evaluate(root,bundle,out,registration):
    m=verify(root,bundle,registration);comp=C.read_json(out/'complete.json')
    if (comp['protocol']!=PROTOCOL or comp['registration_commit']!=registration
        or comp['input_spec_sha256']!=m['spec_sha256'] or len(comp['arms'])!=4):raise ValueError('incomplete experiment')
    reports=[]
    for fs in m['spec']['folds']:
        name=f'fold_{fs["fold"]}_{fs["fold"]+6}';fd=bundle/name;meta=E.read(fd/'evaluation.npz');pred={}
        for arm in ARMS:
            od=out/name/arm;r=C.read_json(od/'manifest.json')
            if r not in comp['arms'] or r['seeds']!=[0,1,2]:raise ValueError('arm receipt differs')
            for p,h in r['files'].items():
                if C.sha(od/p)!=h:raise ValueError('result hash drift')
            z=E.read(od/'prediction.npz')
            for k in ('mvt_id','y','month'):
                if not np.array_equal(z[k],meta[k]):raise ValueError('result heldout metadata differs')
            qb=root/fs['query_bundle'];s=M.verify_bundle(qb,E.OLD_REGISTRATION)['spec']
            expected=(np.mean([np.load(od/f'seed_{i}.npy') for i in (0,1,2)],axis=0)*s['sd']+np.load(qb/'base_p.npy'))[np.load(qb/'control_indices.npy')]
            if not np.array_equal(z['member'],expected):raise ValueError('three-seed mean reconstruction differs')
            ctrl=E.read(fd/'control.npz');expected=E.assemble(ctrl,ctrl['reg']+ctrl['mlp_coefficient']*(expected-ctrl['plain']),ctrl['lobt'])
            if not np.array_equal(z['pred'],expected['pred']):raise ValueError('screen mixture arithmetic differs')
            pred[arm]=z['pred']
        ple=E.read(fd/'ple.npz')
        if not np.array_equal(ple['mvt_id'],meta['mvt_id']):raise ValueError('PLE control IDs differ')
        reports.append(dict(fold=fs['fold'],graph_vs_flat=metrics(pred['flat'],pred['graph'],meta),
                            graph_vs_ple=metrics(ple['pred'],pred['graph'],meta),flat_vs_ple=metrics(ple['pred'],pred['flat'],meta)))
    def passed(comparison):
        return all(all(r[comparison][k]<0 for k in ('body_sse','decided_sse','trim3_sse','trim20_sse')) for r in reports)
    result=dict(protocol=PROTOCOL,registration_commit=registration,reports=reports,
                relation_screen_pass=passed('graph_vs_flat') and passed('graph_vs_ple'),
                observed_input_screen_pass=passed('flat_vs_ple'),accepted_for_submission=False,
                limitation='Two fixed month pairs; LGB5/pre-ADSB screen. Bootstrap descriptive. No complete-chain CV or leaderboard promise.')
    C.write_json(out/'evaluation.json',result);print(json.dumps(result,indent=2),flush=True)


def selftest():
    times=np.array([0,0,0,100,3600,3601,100]);airports=np.array([0,1,1,1,1,1,2])
    nb=neighbors(np.array([2]),times,airports)[0];nz=nb[nb>0]
    assert 2 not in nz and set(nz)=={1,3,4} and 5 not in nz and 6 not in nz
    assert (neighbors(np.array([6]),times,airports)==0).all()
    v=['fit']*20+['heldout']*21+[None];mask=np.arange(len(v))<20
    code,vocab=categorical_codes(v,mask);assert vocab==['fit'] and np.all(code[20:]==1)
    assert np.array_equal(exact_ids([3,1],[1,2,3]),[2,0])
    for want,actual in (([4],[1,2,3]),([1],[1,1,2])):
        try:exact_ids(want,actual)
        except ValueError:pass
        else:raise AssertionError('invalid ID mapping accepted')
    assert not set(RAW_COLS)&{'BLOCK_TIME_UTC_mvt','TAXITIME_SEC_mvt'}
    print('PASS: self exclusion, exact ties/window edges, airport isolation, empty context, train-only vocab, ID mapping and raw label exclusion')


def smoke(root,out):
    fixed_env();import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    N=E.trainer(root/'src');rng=np.random.default_rng(192)
    xn=rng.normal(size=(4096,270)).astype(np.float32);xn[:,-20:]=xn[:,-20:]>0
    cont,scalar,bins=E.numeric_bins(xn);card=[11,49,1509,36,61,2189,155,6,10,7,446,196,586,887]
    xc=torch.zeros((4096,len(card)),dtype=torch.int64,device='cuda:0');tx=torch.from_numpy(xn).to('cuda:0')
    event_card=[12,4,60,2200,160,10,450]
    ec=torch.ones((4096,64,7),dtype=torch.int64,device='cuda:0');ec[:16]=0
    en=torch.randn((4096,64,6),device='cuda:0');dt=torch.arange(64,device='cuda:0')[None,:].expand(4096,-1)*60-1800
    keys=torch.randint(0,12,(4096,64,3),device='cuda:0');qkeys=keys[:,0,:]
    checks=[];counts=[]
    for arm in ARMS:
        torch.manual_seed(192);net=network(N,card,cont,scalar,bins,270,event_card,arm).to('cuda:0')
        counts.append(sum(p.numel() for p in net.parameters()));opt=torch.optim.AdamW(net.parameters(),lr=.002)
        torch.cuda.reset_peak_memory_stats()
        start=None
        cpu_x=torch.from_numpy(xn);cpu_c=torch.zeros((4096,len(card)),dtype=torch.int64)
        for step in range(14):
            if step==2:
                torch.cuda.synchronize();start=time.monotonic()
            tx=cpu_x.to('cuda:0');xc=cpu_c.to('cuda:0')
            p=net(tx,xc,ec,en,dt,keys,qkeys);loss=p.square().mean();opt.zero_grad();loss.backward();opt.step()
            if not torch.isfinite(p).all() or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                raise ValueError('synthetic finite/gradient failure')
        torch.cuda.synchronize();measured=time.monotonic()-start
        net.eval()
        with torch.no_grad():
            ix=torch.tensor([0,16,17,18],device='cuda:0')
            base=net(tx[ix],xc[ix],ec[ix],en[ix],dt[ix],keys[ix],qkeys[ix])
            permutation=torch.arange(63,-1,-1,device='cuda:0')
            swapped=net(tx[ix],xc[ix],ec[ix][:,permutation],en[ix][:,permutation],dt[ix][:,permutation],keys[ix][:,permutation],qkeys[ix])
            if not torch.allclose(base,swapped,atol=1e-5,rtol=1e-4):raise ValueError('token permutation invariance failure')
        torch.cuda.synchronize();checks.append(dict(arm=arm,seconds_12_batches=measured,seconds_per_batch=measured/12,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),parameters=counts[-1]))
        del net,opt,p,loss;gc.collect();torch.cuda.empty_cache()
    if len(set(counts))!=1:raise ValueError('parameter counts differ')
    C.write_json(out,dict(passed=True,synthetic_only=True,worker_sha256=C.sha(__file__),runtime=E.package_runtime(),checks=checks))
    print(json.dumps(checks),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['selftest','pack','verify','smoke','run','evaluate'])
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--input',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--registration-commit');p.add_argument('--smoke',type=Path);a=p.parse_args();root=a.root.resolve()
    if a.action=='selftest':selftest();return
    if a.action=='smoke':smoke(root,a.output.resolve());return
    if not re.fullmatch('[0-9a-f]{40}',a.registration_commit or ''):p.error('full registration commit required before real fit')
    if a.action=='pack':pack(root,a.output.resolve(),a.registration_commit)
    elif a.action=='verify':print(json.dumps(verify(root,a.input.resolve(),a.registration_commit)['spec_sha256']))
    elif a.action=='run':run(root,a.input.resolve(),a.output.resolve(),a.registration_commit,a.smoke.resolve())
    else:evaluate(root,a.input.resolve(),a.output.resolve(),a.registration_commit)


if __name__=='__main__':main()

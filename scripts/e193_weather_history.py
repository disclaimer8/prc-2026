"""E193 fixed METAR history versus repeated-current matched temporal encoder.

Uses only archived observable weather and own-row proxy time. Query tensors are
immutable references. No target-derived neighbor inputs or adaptive model search.
"""
from __future__ import annotations
import argparse
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
import e192_event_context as D
import mlp_gpu_component_oof as M
C=M.C
PROTOCOL='E193-metar6h12bins-history-v-current-3seed8epoch-conditional6pair-v1'
SCREEN=(1,4)
REMAINING=(2,3,5,6)
FOLDS=SCREEN+REMAINING
ARMS=('current','history')
AIRPORTS=('EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LTFM','LSZH')
WXCOLS=('station','valid','tmpc','dwpc','sknt','gust','vsby','skyl1','wxcodes')
SOURCES=('e193_weather_history.py','e192_event_context.py','e190_numeric_embeddings.py',
         'mlp_gpu_component_oof.py','mlp_lowmem_prepare.py','cat_fin1b_oof.py')
CONFIG=dict(screen=list(SCREEN),conditional_remaining=list(REMAINING),arms=list(ARMS),
            seeds=[0,1,2],epochs=8,batch=4096,bins=48,d_embedding=12,ple_version='B',
            activation=False,width=512,weather_bins=12,bin_seconds=1800,
            current_tolerance_seconds=10800,weather_width=32,temporal_kernel=3,
            temporal_layers=2,lr=.002,weight_decay=.00001,optimizer='AdamW',
            schedule='OneCycleLR',pct_start=.1,normalization='fixed physical ranges',
            control='same reporting cadence; repeat latest current values and field masks',
            anchor='AOBT_3_flt or MVT_TIME_UTC_mvt minus900seconds')


def fixed_env():
    D.fixed_env()


def weather_table(root):
    arrays=[];times=[];groups={};availability=[];offset=1
    for ap in AIRPORTS:
        p=root/'data/metar'/f'{ap}.csv'
        d=pl.read_csv(p,columns=list(WXCOLS),infer_schema_length=0,null_values=['null','M'])
        if d['station'].unique().to_list()!=[ap]:raise ValueError('weather station differs')
        d=d.with_columns(pl.col('valid').str.to_datetime('%Y-%m-%d %H:%M',time_zone='UTC').alias('time')).sort('time')
        t=d['time'].dt.epoch('s').to_numpy()
        if len(np.unique(t))!=len(t) or d['time'].null_count():raise ValueError('duplicate/invalid METAR timestamps')
        raw=np.column_stack([d[c].cast(pl.Float64).to_numpy() for c in ('tmpc','dwpc','sknt','gust','vsby','skyl1')])
        raw[:,1]=raw[:,0]-raw[:,1]
        wx=d['wxcodes'].fill_null('')
        flags=np.column_stack([wx.str.contains(x).to_numpy().astype(float) for x in ('SN|SG|PL|GS','FZ','RA|DZ|SH','TS|CB','FG|BR|HZ')])
        raw=np.column_stack([raw,flags]);valid=np.isfinite(raw)
        low=np.array([-50,0,0,0,0,0,0,0,0,0,0]);high=np.array([50,50,100,100,20,30000,1,1,1,1,1])
        values=np.where(valid,np.clip(raw,low,high)/high,0)
        arrays.append(np.column_stack([values,valid]).astype(np.float32));times.append(t)
        groups[ap]=(offset,offset+len(t));offset+=len(t)
        availability.append(dict(airport=ap,rows=len(t),first=str(d['time'].min()),last=str(d['time'].max()),
            rows2025=int((d['time'].dt.year()==2025).sum()),
            rank_month_rows=int(((d['time'].dt.year()==2026)&d['time'].dt.month().is_in([1,7])).sum()),
            field_missing_fraction={c:float(d[c].null_count()/len(d)) for c in WXCOLS[2:]}))
    return np.vstack([np.zeros((1,22),np.float32),*arrays]),np.r_[0,*times].astype(np.int64),groups,availability


def select_history(anchors,times,offset):
    """Latest observation in each disjoint (left,right] 30min interval; no carry."""
    a=np.asarray(anchors,dtype=np.int64)
    ends=a[:,None]-np.arange(11,-1,-1,dtype=np.int64)[None,:]*1800
    ix=np.searchsorted(times,ends,side='right')-1;safe=np.maximum(ix,0)
    ok=(ix>=0)&(times[safe]>ends-1800)&(times[safe]<=ends)
    history=np.where(ok,ix+offset,0).astype(np.int32)
    ci=np.searchsorted(times,a,side='right')-1;safe=np.maximum(ci,0)
    co=(ci>=0)&(a-times[safe]<=10800)&(a-times[safe]>=0)
    current=np.where(co,ci+offset,0).astype(np.int32)
    return history,current


def context_indices(frame,times,groups):
    if frame['MVT_TIME_UTC_mvt'].null_count():raise ValueError('query MVT missing')
    anchor=frame.select(pl.col('AOBT_3_flt').fill_null(pl.col('MVT_TIME_UTC_mvt')-pl.duration(seconds=900)).dt.epoch('s')).to_series().to_numpy()
    airport=frame['ADEP_mvt'].to_numpy();history=np.zeros((len(frame),12),np.int32);current=np.zeros(len(frame),np.int32);stats=[]
    for ap,(lo,hi) in groups.items():
        ix=np.flatnonzero(airport==ap);h,c=select_history(anchor[ix],times[lo:hi],lo);history[ix]=h;current[ix]=c
        count=(h>0).sum(1)
        stats.append(dict(airport=ap,queries=len(ix),current_missing=int((c==0).sum()),empty=int((count==0).sum()),full12=int((count==12).sum()),mean_bins=float(count.mean())))
    if not np.isin(airport,AIRPORTS).all():raise ValueError('unknown query airport')
    valid=history>0
    if np.any((anchor[:,None]-times[history])[valid]<0) or np.any((anchor[:,None]-times[history])[valid]>=21600):raise ValueError('future/outside6h observation')
    return history,current,anchor.astype(np.int64),stats


def pack(root,out,registration):
    if out.exists():raise ValueError('fresh immutable pack destination required')
    fp=root/'data/features_fin1b.parquet'
    if C.sha(fp)!=M.FEATURE_SHA:raise ValueError('FIN1b source differs')
    if shutil.disk_usage(root).free<2*1024**3:raise ValueError('need2GiB disk headroom')
    hashes={str(p.relative_to(root)):C.sha(p) for p in sorted((root/'data/metar').glob('*.csv'))}
    if len(hashes)!=10:raise ValueError('all10 METAR stations required')
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent));(stage/'scripts').mkdir()
    for n in SOURCES:shutil.copy2(root/'scripts'/n,stage/'scripts'/n)
    values,times,groups,av=weather_table(root)
    cols=[C.ID,'ADEP_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt','split']
    queries=pl.read_parquet(fp,columns=cols)
    train=queries.filter(pl.col('split')=='train');rank=queries.filter(pl.col('split')=='rank')
    if train.height!=2085047 or rank.height!=344841:raise ValueError('train/rank coverage differs')
    h,c,anchors,trcov=context_indices(train,times,groups);rh,rc,ra,rkcov=context_indices(rank,times,groups)
    del rh,rc,ra,rank,queries;gc.collect()
    ids=train[C.ID].to_numpy()
    if not C.integral_ids(ids) or len(np.unique(ids))!=len(ids):raise ValueError('invalid query IDs')
    for n,a in dict(weather=values,weather_times=times,history_indices=h,current_indices=c,anchors=anchors,query_ids=ids).items():np.save(stage/(n+'.npy'),a,allow_pickle=False)
    C.write_json(stage/'availability.json',dict(stations=av,training=trcov,ranking=rkcov))
    del train,h,c,anchors,values,times;gc.collect()
    fixed_env();M.runtime()  # frozen trainer bootstrap only; no fitting during pack.
    sys.path.insert(0,str(root/'src'));import e100_nn_partner as N
    frame=pl.read_parquet(fp).filter(pl.col('split')=='train')
    if not np.array_equal(frame[C.ID].to_numpy(),ids):raise ValueError('query row order drift')
    folds=[]
    for fold in FOLDS:
        name=f'fold_{fold}_{fold+6}';fd=stage/name;fd.mkdir();qb=root/E.WORK/'inputs'/name
        qm=M.verify_bundle(qb,E.OLD_REGISTRATION)
        fit=frame.filter(~pl.col('month').is_in([fold,fold+6]));pred=frame.filter(pl.col('month').is_in([fold,fold+6]))
        labels=N.T.regime_label(fit);fit=fit.filter(pl.Series(labels==0))
        trids=fit[C.ID].to_numpy();pids=pred[C.ID].to_numpy()
        xnf=np.load(qb/'xnf.npy',mmap_mode='r');xnp=np.load(qb/'xnp.npy',mmap_mode='r')
        if len(trids)!=len(xnf) or len(pids)!=len(xnp):raise ValueError('query population differs')
        checked=D.verify_query_order(fit,pred,N.T.feature_columns(fit),N.T.CATEGORICAL,xnf,xnp)
        np.save(fd/'train_indices.npy',D.exact_ids(trids,ids).astype(np.int32));np.save(fd/'pred_indices.npy',D.exact_ids(pids,ids).astype(np.int32))
        idx=np.load(qb/'control_indices.npy');meta=E.read(qb/'metadata.npz')
        if not np.array_equal(pids[idx],meta['mvt_id']) or not np.array_equal(pred[C.TARGET].to_numpy()[idx],meta['y']):raise ValueError('heldout metadata differs')
        tail=(pred['ADEP_mvt']=='LIRF').to_numpy()&pred['FLIGHT_ID_mvt'].is_null().to_numpy()
        decided=~(tail&(pred['d_mvt_sched'].to_numpy()>=14400))
        np.savez(fd/'evaluation.npz',**meta,airport=pred['ADEP_mvt'].to_numpy()[idx].astype('U4'),has_nm=pred['FLIGHT_ID_mvt'].is_not_null().to_numpy()[idx],decided=decided[idx],day=pred['MVT_TIME_UTC_mvt'].dt.epoch('s').to_numpy()[idx]//86400)
        shutil.copy2(root/'data/e190_inputs_20261005'/f'{name}.npz',fd/'control.npz')
        shutil.copy2(root/'data/e190_numemb_20261005'/name/'prediction.npz',fd/'ple.npz')
        for source in ('control','ple'):
            z=E.read(fd/(source+'.npz'))
            for k in ('mvt_id','y','month'):
                if not np.array_equal(z[k],meta[k]):raise ValueError('saved control metadata differs')
        folds.append(dict(fold=fold,query_bundle=str(qb.relative_to(root)),query_spec_sha256=qm['spec_sha256'],train_rows=len(trids),pred_rows=len(pids),alignment_numeric_columns_checked=checked))
        print(f'PACKED {name} rows={len(trids)}/{len(pids)}',flush=True)
    for p,h in hashes.items():
        if C.sha(root/p)!=h:raise ValueError('weather source drift')
    spec=dict(protocol=PROTOCOL,registration_commit=registration,configuration=CONFIG,feature_sha256=M.FEATURE_SHA,weather_source_sha256=hashes,script_sha256={n:C.sha(stage/'scripts'/n) for n in SOURCES},folds=folds,files={str(p.relative_to(stage)):C.sha(p) for p in stage.rglob('*') if p.is_file()})
    C.write_json(stage/'manifest.json',dict(spec=spec,spec_sha256=C.digest(spec)));stage.rename(out)
    print(json.dumps(dict(packed=str(out),spec_sha256=C.digest(spec))),flush=True)


def verify(root,bundle,registration):
    m=C.read_json(bundle/'manifest.json');s=m['spec']
    if (m['spec_sha256']!=C.digest(s) or s['protocol']!=PROTOCOL or s['registration_commit']!=registration or s['configuration']!=CONFIG or s['feature_sha256']!=M.FEATURE_SHA or [f['fold'] for f in s['folds']]!=list(FOLDS)):raise ValueError('protocol drift')
    for n,h in s['script_sha256'].items():
        if C.sha(Path(__file__).parent/n)!=h:raise ValueError('active source drift '+n)
    for p,h in s['files'].items():
        if C.sha(bundle/p)!=h:raise ValueError('input drift '+p)
    for fs in s['folds']:
        if M.verify_bundle(root/fs['query_bundle'],E.OLD_REGISTRATION)['spec_sha256']!=fs['query_spec_sha256']:raise ValueError('query tensor drift')
    return m


def network(N,card,continuous,scalar,bins,n_num,arm):
    import torch
    from torch import nn
    class WeatherNet(nn.Module):
        def __init__(self):
            super().__init__();self.arm=arm
            self.query=E.net_factory(N.Net,n_num,card,512,continuous,scalar,bins);self.query.body[-1]=nn.Identity()
            self.token=nn.Sequential(nn.Linear(25,32),nn.SiLU())
            self.temporal=nn.Sequential(nn.Conv1d(32,32,3,padding=1),nn.SiLU(),nn.Conv1d(32,32,3,padding=1),nn.SiLU())
            self.pool=nn.Sequential(nn.Flatten(),nn.Linear(12*32,64),nn.SiLU())
            self.fusion=nn.Sequential(nn.Linear(256+64+24,256),nn.SiLU(),nn.Linear(256,1))
        def forward(self,xn,xc,history,current,aux):
            v=history if self.arm=='history' else current[:,:22].unsqueeze(1).expand(-1,12,-1)
            v=v*aux[:,:,1:2]  # shared observation-bin availability, never labels.
            t=self.token(torch.cat([v,aux],-1)).transpose(1,2)
            h=self.pool(self.temporal(t))
            return self.fusion(torch.cat([self.query(xn,xc),h,current],1)).squeeze(-1)
    return WeatherNet()


def context_tables(bundle,fd,device):
    import torch
    return dict(weather=torch.as_tensor(np.load(bundle/'weather.npy'),device=device),times=torch.as_tensor(np.load(bundle/'weather_times.npy'),device=device),history=np.load(bundle/'history_indices.npy',mmap_mode='r'),current=np.load(bundle/'current_indices.npy',mmap_mode='r'),anchors=np.load(bundle/'anchors.npy',mmap_mode='r'))


def context_batch(t,qi,device):
    import torch
    h=torch.as_tensor(np.asarray(t['history'][qi],dtype=np.int64),device=device);c=torch.as_tensor(np.asarray(t['current'][qi],dtype=np.int64),device=device)
    anchor=torch.as_tensor(np.asarray(t['anchors'][qi]),device=device)
    age=torch.where(h>0,(anchor[:,None]-t['times'][h]).float()/21600,0)
    pos=torch.arange(-11,1,device=device,dtype=torch.float32)[None,:].expand(len(qi),-1)/12
    aux=torch.stack([age,(h>0).float(),pos],-1)
    ca=torch.where(c>0,(anchor-t['times'][c]).float()/10800,0)
    current=torch.cat([t['weather'][c],ca[:,None],(c>0).float()[:,None]],-1)
    return t['weather'][h],current,aux


def fit_predict(root,bundle,fd,fs,out,arm,seed,runtime):
    import torch
    from torch import nn
    qb=root/fs['query_bundle'];qm=M.verify_bundle(qb,E.OLD_REGISTRATION);s=qm['spec']
    arrays,meta=M.load_arrays(qb,s)
    tridx=np.load(fd/'train_indices.npy');pidx=np.load(fd/'pred_indices.npy')
    cont,scalar,bins=E.numeric_bins(arrays['xnf'])
    t=context_tables(bundle,fd,'cuda:0')
    N=E.trainer(qb/'src');torch.manual_seed(seed);np.random.seed(seed)
    net=network(N,s['card'],cont,scalar,bins,arrays['xnf'].shape[1],arm).to('cuda:0')
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

def evaluate(root,bundle,out,registration):
    m=verify(root,bundle,registration);comp=C.read_json(out/'complete.json');folds=comp['folds']
    if (comp['protocol']!=PROTOCOL or comp['registration_commit']!=registration or comp['input_spec_sha256']!=m['spec_sha256'] or folds not in (list(SCREEN),list(REMAINING)) or len(comp['arms'])!=2*len(folds)):raise ValueError('incomplete stage')
    reports=[]
    for fold in folds:
        fs=next(f for f in m['spec']['folds'] if f['fold']==fold);name=f'fold_{fold}_{fold+6}';fd=bundle/name
        meta=E.read(fd/'evaluation.npz');pred={};qb=root/fs['query_bundle'];s=M.verify_bundle(qb,E.OLD_REGISTRATION)['spec']
        for arm in ARMS:
            od=out/name/arm;r=C.read_json(od/'manifest.json')
            if r not in comp['arms'] or r['seeds']!=[0,1,2]:raise ValueError('arm receipt differs')
            for p,h in r['files'].items():
                if C.sha(od/p)!=h:raise ValueError('result hash drift')
            z=E.read(od/'prediction.npz')
            for k in ('mvt_id','y','month'):
                if not np.array_equal(z[k],meta[k]):raise ValueError('result heldout metadata differs')
            expected=(np.mean([np.load(od/f'seed_{i}.npy') for i in (0,1,2)],axis=0)*s['sd']+np.load(qb/'base_p.npy'))[np.load(qb/'control_indices.npy')]
            if not np.array_equal(z['member'],expected) or not np.isfinite(expected).all():raise ValueError('three-seed reconstruction differs')
            ctrl=E.read(fd/'control.npz');expected=E.assemble(ctrl,ctrl['reg']+ctrl['mlp_coefficient']*(expected-ctrl['plain']),ctrl['lobt'])
            if not np.array_equal(z['pred'],expected['pred']) or not np.isfinite(z['pred']).all():raise ValueError('mixture arithmetic differs')
            pred[arm]=z['pred']
        ple=E.read(fd/'ple.npz')
        def metrics(control,candidate):
            r=D.metrics(control,candidate,meta)
            for label,mask in (('body',meta['y']<=7200),('decided',meta['decided'])):
                r[label+'_rmse_control']=float(np.sqrt(np.mean((control[mask]-meta['y'][mask])**2)))
                r[label+'_rmse_candidate']=float(np.sqrt(np.mean((candidate[mask]-meta['y'][mask])**2)))
            return r
        reports.append(dict(fold=fold,history_vs_current=metrics(pred['current'],pred['history']),history_vs_ple=metrics(ple['pred'],pred['history']),current_vs_ple=metrics(ple['pred'],pred['current'])))
    passed=all(all(r[c][k]<0 for c in ('history_vs_current','history_vs_ple') for k in ('body_sse','decided_sse','trim3_sse','trim20_sse')) for r in reports)
    result=dict(protocol=PROTOCOL,registration_commit=registration,folds=folds,reports=reports,weather_history_screen_pass=passed,accepted_for_submission=False,limitation='Fixed LGB5/pre-ADSB screen; matched weather-history contrast; no complete-chain CV or leaderboard promise.')
    C.write_json(out/'evaluation.json',result);print(json.dumps(result,indent=2),flush=True);return result


def run_stage(root,bundle,out,registration,m,runtime,cgroup,folds):
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent));completed=[]
    for fold in folds:
        fs=next(f for f in m['spec']['folds'] if f['fold']==fold);name=f'fold_{fold}_{fold+6}';fd=bundle/name
        s=M.verify_bundle(root/fs['query_bundle'],E.OLD_REGISTRATION)['spec']
        for arm in ARMS:
            od=stage/name/arm;od.mkdir(parents=True);fits=[]
            for seed in (0,1,2):fits.append(fit_predict(root,bundle,fd,fs,od,arm,seed,runtime))
            seeds=[np.load(od/f'seed_{i}.npy') for i in (0,1,2)];qb=root/fs['query_bundle']
            member=(np.mean(seeds,axis=0)*s['sd']+np.load(qb/'base_p.npy'))[np.load(qb/'control_indices.npy')]
            ctrl=E.read(fd/'control.npz');cand=E.assemble(ctrl,ctrl['reg']+ctrl['mlp_coefficient']*(member-ctrl['plain']),ctrl['lobt'])
            np.savez_compressed(od/'prediction.npz',**{k:cand[k] for k in ('mvt_id','y','month','pred','reg','values','proba')},member=member)
            receipt=dict(fold=fold,arm=arm,seeds=[0,1,2],fits=fits,files={p.name:C.sha(p) for p in od.iterdir() if p.is_file()})
            C.write_json(od/'manifest.json',receipt);completed.append(receipt)
        if len({f['parameters'] for r in completed[-2:] for f in r['fits']})!=1:raise ValueError('arms capacity differs')
    verify(root,bundle,registration)
    C.write_json(stage/'complete.json',dict(protocol=PROTOCOL,registration_commit=registration,input_spec_sha256=m['spec_sha256'],runtime=runtime,cgroup=cgroup,folds=list(folds),arms=completed,accepted_for_submission=False))
    stage.rename(out);print(f'COMPLETE STAGE {list(folds)}; exact3seed/8epoch',flush=True)
    return evaluate(root,bundle,out,registration)


def run(root,bundle,out,registration,smoke_path):
    fixed_env();cgroup=M.verify_cgroup()
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if out.exists():raise ValueError('fresh output required; no auto-resume')
    m=verify(root,bundle,registration);sm=C.read_json(smoke_path);runtime=E.package_runtime()
    if not sm['passed'] or sm['worker_sha256']!=C.sha(__file__) or sm['runtime']!=runtime:raise ValueError('synthetic smoke/runtime drift')
    out.mkdir(parents=True);screen=run_stage(root,bundle,out/'screen',registration,m,runtime,cgroup,SCREEN)
    continuation=bool(screen['weather_history_screen_pass']);rest=None
    if continuation:
        print('REGISTERED BOTH-CONTROL SCREEN PASS: starting fixed remaining4pairs',flush=True)
        rest=run_stage(root,bundle,out/'continuation',registration,m,runtime,cgroup,REMAINING)
    else:print('REGISTERED SCREEN NEGATIVE: terminate without tuning or more fits',flush=True)
    C.write_json(out/'complete.json',dict(protocol=PROTOCOL,registration_commit=registration,input_spec_sha256=m['spec_sha256'],screen_evaluation_sha256=C.sha(out/'screen/evaluation.json'),continuation_executed=continuation,continuation_evaluation_sha256=C.sha(out/'continuation/evaluation.json') if continuation else None,all_six_pair_criteria_pass=bool(continuation and rest['weather_history_screen_pass']),fits=36 if continuation else 12,accepted_for_submission=False))
    print('E193 COMPLETE',flush=True)


def selftest():
    t=np.array([0,1799,1800,21600,21601],np.int64)
    h,c=select_history(np.array([21600]),t,1)
    assert h.shape==(1,12) and h[0,0]==3 and h[0,-1]==4 and c[0]==4
    assert 1 not in h and 5 not in h  # left-open6h boundary and future report excluded.
    h,c=select_history(np.array([-1]),t,1);assert not h.any() and not c.any()
    h,c=select_history(np.array([32402]),t,1);assert c[0]==0
    assert not set(WXCOLS)&{'BLOCK_TIME_UTC_mvt','TAXITIME_SEC_mvt'}
    print('PASS: 12disjointbins, endpoints,6hboundary,future exclusion,current3htolerance,empty weather and label-free allowlist')


def smoke(root,out):
    fixed_env();import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    N=E.trainer(root/'src');rng=np.random.default_rng(193)
    xn=rng.normal(size=(4096,270)).astype(np.float32);xn[:,-20:]=xn[:,-20:]>0
    cont,scalar,bins=E.numeric_bins(xn);card=[11,49,1509,36,61,2189,155,6,10,7,446,196,586,887]
    tx=torch.from_numpy(xn).to('cuda:0');xc=torch.zeros((4096,len(card)),dtype=torch.int64,device='cuda:0')
    hist=torch.randn((4096,12,22),device='cuda:0');current=torch.randn((4096,24),device='cuda:0')
    aux=torch.zeros((4096,12,3),device='cuda:0');aux[:,:,1]=1;aux[:16,:,1]=0
    aux[:,:,2]=torch.arange(-11,1,device='cuda:0')/12;checks=[];counts=[];equal_predictions=[]
    for arm in ARMS:
        torch.manual_seed(193);net=network(N,card,cont,scalar,bins,270,arm).to('cuda:0');counts.append(sum(p.numel() for p in net.parameters()))
        net.eval()
        with torch.no_grad():equal_predictions.append(net(tx[:32],xc[:32],current[:32,:22,None].transpose(1,2).expand(-1,12,-1),current[:32],aux[:32]).cpu())
        net.train();opt=torch.optim.AdamW(net.parameters(),lr=.002);torch.cuda.reset_peak_memory_stats();start=None
        for step in range(14):
            if step==2:torch.cuda.synchronize();start=time.monotonic()
            p=net(tx,xc,hist,current,aux);loss=p.square().mean();opt.zero_grad();loss.backward();opt.step()
            if not torch.isfinite(p).all() or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):raise ValueError('synthetic finite/gradient failure')
        torch.cuda.synchronize();checks.append(dict(arm=arm,seconds_per_batch=(time.monotonic()-start)/12,peak_gpu_bytes=torch.cuda.max_memory_allocated(),parameters=counts[-1]))
        del net,opt,p,loss;gc.collect();torch.cuda.empty_cache()
    if len(set(counts))!=1 or not torch.equal(*equal_predictions):raise ValueError('capacity or identical-input control parity failed')
    C.write_json(out,dict(passed=True,synthetic_only=True,worker_sha256=C.sha(__file__),runtime=E.package_runtime(),checks=checks));print(json.dumps(checks),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['selftest','pack','verify','smoke','run','evaluate'])
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--input',type=Path);p.add_argument('--output',type=Path);p.add_argument('--registration-commit');p.add_argument('--smoke',type=Path)
    a=p.parse_args();root=a.root.resolve()
    if a.action=='selftest':selftest();return
    if a.action=='smoke':smoke(root,a.output.resolve());return
    if not re.fullmatch('[0-9a-f]{40}',a.registration_commit or ''):p.error('full registration commit required')
    if a.action=='pack':pack(root,a.output.resolve(),a.registration_commit)
    elif a.action=='verify':print(verify(root,a.input.resolve(),a.registration_commit)['spec_sha256'])
    elif a.action=='run':run(root,a.input.resolve(),a.output.resolve(),a.registration_commit,a.smoke.resolve())
    else:evaluate(root,a.input.resolve(),a.output.resolve(),a.registration_commit)


if __name__=='__main__':main()

"""Fixed mean/Gaussian/4-component conditional density screen on E190 tensors.

No sampling: density predictions use the analytic conditional mean. No shared
trainer changes, target-derived context, hyperparameter search or automatic retry.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import tempfile
import time

for k,v in {'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
            'POLARS_MAX_THREADS':'2','POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR':'load_as_storage'}.items():
    os.environ.setdefault(k,v)
import numpy as np
import e190_numeric_embeddings as E
import e192_event_context as D
M,C=E.M,E.M.C
PROTOCOL='E196-conditional-density-mean-gaussian-mdn4-2pair-3seed-8epoch-v1'
FOLDS=(1,4)
ARMS=('mean','gaussian','mdn4')
CONFIG=dict(folds=list(FOLDS),arms=list(ARMS),seeds=[0,1,2],epochs=8,batch=4096,
            width=512,bins=48,d_embedding=12,ple_version='B',activation=False,
            lr=.002,weight_decay=.00001,log_sigma_bounds=[-4.,3.],mixture_components=4)
FILES=('e196_conditional_density.py','e190_numeric_embeddings.py','e192_event_context.py',
       'mlp_gpu_component_oof.py','mlp_lowmem_prepare.py','cat_fin1b_oof.py')
PARENT='data/e192_event_inputs_20261006'


def fixed_env():
    os.environ.update(M.FIXED_ENV)
    os.environ['FEATURES']='__immutable_query_tensors_only__'


def check_registration(root,registration):
    if not re.fullmatch('[0-9a-f]{40}',registration or ''):raise ValueError('full registration required')
    code=subprocess.check_output(['git','show',registration+':scripts/e196_conditional_density.py'],cwd=root)
    journal=subprocess.check_output(['git','show',registration+':HYPOTHESES_LOG.md'],cwd=root)
    if hashlib.sha256(code).hexdigest()!=C.sha(__file__) or PROTOCOL.encode() not in journal:
        raise ValueError('code+journal registration differs')


def pack(root,out,registration,smoke):
    check_registration(root,registration)
    if out.exists():raise ValueError('fresh pack required')
    sm=C.read_json(smoke)
    if not sm['passed'] or sm['worker_sha256']!=C.sha(__file__):raise ValueError('smoke/code drift')
    parent=C.read_json(root/PARENT/'manifest.json')
    if parent['spec_sha256']!=C.digest(parent['spec']):raise ValueError('parent manifest differs')
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent));(stage/'scripts').mkdir()
    for n in FILES:shutil.copy2(root/'scripts'/n,stage/'scripts'/n)
    folds=[]
    for a in FOLDS:
        name=f'fold_{a}_{a+6}';source=root/PARENT/name;dest=stage/name;dest.mkdir()
        qb=root/E.WORK/'inputs'/name;qm=M.verify_bundle(qb,E.OLD_REGISTRATION)
        fs=next(f for f in parent['spec']['folds'] if f['fold']==a)
        if fs['query_spec_sha256']!=qm['spec_sha256']:raise ValueError('query lineage mismatch')
        meta=E.read(qb/'metadata.npz')
        for n in ('control.npz','evaluation.npz','ple.npz'):
            if C.sha(source/n)!=parent['spec']['files'][name+'/'+n]:raise ValueError('parent input drift')
            z=E.read(source/n)
            if any(not np.array_equal(z[k],v) for k,v in meta.items()):raise ValueError('metadata mismatch')
            shutil.copy2(source/n,dest/n)
        folds.append(dict(fold=a,query_bundle=str(qb.relative_to(root)),query_spec_sha256=qm['spec_sha256']))
    shutil.copy2(smoke,stage/'smoke.json')
    spec=dict(protocol=PROTOCOL,registration_commit=registration,configuration=CONFIG,folds=folds,
              runtime=sm['runtime'],parent_spec_sha256=parent['spec_sha256'],
              files={str(p.relative_to(stage)):C.sha(p) for p in stage.rglob('*') if p.is_file()},
              scripts={n:C.sha(stage/'scripts'/n) for n in FILES})
    C.write_json(stage/'manifest.json',dict(spec=spec,spec_sha256=C.digest(spec)))
    stage.rename(out);print('PACKED',out,'no targets fitted',flush=True)


def verify(root,bundle,registration):
    m=C.read_json(bundle/'manifest.json');s=m['spec']
    if (s['protocol']!=PROTOCOL or s['registration_commit']!=registration or s['configuration']!=CONFIG
        or m['spec_sha256']!=C.digest(s) or [f['fold'] for f in s['folds']]!=list(FOLDS)):
        raise ValueError('protocol/spec mismatch')
    for n,h in s['scripts'].items():
        if C.sha(Path(__file__).parent/n)!=h:raise ValueError('worker/helper drift '+n)
    for n,h in s['files'].items():
        if C.sha(bundle/n)!=h:raise ValueError('packed input drift '+n)
    for f in s['folds']:
        if M.verify_bundle(root/f['query_bundle'],E.OLD_REGISTRATION)['spec_sha256']!=f['query_spec_sha256']:
            raise ValueError('query tensors drift')
    return m


def network(original,n_num,card,continuous,scalar,bins,arm,width=512):
    import torch
    from torch import nn
    net=E.net_factory(original,n_num,card,width,continuous,scalar,bins)
    old=net.body[-1]
    if arm=='mean':return net
    size=2 if arm=='gaussian' else 12
    head=nn.Linear(old.in_features,size)
    with torch.no_grad():
        head.weight.zero_();head.bias.zero_()
        if arm=='gaussian':
            head.weight[0].copy_(old.weight[0]);head.bias[0].copy_(old.bias[0])
        else:
            for j,shift in enumerate((-.75,-.25,.25,.75)):
                head.weight[4+j].copy_(old.weight[0]);head.bias[4+j].copy_(old.bias[0]+shift)
    net.body[-1]=head
    return net


def prediction_and_loss(raw,y,arm):
    import torch
    if arm=='mean':return raw,torch.nn.functional.mse_loss(raw,y)
    if arm=='gaussian':
        mu,logs=raw[:,0],raw[:,1].clamp(-4.,3.)
        nll=(.5*((y-mu)*torch.exp(-logs)).square()+logs+.5*math.log(2*math.pi)).mean()
        return mu,nll
    logits,mu,logs=raw[:,:4],raw[:,4:8],raw[:,8:12].clamp(-4.,3.)
    logp=torch.log_softmax(logits,dim=1)
    component=-.5*((y[:,None]-mu)*torch.exp(-logs)).square()-logs-.5*math.log(2*math.pi)
    nll=-torch.logsumexp(logp+component,dim=1).mean()
    return (torch.softmax(logits,dim=1)*mu).sum(1),nll


def predict_mean(raw,arm):
    if arm=='mean':return raw
    if arm=='gaussian':return raw[:,0]
    import torch
    return (torch.softmax(raw[:,:4],dim=1)*raw[:,4:8]).sum(1)


def smoke(root,out):
    fixed_env();import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    N=E.trainer(root/'data/mlp_split_20261005/inputs/fold_1_7/src')
    rng=np.random.default_rng(196);x=rng.normal(size=(4096,270)).astype(np.float32)
    x[:,-20:]=(x[:,-20:]>0);card=[11,49,1509,36,61,2189,155,6,10,7,446,196,586,887]
    continuous,scalar,bins=E.numeric_bins(x)
    xn=torch.from_numpy(x).to('cuda:0');xc=torch.zeros((4096,len(card)),dtype=torch.int64,device='cuda:0')
    y=torch.from_numpy(rng.normal(size=4096).astype(np.float32)).to('cuda:0');rows=[]
    # One-component zero-mean unit-variance density has the standard normal NLL.
    z=torch.zeros((8,2),device='cuda:0');_,v=prediction_and_loss(z,torch.zeros(8,device='cuda:0'),'gaussian')
    if not torch.allclose(v,torch.tensor(.5*math.log(2*math.pi),device='cuda:0')):raise ValueError('Gaussian formula wrong')
    for arm in ARMS:
        torch.manual_seed(196);net=network(N.Net,270,card,continuous,scalar,bins,arm).to('cuda:0')
        opt=torch.optim.AdamW(net.parameters(),lr=.002);start=time.monotonic()
        for _ in range(4):
            pred,loss=prediction_and_loss(net(xn,xc),y,arm)
            if not torch.isfinite(loss) or not torch.isfinite(pred).all():raise ValueError('synthetic nonfinite')
            opt.zero_grad();loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):raise ValueError('nonfinite gradients')
            opt.step()
        torch.cuda.synchronize();rows.append(dict(arm=arm,seconds=time.monotonic()-start,
             parameters=sum(p.numel() for p in net.parameters()),peak_gpu_bytes=torch.cuda.max_memory_allocated()))
        del net,opt;gc.collect();torch.cuda.empty_cache()
    C.write_json(out,dict(passed=True,synthetic_only=True,worker_sha256=C.sha(__file__),runtime=E.package_runtime(),arms=rows))
    print('SYNTHETIC three-arm forward/backward PASS',flush=True)


def fit(arrays,spec,continuous,scalar,bins,N,arm,seed,od):
    import torch
    torch.manual_seed(seed);np.random.seed(seed)
    net=network(N.Net,arrays['xnf'].shape[1],spec['card'],continuous,scalar,bins,arm).to('cuda:0')
    opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.00001)
    n=len(arrays['y_scaled']);bs=4096;steps=8*(n//bs)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.002,total_steps=steps,pct_start=.1)
    Xn,Xc,Y=(torch.from_numpy(arrays[k]) for k in ('xnf','xcf','y_scaled'));Y=Y.float()
    rng=np.random.default_rng(seed);losses=[];start=time.monotonic()
    for ep in range(8):
        net.train();perm=rng.permutation(n);total=0.
        for i in range(n//bs):
            idx=torch.from_numpy(perm[i*bs:(i+1)*bs]);_,loss=prediction_and_loss(net(Xn[idx].to('cuda:0'),Xc[idx].to('cuda:0')),Y[idx].to('cuda:0'),arm)
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            opt.zero_grad();loss.backward();opt.step();sched.step();total+=loss.item()
        losses.append(total/(n//bs));print(f'{arm} seed{seed} epoch{ep+1}/8 loss={losses[-1]:.6f}',flush=True)
    net.eval();chunks=[]
    with torch.no_grad():
        for i in range(0,len(arrays['xnp']),4096):
            raw=net(torch.from_numpy(arrays['xnp'][i:i+4096]).to('cuda:0'),torch.from_numpy(arrays['xcp'][i:i+4096]).to('cuda:0'))
            chunks.append(predict_mean(raw,arm).cpu().numpy())
    pred=np.concatenate(chunks)
    if not np.isfinite(pred).all():raise ValueError('nonfinite conditional mean')
    np.save(od/f'seed_{seed}.npy',pred);torch.save(net.state_dict(),od/f'model_{seed}.pt')
    result=dict(arm=arm,seed=seed,epochs=8,seconds=time.monotonic()-start,losses=losses,
                parameters=sum(p.numel() for p in net.parameters()),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    del net,opt,sched,Xn,Xc,Y;gc.collect();torch.cuda.empty_cache();return result


def evaluate(root,bundle,out,registration):
    m=verify(root,bundle,registration);c=C.read_json(out/'complete.json');reports=[]
    if c['registration_commit']!=registration or c['input_spec_sha256']!=m['spec_sha256'] or len(c['arms'])!=6:
        raise ValueError('incomplete density screen')
    for f in m['spec']['folds']:
        name=f'fold_{f["fold"]}_{f["fold"]+6}';fd=bundle/name;meta=E.read(fd/'evaluation.npz');pred={}
        qb=root/f['query_bundle'];s=M.verify_bundle(qb,E.OLD_REGISTRATION)['spec'];ctrl=E.read(fd/'control.npz')
        for arm in ARMS:
            ad=out/name/arm;r=C.read_json(ad/'manifest.json')
            if r not in c['arms'] or r['seeds']!=[0,1,2]:raise ValueError('arm receipt mismatch')
            for p,h in r['files'].items():
                if C.sha(ad/p)!=h:raise ValueError('result drift')
            z=E.read(ad/'prediction.npz')
            if any(not np.array_equal(z[k],meta[k]) for k in ('mvt_id','y','month')):raise ValueError('metadata mismatch')
            expected=(np.mean([np.load(ad/f'seed_{i}.npy') for i in (0,1,2)],axis=0)*s['sd']+np.load(qb/'base_p.npy'))[np.load(qb/'control_indices.npy')]
            if not np.array_equal(z['member'],expected):raise ValueError('seed mean mismatch')
            expected=E.assemble(ctrl,ctrl['reg']+ctrl['mlp_coefficient']*(expected-ctrl['plain']),ctrl['lobt'])
            if not np.array_equal(z['pred'],expected['pred']):raise ValueError('mixture arithmetic mismatch')
            pred[arm]=z['pred']
        ple=E.read(fd/'ple.npz')['pred']
        reports.append(dict(fold=f['fold'],gaussian_vs_mean=D.metrics(pred['mean'],pred['gaussian'],meta),
             mdn4_vs_mean=D.metrics(pred['mean'],pred['mdn4'],meta),mdn4_vs_gaussian=D.metrics(pred['gaussian'],pred['mdn4'],meta),
             gaussian_vs_ple=D.metrics(ple,pred['gaussian'],meta),mdn4_vs_ple=D.metrics(ple,pred['mdn4'],meta)))
    def passed(k):return all(all(r[k][n]<0 for n in ('body_sse','decided_sse','trim3_sse','trim20_sse')) for r in reports)
    result=dict(protocol=PROTOCOL,registration_commit=registration,reports=reports,
        gaussian_screen_pass=passed('gaussian_vs_mean') and passed('gaussian_vs_ple'),
        mdn4_screen_pass=passed('mdn4_vs_mean') and passed('mdn4_vs_gaussian') and passed('mdn4_vs_ple'),
        accepted_for_submission=False,limitation='Two-pair LGB5/pre-ADSB retrospective screen; not full-chain or diffusion; no MC sampling.')
    C.write_json(out/'evaluation.json',result);print(json.dumps(result),flush=True)


def run(root,bundle,out,registration):
    fixed_env();import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    cgroup=M.verify_cgroup();m=verify(root,bundle,registration);runtime=E.package_runtime()
    if runtime!=m['spec']['runtime']:raise ValueError('GPU runtime mismatch')
    if out.exists():raise ValueError('fresh output required, no auto resume')
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent));done=[]
    for f in m['spec']['folds']:
        name=f'fold_{f["fold"]}_{f["fold"]+6}';qb=root/f['query_bundle'];qm=M.verify_bundle(qb,E.OLD_REGISTRATION);s=qm['spec']
        arrays,meta=M.load_arrays(qb,s);continuous,scalar,bins=E.numeric_bins(arrays['xnf']);N=E.trainer(qb/'src')
        ctrl=E.read(bundle/name/'control.npz')
        for arm in ARMS:
            od=stage/name/arm;od.mkdir(parents=True);fits=[]
            np.savez(od/'bins.npz',continuous=continuous,scalar=scalar,**{f'bin_{j}':b for j,b in enumerate(bins)})
            for seed in (0,1,2):fits.append(fit(arrays,s,continuous,scalar,bins,N,arm,seed,od))
            mean=np.mean([np.load(od/f'seed_{i}.npy') for i in (0,1,2)],axis=0)
            member=(mean*s['sd']+arrays['base_p'])[arrays['control_indices']]
            cand=E.assemble(ctrl,ctrl['reg']+ctrl['mlp_coefficient']*(member-ctrl['plain']),ctrl['lobt'])
            np.savez_compressed(od/'prediction.npz',**{k:cand[k] for k in ('mvt_id','y','month','pred','reg','values','proba')},member=member)
            receipt=dict(fold=f['fold'],arm=arm,seeds=[0,1,2],fits=fits,files={p.name:C.sha(p) for p in od.iterdir() if p.is_file()})
            C.write_json(od/'manifest.json',receipt);done.append(receipt)
        del arrays;gc.collect()
    verify(root,bundle,registration)
    C.write_json(stage/'complete.json',dict(protocol=PROTOCOL,registration_commit=registration,
           input_spec_sha256=m['spec_sha256'],runtime=runtime,cgroup=cgroup,arms=done,accepted_for_submission=False))
    stage.rename(out);print('COMPLETE 18 fits; evaluating fixed comparisons',flush=True)
    evaluate(root,bundle,out,registration)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['smoke','pack','verify','run','evaluate'])
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--input',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--smoke',type=Path);p.add_argument('--registration-commit');a=p.parse_args();root=a.root.resolve()
    if a.action=='smoke':return smoke(root,a.output.resolve())
    if a.action=='pack':return pack(root,a.output.resolve(),a.registration_commit,a.smoke.resolve())
    if a.action=='verify':return print(verify(root,a.input.resolve(),a.registration_commit)['spec_sha256'])
    return (run if a.action=='run' else evaluate)(root,a.input.resolve(),a.output.resolve(),a.registration_commit)


if __name__=='__main__':main()

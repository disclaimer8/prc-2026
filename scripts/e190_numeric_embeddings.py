"""Fixed PLE-vs-plain MLP screen. E190 is NOT complete production-chain CV.

Reuse immutable FIN1b tensors and three-seed ROCm plain controls. Learn 48-bin,
12-dimensional numerical embeddings with the original 8-epoch optimizer.
Use LGB5-only base, production anchor weights and motion routing in both arms.
"""
from __future__ import annotations
import argparse
import functools
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','POLARS_MAX_THREADS'):
    os.environ.setdefault(key,'2')
import numpy as np
import mlp_gpu_component_oof as M

PROTOCOL='E190-ple48-d12-vB-mlp3-rocm-production-weights-lgb5-screen-v1'
OLD_REGISTRATION='a926f3848baaae7415bd3cb773cbd39728ce1851'
WORK='data/mlp_split_20261005'
ANCHORS={'rawf':1/30,'lobtf':.1,'schedf':.1,'eobtf':.1,'aobtf':.1}
PACKAGE_VERSION='0.0.12'
LGB_LINEAGE='d1749457e64d8a428c59c519036472c7f5894c014280b471b67d5d6e4dade7f7'


def numeric_bins(train,n_bins=48):
    train=np.asarray(train)
    if train.ndim!=2 or len(train)<2 or not np.isfinite(train).all() or n_bins<2:
        raise ValueError('invalid finite numeric training matrix')
    cont,scalar,bins=[],[],[]
    for j in range(train.shape[1]):
        values=train[:,j]
        if np.all((values==values.min()) | (values==values.max())):
            scalar.append(j);continue
        knots=np.unique(np.quantile(values,np.linspace(0,1,n_bins+1)).astype(np.float32))
        # Includes binary indicators and rare/constant columns whose train
        # quantiles have no interior split. No held-out values enter this rule.
        if len(knots)<=2:scalar.append(j)
        else:cont.append(j);bins.append(knots)
    return cont,scalar,bins


def align(z,meta):
    ids=np.asarray(z['mvt_id']); wanted=np.asarray(meta['mvt_id'])
    if not M.C.integral_ids(ids) or len(np.unique(ids))!=len(ids) or not np.array_equal(np.sort(ids),np.sort(wanted)):
        raise ValueError('component IDs differ')
    order=np.argsort(ids);ix=order[np.searchsorted(ids[order],wanted)]
    for key in ('y','month'):
        if key in z and not np.array_equal(z[key][ix],meta[key]):raise ValueError(f'component {key} differs')
    reg=z['reg2'][ix]
    if reg.shape!=wanted.shape or not np.isfinite(reg).all():raise ValueError('invalid member predictions')
    return reg.astype(np.float64)


def regular(base,anchors,mlp,motion,subset):
    reg=(19/60)*base+sum(w*anchors[k] for k,w in ANCHORS.items())+.25*mlp
    return np.where(subset,(23/30)*reg+(7/30)*motion,reg)


def mix(z,reg,lobt):
    v=z['values'].astype(np.float64).copy();v[:,0]=reg;v[:,-1]=reg+86400.
    p=np.sum(z['proba'].astype(np.float64)*np.nan_to_num(v),axis=1)
    return np.maximum(np.where(np.isnan(lobt),p,np.clip(p,lobt-3606.,lobt+3606.)),0.)


def assemble(z,reg,lobt):
    result={k:v.copy() for k,v in z.items()}
    result['pred']=z['pred'].astype(float)+mix(z,reg,lobt)-mix(z,z['reg'],lobt)
    result['reg']=reg.astype(float);result['values']=z['values'].astype(float).copy()
    result['values'][:,0]=reg;result['values'][:,-1]=reg+86400.
    if not np.isfinite(result['pred']).all():raise ValueError('nonfinite assembled prediction')
    return result


def read(path):
    with np.load(path,allow_pickle=False) as z:return dict(z)


def package_runtime():
    import torch
    import rtdl_num_embeddings as R
    version=importlib.metadata.version('rtdl-num-embeddings')
    if version!=PACKAGE_VERSION:raise ValueError('numerical embedding version drift')
    return {'version':version,'source_sha256':M.C.sha(R.__file__),'gpu_runtime':M.runtime(device=True)}


def prepare(root,out,registration,smoke):
    import polars as pl
    sys.path.insert(0,str(root/'src'))
    import e185_on_e182_fixed as E
    import strict_oof_gate as G
    if out.exists():raise ValueError('fresh experiment input directory required')
    s=M.C.read_json(smoke)
    if not s['passed'] or s['worker_sha256']!=M.C.sha(__file__):raise ValueError('smoke/code mismatch')
    source_files=[root/'scripts'/name for name in ('e190_numeric_embeddings.py','mlp_gpu_component_oof.py','cat_fin1b_oof.py','mlp_lowmem_prepare.py')]
    source_files += [root/'src'/name for name in ('e185_on_e182_fixed.py','strict_oof_gate.py')]
    hashes={str(p.relative_to(root)):M.C.sha(p) for p in source_files}
    fp=root/'data/features_fin1b.parquet'
    if M.C.sha(fp)!=M.FEATURE_SHA:raise ValueError('wrong FIN1b feature table')
    hashes[str(fp.relative_to(root))]=M.FEATURE_SHA
    frame=pl.read_parquet(fp,columns=['MVT_ID_mvt','TAXITIME_SEC_mvt','month','split','ADEP_mvt','FLIGHT_ID_mvt','LOBT_flt','d_mvt_lobt'])
    frame=frame.filter(pl.col('split')=='train').sort('MVT_ID_mvt')
    if frame.height!=2085047 or frame['MVT_ID_mvt'].n_unique()!=frame.height:raise ValueError('training coverage differs')
    out.mkdir(parents=True);folds=[]
    for a in range(1,7):
        name=f'fold_{a}_{a+6}';bundle=root/WORK/'inputs'/name
        bm=M.verify_bundle(bundle,OLD_REGISTRATION);spec=bm['spec']
        if spec['fold']!=a:raise ValueError('tensor fold mismatch')
        M.validate_gpu_runtime(s['runtime']['gpu_runtime'],spec['gpu_runtime'])
        meta=read(bundle/'metadata.npz')
        fd=frame.filter(pl.col('month').is_in([a,a+6]));ids=fd['MVT_ID_mvt'].to_numpy()
        if not np.array_equal(np.sort(meta['mvt_id']),ids):raise ValueError('tensor IDs differ from entire feature fold')
        ix=np.searchsorted(ids,meta['mvt_id']);fd=fd[ix]
        if not np.array_equal(fd['TAXITIME_SEC_mvt'].to_numpy(),meta['y']) or not np.array_equal(fd['month'],meta['month']):raise ValueError('tensor targets/months differ')
        po=root/WORK/'outputs'/name
        evidence=dict(protocol=M.PROTOCOL,registration_commit=OLD_REGISTRATION,input_spec_sha256=bm['spec_sha256'],fold=a,seeds=[0,1,2],runtime=spec['gpu_runtime'])
        pm=M.verify_complete(po,meta,evidence);plain=align(read(po/'prediction.npz'),meta)
        parent_path=root/'data/cv_preds_fin1b_hz'/f'{name}.npz'
        if M.C.sha(parent_path)!=spec['control_sha256']:raise ValueError('parent differs from original tensor control')
        parent=E.aligned(E.read_npz(parent_path),meta['mvt_id'],parent_path)
        parent=E.synchronized(parent)
        if not np.array_equal(parent['y'],meta['y']):raise ValueError('base targets differ')
        paths={'base':root/f'data/lgb_fin1b_oof_20261005/{name}/complete/prediction.npz',
               **{k:root/f'data/cv_e130{k}/{name}.npz' for k in ANCHORS},
               'motion':root/f'data/cv_e130motion/{name}.npz'}
        components={k:align(read(p),meta) for k,p in paths.items()}
        lgb=read(paths['base']);lm=M.C.read_json(paths['base'].with_name('manifest.json'))
        if (lm['evidence']!=dict(lineage=LGB_LINEAGE,fold=a,rounds=2319,seeds=list(range(1000,1005)))
            or lm['sha256']!=M.C.sha(paths['base']) or lgb['per_seed'].shape!=(5,len(plain))
            or not np.array_equal(lgb['reg2'],lgb['per_seed'].mean(axis=0))):raise ValueError('LGB mean/provenance differs')
        subset=(fd['LOBT_flt'].is_null() & (fd['ADEP_mvt']!='LIRF')).to_numpy();lobt=fd['d_mvt_lobt'].to_numpy()
        reg=regular(components['base'],components,plain,components['motion'],subset)
        ctrl=assemble(parent,reg,lobt)
        if not G._values_consistency(ctrl)['consistent']:raise ValueError('stale endpoints')
        ctrl['month']=meta['month'];ctrl['plain']=plain;ctrl['lobt']=lobt
        ctrl['mlp_coefficient']=np.where(subset,23/120,.25)
        ctrl['airport']=fd['ADEP_mvt'].to_numpy().astype('U4');ctrl['has_nm']=fd['FLIGHT_ID_mvt'].is_not_null().to_numpy()
        dest=out/f'{name}.npz';np.savez_compressed(dest,**ctrl)
        for p in [parent_path,po/'prediction.npz',po/'manifest.json',bundle/'manifest.json',*paths.values(),paths['base'].with_name('manifest.json')]:hashes[str(p.relative_to(root))]=M.C.sha(p)
        folds.append(dict(fold=a,tensor_spec_sha256=bm['spec_sha256'],plain_sha256=pm['sha256'],input_sha256=M.C.sha(dest),rows=len(plain)))
    for p,h in hashes.items():
        if M.C.sha(root/p)!=h:raise ValueError(f'input changed {p}')
    M.C.write_json(out/'manifest.json',dict(protocol=PROTOCOL,registration_commit=registration,old_registration=OLD_REGISTRATION,
        feature_sha256=M.FEATURE_SHA,source_and_input_sha256=hashes,folds=folds,runtime=s['runtime'],worker_sha256=M.C.sha(__file__),
        configuration=dict(bins=48,d_embedding=12,activation=False,version='B',seeds=[0,1,2],epochs=8,width=512,predict_batch=4096),
        limitation='LGB5-only base, production member weights, same ROCm control. No Cat15 or downstream stages; screen only.'))
    print('PREPARED all six matched controls; no training',flush=True)


def net_factory(original,n_num,card,width,continuous,scalar,bins):
    import torch
    from torch import nn
    from rtdl_num_embeddings import PiecewiseLinearEmbeddings
    plain=original(n_num,card,width)
    encoder=PiecewiseLinearEmbeddings([torch.from_numpy(b.copy()) for b in bins],12,activation=False,version='B')
    plain.body[0]=nn.Linear(12*len(continuous)+len(scalar)+sum(e.embedding_dim for e in plain.emb),width)
    class EmbeddedNet(nn.Module):
        def __init__(self):
            super().__init__();self.emb=plain.emb;self.body=plain.body;self.numeric=encoder
            self.register_buffer('cont',torch.tensor(continuous));self.register_buffer('scalar',torch.tensor(scalar,dtype=torch.int64))
        def forward(self,xn,xc):
            pieces=[self.numeric(xn[:,self.cont]).flatten(1),xn[:,self.scalar]]
            pieces.extend(e(xc[:,j]) for j,e in enumerate(self.emb))
            return self.body(torch.cat(pieces,dim=1)).squeeze(1)
    return EmbeddedNet()


def trainer(root):
    import torch  # precedes LightGBM loaded by original trainer
    sys.path.insert(0,str(root))
    import e100_nn_partner as N
    return N


def smoke(root,out):
    import torch
    torch.set_num_threads(2)
    for k,v in M.FIXED_ENV.items():os.environ[k]=v
    N=trainer(root/'src');rng=np.random.default_rng(190)
    xn=rng.normal(size=(64,3)).astype('float32');xn[:,2]=np.arange(64)%2
    xc=np.zeros((64,1),dtype='int64');cont,scalar,bins=numeric_bins(xn)
    torch.manual_seed(190);net=net_factory(N.Net,3,[3],32,cont,scalar,bins)
    cpu=net.eval()(torch.from_numpy(xn),torch.from_numpy(xc)).detach().numpy()
    net=net.to('cuda:0');gpu=net(torch.from_numpy(xn).to('cuda:0'),torch.from_numpy(xc).to('cuda:0'))
    if not np.allclose(cpu,gpu.detach().cpu().numpy(),rtol=1e-4,atol=1e-5):raise ValueError('synthetic CPU/GPU parity failed')
    gpu.square().mean().backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):raise ValueError('synthetic gradients nonfinite')
    del net,gpu;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    large=rng.normal(size=(4096,270)).astype('float32');large[:,-20:]=(large[:,-20:]>0)
    cont,scalar,bins=numeric_bins(large);card=[11,49,1509,36,61,2189,155,6,10,7,446,196,586,887]
    net=net_factory(N.Net,270,card,512,cont,scalar,bins).to('cuda:0').train()
    opt=torch.optim.AdamW(net.parameters(),lr=.002);start=time.monotonic()
    for _ in range(2):
        loss=net(torch.from_numpy(large).to('cuda:0'),torch.zeros((4096,14),dtype=torch.int64,device='cuda:0')).square().mean()
        opt.zero_grad();loss.backward();opt.step()
        if not torch.isfinite(loss):raise ValueError('full-batch synthetic loss nonfinite')
    torch.cuda.synchronize()
    M.C.write_json(out,dict(passed=True,synthetic_only=True,worker_sha256=M.C.sha(__file__),runtime=package_runtime(),parameters=sum(p.numel() for p in net.parameters()),
        batch4096_training_steps=2,seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
    print('Synthetic PLE CPU/GPU forward and gradients PASS',flush=True)


def verify_inputs(root,bundle,registration):
    m=M.C.read_json(bundle/'manifest.json')
    if m['protocol']!=PROTOCOL or m['registration_commit']!=registration or m['worker_sha256']!=M.C.sha(__file__):raise ValueError('protocol/code drift')
    for p,h in m['source_and_input_sha256'].items():
        if M.C.sha(root/p)!=h:raise ValueError(f'input drift {p}')
    for f in m['folds']:
        if M.C.sha(bundle/f"fold_{f['fold']}_{f['fold']+6}.npz")!=f['input_sha256']:raise ValueError('matched control changed')
    return m


def run(root,bundle,out,registration):
    import torch
    for k,v in M.FIXED_ENV.items():os.environ[k]=v
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    M.verify_cgroup();m=verify_inputs(root,bundle,registration)
    if package_runtime()!=m['runtime']:raise ValueError('runtime changed after smoke')
    out.mkdir(parents=True,exist_ok=False);completed=[]
    for a in range(1,7):
        start=time.monotonic();name=f'fold_{a}_{a+6}';inp=root/WORK/'inputs'/name
        bm=M.verify_bundle(inp,OLD_REGISTRATION);spec=bm['spec']
        M.validate_gpu_runtime(package_runtime()['gpu_runtime'],spec['gpu_runtime'])
        if bm['spec_sha256']!=m['folds'][a-1]['tensor_spec_sha256']:raise ValueError('tensor lineage drift')
        arrays,meta=M.load_arrays(inp,spec);control=read(bundle/f'{name}.npz')
        if any(not np.array_equal(control[k],meta[k]) for k in meta):raise ValueError('held-out metadata mismatch')
        continuous,scalar,bins=numeric_bins(arrays['xnf'])
        if not continuous:raise ValueError('no continuous train columns')
        fd=out/name;fd.mkdir()
        np.savez(fd/'bins.npz',continuous=continuous,scalar=scalar,**{f'bin_{j}':b for j,b in enumerate(bins)})
        N=trainer(inp/'src');original=N.Net
        N.Net=functools.partial(net_factory,original,continuous=continuous,scalar=scalar,bins=bins)
        predictions=[];counts=[]
        try:
            for seed in (0,1,2):
                print(f'{name} seed={seed}: original optimizer, 8 epochs; learned PLE48/d12',flush=True)
                net=N.fit_nn(arrays['xnf'],arrays['xcf'],arrays['y_scaled'],spec['card'],seed=seed)
                counts.append(sum(p.numel() for p in net.parameters()));net.eval();chunks=[]
                with torch.no_grad():
                    for i in range(0,len(arrays['xnp']),4096):
                        chunks.append(net(torch.from_numpy(arrays['xnp'][i:i+4096]).to('cuda:0'),torch.from_numpy(arrays['xcp'][i:i+4096]).to('cuda:0')).cpu().numpy())
                pred=np.concatenate(chunks)
                if not np.isfinite(pred).all():raise ValueError('nonfinite PLE output')
                predictions.append(pred);np.save(fd/f'seed_{seed}.npy',pred)
                torch.save(net.state_dict(),fd/f'model_{seed}.pt')
                del net,chunks;gc.collect();torch.cuda.empty_cache()
        finally:N.Net=original
        member=(np.mean(predictions,axis=0)*spec['sd']+arrays['base_p'])[arrays['control_indices']]
        reg=control['reg']+control['mlp_coefficient']*(member-control['plain'])
        candidate=assemble(control,reg,control['lobt']);candidate['member']=member
        np.savez_compressed(fd/'prediction.npz',**candidate)
        M.verify_bundle(inp,OLD_REGISTRATION)
        result=dict(fold=a,rows=len(member),seconds=time.monotonic()-start,parameters=counts,continuous=len(continuous),scalar=len(scalar),
            input_sha256=m['folds'][a-1]['input_sha256'],tensor_spec_sha256=bm['spec_sha256'],
            files={p.name:M.C.sha(p) for p in fd.iterdir() if p.is_file()},
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),runtime=package_runtime())
        M.C.write_json(fd/'manifest.json',result);completed.append(result)
        print(json.dumps({k:result[k] for k in ('fold','rows','seconds','continuous','scalar','peak_gpu_bytes')}),flush=True)
        del arrays,control,candidate,predictions;gc.collect()
    verify_inputs(root,bundle,registration)
    M.C.write_json(out/'complete.json',dict(protocol=PROTOCOL,registration_commit=registration,folds=completed,accepted_for_submission=False))
    print('COMPLETE: 18 fixed fits, no quality selection; screen evaluation still required',flush=True)


def evaluate(root,bundle,out,registration):
    import shutil
    sys.path.insert(0,str(root/'src'));import strict_oof_gate as G
    m=verify_inputs(root,bundle,registration);c=M.C.read_json(out/'complete.json')
    if c['protocol']!=PROTOCOL or c['registration_commit']!=registration or len(c['folds'])!=6:raise ValueError('incomplete run')
    dirs=[out/'control',out/'candidate']
    for p in dirs:p.mkdir(exist_ok=False)
    for a in range(1,7):
        name=f'fold_{a}_{a+6}';f=M.C.read_json(out/name/'manifest.json')
        if f!=c['folds'][a-1] or f['runtime']!=m['runtime']:raise ValueError('result lineage mismatch')
        for n,h in f['files'].items():
            if M.C.sha(out/name/n)!=h:raise ValueError('result file drift')
        ctrl=read(bundle/f'{name}.npz');cand=read(out/name/'prediction.npz')
        inp=root/WORK/'inputs'/name;bm=M.verify_bundle(inp,OLD_REGISTRATION)
        if bm['spec_sha256']!=f['tensor_spec_sha256']:raise ValueError('result tensor mismatch')
        seeds=[np.load(out/name/f'seed_{seed}.npy',allow_pickle=False) for seed in (0,1,2)]
        if any(v.shape!=ctrl['y'].shape or not np.isfinite(v).all() for v in seeds):raise ValueError('invalid saved seed')
        reconstructed=(np.mean(seeds,axis=0)*bm['spec']['sd']+np.load(inp/'base_p.npy',allow_pickle=False))[np.load(inp/'control_indices.npy',allow_pickle=False)]
        if not np.array_equal(reconstructed,cand['member']):raise ValueError('three-seed mean reconstruction failed')
        expected=assemble(ctrl,ctrl['reg']+ctrl['mlp_coefficient']*(cand['member']-ctrl['plain']),ctrl['lobt'])
        for k in ('mvt_id','y','month','reg','pred','proba','values'):
            if not np.array_equal(cand[k],expected[k],equal_nan=True):raise ValueError(f'candidate arithmetic mismatch {k}')
        for z,d in ((ctrl,dirs[0]),(cand,dirs[1])):
            if not G._values_consistency(z)['consistent']:raise ValueError('stale endpoints')
            np.savez_compressed(d/f'{name}.npz',**{k:z[k] for k in ('mvt_id','y','month','reg','pred','proba','values')})
    report=G.evaluate(*dirs,root/'data/features_fin1b.parquet',require_body=True,check_values=True)
    report.update(protocol=PROTOCOL,registration_commit=registration,accepted_for_submission=False,
                  limitation=m['limitation'],proceed_to_current_chain=report['gate_passed'])
    M.C.write_json(out/'screen_result.json',report);print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','smoke','run','evaluate'])
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--bundle',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--registration-commit');p.add_argument('--smoke',type=Path);a=p.parse_args()
    root=a.root.resolve();out=a.out.resolve()
    if a.action=='smoke':smoke(root,out);return
    if not re.fullmatch('[a-f0-9]{40}',a.registration_commit or ''):p.error('full registration commit required')
    with M.C.writer_lock(out.parent/('.'+out.name+'.lockdir')):
        if a.action=='prepare':prepare(root,out,a.registration_commit,a.smoke.resolve())
        elif a.action=='run':run(root,a.bundle.resolve(),out,a.registration_commit)
        else:evaluate(root,a.bundle.resolve(),out,a.registration_commit)


if __name__=='__main__':main()

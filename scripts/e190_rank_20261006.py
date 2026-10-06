"""Full-training E190 ranking member, separately registered from its screen.

Mac prepare -> immutable tensors -> home ROCm three-seed fit -> atomic member.
No quality selection or CV claim; full-chain integration belongs to the caller.
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
import subprocess
import sys
import tempfile
import time

# Set before NumPy/Polars/Torch; the public launcher also clears ambient options.
for _k, _v in {'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
               'POLARS_MAX_THREADS':'2','VECLIB_MAXIMUM_THREADS':'1'}.items():
    os.environ.setdefault(_k, _v)
import numpy as np
import polars as pl
import e190_numeric_embeddings as E
import mlp_gpu_component_oof as M

C = M.C
PROTOCOL = 'E190-full-fin1b-ranking-ple48-d12-B-width512-8epoch-3seed-rocm-20261006-v1'
ROWS = 344841
TRAIN_ROWS = 2085047
CONFIG = dict(bins=48, d_embedding=12, activation=False, version='B', width=512,
              epochs=8, seeds=[0,1,2], train_batch=4096, predict_batch=4096,
              optimizer='AdamW', lr=.002, weight_decay=.00001,
              schedule='OneCycleLR', pct_start=.1)
ARRAYS = ('xnf','xcf','xnp','xcp','y_scaled','base_p')
SCRIPTS = ('e190_rank_20261006.py','e190_numeric_embeddings.py',
           'mlp_gpu_component_oof.py','mlp_lowmem_prepare.py','cat_fin1b_oof.py')
FIXED_ENV = {**M.FIXED_ENV, 'VECLIB_MAXIMUM_THREADS':'1'}


def environment(feature='__prepared_inputs_only__'):
    env = {k: os.environ[k] for k in ('HOME','PATH','TMPDIR','LANG') if k in os.environ}
    return {**env, **FIXED_ENV, 'FEATURES':str(feature)}


def check_environment(feature='__prepared_inputs_only__'):
    if any(os.environ.get(k) != v for k,v in environment(feature).items()):
        raise ValueError('fixed environment differs')


def check_ids(ids):
    if ids.shape != (ROWS,) or not C.integral_ids(ids) or len(np.unique(ids)) != ROWS:
        raise ValueError('expected 344841 unique integral ranking IDs')


def validate(a, ids, spec):
    check_ids(ids)
    n, p = len(a['xnf']), len(a['xnp'])
    if n < 4096 or p != ROWS or spec['train_rows'] != TRAIN_ROWS:
        raise ValueError('incomplete full-training/ranking coverage')
    if not np.isfinite(spec['sd']) or spec['sd'] <= 0 or not spec['card']:
        raise ValueError('invalid target scale/cardinality')
    for name, rows in (('xnf',n),('xnp',p)):
        x = a[name]
        if x.ndim != 2 or x.dtype != np.float32 or not x.flags.c_contiguous:
            raise ValueError('numeric shape/dtype/layout differs')
        for i in range(0,rows,65536):
            if not np.isfinite(x[i:i+65536]).all(): raise ValueError('nonfinite numeric tensor')
    if a['xnf'].shape[1] != a['xnp'].shape[1]: raise ValueError('numeric widths differ')
    for name, rows in (('xcf',n),('xcp',p)):
        x=a[name]
        if x.shape != (rows,len(spec['card'])) or x.dtype != np.int64 or not x.flags.c_contiguous:
            raise ValueError('categorical shape/dtype/layout differs')
        for j,k in enumerate(spec['card']):
            if k < 1 or x[:,j].min() < 0 or x[:,j].max() >= k: raise ValueError('bad categorical code')
    for name, rows in (('y_scaled',n),('base_p',p)):
        if a[name].shape != (rows,) or a[name].dtype != np.float64 or not np.isfinite(a[name]).all():
            raise ValueError('target/proxy tensor differs')


def verify(bundle, registration):
    m=C.read_json(bundle/'manifest.json'); s=m['spec']
    if (m['spec_sha256'] != C.digest(s) or s['protocol'] != PROTOCOL
        or s['registration_commit'] != registration or s['config'] != CONFIG
        or s['feature_sha256'] != M.FEATURE_SHA or s['source_sha256'] != M.SOURCES
        or s['fixed_environment'] != FIXED_ENV or set(s['array_sha256']) != set(ARRAYS)
        or set(s['script_sha256']) != set(SCRIPTS)):
        raise ValueError('bundle protocol/source/configuration differs')
    for n,h in M.SOURCES.items():
        if C.sha(bundle/'src'/n) != h: raise ValueError('frozen trainer source drift: '+n)
    for n,h in s['script_sha256'].items():
        if C.sha(bundle/'scripts'/n) != h or C.sha(Path(__file__).parent/n) != h:
            raise ValueError('snapshot or active script drift: '+n)
    for n,h in s['array_sha256'].items():
        if C.sha(bundle/(n+'.npy')) != h: raise ValueError('tensor drift: '+n)
    if C.sha(bundle/'mvt_id.npy') != s['ids_sha256']: raise ValueError('ranking ID drift')
    if C.sha(bundle/'admitted_runtime.json') != s['admitted_runtime_sha256']: raise ValueError('runtime evidence drift')
    check_ids(np.load(bundle/'mvt_id.npy',allow_pickle=False))
    return m


def prepare(root, bundle, registration, smoke):
    feature=root/'data/features_fin1b.parquet'
    check_environment(feature)
    if bundle.exists(): raise ValueError('fresh bundle directory required')
    if shutil.disk_usage(root).free < 6*1024**3: raise ValueError('need >=6GiB disk for retained immutable tensors')
    if C.sha(feature) != M.FEATURE_SHA: raise ValueError('FIN1b feature hash differs')
    if any(C.sha(root/'src'/n) != h for n,h in M.SOURCES.items()): raise ValueError('production trainer source drift')
    evidence=C.read_json(smoke)
    if (not evidence['passed'] or not evidence['synthetic_only']
        or evidence['worker_sha256'] != C.sha(E.__file__)
        or evidence['runtime']['version'] != E.PACKAGE_VERSION):
        raise ValueError('screen synthetic runtime evidence differs')
    M.validate_gpu_runtime(evidence['runtime']['gpu_runtime'])
    bundle.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+bundle.name+'.partial-',dir=bundle.parent))
    (stage/'src').mkdir(); (stage/'scripts').mkdir()
    for n in M.SOURCES: shutil.copy2(root/'src'/n,stage/'src'/n)
    for n in SCRIPTS: shutil.copy2(root/'scripts'/n,stage/'scripts'/n)
    shutil.copy2(smoke,stage/'admitted_runtime.json')
    producer=M.runtime()  # Imports torch before LightGBM.
    sys.path.insert(0,str(stage/'src'))
    import e100_nn_partner as N
    frame=pl.read_parquet(feature)
    fit=frame.filter(pl.col('split')=='train'); pred=frame.filter(pl.col('split')=='rank')
    del frame
    ids=pred[C.ID].to_numpy().astype(np.float64)
    check_ids(ids)
    if (fit.height != TRAIN_ROWS or fit[C.ID].n_unique() != TRAIN_ROWS
        or not np.array_equal(np.sort(fit['month'].unique().to_numpy()),np.arange(1,13))
        or np.intersect1d(fit[C.ID].to_numpy(),ids).size):
        raise ValueError('full train coverage or train/ranking separation differs')
    if pred[C.TARGET].is_not_null().any(): raise ValueError('ranking unexpectedly contains targets')
    train_id_sha=C.digest(fit[C.ID].to_list())
    print('Preparing full FIN1b month-out target encodings, no fitting',flush=True)
    lab=N.T.regime_label(fit)
    fit,pred=N.add_encodings(fit,pred,lab==0,fit_labels=lab)
    cols=N.T.feature_columns(fit)
    if C.TARGET in cols or C.ID in cols: raise ValueError('target or ID included as model feature')
    if not np.array_equal(pred[C.ID].to_numpy(),ids): raise ValueError('encoding reordered ranking IDs')
    regular=fit.filter(pl.Series(lab==0))
    base_f,base_p=N.T.proxy_base(regular,pred)
    y=np.clip(np.clip(regular[C.TARGET].to_numpy(),*N.T.REG_CLIP)-base_f,-7200,7200)
    sd=float(y.std())
    print(f'Preparing quantile/category tensors for {regular.height} regular training rows',flush=True)
    xnf,xcf,xnp,xcp,card=M.L.prepare(regular,pred,cols,categorical=N.T.CATEGORICAL)
    a=dict(xnf=xnf,xcf=xcf,xnp=xnp,xcp=xcp,y_scaled=y/sd,base_p=base_p)
    s=dict(protocol=PROTOCOL,registration_commit=registration,config=CONFIG,
           feature_sha256=M.FEATURE_SHA,source_sha256=M.SOURCES,
           fixed_environment=FIXED_ENV,train_rows=TRAIN_ROWS,regular_train_rows=len(y),
           train_id_sha256=train_id_sha,ranking_rows=ROWS,columns=cols,sd=sd,card=card,
           producer_runtime=producer,admitted_runtime=evidence['runtime'],
           admitted_runtime_sha256=C.sha(stage/'admitted_runtime.json'),
           script_sha256={n:C.sha(stage/'scripts'/n) for n in SCRIPTS})
    validate(a,ids,s)
    for n,v in a.items(): np.save(stage/(n+'.npy'),v,allow_pickle=False)
    np.save(stage/'mvt_id.npy',ids,allow_pickle=False)
    s.update(array_sha256={n:C.sha(stage/(n+'.npy')) for n in ARRAYS},ids_sha256=C.sha(stage/'mvt_id.npy'))
    C.write_json(stage/'manifest.json',dict(spec=s,spec_sha256=C.digest(s)))
    verify(stage,registration)
    if C.sha(feature) != M.FEATURE_SHA: raise ValueError('feature input changed during prepare')
    stage.rename(bundle)
    print(json.dumps(dict(prepared=str(bundle),regular_rows=len(y),numeric_shape=list(xnf.shape),
                         array_bytes=sum(v.nbytes for v in a.values()),spec_sha256=C.digest(s))),flush=True)


def verify_output(bundle, out, registration):
    m=verify(bundle,registration); s=m['spec']; r=C.read_json(out/'manifest.json')
    if (r['protocol'] != PROTOCOL or r['registration_commit'] != registration
        or r['input_spec_sha256'] != m['spec_sha256'] or r['runtime'] != s['admitted_runtime']
        or r['seeds'] != [0,1,2]): raise ValueError('output evidence differs')
    for n,h in r['files'].items():
        if C.sha(out/n) != h: raise ValueError('output artifact hash differs: '+n)
    ids=np.load(bundle/'mvt_id.npy',allow_pickle=False)
    seeds=[np.load(out/f'seed_{i}.npy',allow_pickle=False) for i in (0,1,2)]
    if any(p.shape!=(ROWS,) or p.dtype!=np.float32 or not np.isfinite(p).all() for p in seeds):
        raise ValueError('incomplete seed predictions')
    expected=np.mean(seeds,axis=0)*s['sd']+np.load(bundle/'base_p.npy',allow_pickle=False)
    with np.load(out/'prediction.npz',allow_pickle=False) as z:
        if (set(z.files) != {'mvt_id','reg','reg2'} or not np.array_equal(ids,z['mvt_id'])
            or not np.array_equal(expected,z['reg']) or not np.array_equal(z['reg'],z['reg2'])
            or not np.isfinite(z['reg']).all()): raise ValueError('complete member mean/IDs differ')
    return r


def run(bundle, out, registration):
    check_environment(); cgroup=M.verify_cgroup()
    if out.exists(): raise ValueError('fresh output required; no automatic resume')
    m=verify(bundle,registration); s=m['spec']
    actual=E.package_runtime()
    if actual != s['admitted_runtime']: raise ValueError('admitted ROCm/package runtime drift')
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    a={n:np.load(bundle/(n+'.npy'),allow_pickle=False) for n in ARRAYS}
    ids=np.load(bundle/'mvt_id.npy',allow_pickle=False); validate(a,ids,s)
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent))
    started=time.monotonic()
    continuous,scalar,bins=E.numeric_bins(a['xnf'])
    if not continuous: raise ValueError('no continuous numeric columns')
    np.savez(stage/'bins.npz',continuous=continuous,scalar=scalar,**{f'bin_{j}':b for j,b in enumerate(bins)})
    N=E.trainer(bundle/'src'); original=N.Net
    if str(N.DEV) != 'cuda:0' or N.EPOCHS != 8: raise ValueError('trainer config differs')
    N.Net=functools.partial(E.net_factory,original,continuous=continuous,scalar=scalar,bins=bins)
    predictions=[]; counts=[]
    try:
        for seed in (0,1,2):
            print(f'FULL RANK seed={seed}, original optimizer, 8 epochs, PLE48/d12/B, width512',flush=True)
            net=N.fit_nn(a['xnf'],a['xcf'],a['y_scaled'],s['card'],seed=seed)
            counts.append(sum(p.numel() for p in net.parameters())); net.eval(); chunks=[]
            with torch.no_grad():
                for i in range(0,ROWS,4096):
                    chunks.append(net(torch.from_numpy(a['xnp'][i:i+4096]).to('cuda:0'),
                                      torch.from_numpy(a['xcp'][i:i+4096]).to('cuda:0')).cpu().numpy())
            p=np.concatenate(chunks)
            if p.shape != (ROWS,) or not np.isfinite(p).all(): raise ValueError('bad seed output')
            predictions.append(p); np.save(stage/f'seed_{seed}.npy',p,allow_pickle=False)
            torch.save(net.state_dict(),stage/f'model_{seed}.pt')
            del net,chunks; gc.collect(); torch.cuda.empty_cache()
    finally:
        N.Net=original
    reg=np.mean(predictions,axis=0)*s['sd']+a['base_p']
    if not np.isfinite(reg).all(): raise ValueError('bad complete prediction')
    np.savez(stage/'prediction.npz',mvt_id=ids,reg=reg,reg2=reg)
    verify(bundle,registration)
    r=dict(protocol=PROTOCOL,registration_commit=registration,input_spec_sha256=m['spec_sha256'],
           runtime=actual,seeds=[0,1,2],rows=ROWS,parameters=counts,continuous=len(continuous),scalar=len(scalar),
           seconds=time.monotonic()-started,cgroup=cgroup,
           peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
           peak_gpu_bytes=torch.cuda.max_memory_allocated(),
           files={p.name:C.sha(p) for p in stage.iterdir() if p.is_file()},
           limitation='Full-training member; user waived waiting for full production-chain CV. No score claim.')
    C.write_json(stage/'manifest.json',r); verify_output(bundle,stage,registration)
    stage.rename(out)
    print(json.dumps(dict(published=str(out),rows=ROWS,seconds=r['seconds'],
                         peak_rss_gib=r['peak_rss_gib'],sha256=r['files']['prediction.npz'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','produce','run','consume','verify','verify-output','runtime'])
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--input',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--registration-commit');p.add_argument('--smoke',type=Path)
    a=p.parse_args(); root=a.root.resolve()
    if a.action=='runtime': print(json.dumps(E.package_runtime(),sort_keys=True));return
    if not a.input or not re.fullmatch('[a-f0-9]{40}',a.registration_commit or ''):
        p.error('immutable input path and full registration commit required')
    bundle=a.input.resolve()
    if a.action in ('prepare','run'):
        if a.action=='run' and not a.output: p.error('run needs output')
        target=bundle if a.action=='prepare' else a.output.resolve()
        with C.writer_lock(target.parent/('.'+target.name+'.lockdir')):
            cmd='produce' if a.action=='prepare' else 'consume'
            env=environment(root/'data/features_fin1b.parquet') if cmd=='produce' else environment()
            subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),cmd,*sys.argv[2:]],env=env,check=True)
    elif a.action=='produce': prepare(root,bundle,a.registration_commit,a.smoke.resolve())
    elif a.action=='consume': run(bundle,a.output.resolve(),a.registration_commit)
    elif a.action=='verify-output': print(json.dumps(verify_output(bundle,a.output.resolve(),a.registration_commit),sort_keys=True))
    else: print(json.dumps(dict(verified=str(bundle),spec_sha256=verify(bundle,a.registration_commit)['spec_sha256'])))


if __name__=='__main__': main()

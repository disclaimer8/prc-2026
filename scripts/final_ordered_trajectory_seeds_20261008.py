"""Extra seeds 3..9 for the final original ordered E203 member; ten-seed mean.

Variance reduction only. The frozen adapter `final_ordered_trajectory_20261008`
is imported unchanged and its registration №387 (b9e37f30) receipt remains the
admission key for the completed raw/queries/context/smoke/models artifacts. The
same `S.fit` recipe (8 epochs, width512, batch4096, AdamW lr.002, OneCycle)
is called with seeds 3..9 only. No new features, capacity, selection, weight
search, tail assembly, submission or environment change. Seeds 0..2 are not
retrained; their saved arrays are reused after hash admission.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import polars as pl
import final_ordered_trajectory_20261008 as F

S,E,C=F.S,F.E,F.C
PROTOCOL='final-original-ordered-E203-fresh2025-extra-seeds-3to9-eight-epoch-v1'
SELF='scripts/final_ordered_trajectory_seeds_20261008.py'
TEST='tests/test_final_ordered_trajectory_seeds_20261008.py'
KIND='models_seeds3to9'
NEW_SEEDS=(3,4,5,6,7,8,9)
ALL_SEEDS=(0,1,2,*NEW_SEEDS)


def seed_mean_n(seeds,sd,base,count):
    """Same arithmetic as the frozen `F.seed_mean`, for `count` members."""
    seeds=list(seeds);base=np.asarray(base)
    if (len(seeds)!=count or not np.isfinite(sd) or sd<=0 or base.ndim!=1 or not np.isfinite(base).all()
        or any(s.shape!=base.shape or s.dtype!=np.float32 or not np.isfinite(s).all() for s in seeds)):
        raise ValueError(f'{count} finite float32 seeds and residual scale required')
    return np.mean(seeds,axis=0)*sd+base


def pairwise_rms(seeds):
    a=np.stack([np.asarray(s,np.float64) for s in seeds]);n=len(a);out=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):out[i,j]=out[j,i]=float(np.sqrt(np.mean((a[i]-a[j])**2)))
    return out


def rms(a,b):return float(np.sqrt(np.mean((np.asarray(a,np.float64)-np.asarray(b,np.float64))**2)))


def attest(root,commit,parent_receipt,out):
    """Local Git attestation of this helper, its test and the journal line; binds the parent receipt."""
    if not re.fullmatch('[a-f0-9]{40}',commit or ''):raise ValueError('full registration required')
    r=F.read_json(parent_receipt)
    if r['protocol']!=F.PROTOCOL or r['config']!=F.CONFIG:raise ValueError('parent receipt is not the frozen adapter registration')
    pins={}
    for name in (SELF,TEST):
        data=subprocess.check_output(['git','show',commit+':'+name],cwd=root)
        pins[name]=hashlib.sha256(data).hexdigest()
        if C.sha(root/name)!=pins[name]:raise ValueError('registered source drift '+name)
    frozen=hashlib.sha256(subprocess.check_output(['git','show',commit+':'+F.SELF],cwd=root)).hexdigest()
    if frozen!=r['source_sha256'][F.SELF] or C.sha(root/F.SELF)!=frozen:raise ValueError('frozen adapter differs from parent registration')
    if PROTOCOL.encode() not in subprocess.check_output(['git','show',commit+':HYPOTHESES_LOG.md'],cwd=root):
        raise ValueError('registered journal protocol absent')
    F.write_json(out,dict(protocol=PROTOCOL,registration_commit=commit,parent_protocol=F.PROTOCOL,
        parent_registration_commit=r['registration_commit'],parent_receipt_sha256=C.sha(parent_receipt),
        seeds=list(NEW_SEEDS),all_seeds=list(ALL_SEEDS),epochs=8,config=F.CONFIG,source_sha256=pins))


def admit(root,receipt,parent_receipt):
    s=F.read_json(receipt)
    if (s['protocol']!=PROTOCOL or s['config']!=F.CONFIG or s['seeds']!=list(NEW_SEEDS) or s['all_seeds']!=list(ALL_SEEDS)
        or s['epochs']!=8 or not re.fullmatch('[a-f0-9]{40}',s['registration_commit']) or s['parent_protocol']!=F.PROTOCOL
        or set(s['source_sha256'])!={SELF,TEST}):raise ValueError('extra-seed registration contract differs')
    if s['parent_receipt_sha256']!=C.sha(parent_receipt):raise ValueError('parent receipt differs from extra-seed registration')
    for name,h in s['source_sha256'].items():
        if C.sha(F.safe(root,name))!=h:raise ValueError('active registered source drift '+name)
    r=F.admit(root,parent_receipt)
    if s['parent_registration_commit']!=r['registration_commit']:raise ValueError('parent registration commit differs')
    if (Path(F.__file__).resolve()!=F.safe(root,F.SELF) or Path(__file__).resolve()!=F.safe(root,SELF)):
        raise ValueError('active helper loaded from another source tree')
    return s,r


def month_table(feature,ids_final):
    frame=pl.scan_parquet(feature).filter(pl.col('split')=='rank').select(F.ID,'month').collect()
    if not np.array_equal(frame[F.ID].to_numpy().astype(np.float64),ids_final):raise ValueError('final month rows differ from prediction order')
    return frame['month'].to_numpy()


def statistics(all_seeds,reg3,reg10,sd,months):
    pw=pairwise_rms(all_seeds);off=pw[np.triu_indices(len(all_seeds),1)]
    per_month={}
    for m in np.unique(months):
        ix=months==m
        per_month[str(int(m))]=dict(rows=int(ix.sum()),rms_3seed_vs_10seed_sec=rms(reg3[ix],reg10[ix]),
            mean_shift_sec=float(np.mean(reg10[ix]-reg3[ix])),mean_3seed_sec=float(np.mean(reg3[ix])),mean_10seed_sec=float(np.mean(reg10[ix])))
    return dict(sd_sec=float(sd),pairwise_seed_rms_std_units=pw.tolist(),pairwise_seed_rms_std_units_mean=float(off.mean()),
        pairwise_seed_rms_std_units_min=float(off.min()),pairwise_seed_rms_std_units_max=float(off.max()),
        pairwise_seed_rms_sec_mean=float(off.mean()*sd),rms_3seed_vs_10seed_sec=rms(reg3,reg10),
        mean_shift_10_minus_3_sec=float(np.mean(reg10-reg3)),per_month=per_month)


def run(root,work,receipt,parent_receipt,qb,prepared,smoke_dir,models,out):
    s,r=admit(root,receipt,parent_receipt);guard=F.capacity(2,12,25,out);F.fixed_environment()
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    q=F.checked(qb,'queries',parent_receipt);sm=F.checked(smoke_dir,'smoke',parent_receipt);m=F.checked(models,'models',parent_receipt)
    runtime=E.package_runtime()
    if (sm['passed'] is not True or sm['synthetic_only'] is not True or sm['runtime']!=runtime
        or sm['query_complete_sha256']!=C.sha(qb/'complete.json')):raise ValueError('full-shape runtime smoke not admitted')
    if (m['query_complete_sha256']!=C.sha(qb/'complete.json') or m['context_complete_sha256']!=C.sha(prepared/'complete.json')
        or m['smoke_complete_sha256']!=C.sha(smoke_dir/'complete.json') or m['runtime']!=runtime or m['seeds']!=[0,1,2]
        or m['checkpoint_replay_exact'] is not True or len(m['fits'])!=3):raise ValueError('three-seed model artifact lineage differs')
    _,fc,feature,template=F.feature_inputs(work,r)
    if fc['feature_sha256']!=q['feature_sha256']:raise ValueError('feature differs from admitted queries')
    arrays,spec,ids,tr,pi=F.query_arrays(qb);F.checked_context(prepared,qb,parent_receipt,ids,tr)
    if spec!=F.read_json(models/'spec.json'):raise ValueError('query specification differs from three-seed fit')
    F.validate_ids(ids[pi],pl.read_parquet(template,columns=[F.ID])[F.ID].to_numpy(),F.FINAL_ROWS)
    months=month_table(feature,ids[pi])
    cont,scalar,bins=E.numeric_bins(arrays['xnf']);sc,ss,sb=F.load_bins(models/'bins.npz')
    if (list(cont)!=list(sc) or list(scalar)!=list(ss) or len(bins)!=len(sb) or any(not np.array_equal(a,b) for a,b in zip(bins,sb))):
        raise ValueError('PLE bins differ from the registered three-seed fit')
    orig=[np.load(models/f'seed_{i}.npy',allow_pickle=False) for i in (0,1,2)]
    reg3=F.seed_mean(orig,spec['sd'],arrays['base_p'])
    with np.load(models/'prediction.npz',allow_pickle=False) as z:
        if not np.array_equal(reg3,z['reg']) or not np.array_equal(z['mvt_id'],ids[pi]):raise ValueError('three-seed mean does not replay')
    parameters=m['fits'][0]['parameters'];N=F.trainer(root);tables=F.gpu_tables(prepared,len(ids))
    with F.staging(out) as stage:
        F.save_bins(stage/'bins.npz',cont,scalar,bins);fits=[]
        for seed in NEW_SEEDS:
            fits.append(S.fit(arrays,spec,N,cont,scalar,bins,tables,tr,pi,stage,'ordered',seed))
            saved=np.load(stage/f'seed_{seed}.npy',allow_pickle=False)
            replay=F.predict_checkpoint(root,qb,stage/f'model_{seed}.pt',arrays,spec,(cont,scalar,bins),tables,pi)
            if not np.array_equal(replay,saved):raise ValueError('persisted target checkpoint does not replay exact predictions')
            print(json.dumps(dict(seed=seed,checkpoint_replay_exact=True,seconds=fits[-1]['seconds'])),flush=True)
        if any(f['parameters']!=parameters or len(f['losses'])!=8 or not np.isfinite(f['losses']).all() for f in fits):
            raise ValueError('seven complete same-capacity eight-epoch fits required')
        new=[np.load(stage/f'seed_{i}.npy',allow_pickle=False) for i in NEW_SEEDS]
        reg10=seed_mean_n([*orig,*new],spec['sd'],arrays['base_p'],len(ALL_SEEDS))
        if not np.isfinite(reg10).all() or reg10.shape!=(F.FINAL_ROWS,):raise ValueError('ten-seed mean incomplete')
        np.savez_compressed(stage/'prediction_10seed.npz',mvt_id=ids[pi],reg=reg10,reg2=reg10)
        shutil.copyfile(models/'prediction.npz',stage/'prediction_3seed.npz')
        for name in ('spec.json','transforms.json','normalization.json'):shutil.copyfile(models/name,stage/name)
        stats=statistics([*orig,*new],reg3,reg10,spec['sd'],months)
        F.write_json(stage/'statistics.json',stats);admit(root,receipt,parent_receipt)
        F.write_json(stage/'complete.json',dict(protocol=PROTOCOL,kind=KIND,registration_commit=s['registration_commit'],
            receipt_sha256=C.sha(receipt),parent_protocol=F.PROTOCOL,parent_registration_commit=r['registration_commit'],
            parent_receipt_sha256=C.sha(parent_receipt),parent_models_complete_sha256=C.sha(models/'complete.json'),
            query_complete_sha256=C.sha(qb/'complete.json'),context_complete_sha256=C.sha(prepared/'complete.json'),
            smoke_complete_sha256=C.sha(smoke_dir/'complete.json'),config=F.CONFIG,runtime=runtime,seeds=list(NEW_SEEDS),
            all_seeds=list(ALL_SEEDS),fits=fits,checkpoint_replay_exact=True,three_seed_mean_exact=True,bins_equal_registered=True,
            template_ids_equal=True,rows=F.FINAL_ROWS,raw_member_only=True,statistics=stats,
            files={str(p.relative_to(stage)):C.sha(p) for p in sorted(stage.rglob('*')) if p.is_file()},cgroup=guard))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['attest','run'])
    for name in ('root','work','receipt','parent-receipt','output','queries','prepared','smoke','models'):
        p.add_argument('--'+name,type=Path,default=Path.cwd() if name=='root' else None)
    p.add_argument('--registration-commit')
    a=p.parse_args();root=a.root.resolve()
    if a.output is None or a.parent_receipt is None:p.error('--output and --parent-receipt are required')
    out=a.output.resolve()
    if a.action=='attest':attest(root,a.registration_commit,a.parent_receipt,out);return
    if a.receipt is None:p.error('--receipt is required')
    def stopped(signum,frame):raise SystemExit('bounded worker stopped by signal '+str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,stopped)
    failure=out.with_name(out.name+'.failure.json')
    if out.exists() or failure.exists():raise ValueError('fresh output namespace required; no automatic retry')
    signal.alarm(6*3600)
    try:run(root,a.work,a.receipt,a.parent_receipt,a.queries,a.prepared,a.smoke,a.models,out)
    except BaseException as error:
        out.parent.mkdir(parents=True,exist_ok=True)
        F.write_json(failure,dict(protocol=PROTOCOL,action=a.action,error=type(error).__name__,message=str(error),time=time.time()))
        raise


if __name__=='__main__':main()

"""E204: fixed two-scale trajectory encoder, saved ordered control, gated queue.

Only message offsets change: (1,1) -> (1,8). All input/target tensors,
normalizations, parameter shapes and frozen E197 fit mechanics are reused.
No ranking fits, submission, retries, environment mutation or source mutation.
"""
from __future__ import annotations

import argparse
import datetime as dt
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
import types

import numpy as np
import e197_trajectory_encoder as S
import e202_trajectory_extension as Q

E, M, C, D = S.E, S.M, S.C, S.D
PROTOCOL = 'E204-trajectory-multiscale-d1d8-fixed-ordered-control-gated6pair-v1'
SELF = 'scripts/night_gpu_20261007.py'
DOC = 'analysis/night_gpu_20261007/PROTOCOL.md'
TEST = 'tests/test_night_gpu_20261007.py'
BASE = 'analysis/night_gpu_20261007'
FOLDS = (1, 4, 2, 3, 5, 6)
CONFIG = dict(message_dilations=[1, 8], seeds=[0, 1, 2], epochs=8, batch=4096,
              screen_folds=[1, 4], confirmation_folds=[2, 3, 5, 6],
              max_fits=18, max_gpu_bytes=20*1024**3, max_ram_gib=20,
              inherited_recipe=S.CONFIG, trim_population='decided',
              ranking_authorized=False)
SOURCES = tuple(dict.fromkeys([SELF, 'scripts/e202_trajectory_extension.py',
                               *['scripts/'+n for n in S.FILES]]))
KEYS = ('body_sse', 'decided_sse', 'decided_trim3_sse', 'decided_trim20_sse')
ORIGINAL_NETWORK = S.network


def dilated_triplets(h, valid, dilation):
    import torch
    if not isinstance(dilation, int) or not 1 <= dilation < h.shape[-2]:
        raise ValueError('dilation must fit the observed stream')
    h = h * valid.unsqueeze(-1)
    zero = torch.zeros_like(h[..., :dilation, :])
    return torch.cat([torch.cat([zero, h[..., :-dilation, :]], -2), h,
                      torch.cat([h[..., dilation:, :], zero], -2)], -1)


def network(N, card, continuous, scalar, bins, n_num, arm, dilations=(1, 8)):
    """Keep exact E197 parameter creation/RNG order and replace only its forward."""
    import torch
    if arm != 'multiscale': raise ValueError('fixed multiscale arm required')
    net = ORIGINAL_NETWORK(N, card, continuous, scalar, bins, n_num, 'ordered')
    def forward(self, xn, xc, v, k, p):
        valid = p.bool(); known = k.float() * valid[..., None]
        h = self.token(torch.cat([v.float()*known, known], -1)) * valid[..., None]
        for block, dilation in zip(self.messages, dilations):
            h = (h + block(dilated_triplets(h, valid, dilation))) * valid[..., None]
        mean, maximum = S.pools(h, valid)
        context = torch.cat([mean.flatten(1), maximum.flatten(1), valid.float().mean(-1)], -1)
        return self.fusion(torch.cat([self.query(xn, xc), context], -1)).squeeze(-1)
    net.forward = types.MethodType(forward, net)
    return net


def check_prediction(z, meta):
    n = len(meta['mvt_id'])
    if len(np.unique(meta['mvt_id'])) != n: raise ValueError('duplicate evaluation IDs')
    if any(not np.array_equal(z[k], meta[k]) for k in ('mvt_id', 'y', 'month')):
        raise ValueError('exact heldout metadata differs')
    if any(np.asarray(z[k]).shape != (n,) or not np.isfinite(z[k]).all() for k in ('member', 'pred')):
        raise ValueError('incomplete/nonfinite prediction')


def check_fit(fit, seed, parameters):
    if (fit['seed'] != seed or len(fit['losses']) != 8 or not np.isfinite(fit['losses']).all()
            or fit['parameters'] != parameters or fit['peak_gpu_bytes'] > CONFIG['max_gpu_bytes']
            or not 0 < fit['peak_rss_gib'] <= CONFIG['max_ram_gib']):
        raise ValueError('fit, capacity or resource contract failed')


def gate(reports, phase):
    expected = {1, 4} if phase == 'screen' else set(range(1, 7)) if phase == 'six' else None
    if expected is None or len(reports) != len(expected) or {r['fold'] for r in reports} != expected:
        raise ValueError('gate requires exact unique fold inventory')
    if any(not np.isfinite([r['metrics'][k] for k in KEYS]).all() for r in reports):
        raise ValueError('nonfinite gate metrics')
    if phase == 'screen': return all(r['metrics'][k] < 0 for r in reports for k in KEYS)
    selected = [r for r in reports if r['fold'] in (1, 4)]
    confirmation = [r for r in reports if r['fold'] not in (1, 4)]
    return (gate(selected, 'screen')
            and sum(r['metrics']['decided_sse'] < 0 for r in reports) >= 4
            and all(sum(r['metrics'][k] for r in group) < 0
                    for group in (reports, confirmation) for k in KEYS))


def measured(control, candidate, meta):
    r = D.metrics(control, candidate, meta)
    delta = (candidate-meta['y'])**2 - (control-meta['y'])**2
    selected = np.sort(delta[meta['decided']])
    if len(selected) <= 40: raise ValueError('insufficient decided population for trim20')
    r.update(decided_trim3_sse=float(selected[3:-3].sum()),
             decided_trim20_sse=float(selected[20:-20].sum()))
    for label, mask in [('body', meta['y'] <= 7200), ('decided', meta['decided'])]:
        ec = (control[mask]-meta['y'][mask])**2; ep = (candidate[mask]-meta['y'][mask])**2
        r[label+'_rows'] = int(mask.sum())
        r[label+'_control_sse'] = float(ec.sum()); r[label+'_candidate_sse'] = float(ep.sum())
        r[label+'_rmse_delta'] = float(np.sqrt(ep.mean())-np.sqrt(ec.mean()))
    return r


def paths(fold):
    parent = 'e197' if fold in (1, 4) else 'e202'
    name = f'fold_{fold}_{fold+6}'
    return dict(bundle=f'data/{parent}_inputs_20261007',
                prepared=f'data/{parent}_prepared_20261007/{name}',
                output=f'data/{parent}_trajectory_20261007', name=name)


def registration(root, reg):
    if not re.fullmatch('[0-9a-f]{40}', reg or ''): raise ValueError('full registration SHA required')
    for name in (SELF, DOC, TEST):
        raw = subprocess.check_output(['git', 'show', reg+':'+name], cwd=root)
        if hashlib.sha256(raw).hexdigest() != C.sha(root/name):
            raise ValueError('registered source/protocol/test drift: '+name)
    if PROTOCOL.encode() not in subprocess.check_output(['git','show',reg+':HYPOTHESES_LOG.md'],cwd=root):
        raise ValueError('journal missing exact protocol')


def pack(root, reg):
    registration(root, reg)
    out = root/BASE/'input'
    if out.exists(): raise ValueError('fresh sealed input required')
    files = {}; folds = []; runtime = None
    for fold in FOLDS:
        p = paths(fold); bundle = root/p['bundle']; complete = C.read_json(root/p['output']/'complete.json')
        manifest = C.read_json(bundle/'manifest.json'); spec = manifest['spec']
        if manifest['spec_sha256'] != C.digest(spec): raise ValueError('parent digest differs')
        expected_reg = Q.OLD_REG if fold in (1,4) else '55fe9b68f7cc8405681f6c85d576979e067b55c3'
        expected_spec = Q.OLD_SPEC if fold in (1,4) else '6d604eaaca830fa83c18ebb78986b028d338869123d5f8c10b2450dde1ec4309'
        if spec['registration_commit'] != expected_reg or manifest['spec_sha256'] != expected_spec:
            raise ValueError('unexpected scientific comparator lineage')
        for name, sha in spec['scripts'].items():
            if C.sha(root/'scripts'/name) != sha: raise ValueError('frozen parent helper drift: '+name)
        if complete['registration_commit'] != expected_reg or complete['input_spec_sha256'] != expected_spec:
            raise ValueError('comparator complete lineage differs')
        if runtime is not None and runtime != complete['runtime']: raise ValueError('control runtime differs')
        runtime = complete['runtime']
        fs = next(f for f in spec['folds'] if f['fold'] == fold)
        od = root/p['output']/p['name']/'ordered'; fd = bundle/p['name']; qb = root/fs['query_bundle']
        Q.checked_arm(root,fd,qb,od,'ordered',complete)
        prep = next(r for r in complete['prepared_receipts'] if r['fold'] == fold)
        if prep['input_spec_sha256'] != expected_spec: raise ValueError('prepared parent differs')
        qp = M.verify_bundle(qb,E.OLD_REGISTRATION)
        if qp['spec_sha256'] != fs['query_spec_sha256']: raise ValueError('query spec differs')
        for path in (bundle/'manifest.json',bundle/'query_ids.npy',root/p['output']/'complete.json',
                     od/'manifest.json',qb/'manifest.json'):
            files[str(path.relative_to(root))] = C.sha(path)
        for filename in ('train_indices.npy','pred_indices.npy','control.npz','evaluation.npz'):
            path=fd/filename
            if C.sha(path) != spec['files'][p['name']+'/'+filename]: raise ValueError('parent input drift')
            files[str(path.relative_to(root))] = C.sha(path)
        for filename, sha in C.read_json(od/'manifest.json')['files'].items():
            files[str((od/filename).relative_to(root))] = sha
        folds.append(dict(fold=fold,**p,query_bundle=fs['query_bundle'],
                          query_spec_sha256=fs['query_spec_sha256'],prepared_receipt=prep,
                          comparator_parameters=complete['arms'][next(i for i,r in enumerate(complete['arms'])
                              if r['fold']==fold and r['arm']=='ordered')]['fits'][0]['parameters']))
    spec = dict(protocol=PROTOCOL,registration_commit=reg,configuration=CONFIG,folds=folds,runtime=runtime,
                source_sha256={n:C.sha(root/n) for n in SOURCES},files=files,
                protocol_sha256=C.sha(root/DOC),test_sha256=C.sha(root/TEST))
    out.mkdir();(out/'scripts').mkdir()
    for name in SOURCES:shutil.copy2(root/name,out/'scripts'/Path(name).name)
    C.write_json(out/'manifest.json',dict(spec=spec,spec_sha256=C.digest(spec)))
    print('SEALED',out,C.digest(spec),flush=True)


def verify(root, reg):
    m = C.read_json(root/BASE/'input/manifest.json'); s = m['spec']
    if (m['spec_sha256'] != C.digest(s) or s['protocol'] != PROTOCOL
            or s['registration_commit'] != reg or s['configuration'] != CONFIG
            or [f['fold'] for f in s['folds']] != list(FOLDS)):
        raise ValueError('E204 sealed specification differs')
    for name, sha in s['source_sha256'].items():
        if C.sha(Path(__file__).parent/Path(name).name) != sha: raise ValueError('active source drift: '+name)
    for name, sha in s['files'].items():
        if C.sha(root/name) != sha: raise ValueError('input drift: '+name)
    if C.sha(root/DOC) != s['protocol_sha256']: raise ValueError('protocol drift')
    return m


def verify_fold(root, fs):
    bundle=root/fs['bundle'];fd=bundle/fs['name'];qb=root/fs['query_bundle'];prep=root/fs['prepared']
    if C.read_json(prep/'complete.json') != fs['prepared_receipt']: raise ValueError('prepared receipt drift')
    for name,sha in fs['prepared_receipt']['files'].items():
        if C.sha(prep/name)!=sha: raise ValueError('prepared tensor drift: '+name)
    query=M.verify_bundle(qb,E.OLD_REGISTRATION)
    if query['spec_sha256']!=fs['query_spec_sha256']:raise ValueError('original query drift')
    ids=np.load(prep/'query_ids.npy');tr=np.load(fd/'train_indices.npy');pr=np.load(fd/'pred_indices.npy')
    if not np.array_equal(ids,np.load(bundle/'query_ids.npy')):raise ValueError('prepared query order differs')
    if len(np.unique(tr))!=len(tr) or len(np.unique(pr))!=len(pr) or np.intersect1d(tr,pr).size:
        raise ValueError('train/holdout overlap or duplication')
    norm=C.read_json(prep/'normalization.json')
    if norm['heldout_months']!=[fs['fold'],fs['fold']+6] or norm['fitted_query_ids_sha256']!=C.digest(ids[tr].tolist()):
        raise ValueError('normalization uses wrong population')
    meta=E.read(fd/'evaluation.npz');ix=np.load(qb/'control_indices.npy')
    if not np.array_equal(ids[pr][ix],meta['mvt_id']):raise ValueError('heldout query order differs')
    if any(not np.array_equal(meta[k],v) for k,v in E.read(qb/'metadata.npz').items()):
        raise ValueError('heldout labels differ')
    return query['spec'],tr,pr,meta


def smoke(root, reg):
    S.fixed_env();import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);guard=M.verify_cgroup()
    torch.cuda.set_per_process_memory_fraction(CONFIG['max_gpu_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
    m=verify(root,reg)
    if E.package_runtime()!=m['spec']['runtime']:raise ValueError('qualified runtime drift')
    out=root/BASE/'smoke.json'
    if out.exists():raise ValueError('fresh synthetic smoke required')
    rng=np.random.default_rng(204);x=rng.normal(size=(4096,270)).astype(np.float32);x[:,-20:]=x[:,-20:]>0
    cont,scalar,bins=E.numeric_bins(x);card=[11,49,1509,36,61,2189,155,6,10,7,446,196,586,887]
    N=E.trainer(root/'data/mlp_split_20261005/inputs/fold_1_7/src')
    xn=torch.from_numpy(x).to('cuda:0');xc=torch.zeros((4096,len(card)),dtype=torch.int64,device='cuda:0')
    v=torch.randn((4096,2,64,12),dtype=torch.float16,device='cuda:0');k=torch.ones_like(v,dtype=torch.uint8)
    p=torch.ones((4096,2,64),dtype=torch.uint8,device='cuda:0');p[:8]=0;p[8:16,1]=0;p[16:24,:,32:]=0;k[24:32,:,:,4]=0
    reserve=torch.empty((2085047*2*64*37,),dtype=torch.uint8,device='cuda:0')
    torch.manual_seed(204);control=ORIGINAL_NETWORK(N,card,cont,scalar,bins,270,'ordered').to('cuda:0').eval()
    torch.manual_seed(204);identity=network(N,card,cont,scalar,bins,270,'multiscale',dilations=(1,1)).to('cuda:0').eval()
    torch.manual_seed(204);net=network(N,card,cont,scalar,bins,270,'multiscale').to('cuda:0').eval()
    capacity=sum(t.numel() for t in control.parameters())
    if capacity!=sum(t.numel() for t in net.parameters()):raise ValueError('capacity mismatch')
    if any(not torch.equal(a,b) for a,b in zip(control.state_dict().values(),net.state_dict().values())):
        raise ValueError('initialization changed')
    with torch.no_grad():
        args=[a[:32] for a in (xn,xc,v,k,p)];base=control(*args)
        if not torch.equal(base,identity(*args)):raise ValueError('d1d1 adapter does not reproduce ordered exactly')
        candidate=net(*args)
        if torch.allclose(base[8:],candidate[8:],atol=1e-7,rtol=1e-7):raise ValueError('dilation has no synthetic effect')
        altered=v[:32].clone();altered[k[:32]==0]=1234;altered[p[:32]==0]=1234
        if not torch.equal(candidate,net(xn[:32],xc[:32],altered,k[:32],p[:32])):
            raise ValueError('unknown/PAD payload affected prediction')
    del control,identity;gc.collect();torch.cuda.empty_cache()
    opt=torch.optim.AdamW(net.parameters(),lr=.002);torch.cuda.reset_peak_memory_stats()
    for step in range(6):
        if step==2:torch.cuda.synchronize();start=time.monotonic()
        pred=net(xn,xc,v,k,p);loss=pred.square().mean();opt.zero_grad();loss.backward();opt.step()
        if not torch.isfinite(pred).all() or any(t.grad is not None and not torch.isfinite(t.grad).all() for t in net.parameters()):
            raise ValueError('nonfinite synthetic prediction/gradient')
    torch.cuda.synchronize();seconds=(time.monotonic()-start)/4;peak=torch.cuda.max_memory_allocated()
    if peak>CONFIG['max_gpu_bytes']:raise ValueError('synthetic memory exceeds20GiB')
    C.write_json(out,dict(passed=True,synthetic_only=True,registration_commit=reg,protocol=PROTOCOL,
        input_spec_sha256=m['spec_sha256'],worker_sha256=C.sha(__file__),runtime=E.package_runtime(),
        seconds_per_batch=seconds,peak_gpu_bytes=peak,parameters=capacity,cgroup=guard,
        ordered_d1d1_exact=True,initial_parameters_exact=True,pad_unknown_invariant=True))
    print('SYNTHETIC PASS',seconds,'seconds/batch',peak,'bytes',flush=True)


def event(out, label, **extra):
    row=dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),event=label,**extra)
    with (out/'events.jsonl').open('a') as f:
        f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(row,sort_keys=True),flush=True)


def result(root, out, m, reports, status):
    phase='screen' if len(reports)==2 else 'six'
    passed=gate(reports,phase)
    C.write_json(out/'result.json',dict(protocol=PROTOCOL,registration_commit=m['spec']['registration_commit'],
        input_spec_sha256=m['spec_sha256'],status=status,phase=phase,gate_pass=passed,reports=reports,
        ranking_authorized=False,accepted_for_submission=False,
        limitation='Saved ordered LGB5/preADSB comparator. Two E204-selected pairs plus four confirmation pairs; original ordered model had already been selected on these six pairs. No current-chain or ranking claim.',
        files={str(p.relative_to(out)):C.sha(p) for p in out.glob('fold_*/*') if p.is_file()}))
    event(out,status,gate_pass=passed,folds=len(reports))


def run(root, reg):
    S.fixed_env();import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);guard=M.verify_cgroup()
    torch.cuda.set_per_process_memory_fraction(CONFIG['max_gpu_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
    m=verify(root,reg);sm=C.read_json(root/BASE/'smoke.json')
    if (not sm['passed'] or sm['registration_commit']!=reg or sm['input_spec_sha256']!=m['spec_sha256']
            or sm['worker_sha256']!=C.sha(__file__) or sm['runtime']!=E.package_runtime()
            or sm['runtime']!=m['spec']['runtime']):raise ValueError('qualified smoke/runtime differs')
    out=root/BASE/'run'
    out.mkdir(exist_ok=False);reports=[];event(out,'started',registration_commit=reg,cgroup=guard)
    try:
        for fs in m['spec']['folds']:
            fold=fs['fold'];event(out,'verifying_fold',fold=fold)
            spec,tr,pr,meta=verify_fold(root,fs);qb=root/fs['query_bundle'];fd=root/fs['bundle']/fs['name']
            arrays,_=M.load_arrays(qb,spec)
            if len(tr)!=len(arrays['xnf']) or len(pr)!=len(arrays['xnp']):raise ValueError('query/context shape mismatch')
            cont,scalar,bins=E.numeric_bins(arrays['xnf']);N=E.trainer(qb/'src');tables=S.gpu_tables(root/fs['prepared'])
            od=out/fs['name'];od.mkdir();fits=[]
            np.savez(od/'bins.npz',continuous=cont,scalar=scalar,**{f'bin_{i}':b for i,b in enumerate(bins)})
            for seed in (0,1,2):
                event(out,'fit_started',fold=fold,seed=seed)
                # Process-local adapter: the frozen eight-epoch fit implementation stays byte-identical.
                S.network=network
                try: fit=S.fit(arrays,spec,N,cont,scalar,bins,tables,tr,pr,od,'multiscale',seed)
                finally: S.network=ORIGINAL_NETWORK
                check_fit(fit,seed,fs['comparator_parameters']);fits.append(fit)
                C.write_json(od/f'fit_{seed}.json',fit);event(out,'fit_complete',fold=fold,seed=seed,seconds=fit['seconds'])
            member=(np.mean([np.load(od/f'seed_{s}.npy') for s in (0,1,2)],axis=0)*spec['sd']+arrays['base_p'])[arrays['control_indices']]
            ctrl=E.read(fd/'control.npz');cand=E.assemble(ctrl,ctrl['reg']+ctrl['mlp_coefficient']*(member-ctrl['plain']),ctrl['lobt'])
            z={k:cand[k] for k in ('mvt_id','y','month','pred','reg','values','proba')};z['member']=member
            check_prediction(z,meta);np.savez_compressed(od/'prediction.npz',**z)
            saved=E.read(root/fs['output']/fs['name']/'ordered/prediction.npz');check_prediction(saved,meta)
            metrics=measured(saved['pred'],z['pred'],meta)
            report=dict(fold=fold,selection_group='screen' if fold in (1,4) else 'confirmation',metrics=metrics,
                        comparator_sha256=C.sha(root/fs['output']/fs['name']/'ordered/prediction.npz'))
            C.write_json(od/'complete.json',dict(protocol=PROTOCOL,registration_commit=reg,fold=fold,seeds=[0,1,2],fits=fits,
                report=report,files={p.name:C.sha(p) for p in od.iterdir() if p.is_file()}))
            reports.append(report);event(out,'fold_complete',fold=fold,body_delta=metrics['body_rmse_delta'],decided_delta=metrics['decided_rmse_delta'])
            del arrays,tables;gc.collect();torch.cuda.empty_cache()
            if len(reports)==2:
                result(root,out,m,reports,'screen_pass' if gate(reports,'screen') else 'screen_fail')
                if not gate(reports,'screen'):return
        verify(root,reg)
        result(root,out,m,reports,'six_pair_pass' if gate(reports,'six') else 'six_pair_fail')
    except BaseException as exc:
        C.write_json(out/'failed.json',dict(protocol=PROTOCOL,registration_commit=reg,error=repr(exc),completed_folds=len(reports)))
        event(out,'technical_failure',error=repr(exc));raise


def audit(root, reg):
    """Independently reconstruct saved seed means, predictions and all gate metrics."""
    m=verify(root,reg);out=root/BASE/'run';r=C.read_json(out/'result.json');reports=[]
    if (out/'failed.json').exists():raise ValueError('technical failure is not a scientific verdict')
    if r['registration_commit']!=reg or r['protocol']!=PROTOCOL or r['input_spec_sha256']!=m['spec_sha256']:
        raise ValueError('result lineage differs')
    for name,sha in r['files'].items():
        if C.sha(out/name)!=sha:raise ValueError('result artifact drift')
    for fs in m['spec']['folds'][:len(r['reports'])]:
        od=out/fs['name'];fd=root/fs['bundle']/fs['name'];qb=root/fs['query_bundle']
        done=C.read_json(od/'complete.json');spec=M.verify_bundle(qb,E.OLD_REGISTRATION)['spec']
        if done['fold']!=fs['fold'] or done['registration_commit']!=reg or done['seeds']!=[0,1,2] or len(done['fits'])!=3:
            raise ValueError('fold receipt differs')
        for name,sha in done['files'].items():
            if C.sha(od/name)!=sha:raise ValueError('fold artifact drift')
        for seed,fit in enumerate(done['fits']):check_fit(fit,seed,fs['comparator_parameters'])
        seeds=[np.load(od/f'seed_{s}.npy') for s in (0,1,2)]
        if any(not np.isfinite(a).all() for a in seeds):raise ValueError('nonfinite seed')
        member=(np.mean(seeds,axis=0)*spec['sd']+np.load(qb/'base_p.npy'))[np.load(qb/'control_indices.npy')]
        z=E.read(od/'prediction.npz');meta=E.read(fd/'evaluation.npz');check_prediction(z,meta)
        ctrl=E.read(fd/'control.npz');pred=E.assemble(ctrl,ctrl['reg']+ctrl['mlp_coefficient']*(member-ctrl['plain']),ctrl['lobt'])['pred']
        if not np.array_equal(member,z['member']) or not np.array_equal(pred,z['pred']):raise ValueError('seed/mixture arithmetic differs')
        saved=E.read(root/fs['output']/fs['name']/'ordered/prediction.npz');check_prediction(saved,meta)
        actual=measured(saved['pred'],pred,meta)
        if actual!=done['report']['metrics']:raise ValueError('metric reconstruction differs')
        reports.append(done['report'])
    if reports!=r['reports'] or gate(reports,r['phase'])!=r['gate_pass']:raise ValueError('saved gate differs')
    proof=dict(status='PASS',protocol=PROTOCOL,registration_commit=reg,result_sha256=C.sha(out/'result.json'),
               folds=len(reports),fits=3*len(reports),scientific_gate_pass=r['gate_pass'],ranking_authorized=False)
    C.write_json(out/'audit.json',proof);print(json.dumps(proof),flush=True)


def queue(root, reg):
    """Fresh process per phase, preventing smoke allocations/thread state from leaking."""
    for action in ('smoke','run','audit'):
        subprocess.run([str(root/'.venv-numemb/bin/python'),'-u',str(Path(__file__).resolve()),action,
                        '--root',str(root),'--registration-commit',reg],cwd=root,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['pack','verify','smoke','run','audit','queue'])
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--registration-commit',required=True)
    a=p.parse_args();root=a.root.resolve()
    globals()[a.action](root,a.registration_commit)


if __name__=='__main__':main()

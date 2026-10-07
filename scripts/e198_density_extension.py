"""Fixed remaining-four-pair replication of selected E196 mean versus MDN4.

Calls the immutable E196 fitter unchanged. No additional density search or
Gaussian fit; selected screening pairs are kept separate from confirmation.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import numpy as np
import e196_conditional_density as P
E,D,M,C=P.E,P.D,P.M,P.C
PROTOCOL='E198-mdn4-remaining4pair-confirmation-mean-control-3seed8epoch-v1'
FOLDS=(2,3,5,6)
SCREEN=(1,4)
ARMS=('mean','mdn4')
OLD_REG='9078e3508e81bb7c1e6cee432226063b0067b44e'
META_REG='fc7107de145a87a5a1ff93314f318b440b1de825'
META_SPEC='2b8f65e7212c91523d87a166ceb955376784223ee1dfb9cf1b403fa2f88a0f70'
SOURCE='data/e193_weather_inputs_20261007'
OLD_INPUT='data/e196_density_inputs_20261007'
OLD_OUTPUT='data/e196_density_20261007'
FILES=('e198_density_extension.py',*P.FILES)
CONFIG={**P.CONFIG,'folds':list(FOLDS),'arms':list(ARMS),'selected_screen_folds':list(SCREEN),
        'gate_decided_folds_minimum':4,'gate_pooled_negative':['body_sse','trim3_sse','trim20_sse'],
        'trim_pooling':'sum per-fold symmetric trimmed SSE, as repository gate'}


def check_registration(root,registration):
    if not re.fullmatch('[0-9a-f]{40}',registration or ''):raise ValueError('full registration required')
    code=subprocess.check_output(['git','show',registration+':scripts/e198_density_extension.py'],cwd=root)
    journal=subprocess.check_output(['git','show',registration+':HYPOTHESES_LOG.md'],cwd=root)
    if hashlib.sha256(code).hexdigest()!=C.sha(__file__) or PROTOCOL.encode() not in journal:raise ValueError('code+journal registration differs')


def checked_arm(root,fd,qb,od,arm,complete):
    meta=E.read(fd/'evaluation.npz');ctrl=E.read(fd/'control.npz');s=M.verify_bundle(qb,E.OLD_REGISTRATION)['spec']
    r=C.read_json(od/'manifest.json')
    if r not in complete['arms'] or r['arm']!=arm or r['seeds']!=[0,1,2] or len(r['fits'])!=3:raise ValueError('arm receipt mismatch')
    if [f['seed'] for f in r['fits']]!=[0,1,2] or any(f['epochs']!=8 or f['arm']!=arm for f in r['fits']):raise ValueError('fit recipe receipt differs')
    for n,h in r['files'].items():
        if C.sha(od/n)!=h:raise ValueError('prediction/model hash drift')
    z=E.read(od/'prediction.npz')
    for k in ('mvt_id','y','month'):
        if not np.array_equal(z[k],meta[k]):raise ValueError('heldout metadata differs')
    expected=(np.mean([np.load(od/f'seed_{i}.npy') for i in (0,1,2)],axis=0)*s['sd']+np.load(qb/'base_p.npy'))[np.load(qb/'control_indices.npy')]
    if not np.isfinite(expected).all() or not np.array_equal(z['member'],expected):raise ValueError('exact seed mean differs')
    cand=E.assemble(ctrl,ctrl['reg']+ctrl['mlp_coefficient']*(expected-ctrl['plain']),ctrl['lobt'])
    if not np.isfinite(z['pred']).all() or not np.array_equal(z['pred'],cand['pred']):raise ValueError('mixture arithmetic differs')
    return z


def check_selected_screen(root):
    bm=P.verify(root,root/OLD_INPUT,OLD_REG);out=root/OLD_OUTPUT;c=C.read_json(out/'complete.json');ev=C.read_json(out/'evaluation.json')
    if c['protocol']!=P.PROTOCOL or c['registration_commit']!=OLD_REG or c['input_spec_sha256']!=bm['spec_sha256'] or len(c['arms'])!=6 or not ev['mdn4_screen_pass']:raise ValueError('selected screen incomplete/failed')
    if c['runtime']!=bm['spec']['runtime'] or ev['registration_commit']!=OLD_REG or ev['protocol']!=P.PROTOCOL:raise ValueError('selected screen runtime/protocol differs')
    reports=[]
    for fs in bm['spec']['folds']:
        f=fs['fold'];name=f'fold_{f}_{f+6}';fd=root/OLD_INPUT/name;meta=E.read(fd/'evaluation.npz');qb=root/fs['query_bundle']
        pred={arm:checked_arm(root,fd,qb,out/name/arm,arm,c)['pred'] for arm in P.ARMS};ple=E.read(fd/'ple.npz')['pred']
        reports.append(dict(fold=f,gaussian_vs_mean=D.metrics(pred['mean'],pred['gaussian'],meta),mdn4_vs_mean=D.metrics(pred['mean'],pred['mdn4'],meta),mdn4_vs_gaussian=D.metrics(pred['gaussian'],pred['mdn4'],meta),gaussian_vs_ple=D.metrics(ple,pred['gaussian'],meta),mdn4_vs_ple=D.metrics(ple,pred['mdn4'],meta)))
    if reports!=ev['reports']:raise ValueError('selected screen metrics differ')
    if not all(r[k][n]<0 for r in reports for k in ('mdn4_vs_mean','mdn4_vs_gaussian','mdn4_vs_ple') for n in ('body_sse','decided_sse','trim3_sse','trim20_sse')):raise ValueError('selected screen gate differs')
    return bm,c,ev


def pack(root,out,registration):
    check_registration(root,registration)
    if out.exists():raise ValueError('fresh pack required')
    old,complete,ev=check_selected_screen(root);meta=C.read_json(root/SOURCE/'manifest.json');ms=meta['spec']
    if meta['spec_sha256']!=META_SPEC or C.digest(ms)!=META_SPEC or ms['registration_commit']!=META_REG or ms['feature_sha256']!=M.FEATURE_SHA:raise ValueError('metadata parent drift')
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent));(stage/'scripts').mkdir()
    for n in FILES:shutil.copy2(root/'scripts'/n,stage/'scripts'/n)
    folds=[]
    for fold in FOLDS:
        name=f'fold_{fold}_{fold+6}';source=root/SOURCE/name;dest=stage/name;dest.mkdir();fs=next(f for f in ms['folds'] if f['fold']==fold)
        qb=root/fs['query_bundle'];qm=M.verify_bundle(qb,E.OLD_REGISTRATION)
        if fs['query_spec_sha256']!=qm['spec_sha256']:raise ValueError('query lineage drift')
        reference=E.read(qb/'metadata.npz')
        for n in ('control.npz','evaluation.npz','ple.npz'):
            if C.sha(source/n)!=ms['files'][name+'/'+n]:raise ValueError('metadata/control drift')
            z=E.read(source/n)
            if any(not np.array_equal(z[k],v) for k,v in reference.items()):raise ValueError('control metadata mismatch')
            shutil.copy2(source/n,dest/n)
        folds.append(dict(fold=fold,query_bundle=fs['query_bundle'],query_spec_sha256=qm['spec_sha256']))
    for fold in SCREEN:
        name=f'fold_{fold}_{fold+6}';dest=stage/'selected_screen'/name;dest.mkdir(parents=True)
        for n in ('evaluation.npz','ple.npz'):shutil.copy2(root/OLD_INPUT/name/n,dest/n)
        for arm in ARMS:shutil.copy2(root/OLD_OUTPUT/name/arm/'prediction.npz',dest/(arm+'.npz'))
    C.write_json(stage/'selected_screen/evaluation.json',ev)
    spec=dict(protocol=PROTOCOL,registration_commit=registration,configuration=CONFIG,folds=folds,runtime=old['spec']['runtime'],metadata_parent_spec_sha256=META_SPEC,selected_screen_registration=OLD_REG,selected_screen_input_sha256=old['spec_sha256'],selected_screen_complete_sha256=C.sha(root/OLD_OUTPUT/'complete.json'),selected_screen_evaluation_sha256=C.sha(root/OLD_OUTPUT/'evaluation.json'),files={str(p.relative_to(stage)):C.sha(p) for p in stage.rglob('*') if p.is_file()},scripts={n:C.sha(stage/'scripts'/n) for n in FILES})
    C.write_json(stage/'manifest.json',dict(spec=spec,spec_sha256=C.digest(spec)));stage.rename(out);print('PACKED fixed remaining4pairs',flush=True)


def verify(root,bundle,registration):
    m=C.read_json(bundle/'manifest.json');s=m['spec']
    if s['protocol']!=PROTOCOL or s['registration_commit']!=registration or s['configuration']!=CONFIG or m['spec_sha256']!=C.digest(s) or [f['fold'] for f in s['folds']]!=list(FOLDS):raise ValueError('protocol/input mismatch')
    for n,h in s['scripts'].items():
        if C.sha(Path(__file__).parent/n)!=h:raise ValueError('active source drift '+n)
    for n,h in s['files'].items():
        if C.sha(bundle/n)!=h:raise ValueError('packed input drift '+n)
    for f in s['folds']:
        if M.verify_bundle(root/f['query_bundle'],E.OLD_REGISTRATION)['spec_sha256']!=f['query_spec_sha256']:raise ValueError('query tensor drift')
    return m


def measured(control,candidate,meta):
    r=D.metrics(control,candidate,meta)
    for label,mask in (('body',meta['y']<=7200),('decided',meta['decided'])):
        r[label+'_rows']=int(mask.sum());r[label+'_rmse_control']=float(np.sqrt(np.mean((control[mask]-meta['y'][mask])**2)));r[label+'_rmse_candidate']=float(np.sqrt(np.mean((candidate[mask]-meta['y'][mask])**2)))
    return r


def pool(reports,key):
    r=dict(folds=len(reports),decided_folds_improved=sum(x[key]['decided_sse']<0 for x in reports))
    for k in ('all_sse','body_sse','decided_sse','trim3_sse','trim20_sse'):r[k]=float(sum(x[key][k] for x in reports))
    return r


def gate(summary):
    return summary['folds']==6 and summary['decided_folds_improved']>=4 and all(summary[k]<0 for k in ('body_sse','trim3_sse','trim20_sse'))


def evaluate(root,bundle,out,registration):
    m=verify(root,bundle,registration);complete=C.read_json(out/'complete.json')
    if complete['protocol']!=PROTOCOL or complete['registration_commit']!=registration or complete['input_spec_sha256']!=m['spec_sha256'] or len(complete['arms'])!=8:raise ValueError('extension incomplete')
    reports=[]
    for fold in range(1,7):
        name=f'fold_{fold}_{fold+6}';selected=fold in SCREEN
        fd=bundle/'selected_screen'/name if selected else bundle/name;meta=E.read(fd/'evaluation.npz')
        if selected:
            pred={arm:E.read(fd/(arm+'.npz')) for arm in ARMS}
        else:
            fs=next(f for f in m['spec']['folds'] if f['fold']==fold);qb=root/fs['query_bundle'];pred={arm:checked_arm(root,fd,qb,out/name/arm,arm,complete) for arm in ARMS}
        for z in pred.values():
            if any(not np.array_equal(z[k],meta[k]) for k in ('mvt_id','y','month')) or not np.isfinite(z['pred']).all():raise ValueError('combined metadata/finite mismatch')
        ple=E.read(fd/'ple.npz')
        if any(not np.array_equal(ple[k],meta[k]) for k in ('mvt_id','y','month')):raise ValueError('saved PLE metadata differs')
        reports.append(dict(fold=fold,role='selected_screen' if selected else 'remaining_confirmation',mdn4_vs_mean=measured(pred['mean']['pred'],pred['mdn4']['pred'],meta),mdn4_vs_ple=measured(ple['pred'],pred['mdn4']['pred'],meta),mean_vs_ple=measured(ple['pred'],pred['mean']['pred'],meta)))
    pooled={key:pool(reports,key) for key in ('mdn4_vs_mean','mdn4_vs_ple')};confirmation=[r for r in reports if r['role']=='remaining_confirmation']
    result=dict(protocol=PROTOCOL,registration_commit=registration,reports=reports,pooled_six_pairs=pooled,remaining_four_pairs={key:pool(confirmation,key) for key in pooled},six_pair_gate_pass=all(gate(s) for s in pooled.values()),accepted_for_submission=False,limitation='Pairs1/7 and4/10 selected MDN4; remaining4pairs are confirmation. Six-pair result includes selection and is LGB5/preADSB, not current-chain acceptance.')
    C.write_json(out/'evaluation.json',result);print(json.dumps(result),flush=True)


def run(root,bundle,out,registration):
    P.fixed_env();import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);cgroup=M.verify_cgroup();m=verify(root,bundle,registration);runtime=E.package_runtime()
    if runtime!=m['spec']['runtime']:raise ValueError('GPU runtime drift')
    if out.exists():raise ValueError('fresh output required, no resume')
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent));done=[]
    for fs in m['spec']['folds']:
        name=f'fold_{fs["fold"]}_{fs["fold"]+6}';qb=root/fs['query_bundle'];s=M.verify_bundle(qb,E.OLD_REGISTRATION)['spec'];arrays,meta=M.load_arrays(qb,s)
        continuous,scalar,bins=E.numeric_bins(arrays['xnf']);N=E.trainer(qb/'src');ctrl=E.read(bundle/name/'control.npz')
        for arm in ARMS:
            print(f'START {name} {arm}',flush=True);od=stage/name/arm;od.mkdir(parents=True);fits=[]
            np.savez(od/'bins.npz',continuous=continuous,scalar=scalar,**{f'bin_{j}':b for j,b in enumerate(bins)})
            for seed in (0,1,2):fits.append(P.fit(arrays,s,continuous,scalar,bins,N,arm,seed,od))
            member=(np.mean([np.load(od/f'seed_{i}.npy') for i in (0,1,2)],axis=0)*s['sd']+arrays['base_p'])[arrays['control_indices']]
            cand=E.assemble(ctrl,ctrl['reg']+ctrl['mlp_coefficient']*(member-ctrl['plain']),ctrl['lobt'])
            np.savez_compressed(od/'prediction.npz',**{k:cand[k] for k in ('mvt_id','y','month','pred','reg','values','proba')},member=member)
            receipt=dict(fold=fs['fold'],arm=arm,seeds=[0,1,2],fits=fits,files={p.name:C.sha(p) for p in od.iterdir() if p.is_file()});C.write_json(od/'manifest.json',receipt);done.append(receipt)
        del arrays;gc.collect()
    verify(root,bundle,registration);C.write_json(stage/'complete.json',dict(protocol=PROTOCOL,registration_commit=registration,input_spec_sha256=m['spec_sha256'],runtime=runtime,cgroup=cgroup,arms=done,accepted_for_submission=False))
    stage.rename(out);print('COMPLETE24fits; fixed six-pair evaluation follows',flush=True);evaluate(root,bundle,out,registration)


def selftest():
    base=dict(folds=6,decided_folds_improved=4,body_sse=-1.,trim3_sse=-1.,trim20_sse=-1.)
    assert gate(base) and not gate({**base,'decided_folds_improved':3}) and not gate({**base,'body_sse':0.}) and not gate({**base,'folds':4})
    for k in ('body_sse','trim3_sse','trim20_sse'):assert not gate({**base,k:1.})
    assert ARMS==('mean','mdn4') and set(FOLDS).isdisjoint(SCREEN) and sorted(FOLDS+SCREEN)==list(range(1,7))
    print('PASS fixedfold coverage, no Gaussian, gate boundaries; training directly reuses immutable E196.fit')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['selftest','pack','verify','run','evaluate']);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--input',type=Path);p.add_argument('--output',type=Path);p.add_argument('--registration-commit');a=p.parse_args();root=a.root.resolve()
    if a.action=='selftest':return selftest()
    if not re.fullmatch('[0-9a-f]{40}',a.registration_commit or ''):p.error('full registration required')
    if a.action=='pack':pack(root,a.output.resolve(),a.registration_commit)
    elif a.action=='verify':print(verify(root,a.input.resolve(),a.registration_commit)['spec_sha256'])
    elif a.action=='run':run(root,a.input.resolve(),a.output.resolve(),a.registration_commit)
    else:evaluate(root,a.input.resolve(),a.output.resolve(),a.registration_commit)


if __name__=='__main__':main()

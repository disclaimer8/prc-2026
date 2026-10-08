"""Registered, CV-only E207 member substitution through saved E203 pre-E81.

No fitting, ranking, scheduler, source mutation, or automatic next stage.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

for k,v in {'POLARS_MAX_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
            'OMP_WAIT_POLICY':'PASSIVE','GOMP_SPINCOUNT':'0'}.items(): os.environ.setdefault(k,v)
import numpy as np
import polars as pl
import e203_trajectory_production as W
import e207_full_token_sampler_v3 as P

A,C,E,Q,G=W.A,P.C,P.E,P.Q,P.G
PROTOCOL='E207-fullK64-fixed025-current-E203-preE81-sixpair-zero-fit-v1'
SELF='scripts/e207_current_chain_20261008.py'
TEST='tests/test_e207_current_chain_20261008.py'
AREA='analysis/e207_current_chain_20261008'
DOC=AREA+'/PROTOCOL.md'
SPEC=AREA+'/inventory.json'
AUDIT=AREA+'/parent_audit.json'
ATTEST=AREA+'/registration.json'
EVENTS='analysis/e207_full_token_v3_20261008/execution/events.jsonl'
GPU_UNIT='prc-e207-v3-trajectory-20261008.service'
PARENT_REG='f43405e7c6836230862837b66fad081ae4d69777'
PARENT_PROTOCOL='E207-full-K64-retain-original-raw-index-fill-sixpair-3seed8epoch-v3'
PACKED_SPEC='598b6ff1398e85166665ed11cab066f402131cc883d5707ce25be14963c06eae'
MATERIALIZED_SPEC='56e1417bb71aa917847a7536ffbf8f0a9110c768a66bbaa6bf32e851fbffc4ca'
PARENT_KEYS=('body_sse','decided_sse','decided_trim3_sse','decided_trim20_sse')
E203_REG='6e95e3a65936a3401e41499bb340f2d54a1a2121'
E203_RECEIPT_SHA='c137929b58c498195d010e24326ea11a6de038bd3d07daebedbf276aac330ba4'
E203_RESULT_SHA='455c6d0f6242e649817f743d90262f67d5422ffe81e537b3324c1e1be7027cab'
E203_PRED_SHA='381563def761db5e201efbb6349ad6668ff4dc8c2ae00b39eed19e8d3694d4b7'
SOURCES=tuple(dict.fromkeys([SELF,*('scripts/'+n for n in (*W.FILES,*P.FILES))]))
ROWS,EXCLUDED=2085047,126


def check_files(root, files):
    for name, digest in files.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or not re.fullmatch('[0-9a-f]{64}',digest):
            raise ValueError('unsafe artifact path/digest')
        if A.sha(root/p)!=digest: raise ValueError('artifact SHA differs: '+name)


def six_reports(reports):
    if len(reports)!=6 or {r['fold'] for r in reports}!=set(range(1,7)):
        raise ValueError('exact six unique pairs required')


def chain_gate(reports):
    six_reports(reports)
    keys=(('body','delta_sse'),('decided','delta_sse'),('decided','trim3_sse'),('decided','trim20_sse'))
    if any(not np.isfinite(r['metrics'][p][k]) for r in reports for p,k in keys):
        raise ValueError('nonfinite chain gate')
    return (sum(r['metrics']['decided']['delta_sse']<0 for r in reports)>=4
            and all(sum(r['metrics'][p][k] for r in reports)<0 for p,k in keys))


def parent_gate(evaluation):
    e=evaluation; reports=e['reports']; six_reports(reports)
    if any(not np.isfinite(r['metrics'][k]) for r in reports for k in PARENT_KEYS):
        raise ValueError('nonfinite parent gate')
    pooled={k:float(sum(r['metrics'][k] for r in reports)) for k in PARENT_KEYS}
    wins=sum(r['metrics']['decided_sse']<0 for r in reports)
    if (e['protocol']!=PARENT_PROTOCOL or e['registration_commit']!=PARENT_REG
        or e['accepted_for_submission'] is not False or e['six_pair_gate_pass'] is not True
        or e['pooled']!=pooled or e['decided_pairs_improved']!=wins
        or wins<4 or not all(v<0 for v in pooled.values())):
        raise ValueError('completed E207 six-pair PASS required')


def terminal_gate(rows, evaluation_sha):
    if (not rows or rows[0].get('event')!='controller_started'
        or rows[0].get('registration_commit')!=PARENT_REG
        or rows[-1].get('event')!='complete_no_ranking' or rows[-1].get('gate') is not True
        or rows[-1].get('evaluation_sha256')!=evaluation_sha
        or any(r.get('event')=='failed_no_retry' for r in rows)):
        raise ValueError('successful terminal E207 controller receipt required')
    starts=[r for r in rows if r.get('event')=='started' and r.get('unit')==GPU_UNIT]
    ends=[r for r in rows if r.get('event')=='exited' and r.get('unit')==GPU_UNIT]
    if len(starts)!=1 or len(ends)!=1: raise ValueError('exact owned GPU lifecycle required')
    initial=starts[0]['properties']; final=ends[0]['properties']
    if (not initial.get('InvocationID') or final.get('InvocationID')!=initial['InvocationID']
        or final.get('Result')!='success' or final.get('ExecMainStatus')!='0'
        or final.get('MainPID')!='0' or final.get('ActiveState') not in ('inactive','active')
        or final.get('SubState') not in ('dead','exited')):
        raise ValueError('GPU terminal identity/status differs')


def fixed_feasibility(before,after,excluded):
    if before.shape!=after.shape or excluded.shape!=(len(before),): raise ValueError('feasibility shape differs')
    if np.any(np.any(before!=after,axis=1)&~excluded): raise ValueError('candidate would broaden frozen exclusions')


def prepared_gate(receipts,worker_sha):
    six_reports(receipts)
    for r in receipts:
        if (r['protocol']!=PARENT_PROTOCOL or r['registration_commit']!=PARENT_REG
            or r['input_spec_sha256']!=MATERIALIZED_SPEC or r['source_sha256']!=worker_sha
            or r['rows']!=ROWS): raise ValueError('prepared parent lineage/population differs')


def regular_member(cat,lgb,anchors,member):
    if len(anchors)!=5 or any(np.asarray(a).shape!=np.asarray(member).shape for a in (cat,lgb,*anchors)):
        raise ValueError('regular component shapes differ')
    r=(1-sum(A.WEIGHTS))*(.65*cat+.35*lgb)+sum(w*g for w,g in zip(A.WEIGHTS,[*anchors,member]))
    if not np.isfinite(r).all(): raise ValueError('nonfinite regular member')
    return r


def registration(root,reg):
    if not re.fullmatch('[0-9a-f]{40}',reg or ''): raise ValueError('full child prereg commit required')
    for name in (SELF,TEST,DOC,SPEC):
        raw=subprocess.check_output(['git','show',reg+':'+name],cwd=root)
        if hashlib.sha256(raw).hexdigest()!=A.sha(root/name): raise ValueError('registered file differs: '+name)
    if PROTOCOL.encode() not in subprocess.check_output(['git','show',reg+':HYPOTHESES_LOG.md'],cwd=root):
        raise ValueError('child protocol absent from registered journal')


def attest(root,reg):
    verify(root,reg)
    A.save(root/ATTEST,dict(protocol=PROTOCOL,registration_commit=reg,journal_protocol_verified=True,
        files={n:A.sha(root/n) for n in (SELF,TEST,DOC,SPEC)}))


def runtime_registration(root,reg):
    """Root-produced Git admission receipt, transported with frozen files to Home."""
    r=A.read(root/ATTEST)
    if (not re.fullmatch('[0-9a-f]{40}',reg or '') or r['protocol']!=PROTOCOL
        or r['registration_commit']!=reg or r['journal_protocol_verified'] is not True
        or set(r['files'])!={SELF,TEST,DOC,SPEC}): raise ValueError('remote registration receipt differs')
    check_files(root,r['files'])


def seal(root):
    """Only metadata and opaque hashes; never deserialize labels/predictions/errors."""
    if (root/SPEC).exists(): raise ValueError('fresh inventory required')
    check_files(root,{W.RECEIPT:E203_RECEIPT_SHA,W.CV_OUT+'/result.json':E203_RESULT_SHA,
                      W.CV_OUT+'/predictions.parquet':E203_PRED_SHA})
    saved=A.read(root/W.RECEIPT)['inputs']; packed=A.read(root/P.INPUT/'manifest.json')
    if packed['spec_sha256']!=PACKED_SPEC or C.digest(packed['spec'])!=PACKED_SPEC:
        raise ValueError('frozen E207 packed specification differs')
    names=['data/features_fin1b.parquet',A.BUNDLE+'/manifest.json',A.BUNDLE+'/natural_features.parquet']
    canonical=sorted(str(p.relative_to(root)) for p in (root/'data/prc-2026-datasets').glob('training_*.parquet'))
    if len(canonical)!=12: raise ValueError('exact 12 training sources required')
    names.extend(canonical)
    for fold in range(1,7):
        name=f'fold_{fold}_{fold+6}'
        names += [f'data/cv_preds_fin1b_hz/{name}.npz',f'data/cat_extra175_20261006/merged/{name}.npz',
                  f'data/lgb_fin1b_oof_20261005/{name}/complete/prediction.npz',
                  *[f'data/cv_e130{x}/{name}.npz' for x in [*A.ANCHORS,'motion']]]
        names += [f'{A.MODELS}/{name}/{stage}/seed{seed}.txt' for stage in ('e141','e64','e188') for seed in A.SEEDS]
    files={n:saved[n] for n in names}
    files.update({W.RECEIPT:E203_RECEIPT_SHA,W.CV_OUT+'/result.json':E203_RESULT_SHA,
                  W.CV_OUT+'/predictions.parquet':E203_PRED_SHA,P.INPUT+'/manifest.json':A.sha(root/P.INPUT/'manifest.json')})
    for fold in range(1,7):
        files[str(W.trajectory_path(Path('.'),fold)/'prediction.npz')]=packed['spec']['files'][f'fold_{fold}_{fold+6}/ordered.npz']
    check_files(root,files)
    value=dict(protocol=PROTOCOL,parent_registration=PARENT_REG,parent_protocol=PARENT_PROTOCOL,
        packed_spec=PACKED_SPEC,materialized_spec=MATERIALIZED_SPEC,e203_registration=E203_REG,
        source_sha256={n:A.sha(root/n) for n in SOURCES},files=files,canonical_training_files=canonical,
        rows=ROWS,excluded=EXCLUDED,weight=.25,new_fits=0,ranking_authorized=False,
        prior_body_delta_seconds=0,expert_range_seconds=[-1,1])
    A.save(root/SPEC,value)
    print('SEALED METADATA ONLY',len(files),'inputs',flush=True)


def verify(root,reg,inputs=False,runtime=False):
    (runtime_registration if runtime else registration)(root,reg); s=A.read(root/SPEC)
    if (s['protocol']!=PROTOCOL or s['parent_registration']!=PARENT_REG
        or s['parent_protocol']!=PARENT_PROTOCOL or s['packed_spec']!=PACKED_SPEC
        or s['materialized_spec']!=MATERIALIZED_SPEC or s['e203_registration']!=E203_REG
        or s['rows']!=ROWS or s['excluded']!=EXCLUDED or s['weight']!=.25
        or s['new_fits']!=0 or s['ranking_authorized'] is not False): raise ValueError('child inventory differs')
    check_files(root,s['source_sha256'])
    for n,h in s['source_sha256'].items():
        if A.sha(Path(__file__).parent/Path(n).name)!=h: raise ValueError('active helper differs: '+n)
    if inputs:
        check_files(root,s['files'])
        actual=sorted(str(p.relative_to(root)) for p in (root/'data/prc-2026-datasets').glob('training_*.parquet'))
        if actual!=s['canonical_training_files']: raise ValueError('canonical training set changed')
    return s


def parent_metadata(root):
    packed=A.read(root/P.INPUT/'manifest.json'); material=A.read(root/P.MATERIALIZED)
    if (packed['spec_sha256']!=PACKED_SPEC or C.digest(packed['spec'])!=PACKED_SPEC
        or material['spec_sha256']!=MATERIALIZED_SPEC or C.digest(material['spec'])!=MATERIALIZED_SPEC
        or material['spec'].get('parent_spec_sha256')!=PACKED_SPEC): raise ValueError('parent spec lineage differs')
    expected={**packed['spec'],'parent_spec_sha256':PACKED_SPEC,**P.derived_raw(root,packed,PARENT_REG)}
    if material['spec']!=expected: raise ValueError('materialized raw receipts differ')
    complete=A.read(root/P.OUTPUT/'complete.json'); evaluation=A.read(root/P.OUTPUT/'evaluation.json')
    parent_gate(evaluation)
    terminal_gate([json.loads(line) for line in (root/EVENTS).read_text().splitlines()],A.sha(root/P.OUTPUT/'evaluation.json'))
    if (complete['protocol']!=PARENT_PROTOCOL or complete['registration_commit']!=PARENT_REG
        or complete['input_spec_sha256']!=MATERIALIZED_SPEC or complete['runtime']!=material['spec']['runtime']
        or complete['accepted_for_submission'] is not False): raise ValueError('parent completion/runtime differs')
    six_reports(complete['arms']); prepared_gate(complete['prepared_receipts'],packed['spec']['scripts'][Path(P.SELF).name])
    for fs in material['spec']['folds']:
        r=next(r for r in complete['arms'] if r['fold']==fs['fold'])
        if r['arm']!='ordered' or r['seeds']!=[0,1,2] or [f['seed'] for f in r['fits']]!=[0,1,2]:
            raise ValueError('parent fixed seeds/arm differ')
        for f in r['fits']:
            if (len(f['losses'])!=8 or not np.isfinite(f['losses']).all()
                or f['parameters']!=fs['parameters'] or not np.isfinite(f['peak_gpu_bytes'])
                or not 0<f['peak_gpu_bytes']<=20*1024**3): raise ValueError('parent fit evidence differs')
    return packed,material,complete,evaluation


def portable_parent_files(root,complete):
    names=[P.INPUT+'/manifest.json',P.MATERIALIZED,EVENTS,P.OUTPUT+'/complete.json',P.OUTPUT+'/evaluation.json']
    names += [f'{P.RAW}/s{shard}/complete.json' for shard in (0,1)]
    for fold in range(1,7):
        name=f'fold_{fold}_{fold+6}'
        names += [f'{P.INPUT}/{name}/{n}' for n in ('evaluation.npz','ordered.npz')]
        names += [f'{P.OUTPUT}/{name}/ordered/{n}' for n in ('manifest.json','prediction.npz')]
    return {n:A.sha(root/n) for n in names}


def recompute_parent(root,complete,evaluation,check_seeds=False):
    ids=[]
    packed=A.read(root/P.INPUT/'manifest.json')
    for fs in packed['spec']['folds']:
        fold=fs['fold']; name=f'fold_{fold}_{fold+6}'; fd=root/P.INPUT/name; od=root/P.OUTPUT/name/'ordered'
        r=A.read(od/'manifest.json')
        if r not in complete['arms'] or r['fold']!=fold: raise ValueError('parent arm manifest differs')
        if A.sha(od/'prediction.npz')!=r['files']['prediction.npz']: raise ValueError('parent prediction drift')
        for n in ('evaluation.npz','ordered.npz'):
            if A.sha(fd/n)!=packed['spec']['files'][name+'/'+n]: raise ValueError('packed heldout metadata differs')
        if check_seeds:
            cand=Q.checked_arm(root,fd,root/fs['query_bundle'],od,'ordered',complete)
        else: cand=E.read(od/'prediction.npz')
        old=E.read(fd/'ordered.npz'); meta=E.read(fd/'evaluation.npz')
        G.check_prediction(old,meta);G.check_prediction(cand,meta)
        if G.measured(old['pred'],cand['pred'],meta)!=next(r['metrics'] for r in evaluation['reports'] if r['fold']==fold):
            raise ValueError('parent metrics cannot be reconstructed')
        ids.append(meta['mvt_id'])
    ids=np.concatenate(ids)
    if len(ids)!=ROWS or len(np.unique(ids))!=ROWS: raise ValueError('parent six-pair IDs differ')


def audit_parent(root,reg):
    """After prereg and parent PASS, on Home; no prepared F table or model call."""
    verify(root,reg,runtime=True); packed,material,complete,evaluation=parent_metadata(root)
    P.verify_bundle(root,root/P.INPUT,PARENT_REG)
    recompute_parent(root,complete,evaluation,check_seeds=True)
    A.save(root/AUDIT,dict(protocol=PROTOCOL,registration_commit=reg,status='PASS',
        inventory_sha256=A.sha(root/SPEC),worker_sha256=A.sha(root/SELF),registration_sha256=A.sha(root/ATTEST),parent_registration=PARENT_REG,
        packed_spec=PACKED_SPEC,materialized_spec=MATERIALIZED_SPEC,folds=6,fits_verified=18,
        seed_mean_and_mixture_exact=True,files=portable_parent_files(root,complete),
        checked_arm_artifacts={f"{P.OUTPUT}/fold_{r['fold']}_{r['fold']+6}/ordered/{n}":h
                               for r in complete['arms'] for n,h in r['files'].items()},new_fits=0))
    print('PARENT AUDIT PASS; collect receipt and files keys, not prepared tables',flush=True)


def checked_parent(root,reg):
    runtime_registration(root,reg)
    r=A.read(root/AUDIT)
    if (r['protocol']!=PROTOCOL or r['registration_commit']!=reg or r['status']!='PASS'
        or r['inventory_sha256']!=A.sha(root/SPEC) or r['worker_sha256']!=A.sha(root/SELF)
        or r['registration_sha256']!=A.sha(root/ATTEST)
        or r['parent_registration']!=PARENT_REG or r['packed_spec']!=PACKED_SPEC
        or r['materialized_spec']!=MATERIALIZED_SPEC or r['folds']!=6 or r['fits_verified']!=18
        or r['seed_mean_and_mixture_exact'] is not True or r['new_fits']!=0): raise ValueError('parent audit differs')
    check_files(root,r['files']); _,_,complete,evaluation=parent_metadata(root)
    if portable_parent_files(root,complete)!=r['files']: raise ValueError('parent artifact inventory differs')
    artifacts={f"{P.OUTPUT}/fold_{a['fold']}_{a['fold']+6}/ordered/{n}":h for a in complete['arms'] for n,h in a['files'].items()}
    if artifacts!=r['checked_arm_artifacts']: raise ValueError('audited arm inventory differs')
    recompute_parent(root,complete,evaluation)
    return dict(audit_sha256=A.sha(root/AUDIT),evaluation_sha256=A.sha(root/P.OUTPUT/'evaluation.json'),
                complete_sha256=A.sha(root/P.OUTPUT/'complete.json'),controller_sha256=A.sha(root/EVENTS))


def cv(root,reg,threads=4):
    if not 1<=threads<=4: raise ValueError('1..4 prediction threads only')
    verify(root,reg,inputs=True); parent=checked_parent(root,reg)
    out=root/AREA/'cv';out.mkdir(exist_ok=False);start=time.monotonic()
    current=pl.read_parquet(root/W.CV_OUT/'predictions.parquet').sort(A.ID)
    if current.height!=ROWS or current[A.ID].n_unique()!=ROWS or int((~current['evaluation_mask']).sum())!=EXCLUDED:
        raise ValueError('frozen evaluation population differs')
    natural=pl.read_parquet(root/A.BUNDLE/'natural_features.parquet');stage_spec=A.read(root/A.BUNDLE/'manifest.json')['spec']
    cols=[A.ID,A.Y,'split','month','ADEP_mvt','FLIGHT_ID_mvt','LOBT_flt','d_mvt_lobt','d_mvt_sched','d_mvt_iobt','d_mvt_eobt']
    feature=pl.read_parquet(root/'data/features_fin1b.parquet',columns=cols).filter(pl.col('split')=='train').sort(A.ID)
    ra,rb=A.rules(root);frames=[];reports=[]
    for fold in range(1,7):
        name=f'fold_{fold}_{fold+6}';f=feature.filter(pl.col('month').is_in([fold,fold+6]))
        ids=f[A.ID].to_numpy();y=f[A.Y].to_numpy();mo=f['month'].to_numpy()
        old=current.filter(pl.col('month').is_in([fold,fold+6]))
        if any(not np.array_equal(old[k].to_numpy(),f[k].to_numpy()) for k in (A.ID,A.Y,'month')): raise ValueError('E203 IDs/labels differ')
        excluded=~old['evaluation_mask'].to_numpy()
        d=f.select(A.ID).join(natural,on=A.ID,how='left',validate='1:1',maintain_order='left')
        z=A.aligned(root/'data/cv_preds_fin1b_hz'/(name+'.npz'),ids,y)
        cat=A.aligned(root/'data/cat_extra175_20261006/merged'/(name+'.npz'),ids,y,mo)['reg2']
        lgb=A.aligned(root/'data/lgb_fin1b_oof_20261005'/name/'complete/prediction.npz',ids,y,mo)['reg2']
        ordered=A.aligned(W.trajectory_path(root,fold)/'prediction.npz',ids,y,mo)['member']
        full=A.aligned(root/P.OUTPUT/name/'ordered/prediction.npz',ids,y,mo)['member']
        anchors=[A.aligned(root/'data'/('cv_e130'+x)/(name+'.npz'),ids)['reg2'] for x in A.ANCHORS]
        motion=A.aligned(root/'data/cv_e130motion'/(name+'.npz'),ids)['reg2'];pred=[]
        for member in (ordered,full):
            r=regular_member(cat,lgb,anchors,member);v=z['values'].astype(float).copy();v[:,0]=r;v[:,-1]=r+86400
            fixed_feasibility(A.feasibility(z['values'],f['d_mvt_lobt'].to_numpy()),A.feasibility(v,f['d_mvt_lobt'].to_numpy()),excluded)
            base,proba=A.upstream(z,r,motion,f,np.isin(ids,ra),np.isin(ids,rb),excluded)
            _,aug=A.stage_predictions(root,name,d,base,proba,stage_spec,threads)
            aug[excluded]=np.rint(z['pred'][excluded]);pred.append(aug)
        if not np.array_equal(pred[0],old['ordered'].to_numpy()): raise ValueError('current E203 control is not exact')
        if not all(np.isfinite(p).all() for p in pred): raise ValueError('nonfinite chain predictions')
        frame=old.with_columns(pl.Series('current_control',pred[0]),pl.Series('full_tokens',pred[1]))
        metrics={k:W.paired(y,pred[0],pred[1],mask) for k,mask in W.masks(frame).items()}
        reports.append(dict(fold=fold,control_exact=True,excluded=int(excluded.sum()),additional_excluded=0,metrics=metrics))
        frame.write_parquet(out/(name+'.parquet'));frames.append(frame)
    allf=pl.concat(frames).sort(A.ID)
    if allf.height!=ROWS or allf[A.ID].n_unique()!=ROWS: raise ValueError('pooled child IDs differ')
    allf.write_parquet(out/'predictions.parquet')
    result=dict(protocol=PROTOCOL,registration_commit=reg,parent=parent,folds=reports,control_exact=True,
        gate_pass=chain_gate(reports),rows=ROWS,excluded=EXCLUDED,additional_excluded=0,
        metrics={k:W.paired(allf[A.Y].to_numpy(),allf['current_control'].to_numpy(),allf['full_tokens'].to_numpy(),mask)
                 for k,mask in W.masks(allf).items()},files={p.name:A.sha(p) for p in out.glob('*.parquet')},
        seconds=time.monotonic()-start,new_cpu_fits=0,new_gpu_fits=0,full_tail_oof_validated=False,
        selection_informed=True,accepted_for_submission=False,ranking_authorized=False)
    verify(root,reg,inputs=True)
    if parent!=checked_parent(root,reg): raise ValueError('parent changed during comparison')
    A.save(out/'result.json',result)
    checked_cv(root,reg)
    A.save(out/'complete.json',dict(protocol=PROTOCOL,registration_commit=reg,result_sha256=A.sha(out/'result.json'),
        gate_pass=result['gate_pass'],accepted_for_submission=False,ranking_authorized=False))
    print('E207 CURRENT CHAIN GATE',result['gate_pass'],flush=True)


def checked_cv(root,reg):
    verify(root,reg,inputs=True);out=root/AREA/'cv';r=A.read(out/'result.json')
    if (r['protocol']!=PROTOCOL or r['registration_commit']!=reg or r['control_exact'] is not True
        or r['rows']!=ROWS or r['excluded']!=EXCLUDED or r['additional_excluded']!=0
        or r['new_cpu_fits']!=0 or r['new_gpu_fits']!=0 or r['full_tail_oof_validated'] is not False
        or r['accepted_for_submission'] is not False or r['ranking_authorized'] is not False
        or r['parent']!=checked_parent(root,reg) or r['gate_pass']!=chain_gate(r['folds'])):
        raise ValueError('child result lineage differs')
    expected={'predictions.parquet',*[f'fold_{f}_{f+6}.parquet' for f in range(1,7)]}
    if set(r['files'])!=expected: raise ValueError('incomplete child output inventory')
    check_files(out,r['files']);frames=[]
    current=pl.read_parquet(root/W.CV_OUT/'predictions.parquet').sort(A.ID)
    for report in sorted(r['folds'],key=lambda x:x['fold']):
        fold=report['fold'];f=pl.read_parquet(out/f'fold_{fold}_{fold+6}.parquet');frames.append(f)
        old=current.filter(pl.col('month').is_in([fold,fold+6]))
        if not f.select(old.columns).equals(old) or not np.array_equal(f['current_control'].to_numpy(),old['ordered'].to_numpy()):
            raise ValueError('frozen control/population changed')
        if (report['control_exact'] is not True or report['additional_excluded']!=0
            or report['excluded']!=int((~f['evaluation_mask']).sum())): raise ValueError('fold exclusions differ')
        metrics={k:W.paired(f[A.Y].to_numpy(),f['current_control'].to_numpy(),f['full_tokens'].to_numpy(),mask) for k,mask in W.masks(f).items()}
        if metrics!=report['metrics']: raise ValueError('child fold metrics differ')
    f=pl.concat(frames).sort(A.ID)
    if (f.height!=ROWS or f[A.ID].n_unique()!=ROWS or int((~f['evaluation_mask']).sum())!=EXCLUDED
        or not f.equals(pl.read_parquet(out/'predictions.parquet'))): raise ValueError('pooled child output differs')
    metrics={k:W.paired(f[A.Y].to_numpy(),f['current_control'].to_numpy(),f['full_tokens'].to_numpy(),mask) for k,mask in W.masks(f).items()}
    if metrics!=r['metrics']: raise ValueError('pooled descriptive metrics differ')
    return r


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['seal','attest','audit-parent','cv','check'])
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--registration-commit');p.add_argument('--threads',type=int,default=4)
    args=p.parse_args();root=args.root.resolve()
    if args.action=='seal': seal(root)
    elif args.action=='attest': attest(root,args.registration_commit)
    elif args.action=='audit-parent': audit_parent(root,args.registration_commit)
    elif args.action=='cv': cv(root,args.registration_commit,args.threads)
    else: checked_cv(root,args.registration_commit)


if __name__=='__main__': main()

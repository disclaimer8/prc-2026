"""Fixed MDN4 substitution: pre-E81 paired check, conditional ranking, full tail.

No search, upload, retry, altered weights, or shared-source/environment changes.
The two CPU stage passes capture/replay identical LightGBM training calls.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import time

for k,v in {'POLARS_MAX_THREADS':'4','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
            'POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR':'load_as_storage',
            'OMP_WAIT_POLICY':'PASSIVE','GOMP_SPINCOUNT':'0'}.items():os.environ.setdefault(k,v)
import numpy as np
import polars as pl
import e195_pre_e81_comparison as A
import e196_conditional_density as P
import e190_rank_20261006 as R
C,E,M=P.C,P.E,P.M
PROTOCOL='E200-MDN4-fixed-joint-preE81-gated-ranking-fulltail-20261007-v1'
SELF='scripts/e200_mdn_production.py'
DOC='analysis/e200_mdn_protocol_20261007.md'
RECEIPT='analysis/e200_mdn_input_receipt_20261007.json'
CV_PARENT='data/e195_pre_e81_20261007_v2'
CV_PARENT_REG='bc19847d23d4485717bafa84744a14b49af0da19'
CV_OUT='data/e200_pre_e81_20261007'
RANK_INPUT='data/e190_rank_inputs_20261006'
RANK_REG='f40a465633cc32bff959c598d27d59f9bdc11cc3'
RANK_SPEC='9c9e67f37c16db24d1e4a40245c28f56fdb206f8a5a85fca5ff47415cb2c9399'
PACK='data/e200_rank_inputs_20261007'
RANK_OUT='data/e200_rank_20261007'
WORK='data/e200_chain_20261007'
FINAL='data/e200_candidate_20261007'
BEST='submissions/gentle-igloo_v12074.parquet'
BEST_SHA='e5c78ab061492b69fe72d4a46d2e286f7a2f6b3129ebf2ae4a34731bb662374b'
E197_REG='bfe793c62f7cecf633c7cc31c747f48eb0676f2d'
E197_UNIT='prc-e197-trajectory-20261007.service'
E197_INVOCATION='0b48f00de65a4f1daada3b2fdcd094cc'
E197_OUTPUT='data/e197_trajectory_20261007'
E197_SPEC='93d90131cdfa32c04b8f8c3513f3e7f7cc19c9695d8f47d3fe98b4cc8c0a407d'
FILES=tuple(dict.fromkeys([Path(SELF).name,'e200_execution_queue_20261007.py','e195_pre_e81_comparison.py',*P.FILES,*R.SCRIPTS]))


def registration(root,reg):
    if not re.fullmatch('[a-f0-9]{40}',reg or ''):raise ValueError('full registration required')
    for name in (SELF,DOC,RECEIPT):
        b=subprocess.check_output(['git','show',reg+':'+name],cwd=root)
        if hashlib.sha256(b).hexdigest()!=A.sha(root/name):raise ValueError('registration differs '+name)
    if PROTOCOL.encode() not in subprocess.check_output(['git','show',reg+':HYPOTHESES_LOG.md'],cwd=root):raise ValueError('journal registration missing')


def mdn_path(root,fold):
    return root/('data/e196_density_20261007' if fold in (1,4) else 'data/e198_density_20261007')/f'fold_{fold}_{fold+6}'/'mdn4'


def seal(root):
    inventory={}
    A.registration(root,CV_PARENT_REG)
    def add(p,h=None):
        p=Path(p);actual=A.sha(p)
        if h is not None and actual!=h:raise ValueError('input drift '+str(p))
        inventory[str(p.relative_to(root))]=actual
    old=A.read(root/A.RECEIPT)
    for n,h in old['inputs'].items():add(root/n,h)
    add(root/A.RECEIPT);add(root/CV_PARENT/'result.json')
    cv=A.read(root/CV_PARENT/'result.json')
    if cv['registration']!=CV_PARENT_REG or cv['protocol']!=A.PROTOCOL:raise ValueError('wrong E195 comparison')
    for n,h in cv['outputs'].items():add(root/CV_PARENT/n,h)
    for base in ('data/e196_density_20261007','data/e198_density_20261007'):
        add(root/base/'complete.json');add(root/base/'evaluation.json')
    if not A.read(root/'data/e198_density_20261007/evaluation.json')['six_pair_gate_pass']:raise ValueError('MDN extension did not pass')
    for fold in range(1,7):
        d=mdn_path(root,fold);r=A.read(d/'manifest.json');add(d/'manifest.json')
        if r not in A.read(d.parent.parent/'complete.json')['arms'] or r['seeds']!=[0,1,2] or len(r['fits'])!=3 or any(f['epochs']!=8 or f['arm']!='mdn4' for f in r['fits']):raise ValueError('MDN recipe receipt')
        for n,h in r['files'].items():add(d/n,h)
    rm=R.verify(root/RANK_INPUT,RANK_REG)
    if rm['spec_sha256']!=RANK_SPEC:raise ValueError('full-ranking query changed')
    add(root/RANK_INPUT/'manifest.json')
    original_member=root/'data/e190_rank_20261006/complete'
    original_receipt=R.verify_output(root/RANK_INPUT,original_member,RANK_REG)
    add(original_member/'manifest.json')
    for n,h in original_receipt['files'].items():add(original_member/n,h)
    setup=A.read(root/'data/today_chain_20261006/setup.json');add(root/'data/today_chain_20261006/setup.json')
    for n,h in setup['inputs'].items():add(root/n,h)
    for n,h in setup['source_after'].items():add(root/'data/today_chain_20261006'/n,h)
    add(root/BEST,BEST_SHA);add(root/BEST.replace('.parquet','.manifest.json'))
    for p in sorted((root/'data/e188_rank_20261006_v2').iterdir()):
        if p.is_file():add(p)
    for name in (*FILES,'today_submissions_20261006.py','e188_rank_20261006.py'):
        add(root/'scripts'/name)
    for name in ('e185_submission.py','submission_guard.py'):add(root/'src'/name)
    add(root/DOC)
    A.save(root/RECEIPT,dict(protocol=PROTOCOL,inputs=inventory,ranking_input_spec_sha256=RANK_SPEC,
        parent_e195_registration=CV_PARENT_REG,parent_ranking=BEST,parent_ranking_sha256=BEST_SHA,
        gate_performed=False,full_tail_oof_available=False))
    print('SEALED',len(inventory),'inputs; no new predictions/errors',flush=True)


def inventory(root):
    r=A.read(root/RECEIPT)
    if r['protocol']!=PROTOCOL:raise ValueError('wrong inventory')
    for n,h in r['inputs'].items():
        if A.sha(root/n)!=h:raise ValueError('source/input changed '+n)
    return r


def paired(y,control,candidate,mask):
    ec=(control-y)**2;ep=(candidate-y)**2;n=int(mask.sum());d=np.sort((ep-ec)[mask])
    if n<=40:raise ValueError('unexpected empty primary population')
    a=float(ec[mask].sum());b=float(ep[mask].sum())
    return dict(n=n,control_sse=a,candidate_sse=b,delta_sse=float((ep-ec)[mask].sum()),
        control_rmse=float(np.sqrt(a/n)),candidate_rmse=float(np.sqrt(b/n)),
        delta_rmse=float(np.sqrt(b/n)-np.sqrt(a/n)),trim3_sse=float(d[3:-3].sum()),trim20_sse=float(d[20:-20].sum()))


def masks(frame):
    ok=frame['evaluation_mask'].to_numpy()
    airport=frame['ADEP_mvt'].to_numpy();nm=frame['has_nm'].to_numpy();D=frame['D'].to_numpy()
    return dict(included=ok,body=ok&(frame[A.Y].to_numpy()<=7200),decided=ok&~((airport=='LIRF')&~nm&(D>=14400)))


def gate(folds):
    return len(folds)==6 and sum(f['metrics']['decided']['delta_sse']<0 for f in folds)>=4 and sum(f['metrics']['body']['delta_sse'] for f in folds)<0 and all(sum(f['metrics']['decided'][k] for f in folds)<0 for k in ('trim3_sse','trim20_sse'))


def cv(root,reg,threads):
    registration(root,reg);inventory(root);out=root/CV_OUT
    if out.exists() or not 1<=threads<=8:raise ValueError('fresh output and 1..8 threads required')
    start=time.monotonic();tmp=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent))
    spec=A.read(root/A.BUNDLE/'manifest.json')['spec'];natural=pl.read_parquet(root/A.BUNDLE/'natural_features.parquet')
    cols=[A.ID,A.Y,'split','month','ADEP_mvt','FLIGHT_ID_mvt','LOBT_flt','d_mvt_lobt','d_mvt_sched','d_mvt_iobt','d_mvt_eobt']
    feature=pl.read_parquet(root/'data/features_fin1b.parquet',columns=cols).filter(pl.col('split')=='train').sort(A.ID)
    ra,rb=A.rules(root);frames=[];reports=[]
    for fold in range(1,7):
        name=f'fold_{fold}_{fold+6}';f=feature.filter(pl.col('month').is_in([fold,fold+6]));ids=f[A.ID].to_numpy();y=f[A.Y].to_numpy();mo=f['month'].to_numpy()
        old=pl.read_parquet(root/CV_PARENT/(name+'.parquet')).sort(A.ID)
        if any(not np.array_equal(old[k].to_numpy(),f[k].to_numpy()) for k in (A.ID,A.Y,'month')):raise ValueError('E195 population changed')
        d=f.select(A.ID).join(natural,on=A.ID,how='left',validate='1:1',maintain_order='left')
        z=A.aligned(root/'data/cv_preds_fin1b_hz'/(name+'.npz'),ids,y)
        cat=A.aligned(root/'data/cat_extra175_20261006/merged'/(name+'.npz'),ids,y,mo)['reg2']
        lgb=A.aligned(root/'data/lgb_fin1b_oof_20261005'/name/'complete/prediction.npz',ids,y,mo)['reg2']
        ple=A.aligned(root/'data/e190_numemb_20261005'/name/'prediction.npz',ids,y,mo)['member']
        mdn=A.aligned(mdn_path(root,fold)/'prediction.npz',ids,y,mo)['member']
        anchors=[A.aligned(root/'data'/('cv_e130'+x)/(name+'.npz'),ids)['reg2'] for x in A.ANCHORS]
        motion=A.aligned(root/'data/cv_e130motion'/(name+'.npz'),ids)['reg2']
        regular=[];unstable=np.zeros(len(ids),bool);original_excluded=~old['evaluation_mask'].to_numpy()
        for neural in (ple,mdn):
            r=.65*cat+(1-.65)*lgb;r=(1-sum(A.WEIGHTS))*r+sum(w*g for w,g in zip(A.WEIGHTS,[*anchors,neural]))
            if not np.isfinite(r).all():raise ValueError('nonfinite regular member')
            regular.append(r);v=z['values'].astype(float).copy();v[:,0]=r;v[:,-1]=r+86400
            unstable|=np.any(A.feasibility(v,f['d_mvt_lobt'].to_numpy())!=A.feasibility(z['values'],f['d_mvt_lobt'].to_numpy()),axis=1)
        excluded=original_excluded|unstable;pred=[]
        for i,r in enumerate(regular):
            # Control must reproduce original joint on its full original supported population.
            base,p=A.upstream(z,r,motion,f,np.isin(ids,ra),np.isin(ids,rb),original_excluded if i==0 else excluded)
            _,aug=A.stage_predictions(root,name,d,base,p,spec,threads)
            if i==0 and not np.array_equal(aug[~original_excluded],old['joint'].to_numpy()[~original_excluded]):raise ValueError('E195 joint reproduction failed')
            aug[excluded]=np.rint(z['pred'][excluded])
            if not np.isfinite(aug).all() or (aug<0).any() or not np.array_equal(aug,np.rint(aug)):raise ValueError('invalid stage output')
            pred.append(aug)
        frame=old.select(A.ID,A.Y,'month','ADEP_mvt','has_nm','D','excluded_tabpfn').with_columns(
            pl.Series('parent_excluded',original_excluded),pl.Series('mdn_union_feasibility',unstable),
            pl.Series('evaluation_mask',~excluded),pl.Series('joint',pred[0]),pl.Series('mdn4',pred[1]))
        mm=masks(frame);report=dict(fold=fold,control_exact=True,original_excluded=int(original_excluded.sum()),
            additional_excluded=int((excluded&~original_excluded).sum()),excluded=int(excluded.sum()),
            metrics={k:paired(y,pred[0],pred[1],v) for k,v in mm.items()})
        frame.write_parquet(tmp/(name+'.parquet'));frames.append(frame);reports.append(report)
        print('COMPLETE PRE-E81',name,'same-mask rows',int((~excluded).sum()),flush=True)
    allf=pl.concat(frames).sort(A.ID)
    if allf.height!=2085047 or allf[A.ID].n_unique()!=allf.height:raise ValueError('sixfold ID coverage')
    allf.write_parquet(tmp/'predictions.parquet');mm=masks(allf);ok=mm['included']
    mm.update({f'month_{i}':ok&(allf['month'].to_numpy()==i) for i in range(1,13)})
    mm.update({f'airport_{a}':ok&(allf['ADEP_mvt'].to_numpy()==a) for a in A.HUBS})
    result=dict(protocol=PROTOCOL,registration_commit=reg,inventory_sha256=A.sha(root/RECEIPT),folds=reports,
        metrics={k:paired(allf[A.Y].to_numpy(),allf['joint'].to_numpy(),allf['mdn4'].to_numpy(),v) for k,v in mm.items()},
        sum_fold_trims={population:{k:sum(f['metrics'][population][k] for f in reports) for k in ('trim3_sse','trim20_sse')} for population in ('included','decided','body')},
        gate_trim_population='decided',
        gate_pass=gate(reports),ranking_authorized_if_pass=True,rows=allf.height,included=int(ok.sum()),excluded=int((~ok).sum()),
        control_exact=True,all_six_selection_informed=True,final_chain_complete=False,accepted_for_submission=False,
        limitation=A.LIMITATION+' All six pairs are selection-informed after E196/E198.',seconds=time.monotonic()-start,
        files={p.name:A.sha(p) for p in tmp.glob('*.parquet')})
    inventory(root);A.save(tmp/'result.json',result);tmp.rename(out)
    print('PRE-E81 GATE',result['gate_pass'],json.dumps(result['metrics']['body']),flush=True)


def checked_gate(root,reg):
    r=A.read(root/CV_OUT/'result.json')
    if r['protocol']!=PROTOCOL or r['registration_commit']!=reg or not r['control_exact'] or not r['gate_pass'] or not gate(r['folds']):raise ValueError('conditional production gate not passed')
    for n,h in r['files'].items():
        if A.sha(root/CV_OUT/n)!=h:raise ValueError('CV output changed')
    return r


def pack_rank(root,reg):
    registration(root,reg);checked_gate(root,reg);m=R.verify(root/RANK_INPUT,RANK_REG);out=root/PACK
    if out.exists() or m['spec_sha256']!=RANK_SPEC:raise ValueError('fresh pack/original ranking inputs required')
    tmp=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent));(tmp/'scripts').mkdir()
    for name in FILES:shutil.copy2(root/'scripts'/name,tmp/'scripts'/name)
    shutil.copy2(root/CV_OUT/'result.json',tmp/'gate.json')
    s=dict(protocol=PROTOCOL,registration_commit=reg,ranking_parent=RANK_INPUT,ranking_parent_registration=RANK_REG,
        ranking_parent_spec=RANK_SPEC,configuration=dict(arm='mdn4',seeds=[0,1,2],epochs=8,fit_source_sha256=A.sha(P.__file__)),
        e197_unit=E197_UNIT,e197_invocation=E197_INVOCATION,e197_registration=E197_REG,e197_spec=E197_SPEC,
        files={str(p.relative_to(tmp)):A.sha(p) for p in tmp.rglob('*') if p.is_file()})
    A.save(tmp/'manifest.json',dict(spec=s,spec_sha256=C.digest(s)));tmp.rename(out);print('RANK PACK',C.digest(s),flush=True)


def verify_rank_input(root,reg):
    bundle=root/PACK;m=A.read(bundle/'manifest.json');s=m['spec']
    if m['spec_sha256']!=C.digest(s) or s['protocol']!=PROTOCOL or s['registration_commit']!=reg or s['ranking_parent_spec']!=RANK_SPEC:raise ValueError('rank input protocol')
    for n,h in s['files'].items():
        if A.sha(bundle/n)!=h:raise ValueError('rank packed input changed '+n)
        if n.startswith('scripts/') and A.sha(Path(__file__).parent/Path(n).name)!=h:raise ValueError('active GPU source changed '+n)
    gate_receipt=A.read(bundle/'gate.json')
    if not gate_receipt['gate_pass'] or gate_receipt['registration_commit']!=reg or not gate(gate_receipt['folds']):raise ValueError('gate proof changed')
    if R.verify(root/RANK_INPUT,RANK_REG)['spec_sha256']!=RANK_SPEC:raise ValueError('original ranking tensor changed')
    return m


def verify_rank_output(root,reg):
    m=verify_rank_input(root,reg);out=root/RANK_OUT;r=A.read(out/'complete.json')
    if r['registration_commit']!=reg or r['protocol']!=PROTOCOL or r['input_spec_sha256']!=m['spec_sha256'] or [f['seed'] for f in r['fits']]!=[0,1,2] or any(f['epochs']!=8 or f['arm']!='mdn4' for f in r['fits']):raise ValueError('rank incomplete recipe')
    s=A.read(root/RANK_INPUT/'manifest.json')['spec']
    if r['runtime']!=s['admitted_runtime']:raise ValueError('rank runtime differs')
    for n,h in r['files'].items():
        if A.sha(out/n)!=h:raise ValueError('rank output hash changed')
    ids=np.load(root/RANK_INPUT/'mvt_id.npy');R.check_ids(ids)
    preds=[np.load(out/f'seed_{i}.npy') for i in (0,1,2)]
    if any(p.shape!=(R.ROWS,) or p.dtype!=np.float32 or not np.isfinite(p).all() for p in preds):raise ValueError('rank seed missing/nonfinite')
    expected=np.mean(preds,axis=0)*s['sd']+np.load(root/RANK_INPUT/'base_p.npy')
    z=E.read(out/'prediction.npz')
    if set(z)!={'mvt_id','reg','reg2'} or not np.array_equal(z['mvt_id'],ids) or not np.array_equal(z['reg'],expected) or not np.array_equal(z['reg2'],expected) or not np.isfinite(expected).all():raise ValueError('rank exact seed mean/IDs')
    return r


def rank(root,reg):
    P.fixed_env();import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);guard=M.verify_cgroup();m=verify_rank_input(root,reg)
    q=root/RANK_INPUT;s=A.read(q/'manifest.json')['spec'];runtime=E.package_runtime()
    if runtime!=s['admitted_runtime']:raise ValueError('qualified runtime differs')
    out=root/RANK_OUT
    if out.exists():raise ValueError('fresh ranking output; no resume')
    arrays={n:np.load(q/(n+'.npy'),allow_pickle=False) for n in R.ARRAYS};ids=np.load(q/'mvt_id.npy');R.validate(arrays,ids,s)
    cont,scalar,bins=E.numeric_bins(arrays['xnf']);N=E.trainer(q/'src')
    tmp=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent))
    np.savez(tmp/'bins.npz',continuous=cont,scalar=scalar,**{f'bin_{i}':b for i,b in enumerate(bins)})
    fits=[P.fit(arrays,s,cont,scalar,bins,N,'mdn4',seed,tmp) for seed in (0,1,2)]
    pred=np.mean([np.load(tmp/f'seed_{i}.npy') for i in (0,1,2)],axis=0)*s['sd']+arrays['base_p']
    if not np.isfinite(pred).all():raise ValueError('nonfinite full member')
    np.savez(tmp/'prediction.npz',mvt_id=ids,reg=pred,reg2=pred);verify_rank_input(root,reg)
    A.save(tmp/'complete.json',dict(protocol=PROTOCOL,registration_commit=reg,input_spec_sha256=m['spec_sha256'],
        runtime=runtime,cgroup=guard,rows=len(ids),fits=fits,files={p.name:A.sha(p) for p in tmp.iterdir() if p.is_file()}))
    tmp.rename(out);verify_rank_output(root,reg);print('COMPLETE MDN ranking344841 rows3seeds8epochs',flush=True)


def dependency(root):
    p=root/E197_OUTPUT;c=A.read(p/'complete.json');ev=A.read(p/'evaluation.json')
    if c['registration_commit']!=E197_REG or c['input_spec_sha256']!=E197_SPEC or len(c['arms'])!=4 or ev['registration_commit']!=E197_REG:raise ValueError('E197 dependency proof')
    if sorted((a['fold'],a['arm']) for a in c['arms'])!=[(f,a) for f in (1,4) for a in ('flat','ordered')]:raise ValueError('E197 arm inventory')
    for r in c['arms']:
        d=p/f"fold_{r['fold']}_{r['fold']+6}"/r['arm']
        if A.read(d/'manifest.json')!=r:raise ValueError('E197 arm receipt')
        check_e197_fits(r)
        for n,h in r['files'].items():
            if A.sha(d/n)!=h:raise ValueError('E197 output hash changed')
    return dict(complete_sha256=A.sha(p/'complete.json'),evaluation_sha256=A.sha(p/'evaluation.json'))


def queue(root,reg):
    verify_rank_input(root,reg);seen=False;started=time.monotonic()
    while True:
        raw=subprocess.check_output(['systemctl','--user','show',E197_UNIT,'-p','ActiveState','-p','MainPID','-p','InvocationID','-p','Result'],text=True)
        s=dict(x.split('=',1) for x in raw.splitlines() if '=' in x)
        active=s.get('ActiveState') in ('active','activating','deactivating') or int(s.get('MainPID','0'))>0
        if active:
            if s.get('InvocationID')!=E197_INVOCATION:raise ValueError('E197 invocation changed')
            seen=True
        else:
            if s.get('Result') not in (None,'','success'):raise ValueError('E197 failed; no automatic fallback')
            if not seen:raise ValueError('E197 exact running invocation was never observed by this queue')
            proof=dependency(root);A.save(root/'data/e200_gpu_dependency_20261007.json',dict(invocation=E197_INVOCATION,**proof));break
        if time.monotonic()-started>12*3600:raise TimeoutError('E197 did not complete within12h')
        print('WAIT E197 actual invocation',E197_INVOCATION,flush=True);time.sleep(30)
    rank(root,reg)


def tensor_fingerprint(x):
    if x is None:return None
    a=np.ascontiguousarray(x)
    if a.dtype.hasobject:raise ValueError('unexpected object training array')
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())


def check_e197_fits(r):
    if r['seeds']!=[0,1,2] or [f['seed'] for f in r['fits']]!=[0,1,2] or any(len(f['losses'])!=8 or not np.isfinite(f['losses']).all() for f in r['fits']):raise ValueError('E197 fit count/recipe')


def training_signature(params,dataset,rounds,features,lgb_version):
    return dict(params=params,rounds=rounds,data=tensor_fingerprint(dataset.data),
        label=tensor_fingerprint(dataset.label),weight=tensor_fingerprint(dataset.weight),
        categorical=dataset.categorical_feature,feature_names=list(features),dataset_feature_names=dataset.feature_name,
        lightgbm=lgb_version,numpy=np.__version__,polars=pl.__version__,python=sys.version)


def check_signature(expected,actual):
    if expected!=actual:raise ValueError('cached stage training call differs')


def stage_job(root,reg,job_path):
    """Execute immutable production script; capture/replay its exact lgb.train calls."""
    registration(root,reg);job=A.read(job_path)
    if job['registration']!=reg:raise ValueError('stage job registration')
    work=root/WORK;source=work/'src'/job['args'][0]
    expected=A.read(root/'data/today_chain_20261006/setup.json')['source_after']['src/'+source.name]
    if A.sha(source)!=expected:raise ValueError('frozen stage source changed')
    import lightgbm as lgb
    original=lgb.train;calls=[];cache=work/'models'/job['cache']
    if job['mode']=='capture':cache.mkdir(parents=True,exist_ok=False)
    saved=A.read(cache/'complete.json') if job['mode']=='replay' else None
    expected_features=next(s['features'] for s in A.read(root/A.BUNDLE/'manifest.json')['spec']['stages'] if s['name']==job['cache'])
    def wrapped(params,train_set,num_boost_round=100,**kwargs):
        if kwargs:raise ValueError('unexpected LightGBM call options')
        caller=inspect.currentframe().f_back
        features=caller.f_locals.get('feats',caller.f_globals.get('FEATS' if job['cache']!='lszh' else 'F'))
        if list(features or [])!=expected_features or train_set.data.shape[1]!=len(expected_features):raise ValueError('stage feature name/order differs')
        sig=training_signature(params,train_set,num_boost_round,features,lgb.__version__)
        i=len(calls);path=cache/f'model_{i}.txt'
        if job['mode']=='capture':
            model=original(params,train_set,num_boost_round);model.save_model(str(path));item=dict(signature=sig,sha256=A.sha(path));calls.append(item)
        else:
            item=saved['calls'][i]
            check_signature(item['signature'],sig)
            if A.sha(path)!=item['sha256']:raise ValueError('cached stage model changed')
            model=lgb.Booster(model_file=str(path));calls.append(item)
        if model.current_iteration()!=num_boost_round:raise ValueError('incomplete production stage model')
        return model
    lgb.train=wrapped
    try:
        os.chdir(work);sys.path.insert(0,str(work/'src'));sys.argv=[str(source),*map(str,job['args'][1:])];runpy.run_path(str(source),run_name='__main__')
    finally:lgb.train=original
    if len(calls)!=job['expected_calls']:raise ValueError('unexpected stage fit inventory')
    if job['mode']=='capture':A.save(cache/'complete.json',dict(registration=reg,source_sha256=expected,calls=calls))
    elif saved['calls']!=calls:raise ValueError('unused cached models')


def call(root,reg,args,extra=None,cache=None,mode=None):
    import today_submissions_20261006 as T
    work=root/WORK
    if cache is None:return T.stage(work,args,extra)
    index=len(list((work/'logs').glob('*.log')));jobs=work/'jobs';jobs.mkdir(exist_ok=True)
    job=jobs/f'{index:02d}.json';A.save(job,dict(registration=reg,args=args,cache=cache,mode=mode,expected_calls=1 if cache=='lszh' else 5))
    print('STAGE',index,mode,cache,flush=True)
    with (work/'logs'/f'{index:02d}_{cache}_{mode}.log').open('x') as f:
        subprocess.run([sys.executable,'-u',str(root/SELF),'stage','--root',str(root),'--registration-commit',reg,'--job',str(job)],
            cwd=root,env={**T.env(),**(extra or {})},stdout=f,stderr=subprocess.STDOUT,check=True)


def branch(root,reg,member,base,mode):
    import today_submissions_20261006 as T
    work=root/WORK;members=','.join(f'submissions/e130_{m}f_rank.npz:{w}' for m,w in [('raw','0.0333333333333333'),('lobt','.10'),('sched','.10'),('eobt','.10'),('aobt','.10')])
    call(root,reg,['blend.py',870,base,.65],{'FEATURES':'data/features_fin1b.parquet','LGB_SEEDS':'submissions/lgb_reg_seeds_v870.npz','E130_MEMBERS':members+','+member+':0.25'})
    call(root,reg,['e185_apply.py',base,base+1],{'E185_W':str(.35*2/3),'E185_RANK':T.E.RANK})
    call(root,reg,['e167_nm_window_rule.py',base+1,base+2]);T.E.tabpfn_rows(work,base+2,base+3)
    common={'STAGE_SEEDS':'42,1,2,3,4','ADSB_DIR':'data/adsblol_kt1','PARTS':f'submissions/parts_v{base+2}.npz'}
    call(root,reg,['e141_apply.py',base+3,base+4],{**common,'STAGE_KEEP_ALL':'1','DROP_FEATS':T.E.D7,'WALK_DIR':'data/adsblol_sw_kt1','E36_BASE_DIR':'data/cv_preds_e116prod'},'e141',mode)
    call(root,reg,['e64_apply.py',base+4,base+5],{**common,'E64_KEEP_ALL':'1','DROP_FEATS':T.E.D3},'e64',mode)
    subprocess.run([sys.executable,str(root/'scripts/e188_rank_20261006.py'),'apply','--registration-commit',T.E188_REG,'--models',str(root/'data/e188_rank_20261006_v2'),
        '--pre-e64',str(T.sub(work,base+4)),'--post-e64',str(T.sub(work,base+5)),'--parts',str(work/f'submissions/parts_v{base+2}.npz'),'--output',str(T.sub(work,base+6))],cwd=root,env=T.env(),check=True)
    call(root,reg,['e152_stage_gate.py',base+3,base+6,base+2,base+10],{'E152_GATE':'0.95'})
    call(root,reg,['blend_submissions.py',base+11],{'BLEND_BASE':str(base+10),'BLEND_PARTS':'44:0.1214,60:0.0270'})
    call(root,reg,['lobt_edge_lift.py',base+11,base+12],{'LOBT_CLIP':'1'})
    call(root,reg,['protect_tabpfn_rows.py',base+12,base+3,base+13])
    call(root,reg,['e159b_apply.py',base+13,base+14],{'E159_APS':'LSZH','E159_MONTHS':'all','E159_BASE':'data/cv_preds_e150c_e64_keep','E159_SHIFT':'0'},'lszh',mode)
    call(root,reg,['blend_submissions.py',base+15],{'BLEND_BASE':str(base+13),'BLEND_PARTS':f'{base+14}:0.5'})
    return T.sub(work,base+15)


def capture(root,reg):
    registration(root,reg);checked_gate(root,reg);inventory(root);work=root/WORK
    if work.exists():raise ValueError('fresh private production workspace required')
    shutil.copytree(root/'data/today_chain_20261006',work,symlinks=True,copy_function=os.link,ignore=shutil.ignore_patterns('logs'))
    (work/'logs').mkdir();target=work/RANK_OUT;target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(root/RANK_OUT)
    output=branch(root,reg,'data/e190_rank_20261006/complete/prediction.npz',910100,'capture')
    a,b=[pl.read_parquet(p).sort(A.ID) for p in (output,root/BEST)]
    if not a.equals(b):raise ValueError('full production PLE+E188 control differs from v12074; no candidate build')
    inventory(root);A.save(work/'control_verified.json',dict(protocol=PROTOCOL,registration_commit=reg,reference=BEST,reference_sha256=BEST_SHA,
        control_sha256=A.sha(output),control_rows_exact=True,models={n:A.sha(work/'models'/n/'complete.json') for n in ('e141','e64','lszh')}))
    print('FULL CONTROL EXACT v12074;11production stage models cached',flush=True)


def build(root,reg):
    import today_submissions_20261006 as T
    registration(root,reg);checked_gate(root,reg);inventory(root);rank_receipt=verify_rank_output(root,reg);work=root/WORK
    control=A.read(work/'control_verified.json')
    if control['registration_commit']!=reg or not control['control_rows_exact'] or control['reference_sha256']!=BEST_SHA:raise ValueError('exact control proof missing')
    for name,h in control['models'].items():
        if A.sha(work/'models'/name/'complete.json')!=h:raise ValueError('production model receipt changed')
    out=root/FINAL
    if out.exists():raise ValueError('fresh candidate required; no overwrite')
    final=branch(root,reg,RANK_OUT+'/prediction.npz',920100,'replay')
    candidate=pl.read_parquet(final)
    checks=T.G.validate_submission(candidate,pl.read_parquet(root/T.E.TEMPLATE),pl.read_parquet(root/BEST),
        pl.read_parquet(root/'submissions/gentle-igloo_v291.parquet'),pl.read_parquet(root/'submissions/gentle-igloo_v361.parquet'))
    if candidate[A.Y].min()<0:raise ValueError('negative final prediction')
    tmp=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent));shutil.copy2(final,tmp/'candidate.parquet')
    inventory(root);A.save(tmp/'manifest.json',dict(protocol=PROTOCOL,registration_commit=reg,reference=BEST,reference_sha256=BEST_SHA,
        recipe='v12074 fixed full chain; replace PLE member by MDN4 mean at unchanged0.25 slot; originalE188/fulltail/halfLSZH',
        validation=checks,sha256=A.sha(tmp/'candidate.parquet'),rank_complete_sha256=A.sha(root/RANK_OUT/'complete.json'),rank_receipt=rank_receipt,
        control_verified_sha256=A.sha(work/'control_verified.json'),pre_e81_gate_sha256=A.sha(root/CV_OUT/'result.json'),
        all_six_selection_informed=True,full_tail_oof_validated=False,uploaded=False,accepted_for_submission=False))
    tmp.rename(out);print('FULL CANDIDATE READY; no upload',out,flush=True)


def selftest():
    f=[dict(metrics=dict(decided=dict(delta_sse=-1.,trim3_sse=-1.,trim20_sse=-1.),body=dict(delta_sse=-1.))) for _ in range(6)]
    assert gate(f) and not gate(f[:4]);f[0]['metrics']['body']['delta_sse']=6.;assert not gate(f)
    x=np.arange(12).reshape(3,4);assert tensor_fingerprint(x)==tensor_fingerprint(x.copy())
    assert tensor_fingerprint(x)!=tensor_fingerprint(x.astype(float))
    check_e197_fits(dict(seeds=[0,1,2],fits=[dict(seed=i,losses=[1.]*8) for i in (0,1,2)]))
    import lightgbm as lgb
    rng=np.random.default_rng(200);x=rng.normal(size=(128,3));y=x[:,0]-.3*x[:,2]
    params=dict(objective='regression',num_threads=1,num_leaves=3,min_data_in_leaf=5,verbose=-1,deterministic=True)
    ds=lgb.Dataset(x,y,free_raw_data=False);sig=training_signature(params,ds,3,['a','b','c'],lgb.__version__)
    model=lgb.train(params,ds,num_boost_round=3)
    with tempfile.TemporaryDirectory(prefix='e200-synthetic-') as d:
        path=Path(d)/'model.txt';model.save_model(str(path));loaded=lgb.Booster(model_file=str(path))
        assert np.array_equal(model.predict(x,num_threads=1),loaded.predict(x,num_threads=1))
        fresh=lgb.Dataset(x.copy(),y.copy(),free_raw_data=False);check_signature(sig,training_signature(params,fresh,3,['a','b','c'],lgb.__version__))
        changed=lgb.Dataset(x,y+.01,free_raw_data=False)
        try:check_signature(sig,training_signature(params,changed,3,['a','b','c'],lgb.__version__))
        except ValueError:pass
        else:raise AssertionError('changed labels accepted')
        try:check_signature(sig,training_signature(params,fresh,3,['c','b','a'],lgb.__version__))
        except ValueError:pass
        else:raise AssertionError('changed feature order accepted')
    print('PASS gate, E1978losses fixture, synthetic capture/save/reload exact prediction and label/feature-order rejection; no real target fit')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['seal','selftest','cv','pack-rank','queue-rank','rank','verify-rank','capture-control','build','stage'])
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--registration-commit');p.add_argument('--threads',type=int,default=8);p.add_argument('--job',type=Path)
    a=p.parse_args();root=a.root.resolve();reg=a.registration_commit
    if a.action=='seal':seal(root)
    elif a.action=='selftest':selftest()
    elif a.action=='cv':cv(root,reg,a.threads)
    elif a.action=='pack-rank':pack_rank(root,reg)
    elif a.action=='queue-rank':queue(root,reg)
    elif a.action=='rank':rank(root,reg)
    elif a.action=='verify-rank':print(json.dumps(verify_rank_output(root,reg)))
    elif a.action=='capture-control':capture(root,reg)
    elif a.action=='build':build(root,reg)
    else:stage_job(root,reg,a.job.resolve())


if __name__=='__main__':main()

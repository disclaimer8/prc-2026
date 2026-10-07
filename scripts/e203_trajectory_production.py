"""E203: fixed ordered-trajectory substitution, six-pair gate, conditional ranking.

Uses unchanged E197 fit/network and frozen E200 full-chain capture/replay arithmetic.
No upload, architecture search, or mutation of either registered predecessor.
"""
from __future__ import annotations
import argparse
import ast
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
for k,v in {'POLARS_MAX_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OMP_WAIT_POLICY':'PASSIVE','GOMP_SPINCOUNT':'0','POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR':'load_as_storage'}.items():os.environ.setdefault(k,v)
import numpy as np
import polars as pl
import e195_pre_e81_comparison as A
import e197_trajectory_encoder as S
import e190_rank_20261006 as R
C,E,M,D=S.C,S.E,S.M,S.D
PROTOCOL='E203-ordered-trajectory-fixed025-joint-preE81-sixpair-gated-fullrank-fulltail-v1'
SELF='scripts/e203_trajectory_production.py'
DOC='analysis/e203_trajectory_protocol_20261007.md'
RECEIPT='analysis/e203_input_receipt_20261007.json'
CV_PARENT='data/e195_pre_e81_20261007_v2'
CV_PARENT_REG='bc19847d23d4485717bafa84744a14b49af0da19'
CV_OUT='data/e203_pre_e81_20261007'
RANK_INPUT='data/e190_rank_inputs_20261006'
RANK_REG='f40a465633cc32bff959c598d27d59f9bdc11cc3'
RANK_SPEC='9c9e67f37c16db24d1e4a40245c28f56fdb206f8a5a85fca5ff47415cb2c9399'
PACK='data/e203_rank_inputs_20261007'
PREPARED='data/e203_rank_prepared_20261007'
RANK_OUT='data/e203_rank_20261007'
WORK='data/e203_chain_20261007'
FINAL='data/e203_candidate_20261007'
BEST='submissions/gentle-igloo_v12074.parquet'
BEST_SHA='e5c78ab061492b69fe72d4a46d2e286f7a2f6b3129ebf2ae4a34731bb662374b'
OLD_REG='bfe793c62f7cecf633c7cc31c747f48eb0676f2d'
OLD_COMPLETE='84842a0734975ae76bc27ebdf0fdc5841bc769a7dc8322255672d0c48c4f75db'
OLD_AUDIT='cae24cc29438a6d3023329a07664f653a94c40b2802737fbc98c93b7362fac62'
EXT_REG='55fe9b68f7cc8405681f6c85d576979e067b55c3'
EXT_SPEC='6d604eaaca830fa83c18ebb78986b028d338869123d5f8c10b2450dde1ec4309'
EXT_INPUT='data/e202_inputs_20261007'
EXT_OUTPUT='data/e202_trajectory_20261007'
EXT_PROTOCOL='E202-trajectory-remaining4pairs-flat-ordered-3seed8epoch-v1'
E200_SHA='2fe1328214128a0f6ded5a178d954fa3ba0634c4043e42e7c69bba461062ad25'
E197_SHA='6be13d401dd2903c50c9584f5ca2d6ca084a1feb0eb1f67fcba0cd94850db54a'
NROWS=2429888
FILES=tuple(dict.fromkeys([Path(SELF).name,'e203_execution_queue_20261007.py','e195_pre_e81_comparison.py',*S.FILES,*R.SCRIPTS]))

def registration(root,reg):
    if not re.fullmatch('[a-f0-9]{40}',reg or ''):raise ValueError('full registration required')
    for name in (SELF,DOC,RECEIPT,'scripts/e203_execution_queue_20261007.py'):
        raw=subprocess.check_output(['git','show',reg+':'+name],cwd=root)
        if hashlib.sha256(raw).hexdigest()!=A.sha(root/name):raise ValueError('registered source changed '+name)
    if PROTOCOL.encode() not in subprocess.check_output(['git','show',reg+':HYPOTHESES_LOG.md'],cwd=root):raise ValueError('missing journal protocol')

def trajectory_path(root,fold):
    return root/('data/e197_trajectory_20261007' if fold in (1,4) else EXT_OUTPUT)/f'fold_{fold}_{fold+6}'/'ordered'

def inventory(root):
    r=A.read(root/RECEIPT)
    if r['protocol']!=PROTOCOL:raise ValueError('wrong inventory')
    for name,h in r['inputs'].items():
        if A.sha(root/name)!=h:raise ValueError('frozen input changed '+name)
    return r

def seal(root):
    A.registration(root,CV_PARENT_REG)
    if A.sha(root/'scripts/e200_mdn_production.py')!=E200_SHA or A.sha(root/'scripts/e197_trajectory_encoder.py')!=E197_SHA:raise ValueError('frozen borrowed algorithm drift')
    inputs=dict(A.read(root/A.RECEIPT)['inputs'])
    def add(p,h=None):
        actual=A.sha(p)
        if h is not None and actual!=h:raise ValueError('source drift '+str(p))
        inputs[str(p.relative_to(root))]=actual
    add(root/A.RECEIPT);cv=A.read(root/CV_PARENT/'result.json');add(root/CV_PARENT/'result.json')
    if cv['registration']!=CV_PARENT_REG:raise ValueError('wrong current-chain parent')
    for n,h in cv['outputs'].items():add(root/CV_PARENT/n,h)
    add(root/'analysis/e197_verified_results_20261007.json',OLD_AUDIT)
    add(root/'data/e197_trajectory_20261007/complete.json',OLD_COMPLETE)
    for fold in (1,4):
        d=trajectory_path(root,fold);r=A.read(d/'manifest.json');add(d/'manifest.json')
        for n,h in r['files'].items():add(d/n,h)
    if R.verify(root/RANK_INPUT,RANK_REG)['spec_sha256']!=RANK_SPEC:raise ValueError('ranking tensor lineage changed')
    add(root/RANK_INPUT/'manifest.json');r=R.verify_output(root/RANK_INPUT,root/'data/e190_rank_20261006/complete',RANK_REG)
    add(root/'data/e190_rank_20261006/complete/manifest.json')
    for n,h in r['files'].items():add(root/'data/e190_rank_20261006/complete'/n,h)
    setup=A.read(root/'data/today_chain_20261006/setup.json');add(root/'data/today_chain_20261006/setup.json')
    for n,h in setup['inputs'].items():add(root/n,h)
    for n,h in setup['source_after'].items():add(root/'data/today_chain_20261006'/n,h)
    add(root/BEST,BEST_SHA)
    for p in (root/'data/e188_rank_20261006_v2').iterdir():
        if p.is_file():add(p)
    for shard in (0,1):add(root/f'data/e194_sequence_20261007_s{shard}/complete.json')
    for name in (*FILES,'e200_mdn_production.py','e202_trajectory_extension.py','today_submissions_20261006.py','e188_rank_20261006.py'):add(root/'scripts'/name)
    for name in ('e185_submission.py','submission_guard.py'):add(root/'src'/name)
    add(root/DOC)
    for n,h in inputs.items():
        if A.sha(root/n)!=h:raise ValueError('inherited input drift '+n)
    A.save(root/RECEIPT,dict(protocol=PROTOCOL,inputs=inputs,extension_registration=EXT_REG,ranking_spec=RANK_SPEC,no_new_prediction=True,full_tail_oof_validated=False))
    print('SEALED',len(inputs),'immutable inputs',flush=True)


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

def tensor_fingerprint(x):
    if x is None:return None
    a=np.ascontiguousarray(x)
    if a.dtype.hasobject:raise ValueError('unexpected object training array')
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())

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
    if shutil.disk_usage(root).free<2500*1024**2:raise ValueError('production stage requires2.5GiB disk headroom')
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
    if shutil.disk_usage(root).free<4*1024**3:raise ValueError('capture needs4GiB free; preserve2GiB floor')
    if work.exists():raise ValueError('fresh private production workspace required')
    shutil.copytree(root/'data/today_chain_20261006',work,symlinks=True,copy_function=os.link,ignore=shutil.ignore_patterns('logs'))
    (work/'logs').mkdir();target=work/RANK_OUT;target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(root/RANK_OUT)
    output=branch(root,reg,'data/e190_rank_20261006/complete/prediction.npz',930100,'capture')
    a,b=[pl.read_parquet(p).sort(A.ID) for p in (output,root/BEST)]
    if not a.equals(b):raise ValueError('full production PLE+E188 control differs from v12074; no candidate build')
    inventory(root);A.save(work/'control_verified.json',dict(protocol=PROTOCOL,registration_commit=reg,reference=BEST,reference_sha256=BEST_SHA,
        control_sha256=A.sha(output),control_rows_exact=True,models={n:A.sha(work/'models'/n/'complete.json') for n in ('e141','e64','lszh')}))
    print('FULL CONTROL EXACT v12074;11production stage models cached',flush=True)


def cv_subset(root,reg,threads,folds,label):
    registration(root,reg);inventory(root);out=root/CV_OUT/label
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() or not 1<=threads<=8:raise ValueError('fresh output and 1..8 threads required')
    start=time.monotonic();tmp=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent))
    spec=A.read(root/A.BUNDLE/'manifest.json')['spec'];natural=pl.read_parquet(root/A.BUNDLE/'natural_features.parquet')
    cols=[A.ID,A.Y,'split','month','ADEP_mvt','FLIGHT_ID_mvt','LOBT_flt','d_mvt_lobt','d_mvt_sched','d_mvt_iobt','d_mvt_eobt']
    feature=pl.read_parquet(root/'data/features_fin1b.parquet',columns=cols).filter(pl.col('split')=='train').sort(A.ID)
    ra,rb=A.rules(root);frames=[];reports=[]
    for fold in folds:
        verify_trajectory(root,fold)
        name=f'fold_{fold}_{fold+6}';f=feature.filter(pl.col('month').is_in([fold,fold+6]));ids=f[A.ID].to_numpy();y=f[A.Y].to_numpy();mo=f['month'].to_numpy()
        old=pl.read_parquet(root/CV_PARENT/(name+'.parquet')).sort(A.ID)
        if any(not np.array_equal(old[k].to_numpy(),f[k].to_numpy()) for k in (A.ID,A.Y,'month')):raise ValueError('E195 population changed')
        d=f.select(A.ID).join(natural,on=A.ID,how='left',validate='1:1',maintain_order='left')
        z=A.aligned(root/'data/cv_preds_fin1b_hz'/(name+'.npz'),ids,y)
        cat=A.aligned(root/'data/cat_extra175_20261006/merged'/(name+'.npz'),ids,y,mo)['reg2']
        lgb=A.aligned(root/'data/lgb_fin1b_oof_20261005'/name/'complete/prediction.npz',ids,y,mo)['reg2']
        ple=A.aligned(root/'data/e190_numemb_20261005'/name/'prediction.npz',ids,y,mo)['member']
        trajectory=A.aligned(trajectory_path(root,fold)/'prediction.npz',ids,y,mo)['member']
        anchors=[A.aligned(root/'data'/('cv_e130'+x)/(name+'.npz'),ids)['reg2'] for x in A.ANCHORS]
        motion=A.aligned(root/'data/cv_e130motion'/(name+'.npz'),ids)['reg2']
        regular=[];unstable=np.zeros(len(ids),bool);original_excluded=~old['evaluation_mask'].to_numpy()
        for neural in (ple,trajectory):
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
            pl.Series('parent_excluded',original_excluded),pl.Series('trajectory_union_feasibility',unstable),
            pl.Series('evaluation_mask',~excluded),pl.Series('joint',pred[0]),pl.Series('ordered',pred[1]))
        mm=masks(frame);report=dict(fold=fold,control_exact=True,original_excluded=int(original_excluded.sum()),
            additional_excluded=int((excluded&~original_excluded).sum()),excluded=int(excluded.sum()),
            metrics={k:paired(y,pred[0],pred[1],v) for k,v in mm.items()})
        frame.write_parquet(tmp/(name+'.parquet'));frames.append(frame);reports.append(report)
        print('COMPLETE PRE-E81',name,'same-mask rows',int((~excluded).sum()),flush=True)
    allf=pl.concat(frames).sort(A.ID)
    if allf[A.ID].n_unique()!=allf.height:raise ValueError('subset ID coverage')
    allf.write_parquet(tmp/'predictions.parquet');mm=masks(allf);ok=mm['included']
    mm.update({f'month_{i}':ok&(allf['month'].to_numpy()==i) for i in sorted(allf['month'].unique().to_list())})
    mm.update({f'airport_{a}':ok&(allf['ADEP_mvt'].to_numpy()==a) for a in A.HUBS})
    result=dict(protocol=PROTOCOL,registration_commit=reg,inventory_sha256=A.sha(root/RECEIPT),folds=reports,
        metrics={k:paired(allf[A.Y].to_numpy(),allf['joint'].to_numpy(),allf['ordered'].to_numpy(),v) for k,v in mm.items()},
        sum_fold_trims={population:{k:sum(f['metrics'][population][k] for f in reports) for k in ('trim3_sse','trim20_sse')} for population in ('included','decided','body')},
        gate_trim_population='decided',
        gate_pass=gate(reports),ranking_authorized_if_pass=False,rows=allf.height,included=int(ok.sum()),excluded=int((~ok).sum()),
        control_exact=True,all_six_selection_informed=True,final_chain_complete=False,accepted_for_submission=False,
        limitation=A.LIMITATION+' Selected E197 pairs and confirmation E202 pairs reported separately; this current-chain adaptation is selection-informed.',seconds=time.monotonic()-start,
        files={p.name:A.sha(p) for p in tmp.glob('*.parquet')})
    inventory(root);A.save(tmp/'result.json',result);tmp.rename(out)
    print('PRE-E81 GATE',result['gate_pass'],json.dumps(result['metrics']['body']),flush=True)

def extension(root,require_gate=False):
    m=A.read(root/EXT_INPUT/'manifest.json');c=A.read(root/EXT_OUTPUT/'complete.json');v=A.read(root/EXT_OUTPUT/'evaluation.json')
    if m['spec_sha256']!=EXT_SPEC or m['spec_sha256']!=C.digest(m['spec']) or m['spec']['registration_commit']!=EXT_REG or c['registration_commit']!=EXT_REG or c['protocol']!=EXT_PROTOCOL or c['input_spec_sha256']!=m['spec_sha256']:raise ValueError('E202 extension lineage differs')
    if {(r['fold'],r['arm']) for r in c['arms']}!={(f,a) for f in (2,3,5,6) for a in ('flat','ordered')}:raise ValueError('E20224fit inventory differs')
    if v['protocol']!=EXT_PROTOCOL or v['registration_commit']!=EXT_REG or len(v['reports'])!=6:raise ValueError('E202 evaluation lineage differs')
    keys=('ordered_vs_flat','ordered_vs_ple','flat_vs_ple');checks={}
    for key in keys:
        reports=[r[key] for r in v['reports']]
        checks[key]=sum(r['decided_sse']<0 for r in reports)>=4 and all(sum(r[k] for r in reports)<0 for k in ('body_sse','trim3_sse','trim20_sse'))
    passed=checks['ordered_vs_flat'] and checks['ordered_vs_ple'] and checks['flat_vs_ple']
    if (checks['ordered_vs_flat'] and checks['ordered_vs_ple'])!=v['local_order_six_pair_pass'] or checks['flat_vs_ple']!=v['context_capacity_six_pair_pass']:raise ValueError('E202 saved gate arithmetic differs')
    if require_gate and not passed:raise ValueError('E202 production prerequisite did not pass')
    return dict(complete_sha256=A.sha(root/EXT_OUTPUT/'complete.json'),evaluation_sha256=A.sha(root/EXT_OUTPUT/'evaluation.json'),input_spec_sha256=m['spec_sha256'],passed=passed)

def verify_trajectory(root,fold):
    import e202_trajectory_extension as Q
    d=trajectory_path(root,fold)
    if fold in (1,4):
        complete=root/'data/e197_trajectory_20261007/complete.json';bundle=root/'data/e197_inputs_20261007'
        if A.sha(complete)!=OLD_COMPLETE:raise ValueError('original selected screen changed')
    else:
        extension(root);complete=root/EXT_OUTPUT/'complete.json';bundle=root/EXT_INPUT
    m=A.read(bundle/'manifest.json');fs=next(f for f in m['spec']['folds'] if f['fold']==fold)
    Q.checked_arm(root,bundle/f'fold_{fold}_{fold+6}',root/fs['query_bundle'],d,'ordered',A.read(complete))
    return dict(fold=fold,manifest_sha256=A.sha(d/'manifest.json'),complete_sha256=A.sha(complete))

def combine_cv(root,reg):
    registration(root,reg);proof=extension(root);base=root/CV_OUT
    if (base/'result.json').exists():raise ValueError('fresh final CV receipt required')
    frames=[];reports=[];inputs={}
    for label,folds in [('selected',(1,4)),('confirmation',(2,3,5,6))]:
        r=A.read(base/label/'result.json')
        if r['registration_commit']!=reg or r['protocol']!=PROTOCOL or [x['fold'] for x in r['folds']]!=list(folds) or not r['control_exact']:raise ValueError('subset lineage differs')
        for n,h in r['files'].items():
            if A.sha(base/label/n)!=h:raise ValueError('subset predictions changed')
        frames.append(pl.read_parquet(base/label/'predictions.parquet'));reports.extend({**x,'selection_group':label} for x in r['folds']);inputs[label]=A.sha(base/label/'result.json')
    frame=pl.concat(frames).sort(A.ID);reports.sort(key=lambda x:x['fold'])
    if frame.height!=2085047 or frame[A.ID].n_unique()!=2085047:raise ValueError('incomplete six-pair population')
    frame.write_parquet(base/'predictions.parquet');mm=masks(frame)
    metrics={k:paired(frame[A.Y].to_numpy(),frame['joint'].to_numpy(),frame['ordered'].to_numpy(),v) for k,v in mm.items()}
    groups={group:{pop:{key:sum(r['metrics'][pop][key] for r in reports if r['selection_group']==group) for key in ('delta_sse','trim3_sse','trim20_sse')} for pop in ('included','body','decided')} for group in ('selected','confirmation')}
    result=dict(protocol=PROTOCOL,registration_commit=reg,subsets=inputs,extension=proof,folds=reports,metrics=metrics,groups=groups,gate_trim_population='decided',current_chain_gate_pass=gate(reports),gate_pass=gate(reports) and proof['passed'],control_exact=True,rows=len(frame),excluded=int((~frame['evaluation_mask']).sum()),files={'predictions.parquet':A.sha(base/'predictions.parquet')},full_tail_oof_validated=False,accepted_for_submission=False,limitation=A.LIMITATION+' Ordered member selected on E197/E202; six-pair chain check is selection-informed, not independent nested confirmation.')
    A.save(base/'result.json',result);print('FINAL CURRENT-CHAIN GATE',result['gate_pass'],json.dumps(metrics),flush=True)

def checked_gate(root,reg):
    r=A.read(root/CV_OUT/'result.json')
    if r['protocol']!=PROTOCOL or r['registration_commit']!=reg or not r['control_exact'] or not r['gate_pass'] or not gate(r['folds']):raise ValueError('six-pair current-chain gate failed/incomplete')
    p=extension(root,True)
    if p!=r['extension']:raise ValueError('E202 gate proof changed')
    for n,h in r['files'].items():
        if A.sha(root/CV_OUT/n)!=h:raise ValueError('current-chain output changed')
    return r

def pack_rank(root,reg):
    """Label-free token transfer spec plus exact pre-existing full query mapping.

    May prepare before the six-pair gate; actual fit still requires both gates.
    """
    registration(root,reg);out=root/PACK
    if out.exists():raise ValueError('fresh rank pack required')
    m=R.verify(root/RANK_INPUT,RANK_REG);s=m['spec']
    if m['spec_sha256']!=RANK_SPEC:raise ValueError('original query changed')
    os.environ.update(R.FIXED_ENV);N=E.trainer(root/RANK_INPUT/'src')
    early=s['columns'][:48];cols=list(dict.fromkeys([A.ID,A.Y,'split','month',*early,'d_mvt_aobt','d_mvt_lobt','d_mvt_eobt','d_mvt_sched','d_mvt_iobt']))
    f=pl.read_parquet(root/'data/features_fin1b.parquet',columns=cols);fit=f.filter(pl.col('split')=='train');pred=f.filter(pl.col('split')=='rank')
    if C.digest(fit[A.ID].to_list())!=s['train_id_sha256'] or pred[A.Y].is_not_null().any():raise ValueError('full training identity/ranking label boundary changed')
    ids=pred[A.ID].to_numpy();R.check_ids(ids)
    if not np.array_equal(ids,np.load(root/RANK_INPUT/'mvt_id.npy')):raise ValueError('ranking tensor order differs')
    lab=N.T.regime_label(fit);tr=np.flatnonzero(lab==0);ordinary=fit.filter(pl.Series(lab==0))
    if len(tr)!=s['regular_train_rows']:raise ValueError('ordinary population changed')
    xnf=np.load(root/RANK_INPUT/'xnf.npy',mmap_mode='r');xnp=np.load(root/RANK_INPUT/'xnp.npy',mmap_mode='r')
    alignment=D.verify_query_order(ordinary,pred,early,N.T.CATEGORICAL,xnf,xnp)
    base_f,base_p=N.T.proxy_base(ordinary,pred);y=np.clip(np.clip(ordinary[A.Y].to_numpy(),*N.T.REG_CLIP)-base_f,-7200,7200)
    if not np.array_equal(y/s['sd'],np.load(root/RANK_INPUT/'y_scaled.npy')) or not np.array_equal(base_p,np.load(root/RANK_INPUT/'base_p.npy')):raise ValueError('ordinary residual or ranking proxy alignment differs')
    query_ids=np.r_[fit[A.ID].to_numpy(),ids].astype(np.int64);pr=np.arange(len(fit),len(query_ids),dtype=np.int64)
    if len(query_ids)!=NROWS or len(np.unique(query_ids))!=NROWS:raise ValueError('full train/rank ID coverage differs')
    tmp=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent));(tmp/'scripts').mkdir()
    for n in FILES:shutil.copy2(root/'scripts'/n,tmp/'scripts'/n)
    for n in ('e202_trajectory_extension.py','e200_mdn_production.py'):shutil.copy2(root/'scripts'/n,tmp/'scripts'/n)
    for name,a in [('query_ids',query_ids),('train_indices',tr),('pred_indices',pr)]:np.save(tmp/(name+'.npy'),a,allow_pickle=False)
    days=[];receipts={}
    for shard in (0,1):
        path=root/f'data/e194_sequence_20261007_s{shard}/complete.json';r=A.read(path)
        if r['registration_commit']!=S.RAW_REG:raise ValueError('wrong raw sequence registration')
        receipts[str(path.relative_to(root))]=A.sha(path)
        for d in r['reports']:days.append(dict(day=d['day'],path=f'data/e194_sequence_20261007_s{shard}/'+d['day']+'/sequence.npz',sha256=d['npz_sha256'],rows=d['rows']))
    days.sort(key=lambda x:x['day'])
    if len(days)!=427 or sum(x['rows'] for x in days)!=NROWS:raise ValueError('raw427day coverage differs')
    spec=dict(protocol=PROTOCOL,registration_commit=reg,ranking_spec=RANK_SPEC,ranking_registration=RANK_REG,rows=NROWS,regular_train_rows=len(tr),ranking_rows=len(pr),alignment_numeric_columns=alignment,ordinary_ids_sha256=C.digest(query_ids[tr].tolist()),raw_registration=S.RAW_REG,raw_days=days,raw_complete_sha256=receipts,configuration=dict(arm='ordered',seeds=[0,1,2],epochs=8,batch=4096,normalization='ordinary_full_training_only_channels0to5_clip8',fit_source_sha256=E197_SHA,prep_workers=1),scripts={n.name:A.sha(n) for n in (tmp/'scripts').iterdir()},files={str(n.relative_to(tmp)):A.sha(n) for n in tmp.rglob('*') if n.is_file()})
    A.save(tmp/'manifest.json',dict(spec=spec,spec_sha256=C.digest(spec)));tmp.rename(out);print('PACKED FULL RANK MAPPING',NROWS,'rows',len(tr),'ordinary rows',flush=True)

def verify_rank_input(root,reg):
    b=root/PACK;m=A.read(b/'manifest.json');s=m['spec']
    if m['spec_sha256']!=C.digest(s) or s['protocol']!=PROTOCOL or s['registration_commit']!=reg or s['ranking_spec']!=RANK_SPEC or s['rows']!=NROWS or s['configuration']['fit_source_sha256']!=E197_SHA:raise ValueError('rank spec differs')
    for n,h in s['files'].items():
        if A.sha(b/n)!=h:raise ValueError('rank packed file changed '+n)
    for n,h in s['scripts'].items():
        if A.sha(Path(__file__).parent/n)!=h:raise ValueError('rank active helper changed '+n)
    return m

def prep_guard():
    rel=next(x[3:] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'));p=Path('/sys/fs/cgroup')/rel.lstrip('/')
    mem,swap=(p/'memory.max').read_text().strip(),(p/'memory.swap.max').read_text().strip();quota,period=(p/'cpu.max').read_text().split()
    if mem=='max' or int(mem)>2*1024**3 or swap!='0' or quota=='max' or int(quota)>int(period):raise ValueError('rank prep needs1CPU/2GiB/swap0')
    return dict(cgroup=rel,memory_max=int(mem),cpu_max=[int(quota),int(period)],swap=0)

def prepare_rank(root,reg):
    m=verify_rank_input(root,reg);guard=prep_guard();b=root/PACK;out=root/PREPARED
    if out.exists():raise ValueError('fresh rank prepared table required')
    if shutil.disk_usage(root).free<14*1024**3:raise ValueError('remote rank prep needs14GiB free')
    s=m['spec'];ids=np.load(b/'query_ids.npy');tr=np.load(b/'train_indices.npy');order=np.argsort(ids);sorted_ids=ids[order];fit=np.zeros(NROWS,bool);fit[tr]=True;seen=np.zeros(NROWS,bool);count=np.zeros(6,np.int64);total=np.zeros(6);squares=np.zeros(6)
    for n,h in s['raw_complete_sha256'].items():
        if A.sha(root/n)!=h:raise ValueError('raw complete changed')
    for d in s['raw_days']:
        path=root/d['path']
        if A.sha(path)!=d['sha256']:raise ValueError('raw sequence changed')
        with np.load(path,allow_pickle=False) as a:
            ix=S.map_day(a['ids'],sorted_ids,order)
            if seen[ix].any():raise ValueError('duplicate raw query')
            seen[ix]=True
            if not d['day'].startswith('2025'):
                if fit[ix].any():raise ValueError('ranking entered normalization')
                continue
            n,t,ss=S.norm_stats(a['values'],a['known'],fit[ix]);count+=n;total+=t;squares+=ss
    if not seen.all():raise ValueError('missing raw mapping')
    mean,std=S.normalization(count,total,squares);tmp=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent));shape=(NROWS,2,64,12)
    arrays={n:np.lib.format.open_memmap(tmp/(n+'.npy'),mode='w+',dtype=dt,shape=sh) for n,dt,sh in [('values','float16',shape),('known','uint8',shape),('present','uint8',shape[:3]),('support','uint8',(NROWS,2))]}
    counts=np.zeros(4,np.int64)
    for j,d in enumerate(s['raw_days']):
        with np.load(root/d['path'],allow_pickle=False) as a:
            ix=S.map_day(a['ids'],sorted_ids,order);v,k,p=a['values'],a['known'],a['present']
            if np.any(k>p[...,None]) or np.any(a['point_times']>a['takeoff'][:,None,None]):raise ValueError('raw cutoff/mask drift')
            sup=(p.sum(2)>0).astype(np.uint8);arrays['values'][ix]=S.normalize(v,k,mean,std);arrays['known'][ix]=k;arrays['present'][ix]=p;arrays['support'][ix]=sup
            counts+=np.array([int(p.sum()),int(sup[:,0].sum()),int(sup[:,1].sum()),int((sup[:,0]&sup[:,1]).sum())])
        for a in arrays.values():S.release_pages(a,write=True)
        if j%30==0:print('PREPARED DAY',j+1,'/427',flush=True)
    del arrays;gc.collect();np.save(tmp/'query_ids.npy',ids)
    A.save(tmp/'normalization.json',dict(count=count.tolist(),mean=mean.tolist(),std=std.tolist(),training_ids_sha256=C.digest(ids[tr].tolist()),ranking_excluded=True))
    for d in s['raw_days']:
        if A.sha(root/d['path'])!=d['sha256']:raise ValueError('raw changed during preparation')
    A.save(tmp/'complete.json',dict(protocol=PROTOCOL,registration_commit=reg,input_spec_sha256=m['spec_sha256'],rows=NROWS,cgroup=guard,observation_counts=counts.tolist(),files={p.name:A.sha(p) for p in tmp.iterdir() if p.is_file()}));tmp.rename(out);print('COMPLETE FULL RANK PREPARATION',flush=True)

def verify_prepared(root,reg):
    m=verify_rank_input(root,reg);p=root/PREPARED;r=A.read(p/'complete.json')
    if r['protocol']!=PROTOCOL or r['registration_commit']!=reg or r['input_spec_sha256']!=m['spec_sha256'] or r['rows']!=NROWS:raise ValueError('prepared rank lineage changed')
    if set(r['files'])!={'values.npy','known.npy','present.npy','support.npy','query_ids.npy','normalization.json'}:raise ValueError('prepared file inventory')
    for n,h in r['files'].items():
        if A.sha(p/n)!=h:raise ValueError('prepared artifact changed '+n)
    ids=np.load(p/'query_ids.npy');tr=np.load(root/PACK/'train_indices.npy');norm=A.read(p/'normalization.json')
    if not np.array_equal(ids,np.load(root/PACK/'query_ids.npy')) or norm['training_ids_sha256']!=C.digest(ids[tr].tolist()) or not norm['ranking_excluded']:raise ValueError('rank normalization provenance')
    if not np.isfinite(norm['mean']).all() or not np.isfinite(norm['std']).all() or min(norm['std'])<=0:raise ValueError('invalid normalization')
    arrays={n:np.load(p/(n+'.npy'),mmap_mode='r') for n in ('values','known','present','support')}
    counts=np.zeros(4,np.int64)
    for n,a in arrays.items():
        shape=(NROWS,2,64,12) if n in ('values','known') else ((NROWS,2,64) if n=='present' else (NROWS,2))
        if a.shape!=shape or a.dtype!=np.dtype('float16' if n=='values' else 'uint8'):raise ValueError('rank tensor shape/dtype')
    for lo in range(0,NROWS,4096):
        v,k,p,s=[np.asarray(arrays[n][lo:lo+4096]) for n in ('values','known','present','support')]
        if not np.isfinite(v).all() or (k>1).any() or (p>1).any() or (k>p[...,None]).any() or (v[k==0]!=0).any() or (np.abs(v[...,:6])>8).any() or not np.array_equal(s,(p.sum(2)>0).astype(np.uint8)):raise ValueError('rank tensor masks/finite contract')
        counts+=np.array([int(p.sum()),int(s[:,0].sum()),int(s[:,1].sum()),int((s[:,0]&s[:,1]).sum())])
        for a in arrays.values():S.release_pages(a)
    if counts.tolist()!=r['observation_counts']:raise ValueError('prepared observation counts changed')
    return r

def gpu_tables(prepared):
    """Identical E197 chunked upload; only total table row count changes."""
    import torch
    tables={}
    for name,dtype in [('values',torch.float16),('known',torch.uint8),('present',torch.uint8)]:
        a=np.load(prepared/(name+'.npy'),mmap_mode='r');shape=(NROWS,2,64,12) if name!='present' else (NROWS,2,64)
        if a.shape!=shape or a.dtype!=np.dtype('float16' if name=='values' else 'uint8'):raise ValueError('full rank GPU shape/dtype')
        target=torch.empty(a.shape,dtype=dtype,device='cuda:0')
        for lo in range(0,len(a),8192):
            chunk=np.array(a[lo:lo+8192],copy=True);target[lo:lo+len(chunk)].copy_(torch.from_numpy(chunk));S.release_pages(a)
        tables[name]=target;del a,chunk
    return tables

def smoke(root,reg):
    """Synthetic full-batch backward with exact full-ranking table reservation."""
    S.fixed_env();import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);M.verify_cgroup();verify_rank_input(root,reg)
    q=root/RANK_INPUT;s=A.read(q/'manifest.json')['spec'];a=np.load(q/'xnf.npy',mmap_mode='r');width=a.shape[1];del a
    N=E.trainer(q/'src');rng=np.random.default_rng(203);x=rng.normal(size=(4096,width)).astype(np.float32);x[:,-20:]=x[:,-20:]>0;cont,scalar,bins=E.numeric_bins(x)
    xn=torch.from_numpy(x).to('cuda:0');xc=torch.zeros((4096,len(s['card'])),dtype=torch.int64,device='cuda:0');v=torch.randn((4096,2,64,12),dtype=torch.float16,device='cuda:0');k=torch.ones_like(v,dtype=torch.uint8);p=torch.ones((4096,2,64),dtype=torch.uint8,device='cuda:0')
    reserve=torch.empty((NROWS*2*64*37,),dtype=torch.uint8,device='cuda:0');torch.manual_seed(203);net=S.network(N,s['card'],cont,scalar,bins,width,'ordered').to('cuda:0');opt=torch.optim.AdamW(net.parameters(),lr=.002);torch.cuda.reset_peak_memory_stats()
    for _ in range(6):
        pred=net(xn,xc,v,k,p);loss=pred.square().mean();opt.zero_grad();loss.backward();opt.step()
        if not torch.isfinite(pred).all() or any(t.grad is not None and not torch.isfinite(t.grad).all() for t in net.parameters()):raise ValueError('full ranking synthetic gradients nonfinite')
    torch.cuda.synchronize();A.save(root/'data/e203_rank_smoke_20261007.json',dict(passed=True,synthetic_only=True,protocol=PROTOCOL,registration_commit=reg,source_sha256=A.sha(__file__),e197_source_sha256=A.sha(S.__file__),runtime=E.package_runtime(),rows=NROWS,table_bytes=reserve.numel(),batch=4096,numeric_width=width,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
    print('PASS FULL RANKING GPU SMOKE',torch.cuda.max_memory_allocated(),flush=True)

def verify_rank_output(root,reg):
    m=verify_rank_input(root,reg);p=root/RANK_OUT;r=A.read(p/'complete.json');s=A.read(root/RANK_INPUT/'manifest.json')['spec']
    if r['protocol']!=PROTOCOL or r['registration_commit']!=reg or r['input_spec_sha256']!=m['spec_sha256'] or r['runtime']!=s['admitted_runtime'] or [f['seed'] for f in r['fits']]!=[0,1,2] or any(len(f['losses'])!=8 or not np.isfinite(f['losses']).all() for f in r['fits']):raise ValueError('ranking fit recipe incomplete')
    for n,h in r['files'].items():
        if A.sha(p/n)!=h:raise ValueError('ranking model/prediction changed')
    ids=np.load(root/RANK_INPUT/'mvt_id.npy');R.check_ids(ids);seeds=[np.load(p/f'seed_{i}.npy') for i in (0,1,2)]
    if any(a.shape!=(344841,) or a.dtype!=np.float32 or not np.isfinite(a).all() for a in seeds):raise ValueError('ranking seed incomplete')
    expected=np.mean(seeds,axis=0)*s['sd']+np.load(root/RANK_INPUT/'base_p.npy');z=E.read(p/'prediction.npz')
    if set(z)!={'mvt_id','reg','reg2'} or not np.array_equal(z['mvt_id'],ids) or not np.array_equal(z['reg'],expected) or not np.array_equal(z['reg2'],expected):raise ValueError('ranking mean/IDs differ')
    return r

def rank(root,reg):
    S.fixed_env();import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);guard=M.verify_cgroup();m=verify_rank_input(root,reg)
    # Small authorisation receipt is transferred after BOTH six-pair gates.
    gateproof=A.read(root/PACK/'gate_authorization.json')
    if gateproof['protocol']!=PROTOCOL or gateproof['registration_commit']!=reg or not gateproof['current_chain_gate_pass'] or not gateproof['extension_pass']:raise ValueError('rank fit not authorized by both gates')
    result=gateproof['result']
    if result['protocol']!=PROTOCOL or result['registration_commit']!=reg or not result['control_exact'] or not gate(result['folds']) or not result['extension']['passed'] or result['extension']!=gateproof['extension']:raise ValueError('transferred current-chain gate arithmetic differs')
    ext=extension(root,True)
    if ext!=gateproof['extension']:raise ValueError('Home E202 extension proof differs')
    q=root/RANK_INPUT;s=R.verify(q,RANK_REG)['spec'];runtime=E.package_runtime();sm=A.read(root/'data/e203_rank_smoke_20261007.json')
    if runtime!=s['admitted_runtime'] or not sm['passed'] or sm['registration_commit']!=reg or sm['source_sha256']!=A.sha(__file__) or sm['e197_source_sha256']!=E197_SHA or sm['runtime']!=runtime or sm['rows']!=NROWS:raise ValueError('rank runtime/full-shape smoke differs')
    prep=verify_prepared(root,reg);out=root/RANK_OUT
    if out.exists():raise ValueError('fresh ranking output required')
    arrays={n:np.load(q/(n+'.npy'),allow_pickle=False) for n in R.ARRAYS};ids=np.load(q/'mvt_id.npy');R.validate(arrays,ids,s);tr=np.load(root/PACK/'train_indices.npy');pr=np.load(root/PACK/'pred_indices.npy')
    if len(tr)!=len(arrays['xnf']) or len(pr)!=len(arrays['xnp']):raise ValueError('ranking context/query alignment')
    cont,scalar,bins=E.numeric_bins(arrays['xnf']);N=E.trainer(q/'src');tables=gpu_tables(root/PREPARED);tmp=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent))
    np.savez(tmp/'bins.npz',continuous=cont,scalar=scalar,**{f'bin_{i}':v for i,v in enumerate(bins)})
    fits=[S.fit(arrays,s,N,cont,scalar,bins,tables,tr,pr,tmp,'ordered',seed) for seed in (0,1,2)]
    pred=np.mean([np.load(tmp/f'seed_{i}.npy') for i in (0,1,2)],axis=0)*s['sd']+arrays['base_p']
    if not np.isfinite(pred).all():raise ValueError('ranking nonfinite')
    np.savez(tmp/'prediction.npz',mvt_id=ids,reg=pred,reg2=pred)
    A.save(tmp/'complete.json',dict(protocol=PROTOCOL,registration_commit=reg,input_spec_sha256=m['spec_sha256'],runtime=runtime,cgroup=guard,rows=len(ids),fits=fits,prepared_complete_sha256=A.sha(root/PREPARED/'complete.json'),gate_authorization_sha256=A.sha(root/PACK/'gate_authorization.json'),files={p.name:A.sha(p) for p in tmp.iterdir() if p.is_file()}));tmp.rename(out);verify_rank_output(root,reg);print('COMPLETE ORDERED RANKING3SEED8EPOCH',flush=True)

def authorize_rank(root,reg):
    registration(root,reg);g=checked_gate(root,reg)
    # Carry actual reports, not just a freely asserted boolean, across transport.
    A.save(root/PACK/'gate_authorization.json',dict(protocol=PROTOCOL,registration_commit=reg,current_chain_gate_pass=True,extension_pass=True,current_chain_result_sha256=A.sha(root/CV_OUT/'result.json'),result=g,extension=g['extension']))

def build(root,reg):
    import today_submissions_20261006 as T
    registration(root,reg);checked_gate(root,reg);inventory(root);rank_receipt=verify_rank_output(root,reg);work=root/WORK;c=A.read(work/'control_verified.json')
    if c['registration_commit']!=reg or not c['control_rows_exact'] or c['reference_sha256']!=BEST_SHA:raise ValueError('v12074 exact control proof absent')
    for name,h in c['models'].items():
        if A.sha(work/'models'/name/'complete.json')!=h:raise ValueError('cached model manifest changed')
    out=root/FINAL
    if out.exists():raise ValueError('fresh full candidate required')
    final=branch(root,reg,RANK_OUT+'/prediction.npz',940100,'replay');candidate=pl.read_parquet(final)
    checks=T.G.validate_submission(candidate,pl.read_parquet(root/T.E.TEMPLATE),pl.read_parquet(root/BEST),pl.read_parquet(root/'submissions/gentle-igloo_v291.parquet'),pl.read_parquet(root/'submissions/gentle-igloo_v361.parquet'))
    if candidate[A.Y].min()<0 or candidate[A.Y].null_count():raise ValueError('invalid final candidate')
    tmp=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent));shutil.copy2(final,tmp/'candidate.parquet');inventory(root)
    A.save(tmp/'manifest.json',dict(protocol=PROTOCOL,registration_commit=reg,reference=BEST,reference_sha256=BEST_SHA,recipe='Fixed v12074 full chain; replace PLE with ordered E197 network at0.25; sameE188/E152/E81/LOBT/42donors/halfLSZH',validation=checks,sha256=A.sha(tmp/'candidate.parquet'),rank_complete_sha256=A.sha(root/RANK_OUT/'complete.json'),rank_receipt=rank_receipt,control_verified_sha256=A.sha(work/'control_verified.json'),current_chain_gate_sha256=A.sha(root/CV_OUT/'result.json'),full_tail_oof_validated=False,selection_informed=True,uploaded=False,accepted_for_submission=False));tmp.rename(out);print('FULL CANDIDATE READY; NO UPLOAD',flush=True)

def selftest():
    assert NROWS*2*64*37==11507949568
    good=[dict(metrics=dict(decided=dict(delta_sse=-1,trim3_sse=-1,trim20_sse=-1),body=dict(delta_sse=-1))) for _ in range(6)]
    assert gate(good) and not gate(good[:2]);bad=json.loads(json.dumps(good));bad[0]['metrics']['decided']['trim20_sse']=10;assert not gate(bad)
    # Capture/replay arithmetic is copied, not monkeypatched into the predecessor.
    original=ast.parse(Path(__file__).with_name('e200_mdn_production.py').read_text());current=ast.parse(Path(__file__).read_text())
    for name in ('paired','masks','gate','tensor_fingerprint','training_signature','check_signature','stage_job','branch'):
        a=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name);b=next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),name
    print('PASS fixedgate/partialgate/DECIDEDtrim and frozenE200 arithmetic AST parity; no target fit',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['seal','selftest','cv-selected','cv-confirmation','combine-cv','pack-rank','prepare-rank','verify-prepared','authorize-rank','smoke','rank','verify-rank','capture-control','build','stage']);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--registration-commit');p.add_argument('--threads',type=int,default=2);p.add_argument('--job',type=Path);a=p.parse_args();root=a.root.resolve();reg=a.registration_commit
    if a.action=='seal':return seal(root)
    if a.action=='selftest':return selftest()
    if a.action=='stage':return stage_job(root,reg,a.job)
    if a.action=='cv-selected':return cv_subset(root,reg,a.threads,(1,4),'selected')
    if a.action=='cv-confirmation':return cv_subset(root,reg,a.threads,(2,3,5,6),'confirmation')
    fn={'combine-cv':combine_cv,'pack-rank':pack_rank,'prepare-rank':prepare_rank,'verify-prepared':verify_prepared,'authorize-rank':authorize_rank,'smoke':smoke,'rank':rank,'verify-rank':verify_rank_output,'capture-control':capture,'build':build}[a.action];fn(root,reg)

if __name__=='__main__':main()

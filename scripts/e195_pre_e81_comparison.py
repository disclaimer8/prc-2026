"""Registered retrospective four-arm application through E152, before E81.

No fitting, Torch, submissions, model selection or full-chain acceptance.
Cached FIN1b probabilities define the upstream boundary. A changed feasibility
mask excludes that row from all comparisons before any labels/errors are used.
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
import tempfile
import time

for k,v in {'POLARS_MAX_THREADS':'4','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
            'OMP_WAIT_POLICY':'PASSIVE','GOMP_SPINCOUNT':'0',
            'POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR':'load_as_storage'}.items():
    os.environ.setdefault(k,v)
import numpy as np
import polars as pl

PROTOCOL='E195-four-fixed-current-main-pre-E81-retrospective-v1-20261007'
SELF='scripts/e195_pre_e81_comparison.py'
RECEIPT='analysis/e195_pre_e81_input_receipt_20261007.json'
BUNDLE='data/e195_stage_inputs_20261007'
MODELS='data/e195_stage_models_20261007'
STAGE_REG='9078e3508e81bb7c1e6cee432226063b0067b44e'
STAGE_SPEC='5c4a0281a6669d856331317128917097b2a87df55a5c44f3cb0f0d26602441b2'
FEATURE_SHA='f061a441f6c97c232d9a403d612471e45020eb4ad56f501c2acfd7220a7d3052'
CAT_LINEAGE='78ae852b4b0fccec580fda8c724ff476e8f7405add1af0cf73f6936c299a2b78'
LGB_LINEAGE='d1749457e64d8a428c59c519036472c7f5894c014280b471b67d5d6e4dade7f7'
ID,Y='MVT_ID_mvt','TAXITIME_SEC_mvt'
ARMS=['control','e190','e188','joint']
SEEDS=[42,1,2,3,4]
WEIGHTS=[.0333333333333333,.10,.10,.10,.10,.25]
ANCHORS=['rawf','lobtf','schedf','eobtf','aobtf']
HUBS=['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LTFM','LSZH']
LIMITATION=('Retrospective historical-upstream crossfit, not independently nested CV. '
            'Exact transforms conditional on saved FIN1b masked OOF probabilities; '
            '126 TabPFN rows and union of changed-feasibility rows excluded. '
            'Ends at E152 before E81. No full-tail, LSZH or leaderboard claim.')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False)


def read(p):return json.loads(Path(p).read_text())


def registration(root,commit):
    if not re.fullmatch('[a-f0-9]{40}',commit or ''):raise ValueError('full registration SHA required')
    for name in (SELF,RECEIPT):
        old=subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=root)
        if hashlib.sha256(old).hexdigest()!=sha(root/name):raise ValueError('registration drift: '+name)
    log=subprocess.check_output(['git','show',f'{commit}:HYPOTHESES_LOG.md'],cwd=root)
    if PROTOCOL.encode() not in log:raise ValueError('protocol missing from registered journal')


def seal(root):
    """Read-only provenance checks and a new immutable inventory; no predictions/metrics."""
    inventory={}
    def add(p,expected=None):
        p=Path(p);h=sha(p)
        if expected is not None and h!=expected:raise ValueError('SHA mismatch: '+str(p))
        inventory[str(p.relative_to(root))]=h
    add(root/'data/features_fin1b.parquet',FEATURE_SHA)
    add(root/BUNDLE/'manifest.json');m=read(root/BUNDLE/'manifest.json')
    if m['spec_sha256']!=STAGE_SPEC or m['spec']['registration_commit']!=STAGE_REG:raise ValueError('stage provenance')
    for name,h in m['spec']['files'].items():add(root/BUNDLE/name,h)
    for p in sorted((root/'data/prc-2026-datasets').glob('training_*.parquet')):add(p,m['spec']['inputs'][str(p.relative_to(root))])
    add(root/'data/e16_lomo.parquet')
    c=read(root/'data/cat_extra175_20261006/merged/complete.json')
    if c['lineage']!=CAT_LINEAGE or c['fits']!=90 or c['folds']!=6:raise ValueError('Cat90 incomplete')
    for name in ['data/cat_extra175_20261006/merged/complete.json','data/lgb_fin1b_oof_20261005/complete.json','data/e190_numemb_20261005/complete.json']:add(root/name)
    e190=read(root/'data/e190_numemb_20261005/complete.json')
    for a in range(1,7):
        fold=f'fold_{a}_{a+6}'
        base=root/MODELS/fold;r=read(base/'complete.json');add(base/'complete.json')
        if r['registration_commit']!=STAGE_REG or r['input_spec_sha256']!=STAGE_SPEC or r['fold']!=a:raise ValueError('stage fold provenance')
        if [x['stage'] for x in r['results']]!=['e141','e64','e188','lszh']:raise ValueError('stage inventory')
        for s in r['results']:
            if s['heldout_months']!=[a,a+6] or [x['seed'] for x in s['models']]!=(SEEDS if s['stage']!='lszh' else [42]):raise ValueError('stage seeds/pair')
            for z in s['models']:add(base/s['stage']/f'seed{z["seed"]}.txt',z['sha256'])
        cat=root/'data/cat_extra175_20261006/merged';cr=read(cat/(fold+'.json'))
        if cr['lineage']!=CAT_LINEAGE or cr['seeds']!=list(range(15)) or len(cr['receipts'])!=15:raise ValueError('Cat fold')
        add(cat/(fold+'.json'));add(cat/(fold+'.npz'))
        for kind,directory in [('lgb','data/lgb_fin1b_oof_20261005'),('mlp','data/mlp_split_20261005/outputs')]:
            p=root/directory/fold
            if kind=='lgb':p=p/'complete'
            mr=read(p/'manifest.json');ev=mr['evidence']
            if ev['fold']!=a or ev['seeds']!=(list(range(1000,1005)) if kind=='lgb' else [0,1,2]):raise ValueError('member fold/seeds')
            if kind=='lgb' and (ev['lineage']!=LGB_LINEAGE or ev['rounds']!=2319):raise ValueError('LGB lineage')
            if kind=='mlp' and (ev['protocol']!='mlp-fin1b-split-rocm-oof-3seed-v1' or ev['registration_commit']!='a926f3848baaae7415bd3cb773cbd39728ce1851'):raise ValueError('MLP lineage')
            add(p/'manifest.json');add(p/'prediction.npz',mr['sha256'])
        p=root/'data/e190_numemb_20261005'/fold;er=read(p/'manifest.json')
        if er['fold']!=a or er not in e190['folds']:raise ValueError('E190 fold receipt')
        add(p/'manifest.json')
        for name,h in er['files'].items():add(p/name,h)
        for name in ['cv_preds_fin1b_hz',*[f'cv_e130{x}' for x in ANCHORS],'cv_e130motion']:add(root/'data'/name/(fold+'.npz'))
    # Pin the actual production arithmetic and donor evidence without importing it.
    for name in ['scripts/today_submissions_20261006.py','src/blend.py','src/e185_apply.py','src/e167_nm_window_rule.py',
                 'src/e141_apply.py','src/e64_apply.py','src/e152_stage_gate.py','src/train.py','src/fold_metrics.py',
                 'src/run_final_any_period.sh','src/run_final_e38.sh','src/run_final_e52.sh','src/after_v42_stage.sh','src/e16_tail_regimes.py']:
        add(root/name)
    record=dict(protocol=PROTOCOL,stage_registration=STAGE_REG,stage_spec_sha256=STAGE_SPEC,inputs=inventory,
                model_count=96,comparison_performed=False,limitation=LIMITATION)
    save(root/RECEIPT,record);print('SEALED',RECEIPT,sha(root/RECEIPT),flush=True)


def aligned(path,ids,y=None,month=None):
    with np.load(path,allow_pickle=False) as zz:z={k:zz[k] for k in zz.files}
    # Preserve the producer's summation layout for this exact provenance check.
    # Advanced indexing below changes strides and NumPy's reduction algorithm.
    if 'per_seed' in z and not np.array_equal(z['per_seed'].mean(axis=0),z['reg2']):raise ValueError('original seed mean mismatch '+str(path))
    src=z['mvt_id'];s=np.argsort(src);pos=np.searchsorted(src[s],ids)
    if len(src)!=len(ids) or len(np.unique(src))!=len(src) or np.any(pos>=len(src)) or not np.array_equal(src[s][pos],ids):raise ValueError('ID mismatch '+str(path))
    ix=s[pos]
    for k,v in z.items():
        if k=='per_seed':z[k]=v[:,ix]
        elif v.ndim and v.shape[0]==len(src):z[k]=v[ix]
    if y is not None and 'y' in z and not np.array_equal(z['y'],y):raise ValueError('label mismatch '+str(path))
    if month is not None and 'month' in z and not np.array_equal(z['month'],month):raise ValueError('month mismatch '+str(path))
    return z


def rules(root):
    cols=[ID,'PHASE_mvt','FLIGHT_ID_mvt','FLIGHT_mvt','ADEP_mvt','ADES_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt','AOBT_3_flt']
    raw=pl.concat([pl.read_parquet(p,columns=cols) for p in sorted((root/'data/prc-2026-datasets').glob('training_*.parquet'))])
    ep=lambda c:pl.col(c).dt.epoch('s').cast(pl.Float64)
    dep=raw.filter((pl.col('PHASE_mvt')=='DEP')&pl.col('FLIGHT_ID_mvt').is_null()&pl.col('ADES_mvt').is_in(HUBS)).select(ID,'FLIGHT_mvt','ADEP_mvt','ADES_mvt',mvt=ep('MVT_TIME_UTC_mvt'),sched=ep('SCHED_TIME_UTC_mvt'))
    arr=raw.filter((pl.col('PHASE_mvt')=='ARR')&pl.col('FLIGHT_ID_mvt').is_not_null()).select('FLIGHT_mvt','ADEP_mvt','ADES_mvt',a_mvt=ep('MVT_TIME_UTC_mvt'),a_lobt=ep('LOBT_flt'),a_aobt=ep('AOBT_3_flt'))
    link=dep.join(arr,on=['FLIGHT_mvt','ADEP_mvt','ADES_mvt']).filter((pl.col('a_mvt')-pl.col('mvt')).is_between(0,16*3600))
    link=link.filter(pl.len().over(ID)==1).with_columns(D=pl.col('mvt')-pl.col('sched'),gap=(pl.col('sched')-pl.col('a_lobt')).abs(),agap=(pl.col('a_aobt')-pl.col('a_lobt')).abs(),d_lobt=pl.col('mvt')-pl.col('a_lobt'))
    link=link.filter(pl.col('a_lobt').is_not_null()&(pl.col('d_lobt')>=0)&~((pl.col('ADEP_mvt')=='LIRF')&pl.col('D').is_between(14400,108000,closed='left')))
    a=link.filter((pl.col('D').abs()>3606)&(pl.col('gap')<=3606))[ID].to_numpy()
    b=link.filter((pl.col('ADEP_mvt')=='LIRF')&(pl.col('D').abs()>3606)&(pl.col('gap')>3606)&(pl.col('agap')<=3606))[ID].to_numpy()
    return a,b


def mixture(p,v,lobt):
    pred=np.sum(p*np.nan_to_num(v),axis=1)
    return np.clip(np.where(np.isnan(lobt),pred,np.clip(pred,lobt-3606,lobt+3606)),0,None)


def feasibility(values,lobt):
    bad=~np.isnan(lobt[:,None])&(np.abs(np.nan_to_num(values)-np.nan_to_num(lobt[:,None]))>3606)
    bad[:,0]=False
    return bad|np.isnan(values)


def upstream(z,r,motion,f,ia,ib,excluded):
    lobt=f['d_mvt_lobt'].to_numpy();v=z['values'].astype(float).copy()
    v[:,0]=r;v[:,-1]=r+86400
    changes=np.any(feasibility(v,lobt)!=feasibility(z['values'],lobt),axis=1)&~excluded
    if changes.any():raise ValueError(f'Feasibility exclusion failed on {changes.sum()} scored rows')
    p=z['proba'].astype(float).copy()
    if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1,atol=2e-7):raise ValueError('invalid cached probabilities')
    p=np.where(np.isnan(v),0,p);p=p/p.sum(axis=1,keepdims=True)
    base=np.rint(mixture(p,v,lobt))
    subset=(f['LOBT_flt'].is_null()&(f['ADEP_mvt']!='LIRF')).to_numpy();w=.35*2/3
    r=r.copy();r[subset]=(1-w)*r[subset]+w*motion[subset]
    v[:,0]=r;v[:,-1]=r+86400
    base[subset]=np.rint(mixture(p,v,lobt)[subset])
    p[ia,1]=0;p[ia,5]=0
    rest=np.clip(1-p[ib,1],1e-9,None);p[ib]*=((1-.9)/rest)[:,None];p[ib,1]=.9
    p=np.where(np.isnan(v),0,p);p/=p.sum(axis=1,keepdims=True)
    changed=ia|ib;base[changed]=np.rint(mixture(p,v,lobt)[changed])
    return base,p


def stage_predictions(root,fold,d,base,p,spec,threads):
    if 'torch' in sys.modules:raise RuntimeError('LGB-only fresh process required')
    import lightgbm as lgb
    features={s['name']:s['features'] for s in spec['stages']}
    d=d.with_columns(pl.Series('base',base),pl.Series('p_nonreg',1-p[:,0])).with_columns(
        base_vs_first_move=pl.col('base')-pl.col('mvt_first_move'),base_vs_first=pl.col('base')-pl.col('mvt_first'))
    covered=(d['coverage'].to_numpy()>=2);eligible=((d['coverage']<2)&(d['vi_has_prev']==1)).fill_null(False).to_numpy()
    target=eligible&(d['ADEP_mvt'].to_numpy()=='LIRF')&(d['has_nm'].to_numpy()==1)
    path=root/MODELS/fold
    def predict(name,mask,data):
        x=data.filter(pl.Series(mask)).select(features[name]).to_numpy()
        if np.isinf(x).any():raise ValueError('infinite stage input')
        out=[]
        for seed in SEEDS:
            m=lgb.Booster(model_file=str(path/name/f'seed{seed}.txt'))
            if m.num_feature()!=len(features[name]) or m.current_iteration()!=(800 if name=='e141' else 400):raise ValueError('stage model dimensions')
            out.append(m.predict(x,num_threads=threads))
        return out
    e141=base.copy()
    e141[covered]=np.rint(np.mean([np.clip(base[covered]+c,0,None) for c in predict('e141',covered,d)],axis=0))
    d=d.with_columns(pl.Series('base',e141))
    ordinary=e141.copy();ordinary[eligible]=np.rint(np.clip(e141[eligible]+np.mean(predict('e64',eligible,d),axis=0),0,None))
    augmented=ordinary.copy();augmented[target]=np.rint(np.clip(e141[target]+np.mean(predict('e188',target,d),axis=0),0,None))
    gate=p[:,1:5].max(axis=1)>=.95
    ordinary[gate]=base[gate];augmented[gate]=base[gate]
    if not np.array_equal(ordinary[~target],augmented[~target]):raise ValueError('E188 population escaped')
    return ordinary,augmented


def metrics(y,pred,masks):
    out={};control=(pred['control']-y)**2
    for name,mask in masks.items():
        n=int(mask.sum());entry={}
        for arm in ARMS:
            se=(pred[arm]-y)**2;s=float(se[mask].sum());cs=float(control[mask].sum())
            e=dict(n=n,sse=s,rmse=float(np.sqrt(s/n)) if n else None,delta_sse=s-cs,
                   delta_rmse=float(np.sqrt(s/n)-np.sqrt(cs/n)) if n else None)
            delta=np.sort((se-control)[mask])
            for k in (3,20):e[f'trim{k}_sse']=float(delta[k:-k].sum()) if n>2*k else None
            if arm!='control' and n>40:
                for k in (3,20):
                    cix=np.flatnonzero(mask);keep=mask.copy()
                    worst=np.union1d(cix[np.argsort(control[cix])[-k:]],cix[np.argsort(se[cix])[-k:]])
                    keep[worst]=False
                    e[f'union_top_error_trim{k}']=dict(n=int(keep.sum()),delta_rmse=float(np.sqrt(se[keep].mean())-np.sqrt(control[keep].mean())))
            entry[arm]=e
        entry['conditional_pairs']={}
        for candidate,reference in [('joint','e190'),('joint','e188')]:
            cs=(pred[reference]-y)**2;se=(pred[candidate]-y)**2;delta=np.sort((se-cs)[mask])
            e=dict(n=n,delta_sse=float(delta.sum()),delta_rmse=float(np.sqrt(se[mask].mean())-np.sqrt(cs[mask].mean())) if n else None)
            for k in (3,20):e[f'trim{k}_sse']=float(delta[k:-k].sum()) if n>2*k else None
            entry['conditional_pairs'][candidate+'_vs_'+reference]=e
        out[name]=entry
    return out


def run(root,out,commit,threads):
    registration(root,commit)
    receipt=read(root/RECEIPT)
    if receipt['protocol']!=PROTOCOL or out.exists():raise ValueError('protocol drift or output exists')
    for name,h in receipt['inputs'].items():
        if sha(root/name)!=h:raise ValueError('input drift '+name)
    temp=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent));start=time.monotonic()
    spec=read(root/BUNDLE/'manifest.json')['spec']
    natural=pl.read_parquet(root/BUNDLE/'natural_features.parquet')
    cols=[ID,Y,'split','month','ADEP_mvt','FLIGHT_ID_mvt','LOBT_flt','d_mvt_lobt','d_mvt_sched','d_mvt_iobt','d_mvt_eobt']
    feature=pl.read_parquet(root/'data/features_fin1b.parquet',columns=cols).filter(pl.col('split')=='train').sort(ID)
    tab=((feature['ADEP_mvt']=='LIRF')&feature['FLIGHT_ID_mvt'].is_null()&(feature['d_mvt_sched']>14400)&(feature['d_mvt_sched']<=108000)).fill_null(False).to_numpy()
    if tab.sum()!=126 or set(feature.filter(pl.Series(tab))[ID])!=set(pl.read_parquet(root/'data/e16_lomo.parquet')[ID]):raise ValueError('126-row TabPFN exclusion differs')
    rule_a,rule_b=rules(root);allrows=[];folds=[]
    for a in range(1,7):
        fold=f'fold_{a}_{a+6}';f=feature.filter(pl.col('month').is_in([a,a+6]));ids=f[ID].to_numpy();y=f[Y].to_numpy();month=f['month'].to_numpy()
        d=f.select(ID).join(natural,on=ID,how='left',validate='1:1',maintain_order='left')
        if d[Y].null_count() or not np.array_equal(d[Y].to_numpy(),y) or not np.array_equal(d['month'].to_numpy(),month):raise ValueError('natural feature alignment')
        z=aligned(root/'data/cv_preds_fin1b_hz'/(fold+'.npz'),ids,y)
        planned=f.select('d_mvt_sched','d_mvt_iobt','d_mvt_lobt','d_mvt_eobt').to_numpy()
        if not np.array_equal(z['values'][:,1:5],planned,equal_nan=True):raise ValueError('planned candidates differ')
        cat=aligned(root/'data/cat_extra175_20261006/merged'/(fold+'.npz'),ids,y,month)
        lgb=aligned(root/'data/lgb_fin1b_oof_20261005'/fold/'complete/prediction.npz',ids,y,month)
        for q,n in [(cat,15),(lgb,5)]:
            if q['per_seed'].shape!=(n,len(ids)):raise ValueError('seed count/shape mismatch')
        plain=aligned(root/'data/mlp_split_20261005/outputs'/fold/'prediction.npz',ids,y,month)['reg2']
        ple=aligned(root/'data/e190_numemb_20261005'/fold/'prediction.npz',ids,y,month)['member']
        anchors=[aligned(root/'data'/('cv_e130'+x)/(fold+'.npz'),ids)['reg2'] for x in ANCHORS]
        motion=aligned(root/'data/cv_e130motion'/(fold+'.npz'),ids)['reg2']
        if not all(np.isfinite(v).all() for v in [cat['reg2'],lgb['reg2'],plain,ple,*anchors,motion]):raise ValueError('nonfinite member')
        tab_excluded=np.isin(ids,feature.filter(pl.Series(tab))[ID].to_numpy());ia=np.isin(ids,rule_a);ib=np.isin(ids,rule_b)
        regular=[];unstable=np.zeros(len(ids),dtype=bool)
        for neural in (plain,ple):
            r=.65*cat['reg2']+(1-.65)*lgb['reg2']
            r=(1-sum(WEIGHTS))*r+sum(w*g for w,g in zip(WEIGHTS,[*anchors,neural]))
            regular.append(r)
            v=z['values'].astype(float).copy();v[:,0]=r;v[:,-1]=r+86400
            unstable|=np.any(feasibility(v,f['d_mvt_lobt'].to_numpy())!=feasibility(z['values'],f['d_mvt_lobt'].to_numpy()),axis=1)
        excluded=tab_excluded|unstable
        print('COVERAGE',fold,'tabpfn',int(tab_excluded.sum()),'feasibility',int(unstable.sum()),'included',int((~excluded).sum()),flush=True)
        pred={}
        for r,names in zip(regular,[('control','e188'),('e190','joint')]):
            base,p=upstream(z,r,motion,f,ia,ib,excluded)
            print('APPLY',fold,names,'rows',len(ids),flush=True)
            ordinary,augmented=stage_predictions(root,fold,d,base,p,spec,threads)
            pred[names[0]]=ordinary;pred[names[1]]=augmented
        # Retain the same finite upstream-control fallback on unsupported rows;
        # these are explicitly excluded, never represented as exact stage output.
        for arm in ARMS:pred[arm][excluded]=np.rint(z['pred'][excluded])
        if any(not np.isfinite(x).all() or (x<0).any() or not np.equal(x,np.rint(x)).all() for x in pred.values()):raise ValueError('invalid predictions')
        frame=f.select(ID,Y,'month','ADEP_mvt').with_columns(pl.Series('has_nm',f['FLIGHT_ID_mvt'].is_not_null()),pl.Series('D',f['d_mvt_sched']),pl.Series('excluded_tabpfn',tab_excluded),pl.Series('excluded_feasibility',unstable),pl.Series('evaluation_mask',~excluded))
        frame=frame.with_columns(*[pl.Series(arm,values).cast(pl.Int64) for arm,values in pred.items()])
        frame.write_parquet(temp/(fold+'.parquet'));allrows.append(frame)
        masks={'included':~excluded,'decided':~excluded&~((f['ADEP_mvt'].to_numpy()=='LIRF')&f['FLIGHT_ID_mvt'].is_null().to_numpy()&(f['d_mvt_sched'].to_numpy()>=14400)), 'body':~excluded&(y<=7200)}
        folds.append(dict(fold=a,metrics=metrics(y,pred,masks),excluded=int(excluded.sum()),excluded_tabpfn=int(tab_excluded.sum()),excluded_feasibility=int(unstable.sum()),e167_rule_a=int(ia.sum()),e167_rule_b=int(ib.sum())))
        print('COMPLETE',fold,flush=True)
    allf=pl.concat(allrows).sort(ID)
    if allf.height!=2085047 or allf[ID].n_unique()!=allf.height:raise ValueError('incomplete six-fold population')
    allf.write_parquet(temp/'predictions.parquet');ok=allf['evaluation_mask'].to_numpy();y=allf[Y].to_numpy();pred={a:allf[a].to_numpy() for a in ARMS}
    ap=allf['ADEP_mvt'].to_numpy();nm=allf['has_nm'].to_numpy();D=allf['D'].to_numpy()
    masks=dict(included=ok,decided=ok&~((ap=='LIRF')&~nm&(D>=14400)),body=ok&(y<=7200),LIRF_NM=ok&(ap=='LIRF')&nm)
    masks.update({f'airport_{a}':ok&(ap==a) for a in HUBS});masks.update({f'month_{m}':ok&(allf['month'].to_numpy()==m) for m in range(1,13)})
    result=dict(protocol=PROTOCOL,registration=commit,input_receipt_sha256=sha(root/RECEIPT),limitation=LIMITATION,
                final_chain_complete=False,accepted_for_submission=False,rows=allf.height,excluded=int((~ok).sum()),excluded_tabpfn=126,excluded_feasibility=int(allf['excluded_feasibility'].sum()),included=int(ok.sum()),
                primary_metrics=['decided','body'],metrics=metrics(y,pred,masks),folds=folds,seconds=time.monotonic()-start)
    result['sum_of_fold_trimmed_sse']={population:{arm:{f'trim{k}_sse':sum(f['metrics'][population][arm][f'trim{k}_sse'] for f in folds) for k in (3,20)} for arm in ARMS} for population in ('included','decided','body')}
    for name,h in receipt['inputs'].items():
        if sha(root/name)!=h:raise ValueError('input changed during apply '+name)
    result['outputs']={p.name:sha(p) for p in temp.glob('*.parquet')};save(temp/'result.json',result)
    temp.rename(out);print('READY PRE-E81 ONLY',out,'seconds',result['seconds'],flush=True)


def selftest():
    rng=np.random.default_rng(195)
    seeds=rng.normal(size=(15,1000));mean=seeds.mean(axis=0);order=rng.permutation(1000)
    rearranged=seeds[:,order]
    assert np.array_equal(mean[order],seeds.mean(axis=0)[order])
    assert not np.array_equal(mean[order],rearranged.mean(axis=0))
    assert np.allclose(mean[order],rearranged.mean(axis=0),rtol=0,atol=1e-15)
    p=np.array([[.5,.5,0,0,0,0],[1.,0,0,0,0,0]])
    v=np.array([[100,200,np.nan,np.nan,np.nan,86500],[300,200,np.nan,np.nan,np.nan,86700.]])
    assert np.array_equal(mixture(p,v,np.array([np.nan,np.nan])),[150,300])
    f=pl.DataFrame({'LOBT_flt':[None,None],'ADEP_mvt':['LIRF','EDDF'],'d_mvt_lobt':[None,None]},schema_overrides={'d_mvt_lobt':pl.Float64})
    z={'values':v,'proba':p};base,q=upstream(z,np.array([100.,300.]),np.array([500.,600.]),f,np.array([True,False]),np.array([False,False]),np.array([False,False]))
    assert np.array_equal(base,[100,370]) and q[0,0]==1
    assert np.array_equal(np.rint([.5,1.5,2.5]),pl.Series([.5,1.5,2.5]).round(0).to_numpy())
    print('synthetic arithmetic and C/F seed-reduction selftest passed; no real predictions or scores')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['seal','run','selftest']);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--registration-commit');p.add_argument('--output',type=Path,default=Path('data/e195_pre_e81_20261007'));p.add_argument('--threads',type=int,default=8);a=p.parse_args()
    if a.action=='selftest':selftest()
    elif a.action=='seal':seal(a.root.resolve())
    else:
        if not 1<=a.threads<=8:raise ValueError('1..8 apply threads')
        run(a.root.resolve(),a.output.resolve(),a.registration_commit,a.threads)


if __name__=='__main__':main()

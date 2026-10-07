"""E201: recipe-faithful v44/v60 donor reconstruction on six held-out pairs.

Historical byte reproduction is NOT asserted. Inner-only round selection,
original source snapshots and current immutable cached feature/sensor inputs.
126 TabPFN rows remain represented but excluded from downstream validation.
"""
from __future__ import annotations
import argparse
import ast
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
for k,v in {'POLARS_MAX_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OMP_WAIT_POLICY':'PASSIVE','GOMP_SPINCOUNT':'0','POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR':'load_as_storage'}.items():os.environ.setdefault(k,v)
import numpy as np
import polars as pl

PROTOCOL='E201-v44-v60-recipe-donors-inner-rounds-six-pair-exclude126-v1'
SELF='scripts/e201_donor_oof.py'
DOC='analysis/e201_donor_protocol_20261007.md'
ID,Y='MVT_ID_mvt','TAXITIME_SEC_mvt'
AP=('EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM')
SNAPSHOTS={'v44':'da209d5ae10edfef6958a16fc1c593a12b936efa','v60':'1c35ed46d70eac38152ea8f3a34e24d4f4b6f258'}
DATA={'v44':'features_e28.parquet','v60':'features_e32.parquet'}
STAGE={'v44':'e42_stand_geometry_per_airport.py','v60':'e36_adsb_stand_geometry.py'}
SEEDS=(1000,1001,1002,1003,1004)
ASSIGN={'171':('v44',(1,2,3),6),'175':('v44',(4,5,6),6),'mac':('v60',(1,2),8),'home':('v60',(3,4),12),'hetzner':('v60',(5,6),6)}
PARENT_RESULT='94ba00d1522b7d598bc9d3b42d3361ad68af9dd41ca19b120c5c07f023e05f9e'
LIMITATION='Recipe-faithful newly fitted donors, not byte-identical historical v44/v60. Outer labels excluded from donor and inner-round fitting. ADS-B stage training uses historical nomonth OOF and inherited inputs, not fully nested end-to-end CV. 126 TabPFN rows excluded; no ranking replacement or submission.'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def check_registration(root,commit):
    if not re.fullmatch('[a-f0-9]{40}',commit or ''):raise ValueError('full registration SHA required')
    for name in (SELF,DOC):
        if hashlib.sha256(subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=root)).hexdigest()!=sha(root/name):raise ValueError('registered source drift '+name)
    if PROTOCOL.encode() not in subprocess.check_output(['git','show',f'{commit}:HYPOTHESES_LOG.md'],cwd=root):raise ValueError('missing journal registration')
def checked_ids(d,expected=None):
    ids=d[ID].to_numpy()
    if not np.isfinite(ids).all() or not np.equal(ids,np.floor(ids)).all() or len(np.unique(ids))!=len(ids) or (expected is not None and len(ids)!=expected):raise ValueError('ID integrity failed')
    return ids
def split_months(fold):
    if fold not in range(1,7):raise ValueError('invalid outer fold')
    outer=(fold,fold+6);inner=(fold%12+1,(fold+6)%12+1)
    if set(outer)&set(inner):raise ValueError('inner/outer overlap')
    return outer,inner
def rounded(x):return pl.Series(x).round(0).cast(pl.Int64).to_numpy()

def definitions(path,names,functions,classes=(),extra=None):
    tree=ast.parse(Path(path).read_text());nodes=[]
    for x in tree.body:
        if isinstance(x,ast.Assign) and all(isinstance(t,ast.Name) and t.id in names for t in x.targets):nodes.append(x)
        elif isinstance(x,ast.FunctionDef) and x.name in functions:nodes.append(x)
        elif isinstance(x,ast.ClassDef) and x.name in classes:nodes.append(x)
    scope=dict(np=np,pl=pl,os=os,**(extra or {}))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    if any(k not in scope for k in set(names)|set(functions)|set(classes)):raise ValueError('missing frozen definition')
    return SimpleNamespace(**scope)

def stage_def(bundle,donor,threads):
    import lightgbm as lgb
    s=definitions(bundle/'sources'/donor/STAGE[donor],{'Y','AIRPORTS','FEATS','SF','FEATS25','FEATS30','GS','PARAMS','PARAMS30','cols'},{'build','fit_stage','correct'},extra={'lgb':lgb})
    s.PARAMS['num_threads']=threads;s.PARAMS30['num_threads']=threads
    return s

def pack(root,out,commit):
    check_registration(root,commit)
    if out.exists():raise FileExistsError(out)
    if shutil.disk_usage(root).free<3*1024**3:raise ValueError('need3GiB free headroom')
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent));(stage/'sources').mkdir();(stage/'scripts').mkdir()
    shutil.copy2(root/SELF,stage/'scripts'/Path(SELF).name);inputs={};source_blobs={}
    def track(p):inputs[str(p.relative_to(root))]=sha(p);return p
    for donor,revision in SNAPSHOTS.items():
        dest=stage/'sources'/donor;dest.mkdir()
        support=('submit_per_airport.py','run_final_e38.sh') if donor=='v44' else ('seed_avg.py','catboost_reg.py','blend.py','run_final_e52.sh')
        for name in ('train.py','features.py','target_encoding.py',STAGE[donor],'e14_flight_history.py',*support):
            raw=subprocess.check_output(['git','show',f'{revision}:src/{name}'],cwd=root);(dest/name).write_bytes(raw);source_blobs[f'{donor}/{name}']=hashlib.sha256(raw).hexdigest()
        source=track(root/'data'/DATA[donor]);os.link(source,stage/DATA[donor]) # read-only hardlink saves Mac disk; SHA guards every run.
    a=pl.read_parquet(stage/DATA['v44'],columns=[ID,'split','month',Y]);b=pl.read_parquet(stage/DATA['v60'],columns=[ID,'split','month',Y])
    if not a.equals(b):raise ValueError('donor feature ID/month/target populations differ')
    checked_ids(a,2429888);train=a.filter(pl.col('split')=='train');checked_ids(train,2085047)
    # A common pinned historical nomonth parent is the original E36/E42 training recipe.
    parents=[]
    for fold in range(1,7):
        p=track(root/f'data/cv_preds_nomonth/fold_{fold}_{fold+6}.npz');z=np.load(p)
        if z['proba'].shape!=(len(z['mvt_id']),6) or not np.isfinite(z['proba']).all() or not np.allclose(z['proba'].sum(1),1,atol=1e-9):raise ValueError('nomonth probabilities incomplete')
        parents.append(pl.DataFrame({ID:z['mvt_id'],'base':z['pred'],'p_nonreg':1-z['proba'][:,0],'_ycheck':z['y']}))
    parent=pl.concat(parents);checked_ids(parent,2085047)
    q=train.join(parent,on=ID,validate='1:1')
    if not np.array_equal(q[Y].to_numpy(),q['_ycheck'].to_numpy()):raise ValueError('nomonth targets differ')
    s44=stage_def(stage,'v44',2);s60=stage_def(stage,'v60',2)
    if s44.FEATS!=s60.FEATS or s44.PARAMS30!=s60.PARAMS30:raise ValueError('historical stage recipes unexpectedly differ')
    files=sorted((root/'data/prc-2026-datasets').glob('training_*.parquet'))
    if len(files)!=12:raise ValueError('twelve raw months required')
    for p in files:track(p)
    raw=pl.concat([pl.read_parquet(p,columns=s44.cols) for p in files],how='vertical_relaxed').filter(pl.col('PHASE_mvt')=='DEP')
    sensors=[]
    for prefix in ('offblock2','offblock3','offblock5'):
        files=sorted((root/'data/adsblol').glob(prefix+'_2025-*.parquet'))
        if len(files)!=364:raise ValueError('expected364 original3kt sensor days')
        for p in files:track(p)
        x=pl.concat([pl.read_parquet(p) for p in files],how='diagonal_relaxed');checked_ids(x);sensors.append(x)
    adsb=sensors[0].join(sensors[1],on=ID,how='left',validate='1:1').join(sensors[2],on=ID,how='left',validate='1:1')
    d=s44.build(raw,parent.drop('_ycheck'),adsb).with_columns(month=pl.col('MVT_TIME_UTC_mvt').dt.month())
    cols=list(dict.fromkeys([ID,Y,'ADEP_mvt','month',*s44.FEATS]));d=d.select(cols);checked_ids(d,2085047)
    # Enforce actual month correspondence before preserving the source recipe.
    mm=train.select(ID,pl.col('month').alias('_month')).join(d.select(ID,'month'),on=ID,validate='1:1')
    if not np.array_equal(mm['month'].to_numpy(),mm['_month'].to_numpy()):raise ValueError('raw/feature calendar month differs')
    d.write_parquet(stage/'stage_inputs.parquet',compression='zstd');del raw,adsb,sensors,d;gc.collect()
    historycols=[ID,'split','month','ADEP_mvt','FLIGHT_ID_mvt','FLIGHT_mvt','d_mvt_sched','local_hour','day_nonm_rate',Y]
    f=pl.read_parquet(stage/DATA['v44'],columns=historycols);f2=pl.read_parquet(stage/DATA['v60'],columns=historycols)
    if not f.equals(f2):raise ValueError('E14 input recipes differ')
    f=f.filter((pl.col('split')=='train')&(pl.col('ADEP_mvt')=='LIRF')&pl.col('FLIGHT_ID_mvt').is_null()).with_columns(sched=((pl.col(Y)-pl.col('d_mvt_sched')).abs()<=6).cast(pl.Int8),delayed=pl.col('d_mvt_sched')>1800)
    f.write_parquet(stage/'e14_inputs.parquet',compression='zstd')
    rpath=track(root/'data/e195_pre_e81_20261007_v2/result.json');r=read(rpath)
    if sha(rpath)!=PARENT_RESULT:raise ValueError('shared126 exclusion provenance drift')
    p=track(root/'data/e195_pre_e81_20261007_v2/predictions.parquet')
    if sha(p)!=r['outputs']['predictions.parquet']:raise ValueError('E195 prediction provenance drift')
    mask=pl.read_parquet(p,columns=[ID,'evaluation_mask','excluded_tabpfn','excluded_feasibility'])
    if mask['excluded_tabpfn'].sum()!=126 or mask['excluded_feasibility'].sum()!=0:raise ValueError('shared exclusions changed')
    train.join(mask,on=ID,validate='1:1').write_parquet(stage/'metadata.parquet',compression='zstd')
    save(stage/'manifest.json',dict(protocol=PROTOCOL,registration_commit=commit,source_sha256=sha(root/SELF),source_snapshots=SNAPSHOTS,source_blobs=source_blobs,inputs=inputs,files={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()},rows=2085047,ranking_not_fitted=True,excluded_tabpfn=126,inner_policy='next pair within outer training',assignments=ASSIGN,limitation=LIMITATION))
    stage.rename(out);print(json.dumps(dict(packed=str(out),manifest_sha256=sha(out/'manifest.json'))),flush=True)

def verify(bundle,commit):
    m=read(bundle/'manifest.json')
    if m['protocol']!=PROTOCOL or m['registration_commit']!=commit or m['source_sha256']!=sha(__file__) or m['source_snapshots']!=SNAPSHOTS:raise ValueError('bundle lineage drift')
    for name,h in m['files'].items():
        if sha(bundle/name)!=h:raise ValueError('bundle artifact drift '+name)
    return m

def trainer(bundle,donor,threads):
    if 'torch' in sys.modules:raise RuntimeError('fresh LGB-only process required')
    if any(k in sys.modules for k in ('train','features','target_encoding')):raise RuntimeError('fresh donor process required')
    for k in ('REGIME_TOL','REGIME_EXCLUDE','AOBT_GATE','PROXY_SCALE_RIDGE','REGIME_RATES','PROXY_SCALE','DROP_PREFIX'):os.environ.pop(k,None)
    os.environ.update(DROP_PREFIX='day_,tq_,wxr_,month',REGIME_RATES='0',AOBT_GATE='1')
    if donor=='v60':os.environ['PROXY_SCALE']='1'
    sys.path.insert(0,str(bundle/'sources'/donor));import train as T
    T.REG_PARAMS['num_threads']=threads;T.CLF_PARAMS['num_threads']=threads
    original=T.add_encodings
    def checked_encoding(fit,pred,*args,**kwargs):
        fit_ids=checked_ids(fit);pred_ids=checked_ids(pred)
        a,b=original(fit,pred,*args,**kwargs)
        if not np.array_equal(a[ID].to_numpy(),fit_ids) or not np.array_equal(b[ID].to_numpy(),pred_ids):raise ValueError('historical encoding changed row order')
        return a,b
    T.add_encodings=checked_encoding
    return T

def inner_rounds(T,fit,fold,out):
    outer,inner=split_months(fold)
    if fit['month'].is_in(outer).any():raise ValueError('outer month in donor training')
    innerfit=fit.filter(~pl.col('month').is_in(inner));innerval=fit.filter(pl.col('month').is_in(inner))
    if not len(innerfit) or not len(innerval):raise ValueError('empty inner split')
    p,parts,best,models=T.fit_predict(innerfit,innerval,T.feature_columns(innerfit),None,val_y=innerval[Y].to_numpy())
    for kind,model in zip(('clf','reg'),models[:2]):model.save_model(str(out/f'inner_{kind}.txt'))
    rounds={k:max(1,int(v*1.1)) for k,v in best.items()}
    receipt=dict(inner_months=list(inner),outer_months=list(outer),inner_train_rows=len(innerfit),inner_valid_rows=len(innerval),inner_train_id_sha256=hashlib.sha256(checked_ids(innerfit).tobytes()).hexdigest(),inner_valid_id_sha256=hashlib.sha256(checked_ids(innerval).tobytes()).hexdigest(),best=best,rounds=rounds)
    save(out/'inner_selection.json',receipt);del p,parts,models;gc.collect();return rounds

def mix(T,pred,raw_proba,reg):
    values=np.column_stack([reg,*[pred[c].to_numpy() for c in T.CANDIDATES.values()],reg+T.DAY])
    proba=np.where(np.isnan(values),0.,raw_proba);proba=T.mask_infeasible(proba,values,pred);proba=proba/proba.sum(1,keepdims=True)
    point=np.clip(T.lobt_clip(np.sum(proba*np.nan_to_num(values),axis=1),pred),0,None)
    if not np.isfinite(point).all() or not np.isfinite(proba).all():raise ValueError('invalid donor mixture')
    return dict(reg=reg,proba=proba,values=values,pred=point)

def base44(T,fit,pred,fold,out):
    rounds=inner_rounds(T,fit,fold,out)
    # Null outer labels before all encoding/inference calls.
    unlabeled=pred.with_columns(pl.lit(None,dtype=pred.schema[Y]).alias(Y))
    point,parts,_,models=T.fit_predict(fit,unlabeled,T.feature_columns(fit),rounds)
    for kind,model in zip(('clf','reg'),models[:2]):model.save_model(str(out/f'outer_{kind}.txt'))
    save(out/'model_inventory.json',dict(inner_models=2,outer_models=2,rounds=rounds,columns=models[2]))
    return dict(reg=parts['reg'],proba=parts['proba'],values=parts['values'],pred=point)

def base60(T,fit,pred,fold,out,threads):
    import lightgbm as lgb
    rounds=inner_rounds(T,fit,fold,out);lab=T.regime_label(fit)
    unlabeled=pred.with_columns(pl.lit(None,dtype=pred.schema[Y]).alias(Y))
    fit,pred=T.add_encodings(fit,unlabeled,lab==0,fit_labels=lab)
    cols=T.feature_columns(fit);catcols=[c for c in T.CATEGORICAL if c in cols]
    xf,cats=T.to_pandas(fit,cols,None);xp,_=T.to_pandas(pred,cols,cats)
    clf=lgb.train(T.CLF_PARAMS,lgb.Dataset(xf,lab,categorical_feature=catcols),num_boost_round=rounds['clf'])
    raw=clf.predict(xp,num_threads=threads);clf.save_model(str(out/'outer_clf.txt'));del clf,xf,xp;gc.collect()
    fit_r=fit.filter(pl.Series(lab==0));base_f,base_p=T.proxy_base(fit_r,pred)
    target=np.clip(np.clip(fit_r[Y].to_numpy(),*T.REG_CLIP)-base_f,-7200,7200)
    # Original seed_avg.py learns categories on REGULAR rows, independently
    # from the classifier's all-row dictionary. Preserve that distinction.
    xr,reg_cats=T.to_pandas(fit_r,cols,None);xpr,_=T.to_pandas(pred,cols,reg_cats)
    ds=lgb.Dataset(xr,target,categorical_feature=catcols,free_raw_data=False);regs=[]
    for k,seed in enumerate(SEEDS):
        model=lgb.train({**T.REG_PARAMS,'seed':seed,'bagging_seed':2000+k,'feature_fraction_seed':3000+k},ds,num_boost_round=rounds['reg'])
        reg=model.predict(xpr,num_threads=threads)+base_p;regs.append(reg);np.save(out/f'lgb_seed{seed}.npy',reg,allow_pickle=False);model.save_model(str(out/f'lgb_seed{seed}.txt'));del model
        print(json.dumps(dict(event='lgb_seed_complete',fold=fold,seed=seed,rounds=rounds['reg'])),flush=True)
    del ds,xr,xpr;gc.collect()
    from catboost import CatBoostRegressor,Pool
    def pool(frame,label=None):
        pdf=frame.select(cols).to_pandas()
        for c in catcols:pdf[c]=pdf[c].fillna('NA').astype(str)
        return Pool(pdf,label,cat_features=catcols)
    cat=CatBoostRegressor(iterations=5600,learning_rate=.08,depth=8,loss_function='RMSE',thread_count=threads,verbose=500,random_seed=0,allow_writing_files=False)
    cat.fit(pool(fit_r,target));cr=cat.predict(pool(pred));cr=cr+base_p;np.save(out/'cat_seed0.npy',cr,allow_pickle=False);cat.save_model(str(out/'cat_seed0.cbm'))
    reg=.65*cr+.35*np.mean(regs,axis=0)
    save(out/'model_inventory.json',dict(inner_models=2,outer_clf=1,lgb_seeds=list(SEEDS),lgb_rounds=rounds['reg'],cat_seed=0,cat_iterations=5600,cat_weight=.65,rounds=rounds,columns=cols))
    return mix(T,pred,raw,reg)

def after_e14(bundle,donor,fold,ids,base,reg,out):
    from sklearn.linear_model import LogisticRegression
    h=definitions(bundle/'sources'/donor/'e14_flight_history.py',{'Y'},{'history','design'})
    f=pl.read_parquet(bundle/'e14_inputs.parquet');outer,_=split_months(fold);fit=f.filter(~pl.col('month').is_in(outer));val=f.filter(pl.col('month').is_in(outer))
    band=(pl.col('d_mvt_sched')>0)&(pl.col('d_mvt_sched')<14400)
    fit_h=pl.concat([h.history(fit.filter(pl.col('month')!=month),fit.filter(pl.col('month')==month)) for month in sorted(fit['month'].unique().to_list())]).filter(band)
    model=LogisticRegression(C=1.,max_iter=2000).fit(h.design(fit_h),fit_h['sched'].to_numpy())
    val=h.history(fit,val).filter(band&(pl.col('n')>=3)).join(pl.DataFrame({ID:ids,'reg':reg}),on=ID,how='inner',validate='1:1')
    prob=model.predict_proba(h.design(val))[:,1] if len(val) else np.array([])
    update=rounded(prob*val['d_mvt_sched'].to_numpy()+(1-prob)*val['reg'].to_numpy())
    lookup={v:i for i,v in enumerate(ids)};result=base.copy()
    for mid,value in zip(val[ID],update):result[lookup[mid]]=value
    save(out/'e14_model.json',dict(classes=model.classes_.tolist(),coef=model.coef_.tolist(),intercept=model.intercept_.tolist(),train_rows=len(fit_h),applied_rows=len(val),heldout_months=list(outer)))
    return result

def donor_stage(bundle,donor,fold,base,threads,out):
    s=stage_def(bundle,donor,threads);outer,_=split_months(fold)
    d=pl.read_parquet(bundle/'stage_inputs.parquet');fit=d.filter(~pl.col('month').is_in(outer));pred=d.filter(pl.col('month').is_in(outer))
    pred=pred.drop(['base','p_nonreg']).join(base.select(ID,'base','p_nonreg'),on=ID,how='inner',validate='1:1').with_columns(base_vs_first_move=pl.col('base')-pl.col('mvt_first_move'),base_vs_first=pl.col('base')-pl.col('mvt_first'))
    if len(pred)!=len(base):raise ValueError('stage donor ID alignment failed')
    ans=pred['base'].to_numpy().copy();models=[]
    groups=AP if donor=='v44' else ('ALL',)
    for ap in groups:
        tr=fit if ap=='ALL' else fit.filter(pl.col('ADEP_mvt')==ap)
        at=np.ones(len(pred),bool) if ap=='ALL' else (pred['ADEP_mvt']==ap).to_numpy()
        if donor=='v44' and tr.filter(pl.col('coverage')>=2).height<=2000:
            models.append(dict(airport=ap,fitted=False,reason='original <=2000 coverage rule'));continue
        model=s.fit_stage(tr);path=out/f'stage_{ap}.txt';model.save_model(str(path));ans[at]=s.correct(model,pred.filter(pl.Series(at)))
        models.append(dict(airport=ap,fitted=True,rounds=800,sha256=sha(path)));del model
    return pred[ID].to_numpy(),rounded(np.clip(ans,0,None)),models

def run_fold(bundle,out,commit,donor,fold,threads):
    if out.exists():raise FileExistsError(out)
    m=verify(bundle,commit);T=trainer(bundle,donor,threads);outer,inner=split_months(fold);start=time.monotonic()
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent));records=[];blocks=[]
    for ap in (AP if donor=='v44' else ('ALL',)):
        ad=stage/ap;ad.mkdir()
        query=pl.scan_parquet(bundle/DATA[donor]).filter(pl.col('split')=='train')
        if ap!='ALL':query=query.filter(pl.col('ADEP_mvt')==ap)
        d=query.collect();checked_ids(d)
        fit=d.filter(~pl.col('month').is_in(outer));pred=d.filter(pl.col('month').is_in(outer));del d
        if not len(pred) or not len(fit):raise ValueError('missing airport/outer population')
        ids=checked_ids(pred);begin=time.monotonic()
        part=base44(T,fit,pred,fold,ad) if donor=='v44' else base60(T,fit,pred,fold,ad,threads)
        if part['proba'].shape!=(len(ids),6) or not np.isfinite(part['proba']).all() or not np.allclose(part['proba'].sum(1),1,atol=1e-9):raise ValueError('invalid six-regime probabilities')
        np.savez_compressed(ad/'base_parts.npz',mvt_id=ids,y=pred[Y].to_numpy(),month=pred['month'].to_numpy(),**part)
        blocks.append(pl.DataFrame({ID:ids,'base':rounded(part['pred']),'reg':part['reg'],'p_nonreg':1-part['proba'][:,0]}))
        record=dict(airport=ap,rows=len(ids),train_rows=len(fit),seconds=time.monotonic()-begin,inner_selection=read(ad/'inner_selection.json'),parts_sha256=sha(ad/'base_parts.npz'));records.append(record);print(json.dumps(dict(event='base_complete',donor=donor,fold=fold,**record)),flush=True)
        del fit,pred,part;gc.collect()
    base=pl.concat(blocks);checked_ids(base);base=base.with_columns(pl.Series('base',after_e14(bundle,donor,fold,base[ID].to_numpy(),base['base'].to_numpy(),base['reg'].to_numpy(),stage)))
    ids,point,stage_models=donor_stage(bundle,donor,fold,base,threads,stage)
    meta=pl.read_parquet(bundle/'metadata.parquet').filter(pl.col('month').is_in(outer))
    pred=meta.join(pl.DataFrame({ID:ids,'donor':point}),on=ID,how='left',validate='1:1')
    if pred['donor'].null_count() or len(pred)!=len(ids):raise ValueError('donor completeness failed')
    pred.write_parquet(stage/'prediction.parquet',compression='zstd')
    base.write_parquet(stage/'pre_stage.parquet',compression='zstd')
    import resource,platform
    runtime={k:version(k) for k in ('numpy','polars','pandas','lightgbm','scikit-learn')}
    if donor=='v60':runtime['catboost']=version('catboost')
    save(stage/'complete.json',dict(protocol=PROTOCOL,registration_commit=commit,input_manifest_sha256=sha(bundle/'manifest.json'),donor=donor,fold=fold,outer_months=list(outer),inner_months=list(inner),rows=len(pred),threads=threads,records=records,stage_models=stage_models,runtime=runtime,python=platform.python_version(),maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,seconds=time.monotonic()-start,files={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()},accepted_for_submission=False,limitation=LIMITATION))
    stage.rename(out);print(json.dumps(dict(event='donor_fold_complete',donor=donor,fold=fold,rows=len(pred),seconds=time.monotonic()-start,receipt_sha256=sha(out/'complete.json'))),flush=True)

def audit(bundle,part,commit):
    """Read-only verification on the original node; publish small transfer proof.
    Models/per-seed arrays remain on that node, avoiding multi-GB Mac copies.
    """
    verify(bundle,commit);r=read(part/'complete.json')
    if r['protocol']!=PROTOCOL or r['registration_commit']!=commit or r['input_manifest_sha256']!=sha(bundle/'manifest.json'):raise ValueError('result lineage differs')
    donor=r['donor'];fold=r['fold'];outer,inner=split_months(fold)
    if r['outer_months']!=list(outer) or r['inner_months']!=list(inner):raise ValueError('result split differs')
    for name,h in r['files'].items():
        if sha(part/name)!=h:raise ValueError('original-node artifact drift '+name)
    groups=AP if donor=='v44' else ('ALL',)
    if [x['airport'] for x in r['records']]!=list(groups):raise ValueError('base model population incomplete')
    for ap in groups:
        directory=part/ap;selection=read(directory/'inner_selection.json');inventory=read(directory/'model_inventory.json')
        if selection['outer_months']!=list(outer) or selection['inner_months']!=list(inner) or selection['rounds']!={k:max(1,int(v*1.1)) for k,v in selection['best'].items()}:raise ValueError('inner selection differs')
        if inventory['rounds']!=selection['rounds']:raise ValueError('outer round receipt differs')
        with np.load(directory/'base_parts.npz') as z:
            if z['proba'].shape!=(len(z['mvt_id']),6) or not np.isfinite(z['pred']).all() or not np.isfinite(z['reg']).all() or not np.isfinite(z['proba']).all() or not np.allclose(z['proba'].sum(1),1,atol=1e-9):raise ValueError('invalid prediction/probability')
            if not np.isin(z['month'],outer).all() or len(np.unique(z['mvt_id']))!=len(z['mvt_id']):raise ValueError('base heldout identity differs')
            if not np.array_equal(z['values'][:,0],z['reg']) or not np.array_equal(z['values'][:,-1],z['reg']+86400):raise ValueError('mixture ordinary/day candidates differ')
            if donor=='v60':
                if inventory['lgb_seeds']!=list(SEEDS) or inventory['cat_seed']!=0 or inventory['cat_iterations']!=5600 or inventory['cat_weight']!=.65:raise ValueError('production blend recipe differs')
                reg=.65*np.load(directory/'cat_seed0.npy')+.35*np.mean([np.load(directory/f'lgb_seed{s}.npy') for s in SEEDS],axis=0)
                if not np.array_equal(reg,z['reg']):raise ValueError('five-seed/Cat mixture arithmetic differs')
    # All heavy artifacts verified above. Every small artifact is copied intact.
    small={name:h for name,h in r['files'].items() if Path(name).suffix in ('.json','.parquet','.npz')}
    import platform
    proof=dict(protocol=PROTOCOL,registration_commit=commit,input_manifest_sha256=sha(bundle/'manifest.json'),source_sha256=sha(__file__),complete_sha256=sha(part/'complete.json'),all_files=r['files'],small_files=small,original_host=platform.node(),original_path=str(part.resolve()),verification='all source-node SHA plus exact donor recipe/rounds/P/seed arithmetic',models_retained_on_original_node=True)
    save(part/'transfer_receipt.json',proof);print(json.dumps(dict(audited=str(part),files=len(r['files']),small_files=len(small),receipt_sha256=sha(part/'transfer_receipt.json'))),flush=True)

def collect(bundle,parts,out,commit):
    verify(bundle,commit)
    if out.exists():raise FileExistsError(out)
    frames=[];receipts=[];seen=[]
    for part in parts:
        r=read(part/'complete.json')
        if r['protocol']!=PROTOCOL or r['registration_commit']!=commit or r['input_manifest_sha256']!=sha(bundle/'manifest.json'):raise ValueError('result lineage differs')
        key=(r['donor'],r['fold']);seen.append(key)
        if r['outer_months']!=list(split_months(r['fold'])[0]) or r['inner_months']!=list(split_months(r['fold'])[1]):raise ValueError('fold split differs')
        if len(r['records'])!=(10 if r['donor']=='v44' else 1):raise ValueError('base inventory differs')
        proof=read(part/'transfer_receipt.json')
        if proof['protocol']!=PROTOCOL or proof['registration_commit']!=commit or proof['input_manifest_sha256']!=sha(bundle/'manifest.json') or proof['source_sha256']!=sha(__file__) or proof['complete_sha256']!=sha(part/'complete.json') or proof['all_files']!=r['files']:raise ValueError('original-node verification proof differs')
        expected={name:h for name,h in r['files'].items() if Path(name).suffix in ('.json','.parquet','.npz')}
        if proof['small_files']!=expected:raise ValueError('small transfer inventory differs')
        for name,h in expected.items():
            if sha(part/name)!=h:raise ValueError('transferred artifact SHA differs')
        d=pl.read_parquet(part/'prediction.parquet');checked_ids(d);frames.append(d.with_columns(pl.lit(r['donor']).alias('donor_name')));receipts.append(dict(donor=r['donor'],fold=r['fold'],sha256=sha(part/'complete.json')))
    if sorted(seen)!=[(donor,fold) for donor in ('v44','v60') for fold in range(1,7)]:raise ValueError('all12 donor folds required')
    meta=pl.read_parquet(bundle/'metadata.parquet');output=meta
    for donor in ('v44','v60'):
        d=pl.concat([f for f in frames if f['donor_name'][0]==donor]);checked_ids(d,2085047)
        check=meta.join(d.select(ID,pl.col(Y).alias('_y'),pl.col('month').alias('_month')),on=ID,validate='1:1')
        if not np.array_equal(check[Y],check['_y']) or not np.array_equal(check['month'],check['_month']):raise ValueError('aggregate metadata mismatch')
        output=output.join(d.select(ID,pl.col('donor').alias(donor)),on=ID,how='left',validate='1:1')
    if output.select(pl.col('v44','v60').null_count()).row(0)!=(0,0):raise ValueError('missing donor predictions')
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent));output.write_parquet(stage/'donors.parquet',compression='zstd')
    save(stage/'complete.json',dict(protocol=PROTOCOL,registration_commit=commit,input_manifest_sha256=sha(bundle/'manifest.json'),rows=len(output),excluded=int((~output['evaluation_mask']).sum()),receipts=receipts,files={'donors.parquet':sha(stage/'donors.parquet')},accepted_for_submission=False,limitation=LIMITATION));stage.rename(out)

def selftest():
    for fold in range(1,7):
        outer,inner=split_months(fold);assert len(set(outer)|set(inner))==4 and len(set(range(1,13))-set(outer)-set(inner))==8
    assert set(f for _,fs,_ in ASSIGN.values() for f in fs)==set(range(1,7))
    for donor in ('v44','v60'):assert sorted(f for dn,fs,_ in ASSIGN.values() if dn==donor for f in fs)==list(range(1,7))
    print('PASS outer/inner disjoint, complete nonoverlapping donor assignments')

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['selftest','pack','verify','run-fold','audit','collect']);p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--input',type=Path);p.add_argument('--output',type=Path);p.add_argument('--registration-commit');p.add_argument('--donor',choices=['v44','v60']);p.add_argument('--fold',type=int,choices=range(1,7));p.add_argument('--threads',type=int,choices=[6,8,12],default=6);p.add_argument('--parts',nargs='+',type=Path);a=p.parse_args()
    if a.action=='selftest':return selftest()
    if a.action=='pack':return pack(a.root.resolve(),a.output.resolve(),a.registration_commit)
    if a.action=='verify':return print(json.dumps(verify(a.input.resolve(),a.registration_commit)),flush=True)
    if a.action=='run-fold':return run_fold(a.input.resolve(),a.output.resolve(),a.registration_commit,a.donor,a.fold,a.threads)
    if a.action=='audit':return audit(a.input.resolve(),a.output.resolve(),a.registration_commit)
    return collect(a.input.resolve(),a.parts,a.output.resolve(),a.registration_commit)

if __name__=='__main__':main()

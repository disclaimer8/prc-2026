"""Fixed-recipe final-period tabular fits, persisted transforms and inference.

No CV, round selection, upload, network transport, implicit resume, or environment
installation. Real execution requires a root-created Git admission receipt and
the completed, hashed final feature contract. Synthetic tests use pure helpers.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

os.environ.setdefault('POLARS_MAX_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OMP_THREAD_LIMIT', '1')
os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR', 'load_as_storage')
import numpy as np
import polars as pl

ID='MVT_ID_mvt'; Y='TAXITIME_SEC_mvt'
PROTOCOL='final-four-month-tabular-fixed-recipe-save-models-v1'
SELF='scripts/final_tabular_models_20261008.py'
TEST='tests/test_final_tabular_models_20261008.py'
DOC='analysis/final_tabular_models_20261008/PROTOCOL.md'
HELPERS=('src/train.py','src/features.py','src/target_encoding.py','src/e185_motion_view.py')
DROP='day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config'
ROUNDS={'cat':5600,'lgb':2319,'raw':2565,'lobt':3065,'sched':2958,'eobt':2736,'aobt':2738,'motion':2481,'base':2319}
# Unit memory ceiling accepted by resource_guard, GiB. Cat was drafted at 12 GiB; the
# production prepare() path measured a 19.46 GiB transient cgroup peak (prc-cloud,
# 08.10.2026 12:42Z, full 2755837-row frame), so the ceiling is 24 GiB. LGB stays 32.
MEMORY_LIMIT_GIB={'cat':24,'lgb':32}
ENV={'PROXY_SCALE':'1','PROXY_SCALE_RIDGE':'1000','CLF_COST_WEIGHT':'1',
     'CLF_COST_MIN_SPREAD':'3600','REG_CLIP_HI':'7200','REGIME_TOL':'6',
     'REGIME_EXCLUDE':'EDDF,EHAM,LSZH','AOBT_GATE':'1','CLF_SEEDS':'1',
     'DROP_PREFIX':DROP}
OPTIONAL=('REG_OVERRIDE','SCHED_TOL_LONG','NO_AOBT','PROXY_ISO','PROXY_AFFINE',
          'ADV_WEIGHT','CLF_ONLY_PREFIX','SERVED_ONLY','REG_COST_WEIGHT',
          'CLF_MIX_OBJ','REGIME_RATES','X_FEATS','MIX_REG_CROSSFIT')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)


def read(path):return json.loads(Path(path).read_text())


def runtime():
    return {'python':sys.version,'platform':sys.platform,
            'packages':{k:importlib.metadata.version(k) for k in ('numpy','polars','pandas','pyarrow','lightgbm','catboost','scikit-learn')},
            'environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','POLARS_MAX_THREADS','OPENBLAS_NUM_THREADS')}}


def resource_guard(root,kind,threads):
    if shutil.disk_usage(root).free<4*1024**3:raise ValueError('4GiB disk floor')
    if sys.platform!='linux':raise ValueError('production fits require bounded Linux unit; Mac fit launcher not qualified')
    cg=next(v.split(':',2)[2].strip() for v in Path('/proc/self/cgroup').read_text().splitlines() if v.startswith('0:'))
    path=Path('/sys/fs/cgroup')/cg.lstrip('/')
    memory=(path/'memory.max').read_text().strip();swap=(path/'memory.swap.max').read_text().strip()
    quota,period=(path/'cpu.max').read_text().split();limit=memory_limit(kind)
    if memory=='max' or int(memory)>limit or swap!='0' or quota=='max' or int(quota)>threads*int(period):
        raise ValueError('unbounded/excessive CPU/RAM/swap resource unit')
    if os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError('nice10 required')
    return {'cgroup':cg,'memory_max':int(memory),'cpu_max':[int(quota),int(period)],'swap_max':0}


def memory_limit(kind):
    return MEMORY_LIMIT_GIB['cat' if kind=='cat' else 'lgb']*1024**3


def ids_sha(ids):
    return hashlib.sha256(np.asarray(ids,dtype='<f8').tobytes()).hexdigest()


def expected_seeds(kind):
    if kind not in ROUNDS:raise ValueError('unknown family')
    return list(range(15 if kind=='cat' else 1 if kind=='base' else 5))


def check_seed_set(kind,seeds):
    if sorted(seeds)!=expected_seeds(kind):raise ValueError('incomplete/duplicate seed set')


def validate_rows(df,ntrain,nrank):
    if df.height!=ntrain+nrank or df[ID].n_unique()!=df.height or df[ID].null_count():
        raise ValueError('row population or duplicate/missing IDs')
    if not np.isfinite(df[ID].to_numpy()).all():raise ValueError('nonfinite IDs')
    fit=df.filter(pl.col('split')=='train');pred=df.filter(pl.col('split')=='rank')
    if fit.height!=ntrain or pred.height!=nrank:raise ValueError('wrong split counts')
    if not np.isfinite(fit[Y].cast(pl.Float64).to_numpy()).all():raise ValueError('missing/nonfinite training labels')
    if pred[Y].null_count()!=nrank:raise ValueError('final targets must all be null')
    if 'BLOCK_TIME_UTC_mvt' in pred and pred['BLOCK_TIME_UTC_mvt'].null_count()!=nrank:
        raise ValueError('final departure block targets present')
    return fit,pred


def validate_predictions(values,n):
    a=np.asarray(values)
    if len(a)!=n or not np.isfinite(a).all():raise ValueError('prediction shape/nonfinite')


def configure(root,features,threads):
    if not 1<=threads<=24:raise ValueError('threads outside resource bound')
    for k in OPTIONAL:
        if os.environ.get(k) not in (None,'','0'):raise ValueError('unregistered environment option '+k)
        # Several original callers test truthiness, not equality to "1":
        # REG_OVERRIDE="0" is parsed as an int and ADV_WEIGHT="0" is a filename.
        os.environ.pop(k,None)
    os.environ.update(ENV)
    os.environ.update(FEATURES=str(features),OMP_NUM_THREADS=str(threads),OMP_THREAD_LIMIT=str(threads),
                      OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',CATBOOST_THREADS=str(threads))
    src=str(Path(root)/'src')
    if src not in sys.path:sys.path.insert(0,src)


def proxy_chain(frame):
    def ok(col):
        condition=pl.col(col).is_between(0,7200)
        if col=='d_mvt_aobt':
            condition=condition & ((pl.col(col)-pl.col('d_mvt_lobt')).abs()<=3606).fill_null(True)
        return pl.when(condition).then(pl.col(col))
    return frame.select('ADEP_mvt',raw=pl.coalesce(ok('d_mvt_aobt'),ok('d_mvt_lobt'),ok('d_mvt_eobt')),
        src=pl.when(ok('d_mvt_aobt').is_not_null()).then(pl.lit('aobt'))
        .when(ok('d_mvt_lobt').is_not_null()).then(pl.lit('lobt'))
        .when(ok('d_mvt_eobt').is_not_null()).then(pl.lit('eobt')).otherwise(pl.lit('none')))


def fit_proxy(ordinary):
    """Only the selected E46 scale recipe; serialized state never contains labels."""
    cf=proxy_chain(ordinary).with_columns(y=ordinary[Y].clip(0,7200))
    med=cf.group_by('ADEP_mvt').agg(med=pl.col('y').median())
    cs=cf.filter(pl.col('raw').is_not_null() & (pl.col('raw')>0))
    g=cs.select(sxy=(pl.col('raw')*pl.col('y')).sum(),sxx=(pl.col('raw')**2).sum())
    global_scale=float(g['sxy'][0]/max(float(g['sxx'][0]),1e-9))
    scale=cs.group_by('ADEP_mvt','src').agg(sxy=(pl.col('raw')*pl.col('y')).sum(),sxx=(pl.col('raw')**2).sum()).with_columns(
        a=(pl.col('sxy')+1000*global_scale*1e6)/(pl.col('sxx')+1000*1e6)).select('ADEP_mvt','src','a')
    return {'version':'E46-scale-ridge1000-window3606','median':med.sort('ADEP_mvt').to_dicts(),
            'scale':scale.sort(['ADEP_mvt','src']).to_dicts(),'global_scale':global_scale}


def apply_proxy(frame,state):
    if state['version']!='E46-scale-ridge1000-window3606':raise ValueError('proxy recipe')
    med=pl.DataFrame(state['median'],schema={'ADEP_mvt':pl.String,'med':pl.Float64})
    scale=pl.DataFrame(state['scale'],schema={'ADEP_mvt':pl.String,'src':pl.String,'a':pl.Float64})
    c=(proxy_chain(frame).with_row_index('_i').join(med,on='ADEP_mvt',how='left')
       .join(scale,on=['ADEP_mvt','src'],how='left').sort('_i'))
    # Original T.proxy_base uses scale 1.0 for an unseen source, not global_scale.
    return c.select(pl.coalesce(pl.col('raw')*pl.col('a').fill_null(1.0),pl.col('med'))).to_series().to_numpy()


def categories(frame,cat_cols):
    return {c:sorted(frame[c].drop_nulls().unique().to_list()) for c in cat_cols}


def pandas_matrix(frame,columns,cats):
    if set(columns)-set(frame.columns):raise ValueError('missing model columns')
    pdf=frame.select(columns).to_pandas()
    for c,levels in cats.items():
        if c not in columns:raise ValueError('category outside columns')
        pdf[c]=pdf[c].astype('category').cat.set_categories(levels)
    return pdf


def cat_pool(frame,columns,cat_cols,y=None):
    from catboost import Pool
    pdf=frame.select(columns).to_pandas()
    for c in cat_cols:pdf[c]=pdf[c].fillna('NA').astype(str)
    return Pool(pdf,y,cat_features=cat_cols)


def anchor(frame,kind):
    if kind=='raw':return np.zeros(frame.height)
    if kind not in ('lobt','sched','eobt','aobt'):raise ValueError('invalid anchor')
    a=frame['d_mvt_'+kind].to_numpy().astype(float)
    return np.where(np.isnan(a),0.,np.clip(np.nan_to_num(a),0,7200))


def predict_saved(directory,frame,state):
    directory=Path(directory)
    if state['engine']=='catboost':
        from catboost import CatBoostRegressor
        model=CatBoostRegressor();model.load_model(str(directory/'model.cbm'))
        out=model.predict(cat_pool(frame,state['columns'],state['categorical_columns']))
    else:
        import lightgbm as lgb
        model=lgb.Booster(model_file=str(directory/'model.txt'))
        if state['kind']=='motion':
            a=np.empty((frame.height,len(state['columns'])),dtype=np.float64,order='F')
            for j,c in enumerate(state['columns']):
                s=frame[c]
                if c in state['categories']:
                    levels=state['categories'][c]
                    s=s.replace_strict(levels,list(range(len(levels))),default=None,return_dtype=pl.Float64)
                a[:,j]=s.cast(pl.Float64).to_numpy()
            out=model.predict(a,num_threads=int(os.environ.get('OMP_NUM_THREADS','1')))
        else:out=model.predict(pandas_matrix(frame,state['columns'],state['categories']),num_threads=int(os.environ.get('OMP_NUM_THREADS','1')))
    if state.get('proxy') is not None:out=out+apply_proxy(frame,state['proxy'])
    elif state['kind'] in ('raw','lobt','sched','eobt','aobt'):out=out+anchor(frame,state['kind'])
    validate_predictions(out,frame.height)
    return out


def predict_base(directory,frame,state,T):
    import lightgbm as lgb
    r=predict_saved(directory,frame,state)
    clf=lgb.Booster(model_file=str(Path(directory)/'classifier.txt'))
    proba=clf.predict(pandas_matrix(frame,state['columns'],state['categories']),num_threads=T.CLF_PARAMS['num_threads'])
    values=np.column_stack([r,*[frame[c].to_numpy() for c in T.CANDIDATES.values()],r+T.DAY])
    proba=np.where(np.isnan(values),0.,proba)
    proba=T.mask_infeasible(proba,values,frame);proba=proba/proba.sum(axis=1,keepdims=True)
    pred=np.clip(T.lobt_clip(np.sum(proba*np.nan_to_num(values),axis=1),frame),0,None)
    return {'reg':r,'p':1-proba[:,0],'proba':proba,'values':values,'pred':pred}


def relative_file(root,name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe contract path')
    full=(root/p).resolve()
    if not full.is_relative_to(root.resolve()):raise ValueError('contract path leaves workspace')
    return full


def attest(root,commit,contract_path,out):
    if not re.fullmatch('[a-f0-9]{40}',commit):raise ValueError('full registration commit required')
    rel=str(contract_path.resolve().relative_to(root.resolve()))
    pins={}
    for name in (SELF,TEST,DOC,rel):
        b=subprocess.check_output(['git','show',commit+':'+name],cwd=root)
        pins[name]=hashlib.sha256(b).hexdigest()
        if sha(root/name)!=pins[name]:raise ValueError('registered source changed '+name)
    if PROTOCOL.encode() not in subprocess.check_output(['git','show',commit+':HYPOTHESES_LOG.md'],cwd=root):
        raise ValueError('no registered journal protocol')
    save_json(out,{'protocol':PROTOCOL,'commit':commit,'files':pins,'contract':rel,'created_utc':time.time()})


def admit(root,receipt_path):
    receipt=read(receipt_path)
    if receipt['protocol']!=PROTOCOL or not re.fullmatch('[a-f0-9]{40}',receipt['commit']):raise ValueError('registration receipt')
    if set(receipt['files'])!={SELF,TEST,DOC,receipt['contract']}:raise ValueError('incomplete registration file set')
    for name,h in receipt['files'].items():
        if sha(relative_file(root,name))!=h:raise ValueError('registered file drift '+name)
    contract=read(relative_file(root,receipt['contract']))
    if contract['protocol']!=PROTOCOL or (contract['train_rows'],contract['rank_rows'])!=(2085047,670790):
        raise ValueError('wrong final data contract')
    required={*HELPERS,contract['features'],contract['motion'],contract['template'],contract['feature_receipt']}
    if not required<=set(contract['files']):raise ValueError('incomplete source/input bindings')
    for name,h in contract['files'].items():
        if sha(relative_file(root,name))!=h:raise ValueError('final input/source changed '+name)
    if contract.get('feature_build_complete') is not True:raise ValueError('feature build incomplete')
    build=read(relative_file(root,contract['feature_receipt']))
    for key,other in [('train_rows','train_rows'),('rank_rows','rank_rows'),
                      ('train_ids_sha256','train_ordered_id_sha256'),('rank_ids_sha256','rank_ordered_id_sha256')]:
        if contract[key]!=build[other]:raise ValueError('feature completion contract differs: '+key)
    if build['feature_sha256']!=contract['files'][contract['features']] or build['template_sha256']!=contract['files'][contract['template']]:
        raise ValueError('feature/template receipt drift')
    template=pl.read_parquet(relative_file(root,contract['template']))
    if template.height!=670790 or template[ID].n_unique()!=670790 or template[Y].null_count()!=670790:
        raise ValueError('final template invalid')
    return receipt,contract


def prepare(root,contract,kind,T):
    df=pl.read_parquet(relative_file(root,contract['features']))
    fit,pred=validate_rows(df,contract['train_rows'],contract['rank_rows'])
    for split,frame in [('train',fit),('rank',pred)]:
        if ids_sha(frame[ID].to_numpy())!=contract[split+'_ids_sha256']:raise ValueError('ordered '+split+' IDs changed')
    del df;gc.collect()  # memory hygiene only: the full frame is not used again (19.4 GiB transient peak measured without it)
    if kind=='motion':
        os.environ['E185_MOTION']=str(relative_file(root,contract['motion']))
        import e185_motion_view as M
        frame,cols,cat_cols,replaced=M.load(['train','rank'])
        cats=M.categories(frame,pl.col('split')=='train',cat_cols)
        p=frame.filter(pl.col('split')=='rank');mask=(pl.col('split')=='train')&(pl.col('_lab')==0)
        xf=M.matrix(frame,mask,cols,cats);xp=M.matrix(frame,pl.col('split')=='rank',cols,cats)
        yf=M.label(frame,mask);ordinary=frame.filter(mask)
        if not np.array_equal(p[ID].to_numpy(),pred[ID].to_numpy()):raise ValueError('motion join reordered final IDs')
        return ordinary,p,xf,xp,yf,dict(columns=cols,categories=cats,categorical_columns=cat_cols,proxy=None,replaced=replaced)
    lab=T.regime_label(fit)
    # Preserve the producer's encoding call and row order. All its fitted columns
    # are excluded in this tabular recipe; neural encodings have a separate recipe.
    fit,pred=T.add_encodings(fit,pred,lab==0,fit_labels=lab)
    if ids_sha(fit[ID].to_numpy())!=contract['train_ids_sha256'] or ids_sha(pred[ID].to_numpy())!=contract['rank_ids_sha256']:
        raise ValueError('encoding join reordered rows')
    cols=T.feature_columns(fit)
    if any(c.startswith(('te_','ter_','tq_','rr_')) for c in cols):raise ValueError('unserialized encoding used')
    cat_cols=[c for c in T.CATEGORICAL if c in cols]
    ordinary=fit.filter(pl.Series(lab==0))
    cats=categories(ordinary if kind=='lgb' else fit,cat_cols)
    del fit,lab;gc.collect()  # memory hygiene only: ordinary/pred carry everything the fit needs
    state=dict(columns=cols,categories=cats,categorical_columns=cat_cols,proxy=None)
    if kind in ('cat','lgb','base'):
        state['proxy']=fit_proxy(ordinary)
        base_f,base_p=T.proxy_base(ordinary,pred)
        np.testing.assert_allclose(apply_proxy(ordinary,state['proxy']),base_f,rtol=0,atol=1e-8)
        np.testing.assert_allclose(apply_proxy(pred,state['proxy']),base_p,rtol=0,atol=1e-8)
    else:base_f=anchor(ordinary,kind)
    yf=np.clip(np.clip(ordinary[Y].to_numpy(),*T.REG_CLIP)-base_f,-7200,7200)
    if kind=='cat':xf=cat_pool(ordinary,cols,cat_cols,yf);xp=cat_pool(pred,cols,cat_cols)
    else:xf=pandas_matrix(ordinary,cols,cats);xp=pandas_matrix(pred,cols,cats)
    return ordinary,pred,xf,xp,yf,state


def fit_family(root,receipt_path,kind,seeds,threads):
    resources=resource_guard(root,kind,threads)
    receipt,contract=admit(root,receipt_path)
    allowed=expected_seeds(kind)
    if not seeds or len(set(seeds))!=len(seeds) or any(s not in allowed for s in seeds):raise ValueError('invalid shard seeds')
    out=root/'models/final_tabular'/kind
    if any((out/f'seed_{s}').exists() for s in seeds):raise FileExistsError('existing or partial seed; no resume')
    configure(root,relative_file(root,contract['features']),threads)
    # CPU-only Linux environments need no Torch installation. Mac production
    # is refused above until its independent RSS/time supervisor is qualified.
    import lightgbm as lgb
    import train as T
    T.REG_PARAMS['num_threads']=threads;T.CLF_PARAMS['num_threads']=threads
    started=time.time()
    if kind=='base':
        df=pl.read_parquet(relative_file(root,contract['features']))
        fit,pred=validate_rows(df,contract['train_rows'],contract['rank_rows'])
        if ids_sha(fit[ID].to_numpy())!=contract['train_ids_sha256'] or ids_sha(pred[ID].to_numpy())!=contract['rank_ids_sha256']:
            raise ValueError('ordered base IDs')
        # Reuse exact classifier/cost/mixture implementation at frozen rounds.
        d=out/'seed_0';d.mkdir(parents=True,exist_ok=False)
        values,parts,_,models=T.fit_predict(fit,pred,T.feature_columns(fit),{'clf':180,'reg':2319})
        clf,reg,cols=models
        if len(clf)!=1:raise ValueError('unexpected classifier ensemble')
        ordinary=fit.filter(pl.Series(T.regime_label(fit)==0))
        cats=categories(fit,[c for c in T.CATEGORICAL if c in cols])
        state=dict(kind=kind,seed=0,engine='lightgbm',columns=cols,categories=cats,proxy=fit_proxy(ordinary),
                   categorical_columns=list(cats),rounds={'clf':180,'reg':2319},params={'clf':T.CLF_PARAMS,'reg':T.REG_PARAMS})
        reg.save_model(str(d/'model.txt'));clf[0].save_model(str(d/'classifier.txt'))
        save_json(d/'state.json',state)
        restored=predict_base(d,pred,read(d/'state.json'),T)
        for key in ('reg','p','proba','values'):np.testing.assert_allclose(restored[key],parts[key],rtol=0,atol=1e-8,equal_nan=True)
        np.testing.assert_allclose(restored['pred'],values,rtol=0,atol=1e-8)
        publish_seed(d,pred[ID].to_numpy(),restored,receipt,contract,kind,0,started,resources)
        return
    ordinary,pred,xf,xp,yf,state=prepare(root,contract,kind,T)
    del ordinary;gc.collect()
    ds=None if kind=='cat' else lgb.Dataset(xf,yf,feature_name=state['columns'],categorical_feature=state['categorical_columns'],free_raw_data=False)
    for seed in seeds:
        resource_guard(root,kind,threads)
        began=time.time();d=out/f'seed_{seed}';d.mkdir(parents=True,exist_ok=False)
        if kind=='cat':
            from catboost import CatBoostRegressor
            params=dict(iterations=5600,learning_rate=.08,depth=8,loss_function='RMSE',thread_count=threads,
                        verbose=500,random_seed=seed,allow_writing_files=False)
            model=CatBoostRegressor(**params);model.fit(xf);value=model.predict(xp)
            model.save_model(str(d/'model.cbm'));engine='catboost'
        else:
            params={**T.REG_PARAMS,'seed':1000+seed,'bagging_seed':2000+seed,'feature_fraction_seed':3000+seed}
            model=lgb.train(params,ds,num_boost_round=ROUNDS[kind]);value=model.predict(xp)
            model.save_model(str(d/'model.txt'));engine='lightgbm'
        if state['proxy'] is not None:value=value+apply_proxy(pred,state['proxy'])
        elif kind!='motion':value=value+anchor(pred,kind)
        saved={**state,'kind':kind,'engine':engine,'rounds':ROUNDS[kind],'seed':seed,'params':params}
        save_json(d/'state.json',saved)
        restored=predict_saved(d,pred,read(d/'state.json'))
        np.testing.assert_array_equal(restored,value)
        publish_seed(d,pred[ID].to_numpy(),{'reg':restored},receipt,contract,kind,seed,began,resources)
        print(f'COMPLETE {kind} seed{seed}: {time.time()-began:.1f}s',flush=True)
        del model;gc.collect()


def publish_seed(directory,ids,parts,receipt,contract,kind,seed,started,resources):
    for key,value in parts.items():
        # Candidate values intentionally retain NaN for unavailable filed times.
        if key!='values':validate_predictions(value,len(ids))
    np.savez(directory/'predictions.npz',mvt_id=ids,**parts)
    files={p.name:sha(p) for p in directory.iterdir() if p.is_file()}
    save_json(directory/'complete.json',dict(protocol=PROTOCOL,registration=receipt['commit'],kind=kind,seed=seed,
        files=files,feature_sha256=contract['files'][contract['features']],rows=len(ids),ids_sha256=ids_sha(ids),
        elapsed_seconds=time.time()-started,reload_parity=True,new_cv_fits=0,accepted_for_submission=False,
        runtime=runtime(),resources=resources))


def merge(root,receipt_path,kind):
    receipt,contract=admit(root,receipt_path);directory=root/'models/final_tabular'/kind
    seeds=expected_seeds(kind);records=[];arrays=[];ids=None
    present=sorted(int(p.name.removeprefix('seed_')) for p in directory.glob('seed_*') if p.is_dir())
    check_seed_set(kind,present)
    for seed in seeds:
        d=directory/f'seed_{seed}';r=read(d/'complete.json')
        if (r['protocol'],r['registration'],r['kind'],r['seed'],r['reload_parity'])!=(PROTOCOL,receipt['commit'],kind,seed,True):raise ValueError('wrong seed provenance')
        expected={'state.json','predictions.npz','model.cbm' if kind=='cat' else 'model.txt'}
        if kind=='base':expected.add('classifier.txt')
        if set(r['files'])!=expected:raise ValueError('incomplete model/state/prediction file binding')
        for name,h in r['files'].items():
            if sha(relative_file(d,name))!=h:raise ValueError('seed artifact changed')
        if r['feature_sha256']!=contract['files'][contract['features']]:raise ValueError('mixed feature lineage')
        state=read(d/'state.json');rounds={'clf':180,'reg':2319} if kind=='base' else ROUNDS[kind]
        if (state['kind'],state['seed'],state['rounds'],state['engine'])!=(kind,seed,rounds,'catboost' if kind=='cat' else 'lightgbm'):
            raise ValueError('model state identity/recipe differs')
        with np.load(d/'predictions.npz',allow_pickle=False) as z:
            this=z['mvt_id'];pred=z['reg'];validate_predictions(pred,contract['rank_rows'])
            if ids_sha(this)!=contract['rank_ids_sha256']:raise ValueError('seed IDs')
            if ids is None:ids=this.copy()
            elif not np.array_equal(ids,this):raise ValueError('seed alignment')
            arrays.append(pred.copy())
        records.append({'seed':seed,'complete_sha256':sha(d/'complete.json')})
    out=directory/'merged';out.mkdir(exist_ok=False)
    stack=np.stack(arrays);np.savez(out/'predictions.npz',mvt_id=ids,per_seed=stack,reg=stack.mean(axis=0))
    save_json(out/'complete.json',dict(protocol=PROTOCOL,registration=receipt['commit'],kind=kind,seeds=records,
        files={'predictions.npz':sha(out/'predictions.npz')},rows=len(ids),accepted_for_submission=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['attest','fit','merge'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--contract',type=Path);p.add_argument('--registration-commit')
    p.add_argument('--kind',choices=sorted(ROUNDS));p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--threads',type=int,default=6);a=p.parse_args();root=a.root.resolve()
    if a.action=='attest':attest(root,a.registration_commit,a.contract,a.receipt)
    elif a.action=='fit':fit_family(root,a.receipt,a.kind,a.seeds,a.threads)
    else:merge(root,a.receipt,a.kind)


if __name__=='__main__':main()

"""E195 phase 1: immutable inputs and held-out-pair production stage models.

No final-chain predictions, scores or submission acceptance are produced by this
phase. Historical upstream OOF is not fully nested. Exact donor-chain assembly
requires a separate registered continuation; it must never be silently omitted.
"""
from __future__ import annotations
import argparse
import ast
import gc
import glob
import hashlib
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

for k,v in {'POLARS_MAX_THREADS':'4','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
            'OMP_WAIT_POLICY':'PASSIVE','GOMP_SPINCOUNT':'0',
            'POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR':'load_as_storage'}.items():
    os.environ.setdefault(k,v)
import numpy as np
import polars as pl

PROTOCOL='E195-current-chain-retrospective-stage-models-six-pair-v1-20261007'
SELF='scripts/e195_current_chain.py'
ID,Y='MVT_ID_mvt','TAXITIME_SEC_mvt'
SEEDS=[42,1,2,3,4]
D7=['n_pts','sf_n_seg','vi_n_pts','vi_prev_pts','vi_alt_first','vm_n_pts','vm_prev_pts']
D3=['vi_n_pts','vi_prev_pts','vi_alt_first']
ACTIVE=['vi_first_seen','vi_turnaround','vi_gap','vi_gs_first','vi_has_prev']
VISIT=['vi_first_seen','vi_turnaround','vi_gap','vi_n_pts','vi_prev_pts','vi_alt_first','vi_gs_first','vi_has_prev']
F64=['ap','coverage','D','p_nonreg','base','has_nm']+ACTIVE
HOSTS={'mac':([1,2],8),'home':([3,4],12),'hetzner':([5,6],6)}
PINNED={
 'data/visit_mask_inputs_20261005_v2/manifest.json':'57c1848c6d7634e87e041cd75bc67e3251e8274d3fa827eef7e72ac7203f40da',
 'data/visit_mask_20261005_v2/extraction.json':'1422ac4b444c260e71c0472a7a63b92c1c8cc2ac7233176e3acbf85bf27c36c3',
 'src/e159_fixed_lineage.json':'2c607b8be32293924e044a1f3be978fb31aac5e0f8e3b3ecc99ee81a2971e7a3',
}
STAGE_SOURCES={
 'src/e141_apply.py':'1b5871e2291ec1cf82a99e6a2304e758c1182aacc0b917ff7aece2f5e3f823d1',
 'src/e36_adsb_stand_geometry.py':'51a1d4544b3539d95dca1354db06833598d288f930e3fc10cdd82c5eec9c7edb',
 'src/e64_visit_stage.py':'bd1238c5d0a4f9980842c7b8c1da8d0ba4d59f9c9aeeebaa0cde4d8b5f142e41',
 'src/e69_visit_merge.py':'63d6ad4326041cc4f4672815ed2bd5e9777557b258734787ed7f8a14fab2ee6e',
}


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,sort_keys=True,indent=2,allow_nan=False)


def registration(root,commit):
    if not re.fullmatch('[a-f0-9]{40}',commit or ''):raise ValueError('full registration commit required')
    source=subprocess.check_output(['git','show',f'{commit}:{SELF}'],cwd=root)
    journal=subprocess.check_output(['git','show',f'{commit}:HYPOTHESES_LOG.md'],cwd=root)
    if hashlib.sha256(source).hexdigest()!=sha(__file__) or PROTOCOL.encode() not in journal:
        raise ValueError('source and journal must share the registration commit')


def definitions(path,names,functions,extra=None):
    """Execute only explicit constant assignments/function definitions, no imports/CV."""
    tree=ast.parse(Path(path).read_text());nodes=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and all(isinstance(t,ast.Name) and t.id in names for t in node.targets):nodes.append(node)
        elif isinstance(node,ast.FunctionDef) and node.name in functions:nodes.append(node)
    scope=dict(np=np,pl=pl,os=os,glob=glob,sys=sys,**(extra or {}))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    if any(n not in scope for n in names|functions):raise ValueError('frozen AST definition missing')
    return SimpleNamespace(**scope)


def checked_rows(d,n=None):
    if d[ID].null_count() or d[ID].n_unique()!=d.height or (n is not None and d.height!=n):
        raise ValueError('incomplete or duplicate stage IDs')
    ids=d[ID].to_numpy()
    if not np.isfinite(ids).all() or not np.equal(ids,np.floor(ids)).all():raise ValueError('invalid IDs')


def pack(root,out,commit):
    registration(root,commit)
    if out.exists():raise ValueError('fresh immutable input directory required')
    os.chdir(root)
    frozen=root/'data/today_chain_source_20261006'
    inputs={}
    def track(p,expected=None):
        p=Path(p);h=sha(p)
        if expected is not None and h!=expected:raise ValueError('input SHA differs: '+str(p))
        inputs[str(p.relative_to(root))]=h
        return p
    for name,h in PINNED.items():track(root/name,h)
    for name,h in STAGE_SOURCES.items():track(frozen/name,h)
    rawfiles=sorted((root/'data/prc-2026-datasets').glob('training_*.parquet'))
    if len(rawfiles)!=12:raise ValueError('all twelve raw months required')
    for p in rawfiles:track(p)
    sensors={}
    for directory,prefix in [('adsblol_kt1',x) for x in ('offblock2','offblock3','offblock5')]+[
        ('adsblol_sw_kt1','offblock2')]+[('adsblol',x) for x in ('visit','visit_m','head')]:
        files=sorted((root/'data'/directory).glob(prefix+'_2025-*.parquet'))
        if len(files)!=364:raise ValueError('expected 364 observed sensor days: '+directory+'/'+prefix)
        sensors[directory+'/'+prefix]=files
        for p in files:track(p)
    historical={}
    for name in ('cv_preds_e116prod','cv_preds_nomonth','cv_preds_e150c_e64_keep'):
        paths=[root/'data'/name/f'fold_{a}_{a+6}.npz' for a in range(1,7)]
        for p in paths:track(p)
        historical[name]=paths
    os.environ.update(ADSB_DIR='data/adsblol_kt1',WALK_DIR='data/adsblol_sw_kt1',
                      VISIT_DIR='data/adsblol',HEAD_DIR='data/adsblol',E36_BASE_DIR='data/cv_preds_e116prod',
                      STAGE_SEEDS='42,1,2,3,4',DROP_FEATS=','.join(D7))
    e36=definitions(frozen/'src/e36_adsb_stand_geometry.py',
        {'Y','AIRPORTS','FEATS','SF','FEATS25','FEATS30','GS','SW','VISIT','PARAMS','PARAMS30','ADSB_DIR','cols'},
        {'build','load_adsb','oof_base'})
    nm=definitions(frozen/'src/e69_visit_merge.py',{'NAMES_M'},set()).NAMES_M
    e141=definitions(frozen/'src/e141_apply.py',{'SW','NEW','HEAD','E141','FEATS','SEEDS','DROP'},
                     {'augment'},{'E36':e36,'NAMES_M':nm})
    e64=definitions(frozen/'src/e64_visit_stage.py',{'Y','AIRPORTS','FEATS','PARAMS','ROUNDS'},set(),{'NAMES':VISIT})
    if [c for c in e64.FEATS if c not in D3]!=F64 or e64.ROUNDS!=400:raise ValueError('E64 recipe drift')
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent))
    (stage/'scripts').mkdir();shutil.copy2(__file__,stage/'scripts'/Path(SELF).name)
    specs=[]
    def store_stage(name,d,features,params,rounds,seeds,parent,clip):
        checked_rows(d)
        cols=list(dict.fromkeys([ID,Y,'month','ADEP_mvt',*features,'weight']))
        d=d.select(cols)
        if d[Y].null_count() or d['base' if 'base' in features else 'pred'].null_count():raise ValueError('stage labels/base missing')
        p=stage/(name+'.parquet');d.write_parquet(p)
        specs.append(dict(name=name,features=features,params=params,rounds=rounds,seeds=seeds,
                          historical_parent=parent,target_clip=clip,rows=d.height,file=p.name,sha256=sha(p)))
        print(f'PACKED {name}: {d.height} rows, {len(features)} features',flush=True)
    raw=pl.concat([pl.read_parquet(p,columns=e36.cols) for p in rawfiles]).filter(pl.col('PHASE_mvt')=='DEP')
    checked_rows(raw,2085047)
    d=e141.augment(e36.build(raw,e36.oof_base(),e36.load_adsb('data/adsblol/offblock2_2025-*.parquet')),2025)
    checked_rows(d,2085047)
    d=d.with_columns(month=pl.col('MVT_TIME_UTC_mvt').dt.month(),weight=pl.lit(1.))
    # Preserve the complete natural feature view for a later registered application.
    d.select(list(dict.fromkeys([ID,Y,'month','ADEP_mvt',*e141.FEATS,*F64]))).write_parquet(stage/'natural_features.parquet')
    store_stage('e141',d.filter(pl.col('coverage')>=2),e141.FEATS,e36.PARAMS30,800,SEEDS,'cv_preds_e116prod',1800)
    ordinary=e36.oof_base('data/cv_preds_nomonth').rename({'base':'nomonth_base','p_nonreg':'nomonth_pnr'})
    d=d.join(ordinary,on=ID,validate='1:1',maintain_order='left').with_columns(
        base=pl.col('nomonth_base'),p_nonreg=pl.col('nomonth_pnr'))
    store_stage('e64',d.filter(((pl.col('coverage')<2)&(pl.col('vi_has_prev')==1)).fill_null(False)),
                F64,e64.PARAMS,400,SEEDS,'cv_preds_nomonth',1800)
    del d,ordinary;gc.collect()
    bm=json.loads((root/'data/visit_mask_inputs_20261005_v2/manifest.json').read_text())
    ex=json.loads((root/'data/visit_mask_20261005_v2/extraction.json').read_text())
    if bm['features']!=F64 or bm['seeds']!=SEEDS or bm['synthetic_weight']!=.25 or len(ex['days'])!=365:
        raise ValueError('E188 registered augmentation differs')
    tp=root/'data/visit_mask_inputs_20261005_v2/training.parquet';track(tp,bm['bundle_sha256'][str(tp.relative_to(root))])
    frames=[]
    for item in ex['days']:
        rel=f"days/{item['day']}.parquet";p=root/'data/visit_mask_20261005_v2'/rel
        track(p,ex['output_sha256'][rel]);frames.append(pl.read_parquet(p))
    audit=pl.concat([f.select(frames[0].columns) for f in frames]);checked_rows(audit,2085047)
    d=pl.read_parquet(tp).join(audit.drop('month','ADEP_mvt','has_nm'),on=ID,validate='1:1',maintain_order='left')
    d=d.with_columns(*[pl.col('base_'+c).alias(c) for c in ACTIVE],pl.col('coverage').fill_null(-1))
    natural=d.filter(((pl.col('coverage')<2)&(pl.col('vi_has_prev')==1)).fill_null(False)).with_columns(weight=pl.lit(1.))
    changed=np.zeros(d.height,dtype=bool)
    for c in ACTIVE:
        a,b=[d[p+c].cast(pl.Float64).to_numpy() for p in ('base_','air_')]
        changed|=~((a==b)|(np.isnan(a)&np.isnan(b)))
    eligible=d.select(((pl.col('ADEP_mvt')=='LIRF')&pl.col('has_nm')&(pl.col('coverage')>=2)&
        (pl.col('air_vi_has_prev')==1)&pl.col('base_icao').is_not_null()&(pl.col('base_icao')==pl.col('air_icao'))).fill_null(False)).to_series().to_numpy()
    extra=d.filter(pl.Series(eligible&changed)).with_columns(coverage=pl.lit(1).cast(pl.Int64),weight=pl.lit(.25),
                                                          **{c:pl.col('air_'+c) for c in ACTIVE})
    if natural.height!=318697 or extra.height!=60327 or extra.height!=ex['synthetic_total']:raise ValueError('E188 population differs')
    store_stage('e188',pl.concat([natural,extra]),F64,e64.PARAMS,400,SEEDS,'cv_preds_nomonth plus train-only mask .25',1800)
    del d,natural,extra,audit,frames;gc.collect()
    # LSZH uses exactly the 44 frozen OPDI sources, never a broad directory glob.
    lineage=json.loads((root/'src/e159_fixed_lineage.json').read_text())
    opaths=[track(root/n,h) for n,h in lineage['inputs'].items() if n.startswith('data/opdi/')]
    if len(opaths)!=44:raise ValueError('OPDI archive inventory differs')
    for n,h in lineage['inputs'].items():
        if n.startswith('data/cv_preds_e150c_e64_keep/'):track(root/n,h)
    opath=root/'data/today_chain_20261006/src/e159_opdi_stage_cv.py'
    track(opath,lineage['sources']['src/e159_opdi_stage_cv.py'])
    op=definitions(opath,{'TYPES'}, {'match'})
    ev=pl.concat([pl.read_parquet(p,columns=['flight_id','flt_id','adep','type','t']) for p in opaths])
    ev=ev.filter(pl.col('type').is_in(op.TYPES)).with_columns(t=pl.col('t').dt.epoch('s').cast(pl.Float64))
    w=ev.group_by('flight_id','type').agg(pl.col('t').min(),pl.col('flt_id').first(),pl.col('adep').first()).pivot(on='type',index=['flight_id','flt_id','adep'],values='t')
    for c in op.TYPES:
        if c not in w.columns:w=w.with_columns(pl.lit(None,dtype=pl.Float64).alias(c))
    w=w.with_columns(anchor=pl.coalesce('take-off','first-xing-fl50','first_seen')).filter(pl.col('anchor').is_not_null())
    cols=[ID,'PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','CALLSIGN_flt','FLIGHT_mvt']
    tr=pl.concat([pl.read_parquet(p,columns=cols) for p in rawfiles]).filter((pl.col('PHASE_mvt')=='DEP')&(pl.col('ADEP_mvt')=='LSZH')).with_columns(
        mvt=pl.col('MVT_TIME_UTC_mvt').dt.epoch('s').cast(pl.Float64),month=pl.col('MVT_TIME_UTC_mvt').dt.month(),
        cs=pl.coalesce('CALLSIGN_flt','FLIGHT_mvt').str.replace_all(' ','')).drop('CALLSIGN_flt','FLIGHT_mvt','MVT_TIME_UTC_mvt','PHASE_mvt')
    prior=[]
    for p in historical['cv_preds_e150c_e64_keep']:
        with np.load(p,allow_pickle=False) as z:prior.append(pl.DataFrame({ID:z['mvt_id'],'pred':z['pred'],Y:z['y']}))
    lszh=op.match(tr,w).filter(pl.col('exit').is_not_null()).join(pl.concat(prior),on=ID,how='inner',validate='1:1').with_columns(
        t_opdi=pl.col('mvt')-pl.col('exit'),twy_s=pl.col('twy')-pl.col('exit'),weight=pl.lit(1.)).with_columns(d_opdi=pl.col('t_opdi')-pl.col('pred'))
    if lszh.height!=84758:raise ValueError('LSZH historical population differs')
    lszh.select(ID,'month','mvt','exit','twy').write_parquet(stage/'lszh_matches.parquet')
    params=dict(objective='regression',learning_rate=.05,num_leaves=15,min_data_in_leaf=100,lambda_l2=10.,feature_fraction=1.,bagging_fraction=.8,bagging_freq=1,verbose=-1,seed=42,num_threads=4,deterministic=True)
    store_stage('lszh',lszh,['d_opdi','t_opdi','pred','twy_s'],params,300,[42],'cv_preds_e150c_e64_keep',None)
    for name,h in inputs.items():
        if sha(root/name)!=h:raise ValueError('input changed during preparation')
    spec=dict(protocol=PROTOCOL,registration_commit=commit,worker_sha256=sha(__file__),inputs=inputs,stages=specs,
              files={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()},
              final_chain_complete=False,limitation='Retrospective historical-upstream crossfit, not independently nested CV. Phase1 stage models only.')
    save(stage/'manifest.json',dict(spec=spec,spec_sha256=digest(spec)))
    stage.rename(out);print('IMMUTABLE STAGE INPUTS READY',out,flush=True)


def verify(bundle,commit):
    m=json.loads((bundle/'manifest.json').read_text());s=m['spec']
    if m['spec_sha256']!=digest(s) or s['protocol']!=PROTOCOL or s['registration_commit']!=commit or s['worker_sha256']!=sha(__file__):raise ValueError('pack/worker registration drift')
    if [x['name'] for x in s['stages']]!=['e141','e64','e188','lszh']:raise ValueError('incomplete stage inventory')
    for name,h in s['files'].items():
        if sha(bundle/name)!=h:raise ValueError('packed input changed: '+name)
    return m


def fit(bundle,out,commit,host,fold):
    if 'torch' in sys.modules:raise ValueError('LGB-only fresh process required')
    folds,threads=HOSTS[host]
    if fold not in folds or out.exists():raise ValueError('wrong host/pair or existing output; no auto-resume')
    if host=='hetzner' and set(os.sched_getaffinity(0))!={0,2,4,6,8,10}:raise ValueError('Hetzner must use six distinct physical cores')
    if host=='home':
        rel=next(x[3:] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
        group=Path('/sys/fs/cgroup')/rel.lstrip('/')
        memory=(group/'memory.max').read_text().strip();swap=(group/'memory.swap.max').read_text().strip()
        quota,period=(group/'cpu.max').read_text().split()
        if memory=='max' or int(memory)>16*1024**3 or swap!='0' or quota=='max' or int(quota)>12*int(period):
            raise ValueError('Home requires MemoryMax<=16GiB, MemorySwapMax0, CPUQuota<=1200%')
    os.environ.update(OMP_NUM_THREADS=str(threads),OMP_THREAD_LIMIT=str(threads))
    m=verify(bundle,commit);out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.partial-',dir=out.parent))
    import lightgbm as lgb
    results=[]
    for spec in m['spec']['stages']:
        d=pl.read_parquet(bundle/spec['file']);tr=d.filter(~pl.col('month').is_in([fold,fold+6]));te=d.filter(pl.col('month').is_in([fold,fold+6]))
        if not tr.height or not te.height:raise ValueError('empty stage outer train/holdout')
        base='base' if 'base' in spec['features'] else 'pred'
        x=tr.select(spec['features']).to_numpy();y=(tr[Y]-tr[base]).to_numpy()
        if spec['target_clip'] is not None:y=np.clip(y,-spec['target_clip'],spec['target_clip'])
        if not np.isfinite(y).all() or np.isinf(x).any():raise ValueError('invalid fit values')
        od=stage/spec['name'];od.mkdir();trained=[]
        for seed in spec['seeds']:
            start=time.monotonic();print(f'FIT fold{fold}+{fold+6} {spec["name"]} seed{seed} rows{tr.height} threads{threads}',flush=True)
            cat=[0] if spec['name']!='lszh' else 'auto'
            ds=lgb.Dataset(x,y,weight=tr['weight'].to_numpy() if spec['name']=='e188' else None,categorical_feature=cat)
            model=lgb.train({**spec['params'],'num_threads':threads,'seed':seed},ds,num_boost_round=spec['rounds'])
            if model.current_iteration()!=spec['rounds']:raise ValueError('fit did not produce every fixed boosting round')
            p=od/f'seed{seed}.txt';model.save_model(str(p))
            trained.append(dict(seed=seed,sha256=sha(p),seconds=time.monotonic()-start))
            print('DONE',spec['name'],seed,trained[-1]['seconds'],flush=True)
            del model,ds;gc.collect()
        results.append(dict(stage=spec['name'],train_rows=tr.height,heldout_rows=te.height,heldout_months=[fold,fold+6],
                            train_id_sha256=digest(tr[ID].to_list()),models=trained))
        del d,tr,te,x,y;gc.collect()
    verify(bundle,commit)
    save(stage/'complete.json',dict(protocol=PROTOCOL,registration_commit=commit,input_spec_sha256=m['spec_sha256'],fold=fold,
        host=host,threads=threads,lightgbm_version=lgb.__version__,results=results,final_chain_complete=False,accepted_for_submission=False))
    stage.rename(out);print('COMPLETE STAGE MODELS ONLY',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['pack','verify','fit'])
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--input',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--registration-commit',required=True);p.add_argument('--host',choices=HOSTS);p.add_argument('--fold',type=int)
    a=p.parse_args()
    if not re.fullmatch('[a-f0-9]{40}',a.registration_commit):p.error('full registration SHA required')
    if a.action=='pack':pack(a.root.resolve(),a.output.resolve(),a.registration_commit)
    elif a.action=='verify':print(verify(a.input.resolve(),a.registration_commit)['spec_sha256'])
    else:fit(a.input.resolve(),a.output.resolve(),a.registration_commit,a.host,a.fold)


if __name__=='__main__':main()

"""E205: fixed current-METAR CB residual test on the E203 pre-E81 OOF.

No production source edits, new environments, rank fit or upload. Every real
target operation requires the immutable bundle of the registered protocol.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
for key,value in {'POLARS_MAX_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
    'OMP_NUM_THREADS':'6','OMP_THREAD_LIMIT':'6','OMP_WAIT_POLICY':'PASSIVE','GOMP_SPINCOUNT':'0'}.items():
    os.environ.setdefault(key,value)
import numpy as np
import polars as pl

PROTOCOL='E205-current-cloud-CB-matched-residual-sixpair-preE81-v1'
SELF='scripts/night_cb_20261007.py'
DOC='analysis/night_cb_20261007/protocol.md'
RECEIPT='analysis/night_cb_20261007/input_receipt.json'
FEATURES='data/features_fin1b.parquet'
PARENT='data/e203_pre_e81_20261007/predictions.parquet'
PARENT_RECEIPT='data/e203_pre_e81_20261007/result.json'
ID='MVT_ID_mvt'; TARGET='TAXITIME_SEC_mvt'
AIRPORTS='EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF LTFM LSZH'.split()
CB_NAMES=['cloud_cb','cloud_cb_missed','cloud_cb_base_ft','cloud_cb_amount']
WEATHER=['wx_temp','wx_spread','wx_wind','wx_gust','wx_vis_mi','wx_ceiling_ft',
    'wx_snow','wx_freezing','wx_rain','wx_thunder','wx_fog','wx_icing','wx_snow_recent','wx_icing_recent','metar_age_min']
CONTEXT=['airport_code','has_nm','local_hour','d_mvt_aobt','d_mvt_eobt','d_mvt_lobt',
    'd_mvt_iobt','d_mvt_sched','rwy_dep_before_30','nb_ap_30_0_n','n_dep_rwys',
    'n_arr_rwys','own_rwy_share_hour','ordered']
BASE_COLS=WEATHER+CONTEXT
CB_RE=re.compile(r'\b(FEW|SCT|BKN|OVC)(\d{3})CB\b')
PARAMS=dict(objective='regression',metric='None',learning_rate=.03,num_leaves=15,
    max_depth=4,min_data_in_leaf=500,lambda_l2=100.,max_bin=127,feature_fraction=1.,
    bagging_fraction=1.,bagging_freq=0,seed=20307,deterministic=True,force_col_wise=True,
    verbosity=-1,num_threads=6)
CONFIG=dict(params=PARAMS,rounds=400,residual_clip=900.,prediction_clip=120.,shrinkage=.5,
    folds=[1,2,3,4,5,6],arms=['weather','cb'],base_columns=BASE_COLS,cb_columns=CB_NAMES,
    train_body=True,train_decided=True,apply_decided=True,tolerance_seconds=10800)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temp.replace(path)


def parse_cb(metar,wxcodes):
    if not metar:return (np.nan,np.nan,np.nan,np.nan)
    groups=CB_RE.findall(metar)
    if not groups:return (0,0,np.nan,np.nan)
    return (1,int(not re.search('TS|CB',wxcodes or '')),
        float(min(int(height)*100 for _,height in groups)),
        float(max({'FEW':1,'SCT':2,'BKN':3,'OVC':4}[amount] for amount,_ in groups)))


def backward_indices(times,anchors):
    times=np.asarray(times);anchors=np.asarray(anchors)
    if len(times)==0 or np.any(np.diff(times)<=0):raise ValueError('unique sorted weather timestamps required')
    ix=np.searchsorted(times,anchors,side='right')-1;safe=np.maximum(ix,0)
    ok=(ix>=0)&(anchors-times[safe]>=0)&(anchors-times[safe]<=10800)
    return np.where(ok,ix,-1)


def align_ids(wanted,source):
    wanted=np.asarray(wanted);source=np.asarray(source)
    if not np.isfinite(wanted).all() or not np.isfinite(source).all() or len(np.unique(wanted))!=len(wanted) or len(np.unique(source))!=len(source):raise ValueError('nonunique/nonfinite IDs')
    order=np.argsort(source);sorted_ids=source[order];ix=np.searchsorted(sorted_ids,wanted)
    if np.any(ix>=len(sorted_ids)) or not np.array_equal(sorted_ids[np.minimum(ix,len(sorted_ids)-1)],wanted):raise ValueError('ID coverage differs')
    return order[ix]


def weather_lookup(root):
    lookup={};statistics=[]
    for airport in AIRPORTS:
        rows={};raw_count=0
        with (root/'data/metar'/f'{airport}.csv').open(newline='') as source:
            for row in csv.DictReader(source):
                if row['station']!=airport:raise ValueError('station differs')
                stamp=int(datetime.strptime(row['valid'],'%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc).timestamp())
                # Duplicate times resolve to last source row, independently of flight labels.
                rows[stamp]=(parse_cb(row.get('metar'),row.get('wxcodes')),int(bool(re.search('TS|CB',row.get('wxcodes') or ''))))
                raw_count+=1
        times=np.array(sorted(rows),dtype=np.int64)
        values=np.array([rows[t][0] for t in times],dtype=np.float32)
        thunder=np.array([rows[t][1] for t in times],dtype=np.float32)
        lookup[airport]=(times,values,thunder)
        statistics.append(dict(airport=airport,raw_reports=raw_count,unique_reports=len(rows),duplicates=raw_count-len(rows)))
    return lookup,statistics


def flight_join(frame,lookup):
    anchor=frame.select(pl.col('AOBT_3_flt').fill_null(pl.col('MVT_TIME_UTC_mvt')-pl.duration(seconds=900)).dt.epoch('s')).to_series().to_numpy()
    if not np.isfinite(anchor).all():raise ValueError('missing observable anchor')
    airport=frame['ADEP_mvt'].to_numpy();new=np.full((len(frame),4),np.nan,np.float32)
    reported_thunder=np.full(len(frame),np.nan);age=np.full(len(frame),np.nan)
    if not np.isin(airport,AIRPORTS).all():raise ValueError('unknown flight airport')
    for ap in AIRPORTS:
        rows=np.flatnonzero(airport==ap);times,values,thunder=lookup[ap]
        ix=backward_indices(times,anchor[rows]);present=ix>=0;selected=rows[present];report=ix[present]
        new[selected]=values[report];reported_thunder[selected]=thunder[report]
        age[selected]=(anchor[selected]-times[report])//60
    if not np.array_equal(reported_thunder,frame['wx_thunder'].to_numpy(),equal_nan=True):raise ValueError('current wx_thunder asof parity failed')
    if not np.array_equal(age,frame['metar_age_min'].to_numpy(),equal_nan=True):raise ValueError('current age asof parity failed')
    return frame.with_columns(*[pl.Series(n,new[:,i]) for i,n in enumerate(CB_NAMES)])


def coverage(frame):
    return frame.group_by('split','ADEP_mvt','month').agg(pl.len().alias('flights'),
        (pl.col('cloud_cb')==1).sum().alias('cb'),(pl.col('cloud_cb_missed')==1).sum().alias('cb_missed'),
        pl.col('cloud_cb').is_nan().sum().alias('cb_unknown')).sort('split','ADEP_mvt','month').to_dicts()


def observable_frame(root):
    columns=[ID,'split','month','ADEP_mvt','FLIGHT_ID_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt',
        *WEATHER,*[c for c in CONTEXT if c not in ('airport_code','has_nm','ordered')]]
    frame=pl.read_parquet(root/FEATURES,columns=list(dict.fromkeys(columns)))
    lookup,statistics=weather_lookup(root);frame=flight_join(frame,lookup)
    frame=frame.with_columns(pl.col('FLIGHT_ID_mvt').is_not_null().alias('has_nm'),
        pl.col('ADEP_mvt').replace_strict({ap:i for i,ap in enumerate(AIRPORTS)},return_dtype=pl.Int16).alias('airport_code'))
    return frame,statistics


def seal(root):
    receipt=json.loads((root/PARENT_RECEIPT).read_text())
    if receipt['files']['predictions.parquet']!=sha(root/PARENT):raise ValueError('E203 OOF differs from its receipt')
    paths=[FEATURES,PARENT,PARENT_RECEIPT,*[f'data/metar/{ap}.csv' for ap in AIRPORTS]]
    inputs={p:sha(root/p) for p in paths}
    frame,statistics=observable_frame(root)
    if frame.filter(pl.col('split')=='train').height!=2085047 or frame.filter(pl.col('split')=='rank').height!=344841:raise ValueError('unexpected flight coverage')
    write_json(root/RECEIPT,dict(protocol=PROTOCOL,inputs=inputs,weather_reports=statistics,
        coverage=coverage(frame),target_independent=True,configuration=CONFIG))
    print('SEALED LABEL-FREE COVERAGE',len(frame),flush=True)


def registration(root,commit):
    if not re.fullmatch('[a-f0-9]{40}',commit or ''):raise ValueError('full preregistration commit required')
    for path in (SELF,DOC,RECEIPT,'scripts/night_cb_20261007_queue.py'):
        frozen=subprocess.check_output(['git','show',commit+':'+path],cwd=root)
        if hashlib.sha256(frozen).hexdigest()!=sha(root/path):raise ValueError('registered file changed '+path)
    if PROTOCOL.encode() not in subprocess.check_output(['git','show',commit+':HYPOTHESES_LOG.md'],cwd=root):raise ValueError('protocol absent from registered journal')


def inputs_unchanged(root,receipt):
    if receipt['protocol']!=PROTOCOL or receipt['configuration']!=CONFIG:raise ValueError('input protocol differs')
    for path,expected in receipt['inputs'].items():
        if sha(root/path)!=expected:raise ValueError('input drift '+path)


def pack(root,output,commit):
    registration(root,commit)
    if output.exists():raise ValueError('fresh pack required')
    receipt=json.loads((root/RECEIPT).read_text());inputs_unchanged(root,receipt)
    if shutil.disk_usage(root).free<2*1024**3:raise ValueError('need 2GiB headroom')
    frame,statistics=observable_frame(root)
    if coverage(frame)!=receipt['coverage'] or statistics!=receipt['weather_reports']:raise ValueError('coverage drift')
    frame=frame.filter(pl.col('split')=='train')
    parent=pl.read_parquet(root/PARENT)
    ix=align_ids(frame[ID].to_numpy(),parent[ID].to_numpy());parent=parent[ix]
    labels=pl.read_parquet(root/FEATURES,columns=[ID,'split',TARGET]).filter(pl.col('split')=='train')
    if not np.array_equal(labels[ID].to_numpy(),frame[ID].to_numpy()):raise ValueError('label ID mismatch')
    for key,other in [('month','month'),('ADEP_mvt','ADEP_mvt'),('has_nm','has_nm'),('d_mvt_sched','D')]:
        if not np.array_equal(frame[key].to_numpy(),parent[other].to_numpy(),equal_nan=key=='d_mvt_sched'):raise ValueError('parent metadata drift '+key)
    if not np.array_equal(labels[TARGET].to_numpy(),parent[TARGET].to_numpy()):raise ValueError('parent labels differ')
    if parent['excluded_tabpfn'].sum()!=126 or (~parent['evaluation_mask']).sum()!=126:raise ValueError('126-row protected support differs')
    frame=frame.with_columns(parent['ordered'],parent['evaluation_mask'],parent['excluded_tabpfn'],labels[TARGET])
    frame=frame.with_columns((~((pl.col('ADEP_mvt')=='LIRF')&~pl.col('has_nm')&(pl.col('d_mvt_sched')>=14400))).alias('decided'))
    output.mkdir(parents=True);keep=[ID,TARGET,'month','ADEP_mvt','evaluation_mask','excluded_tabpfn','decided',*BASE_COLS,*CB_NAMES]
    frame.select(keep).write_parquet(output/'rows.parquet')
    shutil.copy2(root/SELF,output/'night_cb_20261007.py');shutil.copy2(root/DOC,output/'protocol.md');shutil.copy2(root/RECEIPT,output/'input_receipt.json')
    spec=dict(protocol=PROTOCOL,registration_commit=commit,configuration=CONFIG,rows=len(frame),
        columns=keep,inputs=receipt['inputs'],files={p.name:sha(p) for p in output.iterdir() if p.is_file()},
        limitation='Current E203 pre-E81 historical OOF, selection-informed, not fully nested; no full-tail or leaderboard inference.')
    inputs_unchanged(root,receipt);write_json(output/'manifest.json',dict(spec=spec,spec_sha256=digest(spec)))
    print('PACK COMPLETE',len(frame),digest(spec),flush=True)


def verify_bundle(bundle):
    manifest=json.loads((bundle/'manifest.json').read_text());spec=manifest['spec']
    if digest(spec)!=manifest['spec_sha256'] or spec['protocol']!=PROTOCOL or spec['configuration']!=CONFIG:raise ValueError('bundle spec drift')
    for path,expected in spec['files'].items():
        if Path(path).name!=path or sha(bundle/path)!=expected:raise ValueError('bundle file drift '+path)
    if sha(Path(__file__))!=spec['files']['night_cb_20261007.py']:raise ValueError('worker source differs')
    return manifest


def fold_masks(month,y,included,decided,fold):
    if fold not in range(1,7):raise ValueError('unknown fold')
    heldout=np.isin(month,[fold,fold+6])
    return (~heldout)&included&decided&(y<=7200),heldout


def apply_correction(base,residual,included,decided):
    base=np.asarray(base,dtype=float);residual=np.asarray(residual,dtype=float)
    if not np.isfinite(base).all() or not np.isfinite(residual).all():raise ValueError('nonfinite predictions')
    result=base.copy();active=included&decided
    result[active]=np.maximum(0,base[active]+.5*np.clip(residual[active],-120,120))
    return result


def paired(y,control,candidate,mask):
    if not np.isfinite(y).all() or not np.isfinite(control).all() or not np.isfinite(candidate).all():raise ValueError('nonfinite metric input')
    n=int(mask.sum())
    if n<=40:raise ValueError('empty/undersized metric mask')
    c=(control[mask]-y[mask])**2;p=(candidate[mask]-y[mask])**2;difference=np.sort(p-c)
    sc=float(c.sum());sp=float(p.sum())
    if not np.isfinite(sc+sp):raise ValueError('nonfinite SSE')
    return dict(n=n,control_sse=sc,candidate_sse=sp,delta_sse=float((p-c).sum()),
        delta_rmse=float(np.sqrt(sp/n)-np.sqrt(sc/n)),trim3_sse=float(difference[3:-3].sum()),trim20_sse=float(difference[20:-20].sum()))


def comparisons(y,base,weather,cb,included,decided):
    masks={'included':included,'body':included&(y<=7200),'decided':included&decided}
    return {label:{name:paired(y,control,candidate,mask) for name,mask in masks.items()}
        for label,control,candidate in [('cb_vs_weather',weather,cb),('cb_vs_base',base,cb),('weather_vs_base',base,weather)]}


def gate(reports):
    if len(reports)!=6:return False
    values=[float(r[pop][key]) for r in reports for pop,key in [('body','delta_sse'),('decided','delta_sse'),('decided','trim3_sse'),('decided','trim20_sse')]]
    if not np.isfinite(values).all():return False
    return sum(r['body']['delta_sse'] for r in reports)<0 and sum(r['decided']['delta_sse'] for r in reports)<0 and sum(r['decided']['delta_sse']<0 for r in reports)>=4 and all(sum(r['decided'][key] for r in reports)<0 for key in ('trim3_sse','trim20_sse'))


def run(bundle,output,folds,threads):
    manifest=verify_bundle(bundle)
    if output.exists():raise ValueError('fresh node output required; no automatic retry')
    if not folds or len(set(folds))!=len(folds) or any(f not in range(1,7) for f in folds) or not 1<=threads<=6:raise ValueError('invalid fold/thread assignment')
    output.mkdir(parents=True);started=time.time()
    # This process does not import torch and runs in existing CPU LGB environments.
    import lightgbm as lgb
    frame=pl.read_parquet(bundle/'rows.parquet');ids=frame[ID].to_numpy();y=frame[TARGET].to_numpy().astype(float)
    month=frame['month'].to_numpy();base=frame['ordered'].to_numpy();included=frame['evaluation_mask'].to_numpy();decided=frame['decided'].to_numpy()
    if len(frame)!=2085047 or (~included).sum()!=126 or len(np.unique(ids))!=len(ids):raise ValueError('packed support invalid')
    x=frame.select(BASE_COLS+CB_NAMES).to_numpy().astype(np.float32)
    if np.isinf(x).any() or not np.isfinite(y).all():raise ValueError('invalid training values')
    reports=[]
    for fold in folds:
        destination=output/f'fold_{fold}_{fold+6}';destination.mkdir()
        fit,valid=fold_masks(month,y,included,decided,fold)
        if fit.sum()==0 or valid.sum()==0:raise ValueError('empty fit/validation')
        residual=np.clip(y[fit]-base[fit],-900,900);predictions={}
        for arm in ('weather','cb'):
            width=len(BASE_COLS)+(4 if arm=='cb' else 0)
            dataset=lgb.Dataset(x[fit,:width],label=residual,feature_name=(BASE_COLS+CB_NAMES)[:width],categorical_feature=['airport_code'],free_raw_data=True)
            params=dict(PARAMS,num_threads=threads)
            print(f'FIT fold={fold} arm={arm} train={fit.sum()} heldout={valid.sum()} rounds=400',flush=True)
            model=lgb.train(params,dataset,num_boost_round=400)
            model.save_model(str(destination/(arm+'.txt')))
            correction=model.predict(x[valid,:width],num_threads=threads)
            predictions[arm]=apply_correction(base[valid],correction,included[valid],decided[valid])
            print(f'FIT COMPLETE fold={fold} arm={arm}',flush=True)
        payload=dict(mvt_id=ids[valid],y=y[valid],month=month[valid],base=base[valid],included=included[valid],decided=decided[valid],weather=predictions['weather'],cb=predictions['cb'])
        np.savez_compressed(destination/'predictions.npz',**payload)
        metrics=comparisons(payload['y'],payload['base'],payload['weather'],payload['cb'],payload['included'],payload['decided'])
        report=dict(fold=fold,metrics=metrics,train_rows=int(fit.sum()),valid_rows=int(valid.sum()),
            input_spec_sha256=manifest['spec_sha256'],protocol=PROTOCOL,registration_commit=manifest['spec']['registration_commit'],
            files={p.name:sha(p) for p in destination.iterdir() if p.is_file()},
            runtime=dict(python=sys.version,numpy=np.__version__,polars=pl.__version__,lightgbm=lgb.__version__),threads=threads)
        write_json(destination/'complete.json',report);reports.append(report)
    verify_bundle(bundle)
    write_json(output/'complete.json',dict(protocol=PROTOCOL,input_spec_sha256=manifest['spec_sha256'],folds=folds,reports=reports,seconds=time.time()-started))


def evaluate(bundle,nodes,output):
    manifest=verify_bundle(bundle)
    if output.exists():raise ValueError('fresh evaluation required')
    frame=pl.read_parquet(bundle/'rows.parquet');reports=[];arrays=[]
    for fold in range(1,7):
        matches=[node/f'fold_{fold}_{fold+6}' for node in nodes if (node/f'fold_{fold}_{fold+6}'/'complete.json').exists()]
        if len(matches)!=1:raise ValueError('exactly one completed result per fold required')
        directory=matches[0];report=json.loads((directory/'complete.json').read_text())
        if report['fold']!=fold or report['input_spec_sha256']!=manifest['spec_sha256'] or report['protocol']!=PROTOCOL or report['registration_commit']!=manifest['spec']['registration_commit']:raise ValueError('result lineage differs')
        for path,expected in report['files'].items():
            if sha(directory/path)!=expected:raise ValueError('result drift')
        with np.load(directory/'predictions.npz',allow_pickle=False) as source:z={k:source[k] for k in source.files}
        expected=frame.filter(pl.col('month').is_in([fold,fold+6]));ix=align_ids(expected[ID].to_numpy(),z['mvt_id']);z={k:v[ix] for k,v in z.items()}
        for key,column in [('y',TARGET),('month','month'),('base','ordered'),('included','evaluation_mask'),('decided','decided')]:
            if not np.array_equal(z[key],expected[column].to_numpy()):raise ValueError('saved result metadata differs '+key)
        inactive=~(z['included']&z['decided'])
        for arm in ('weather','cb'):
            if not np.array_equal(z[arm][inactive],z['base'][inactive]):raise ValueError('protected prediction modified')
        metrics=comparisons(z['y'],z['base'],z['weather'],z['cb'],z['included'],z['decided'])
        if metrics!=report['metrics']:raise ValueError('saved metrics differ')
        reports.append(dict(fold=fold,metrics=metrics));arrays.append(z)
    pooled={key:np.concatenate([z[key] for z in arrays]) for key in arrays[0]}
    if len(np.unique(pooled['mvt_id']))!=2085047:raise ValueError('incomplete unique OOF')
    gates={name:gate([r['metrics'][name] for r in reports]) for name in ('cb_vs_weather','cb_vs_base')}
    result=dict(protocol=PROTOCOL,registration_commit=manifest['spec']['registration_commit'],input_spec_sha256=manifest['spec_sha256'],folds=reports,gates=gates,gate_pass=all(gates.values()),
        metrics=comparisons(pooled['y'],pooled['base'],pooled['weather'],pooled['cb'],pooled['included'],pooled['decided']),
        trim_rule='Sum within-fold DECIDED symmetric trim3/trim20 SSE; pooled trim is descriptive only',
        protected_126_exact=True,rank_protected_42_not_touched=True,accepted_for_submission=False,full_tail_oof_validated=False,limitation=manifest['spec']['limitation'])
    write_json(output,result);print(json.dumps(dict(gates=gates,gate_pass=result['gate_pass'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['seal','pack','run','evaluate','verify'])
    parser.add_argument('--root',type=Path,default=Path('.'));parser.add_argument('--input',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--registration-commit',default='');parser.add_argument('--folds',default='');parser.add_argument('--threads',type=int,default=6)
    parser.add_argument('--nodes',nargs='*',type=Path);args=parser.parse_args()
    if args.command=='seal':seal(args.root.resolve())
    elif args.command=='pack':pack(args.root.resolve(),args.output,args.registration_commit)
    elif args.command=='run':run(args.input,args.output,[int(f) for f in args.folds.split(',') if f],args.threads)
    elif args.command=='evaluate':evaluate(args.input,args.nodes,args.output)
    else:print(json.dumps(verify_bundle(args.input)['spec_sha256']))


if __name__=='__main__':main()

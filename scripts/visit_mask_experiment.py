"""Registered six-pair screening of train-only ground-mask augmentation.

This uses the production E64 training recipe/nomonth parent and is a screen,
not full-current-chain CV. Synthetic held-out sensor loss is reported separately
from naturally uncovered flights. Never writes ranking or production artifacts.
"""
from __future__ import annotations
import argparse
import calendar
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import time

os.environ.setdefault('POLARS_MAX_THREADS', '6')
os.environ.setdefault('OMP_NUM_THREADS', '6')
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
os.environ.setdefault('GOMP_SPINCOUNT', '0')
import numpy as np
import polars as pl
import visit_airborne_audit as A

ID, Y = 'MVT_ID_mvt', 'TAXITIME_SEC_mvt'
PROTOCOL = 'LIRF-ground-mask-train-augmentation-screen-v1'
MONTHS = [f'2025-{m:02}' for m in range(1,13)]
DAYS = [f'{m}-{d:02}' for m in MONTHS for d in range(1,calendar.monthrange(2025,int(m[5:]))[1]+1)]
# Upstream sensor archive has no Dec31 files on any machine. Keep all 4715
# departures with absent observations, exactly as the production left join does.
# They neither fit E64 nor receive a correction; never impute a physical track.
MISSING_SOURCE_DAYS={'2025-12-31':4715}
SEEDS = [42,1,2,3,4]
AIRPORTS = ['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']
FEATS = ['ap','coverage','D','p_nonreg','base','has_nm'] + A.ACTIVE
CODE = ['scripts/visit_mask_experiment.py','scripts/visit_airborne_audit.py',
        *[f'src/{n}' for n in {**A.SOURCES, **A.CONFIG_SOURCES}]]
PARAMS = dict(objective='regression',learning_rate=.03,num_leaves=31,min_data_in_leaf=200,
              feature_fraction=.9,bagging_fraction=.8,bagging_freq=1,lambda_l2=10.,
              verbose=-1,deterministic=True,force_row_wise=True,num_threads=6)


def synthetic_mask(d):
    changed = np.zeros(d.height,dtype=bool)
    for name in A.ACTIVE:
        old = d['base_'+name].cast(pl.Float64).to_numpy()
        new = d['air_'+name].cast(pl.Float64).to_numpy()
        changed |= ~A.same(old,new)
    observable = d.select(((pl.col('ADEP_mvt')=='LIRF') & pl.col('has_nm') &
        (pl.col('coverage')>=2) & (pl.col('air_vi_has_prev')==1) &
        pl.col('base_icao').is_not_null() & (pl.col('base_icao')==pl.col('air_icao'))).fill_null(False)).to_series().to_numpy()
    return observable & changed


def synthetic_features(d):
    # Canonical offblock extractor: callsign/ICAO exists but no surface -> 1.
    # Eligibility guarantees a surviving known airframe and previous visit.
    return d.with_columns(pl.lit(1).cast(pl.Int64).alias('coverage'),
                          *[pl.col('air_'+c).alias(c) for c in A.ACTIVE])


def missing_observations(meta):
    fields=[pl.lit(None,dtype=pl.Int64).alias('coverage'),pl.lit(False).alias('sensor_available')]
    for prefix in ['base_','air_']:
        fields.append(pl.lit(None,dtype=pl.String).alias(prefix+'icao'))
        for c in A.V.NAMES:
            dtype=pl.Int64 if c in ('vi_n_pts','vi_prev_pts','vi_has_prev') else pl.Float64
            fields.append(pl.lit(0 if c=='vi_has_prev' else None,dtype=dtype).alias(prefix+c))
    return meta.with_columns(*fields)


def split_frames(natural, audit, a):
    if a not in range(1,7): raise ValueError('invalid month pair')
    fit = natural.filter(~pl.col('month').is_in([a,a+6]))
    extra = audit.filter(pl.Series(synthetic_mask(audit)) & ~pl.col('month').is_in([a,a+6]))
    test = natural.filter(pl.col('month').is_in([a,a+6]))
    if set(fit[ID]) & set(test[ID]) or set(extra[ID]) & set(test[ID]):
        raise RuntimeError('held-out ID entered fit')
    return fit,extra,test


def metrics(y,control,candidate):
    y,c,n = [np.asarray(v,dtype=float) for v in (y,control,candidate)]
    if not len(y) or not all(np.isfinite(v).all() for v in (y,c,n)):
        raise RuntimeError('empty or nonfinite comparison')
    ec,en = (c-y)**2,(n-y)**2
    body = y<=7200
    # Symmetric trim of largest benefits and harms, same row set for both arms.
    delta = en-ec; order = np.argsort(delta,kind='stable')
    k = min(3,(len(y)-1)//2); keep = order[k:len(y)-k] if k else order
    rmse = lambda v: float(np.sqrt(np.mean(v)))
    return dict(n=len(y),control_rmse=rmse(ec),candidate_rmse=rmse(en),
        delta_rmse=rmse(en)-rmse(ec),delta_sse=float(delta.sum()),
        body_delta_rmse=rmse(en[body])-rmse(ec[body]) if body.any() else None,
        trimmed_delta_rmse=rmse(en[keep])-rmse(ec[keep]),trimmed_n=len(keep),
        trimmed_control_sse=float(ec[keep].sum()),trimmed_candidate_sse=float(en[keep].sum()))


def aggregate_metrics(rows):
    total=metrics(*[np.concatenate([r[i] for r in rows]) for i in range(3)])
    fold=[metrics(*r) for r in rows]
    n=sum(m['trimmed_n'] for m in fold)
    cs=sum(m['trimmed_control_sse'] for m in fold);ns=sum(m['trimmed_candidate_sse'] for m in fold)
    total.update(trimmed_n=n,trimmed_control_sse=cs,trimmed_candidate_sse=ns,
                 trimmed_delta_rmse=float(np.sqrt(ns/n)-np.sqrt(cs/n)))
    return total


def prepare(root,bundle,registration):
    A.sources(root)
    bundle.mkdir(parents=True,exist_ok=False)
    (bundle/'departures').mkdir()
    inputs = {name:A.sha(root/name) for name in CODE}
    missing=[]
    for day in DAYS:
        for directory,prefix in [('adsblol','surface'),('adsblol','visit'),('adsblol_kt1','offblock2')]:
            name = f'data/{directory}/{prefix}_{day}.parquet'
            if (root/name).exists():inputs[name] = A.sha(root/name)
            else:missing.append(name)
    expected_missing=[f'data/{directory}/{prefix}_{day}.parquet' for day in MISSING_SOURCE_DAYS
                      for directory,prefix in [('adsblol','surface'),('adsblol','visit'),('adsblol_kt1','offblock2')]]
    if set(missing)!=set(expected_missing):raise RuntimeError(f'unregistered missing-source set: {missing}')
    training=[]
    for month in MONTHS:
        files=sorted((root/'data/prc-2026-datasets').glob(f'training_{month}-01_*.parquet'))
        if len(files)!=1: raise RuntimeError(f'raw month missing: {month}')
        path=files[0];inputs[str(path.relative_to(root))]=A.sha(path)
        raw=pl.read_parquet(path,columns=list(dict.fromkeys(A.DEP_COLS+['SCHED_TIME_UTC_mvt',Y]))).filter(pl.col('PHASE_mvt')=='DEP')
        for day,n in MISSING_SOURCE_DAYS.items():
            if day.startswith(month) and raw.filter(pl.col('MVT_TIME_UTC_mvt').dt.strftime('%Y-%m-%d')==day).height!=n:
                raise RuntimeError('missing-source departure count drift')
        raw.select(A.DEP_COLS).write_parquet(bundle/'departures'/f'{month}.parquet')
        training.append(raw.select(ID,Y,'ADEP_mvt',
            pl.col('MVT_TIME_UTC_mvt').dt.month().alias('month'),
            pl.col('FLIGHT_ID_mvt').is_not_null().alias('has_nm'),
            (pl.col('MVT_TIME_UTC_mvt').dt.epoch('s')-pl.col('SCHED_TIME_UTC_mvt').dt.epoch('s')).cast(pl.Float64).alias('D')))
    bases=[]
    for a in range(1,7):
        path=root/f'data/cv_preds_nomonth/fold_{a}_{a+6}.npz'
        inputs[str(path.relative_to(root))]=A.sha(path)
        with np.load(path,allow_pickle=False) as z:
            bases.append(pl.DataFrame({ID:z['mvt_id'],'base':z['pred'],'p_nonreg':1-z['proba'][:,0], 'parent_y':z['y']}))
    raw=pl.concat(training);base=pl.concat(bases)
    if raw[ID].n_unique()!=raw.height or base[ID].n_unique()!=base.height: raise RuntimeError('duplicate IDs')
    raw=raw.join(base,on=ID,how='inner',validate='1:1')
    if raw.height!=base.height or (raw[Y]!=raw['parent_y']).any(): raise RuntimeError('parent y/population mismatch')
    raw.drop('parent_y').with_columns(pl.col('ADEP_mvt').replace_strict(dict(zip(AIRPORTS,range(10))),return_dtype=pl.Int32).alias('ap')).write_parquet(bundle/'training.parquet')
    files={str(p.relative_to(root)):A.sha(p) for p in bundle.rglob('*.parquet')}
    for name,h in inputs.items():
        if A.sha(root/name)!=h: raise RuntimeError(f'input changed while packing: {name}')
    A.save(bundle/'manifest.json',dict(protocol=PROTOCOL,registration_commit=registration,days=DAYS,
        input_sha256=inputs,bundle_sha256=files,features=FEATS,seeds=SEEDS,synthetic_weight=.25,
        missing_source_days=MISSING_SOURCE_DAYS))
    # Raw data and parent OOF are packed into smaller projections; do not transfer them again.
    transfer=[n for n in inputs if n.startswith(('scripts/','src/','data/adsblol/','data/adsblol_kt1/'))]
    transfer+=list(files)+[str((bundle/'manifest.json').relative_to(root))]
    (bundle/'transfer.txt').write_text('\n'.join(transfer)+'\n')
    print(json.dumps(dict(prepared=str(bundle),rows=raw.height,files=len(transfer),manifest_sha=A.sha(bundle/'manifest.json'))),flush=True)


def verify(root,bundle,registration):
    m=A.read(bundle/'manifest.json')
    if (m['protocol']!=PROTOCOL or m['registration_commit']!=registration or m['days']!=DAYS
        or m['missing_source_days']!=MISSING_SOURCE_DAYS):
        raise RuntimeError('registration/calendar drift')
    selected={n:h for n,h in m['input_sha256'].items() if n.startswith(('scripts/','src/','data/adsblol/','data/adsblol_kt1/'))}
    for n,h in {**selected,**m['bundle_sha256']}.items():
        if A.sha(root/n)!=h: raise RuntimeError(f'input hash drift: {n}')
    A.sources(root)
    return m


def extract(root,bundle,out,registration):
    m=verify(root,bundle,registration);out.mkdir(parents=True,exist_ok=False);(out/'days').mkdir()
    counts=[];synthetic_total=0
    for month in MONTHS:
        dep=pl.read_parquet(bundle/'departures'/f'{month}.parquet').with_columns(
            cs=pl.col('CALLSIGN_flt').fill_null(pl.col('FLIGHT_mvt')).str.replace_all(' ',''),
            to=pl.col('MVT_TIME_UTC_mvt').dt.epoch('s').cast(pl.Float64))
        month_n=0
        for day in [d for d in DAYS if d.startswith(month)]:
            d=dep.filter(pl.col('MVT_TIME_UTC_mvt').dt.strftime('%Y-%m-%d')==day)
            if day in MISSING_SOURCE_DAYS:
                if d.height!=MISSING_SOURCE_DAYS[day]:raise RuntimeError('missing-source population changed')
                f=missing_observations(d.select(ID,'ADEP_mvt',pl.col('FLIGHT_ID_mvt').is_not_null().alias('has_nm'),
                                                pl.lit(int(month[5:])).alias('month')))
                f.write_parquet(out/'days'/f'{day}.parquet')
                counts.append(dict(day=day,rows=f.height,synthetic=0,parity_mismatched_cells=None,sensor_available=False))
                continue
            pts=pl.read_parquet(root/f'data/adsblol/surface_{day}.parquet',columns=A.POINT_COLS)
            control,oldid=A.extract(pts,d);air,newid=A.extract(A.airborne(pts),d)
            mismatch=A.compare_control(control,pl.read_parquet(root/f'data/adsblol/visit_{day}.parquet'))
            if sum(mismatch.values()):
                A.save(out/'failure.json',dict(day=day,mismatch=mismatch));raise RuntimeError(f'control parity failed {day}')
            cov=pl.read_parquet(root/f'data/adsblol_kt1/offblock2_{day}.parquet',columns=[ID,'coverage'])
            if cov.height!=d.height or not np.array_equal(np.sort(cov[ID]),np.sort(d[ID])): raise RuntimeError('coverage population mismatch')
            f=(d.select(ID,'ADEP_mvt',pl.col('FLIGHT_ID_mvt').is_not_null().alias('has_nm'))
               .join(cov,on=ID,validate='1:1').join(control.rename({c:'base_'+c for c in A.V.NAMES}),on=ID,validate='1:1')
               .join(air.rename({c:'air_'+c for c in A.V.NAMES}),on=ID,validate='1:1'))
            f=f.with_columns(pl.lit(True).alias('sensor_available'),pl.Series('base_icao',[oldid[i] for i in f[ID]],dtype=pl.String),
                pl.Series('air_icao',[newid[i] for i in f[ID]],dtype=pl.String),pl.lit(int(month[5:])).alias('month'))
            switched=f.filter(pl.col('base_icao').is_not_null() & pl.col('air_icao').is_not_null() & (pl.col('base_icao')!=pl.col('air_icao'))).height
            if switched: raise RuntimeError('previously unique aircraft identity switched')
            n=int(synthetic_mask(f).sum());month_n+=n
            f.write_parquet(out/'days'/f'{day}.parquet')
            counts.append(dict(day=day,rows=f.height,synthetic=n,parity_mismatched_cells=0,sensor_available=True))
        synthetic_total+=month_n
        print(json.dumps(dict(month=month,synthetic=month_n,completed_days=len(counts))),flush=True)
    verify(root,bundle,registration)
    A.save(out/'extraction.json',dict(protocol=PROTOCOL,registration_commit=registration,days=counts,
        synthetic_total=synthetic_total,manifest_sha256=A.sha(bundle/'manifest.json'),
        output_sha256={str(p.relative_to(out)):A.sha(p) for p in (out/'days').glob('*.parquet')}))


def screen(root,bundle,out,registration):
    verify(root,bundle,registration)
    report=A.read(out/'extraction.json')
    if report['registration_commit']!=registration or len(report['days'])!=365: raise RuntimeError('incomplete extraction')
    for n,h in report['output_sha256'].items():
        if A.sha(out/n)!=h: raise RuntimeError('extraction changed')
    import lightgbm as lgb
    raw=pl.read_parquet(bundle/'training.parquet')
    frames=[pl.read_parquet(out/'days'/f'{d}.parquet') for d in DAYS]
    audit=pl.concat([f.select(frames[0].columns) for f in frames])
    # Only one source for each metadata field; identical IDs are checked by 1:1 join.
    d=raw.join(audit.drop('month','ADEP_mvt','has_nm'),on=ID,validate='1:1')
    if d.height!=raw.height: raise RuntimeError('screen join lost IDs')
    d=d.with_columns(*[pl.col('base_'+c).alias(c) for c in A.ACTIVE],pl.col('coverage').fill_null(-1))
    natural=((pl.col('coverage')<2)&(pl.col('vi_has_prev')==1)).fill_null(False)
    target=((pl.col('ADEP_mvt')=='LIRF')&pl.col('has_nm'))
    results=[];all_rows={'natural':[], 'synthetic':[]};(out/'screen').mkdir(exist_ok=False)
    for a in range(1,7):
        start=time.monotonic();fit,extra,test=split_frames(d,d,a)
        train=fit.filter(natural);ex=synthetic_features(extra)
        X=train.select(FEATS).to_numpy();y=np.clip((train[Y]-train['base']).to_numpy(),-1800,1800)
        Xa=np.concatenate([X,ex.select(FEATS).to_numpy()]);ya=np.concatenate([y,np.clip((ex[Y]-ex['base']).to_numpy(),-1800,1800)])
        weights=np.r_[np.ones(len(y)),np.full(ex.height,.25)]
        pools={'natural':test.filter(natural&target),'synthetic':synthetic_features(test.filter(pl.Series(synthetic_mask(test))))}
        predictions={k:[[],[]] for k in pools}
        for seed in SEEDS:
            for arm,xf,yf,w in [(0,X,y,None),(1,Xa,ya,weights)]:
                model=lgb.train({**PARAMS,'seed':seed},lgb.Dataset(xf,yf,weight=w,categorical_feature=[0]),num_boost_round=400)
                for name,pool in pools.items(): predictions[name][arm].append(model.predict(pool.select(FEATS).to_numpy(),num_threads=6))
                model.save_model(str(out/'screen'/f'fold_{a}_{a+6}_arm{arm}_seed{seed}.txt'))
        entry=dict(fold=a,fit_rows=train.height,synthetic_fit_rows=ex.height)
        for name,pool in pools.items():
            base=pool['base'].to_numpy();truth=pool[Y].to_numpy()
            ctrl=np.maximum(base+np.mean(predictions[name][0],axis=0),0)
            cand=np.maximum(base+np.mean(predictions[name][1],axis=0),0)
            entry[name]=metrics(truth,ctrl,cand)
            all_rows[name].append((truth,ctrl,cand))
            np.savez_compressed(out/'screen'/f'{name}_{a}_{a+6}.npz',mvt_id=pool[ID].to_numpy(),y=truth,control=ctrl,candidate=cand)
        entry['seconds']=time.monotonic()-start;results.append(entry);print(json.dumps(entry),flush=True)
    totals={name:aggregate_metrics(rows) for name,rows in all_rows.items()}
    gates={name:sum(r[name]['delta_rmse']<0 for r in results)>=4 and
        totals[name]['delta_rmse']<0 and totals[name]['body_delta_rmse']<0 and totals[name]['trimmed_delta_rmse']<0 for name in totals}
    A.save(out/'screen_result.json',dict(protocol=PROTOCOL,registration_commit=registration,folds=results,totals=totals,gates=gates,
        proceed_to_current_chain=all(gates.values()),accepted_for_submission=False,
        limitation='Production-training-parent nomonth stage screen; synthetic sensor stress is separate from natural OOF. Neither is full v12061 CV.',
        output_sha256={str(p.relative_to(out)):A.sha(p) for p in (out/'screen').glob('*')}))
    print(json.dumps(dict(screen_gates=gates,accepted_for_submission=False)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run'])
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--registration-commit',required=True);a=p.parse_args()
    if not re.fullmatch('[a-f0-9]{40}',a.registration_commit):p.error('full registration commit required')
    root=a.root.resolve();bundle=(root/a.bundle).resolve();out=(root/a.output).resolve() if a.output else None
    if a.command=='run' and out is None:p.error('--output required')
    lockpath=(out or bundle).parent;lockpath.mkdir(parents=True,exist_ok=True)
    with (lockpath/'.visit_mask_experiment.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.command=='prepare':prepare(root,bundle,a.registration_commit)
        else:extract(root,bundle,out,a.registration_commit);screen(root,bundle,out,a.registration_commit)


if __name__=='__main__':main()

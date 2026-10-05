"""Label-free audit of observable off-block brackets for auxiliary supervision.

Candidate sensor measurements, not ground-truth labels or a quality gain.
No target, block time, prediction or model enters the reliability selector.
"""
from __future__ import annotations
import argparse
import calendar
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys
import time

os.environ['POLARS_MAX_THREADS']='2'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR','load_as_storage')
import numpy as np
import polars as pl

PROTOCOL='E191-observable-offblock-bracket-label-free-365days-v2-missing-flags'
ID='MVT_ID_mvt'
DAYS=[f'2025-{m:02}-{d:02}' for m in range(1,13) for d in range(1,calendar.monthrange(2025,m)[1]+1)]
FIELDS=['icao','airport','callsign','t','lat','lon','gs','on_ground','stale']


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def observe(t,lat,lon,gs,ground,stale,takeoff,stand,*,callsigns=None,expected_callsign=None):
    out=dict(reason='unknown_stand',q_mid=None,bracket_s=None,dwell_s=None,return_to_stand=False,n_segment=0)
    if stand is None or not np.isfinite(stand).all():return out
    t,lat,lon,gs=[np.asarray(x,dtype=float) for x in (t,lat,lon,gs)]
    ground,stale=np.asarray(ground,dtype=bool),np.asarray(stale,dtype=bool)
    if (not all(x.shape==t.shape for x in (lat,lon,gs,ground,stale)) or np.any(np.diff(t)<0)
        or not np.isfinite(takeoff)):raise ValueError('malformed time-sorted track')
    use=(t>=takeoff-14400)&(t<=takeoff)
    cs=np.asarray(callsigns,dtype=object)[use] if callsigns is not None else None
    t,lat,lon,gs,ground,stale=[x[use] for x in (t,lat,lon,gs,ground,stale)]
    good=ground & ~stale & np.isfinite(t)&np.isfinite(lat)&np.isfinite(lon)&np.isfinite(gs)
    inds=np.flatnonzero(good)
    if not len(inds) or takeoff-t[inds[-1]]>180:out['reason']='no_recent_ground';return out
    end=inds[-1];start=end
    while start>0 and good[start-1] and t[start]-t[start-1]<=120:start-=1
    if cs is not None:
        observed={str(v).replace(' ','') for v in cs[start:end+1] if v is not None and str(v).replace(' ','')}
        expected=str(expected_callsign).replace(' ','') if expected_callsign is not None else None
        if expected not in observed:out['reason']='no_segment_callsign';return out
        if observed!={expected}:out['reason']='conflicting_segment_callsign';return out
    t,lat,lon,gs=[x[start:end+1] for x in (t,lat,lon,gs)]
    out['n_segment']=len(t)
    if np.any(np.diff(t)<=0):out['reason']='duplicate_segment_timestamps';return out
    distance=np.hypot((lat-stand[0])*111200,(lon-stand[1])*111200*np.cos(np.radians(stand[0])))
    moving=np.flatnonzero(gs>1)
    if not len(moving):out['reason']='no_movement';return out
    k=int(moving[0])
    if k<3 or t[k-1]-t[0]<90 or np.any(distance[:k]>100):out['reason']='no_initial_stand_dwell';return out
    out['dwell_s']=float(t[k-1]-t[0]);out['bracket_s']=float(t[k]-t[k-1])
    if out['bracket_s']>30:out['reason']='wide_bracket';return out
    if k+2>=len(t) or not np.all(gs[k:k+3]>1) or t[k+2]-t[k]<10:out['reason']='movement_not_sustained';return out
    exits=np.flatnonzero((np.arange(len(t))>=k)&(distance>150)&(t-t[k]<=240))
    if not len(exits):out['reason']='no_observed_stand_exit';return out
    leave=int(exits[0]);out['return_to_stand']=bool(np.any((distance[leave+1:]<=100)&(gs[leave+1:]<=1)))
    out['q_mid']=float(takeoff-.5*(t[k-1]+t[k]))
    out['reason']='return_to_stand' if out['return_to_stand'] else 'eligible'
    return out


def pack(root,bundle,registration):
    if bundle.exists():raise ValueError('fresh input path required')
    sys.path.insert(0,str(root/'src'));import geometry as G
    geom=[root/'analysis/sources/samples/osm_overpass'/f'{ap}.json' for ap in G.AIRPORTS]
    fp=root/'data/features_fin1b.parquet';upstream={str(p.relative_to(root)):sha(p) for p in [fp,*geom,root/'src/geometry.py']}
    stands,_=G.load_tables();standmap={(a,k):(la,lo) for a,k,la,lo in stands.iter_rows()}
    # Only observable columns are read. No suffix fallback to a possibly different stand.
    meta=pl.read_parquet(fp,columns=[ID,'split','ADEP_mvt','STAND_mvt']).filter(pl.col('split')=='train')
    coords=[standmap.get((ap,G.norm_stand(s)),(None,None)) if s is not None else (None,None) for ap,s in meta.select('ADEP_mvt','STAND_mvt').iter_rows()]
    meta=meta.select(ID).with_columns(pl.Series('stand_lat',[p[0] for p in coords],dtype=pl.Float64),pl.Series('stand_lon',[p[1] for p in coords],dtype=pl.Float64))
    if meta.height!=2085047 or meta[ID].n_unique()!=meta.height:raise ValueError('metadata IDs differ')
    bundle.mkdir(parents=True);inputs={};n=0
    for month in range(1,13):
        path=root/f'data/visit_mask_inputs_20261005_v2/departures/2025-{month:02}.parquet'
        upstream[str(path.relative_to(root))]=sha(path)
        d=pl.read_parquet(path).join(meta,on=ID,validate='1:1')
        if d.filter(pl.col('MVT_TIME_UTC_mvt').dt.month()!=month).height:raise ValueError('departure UTC partition drift')
        d=d.select(ID,'ADEP_mvt','CALLSIGN_flt','FLIGHT_mvt','MVT_TIME_UTC_mvt',pl.col('FLIGHT_ID_mvt').is_not_null().alias('has_nm'),'stand_lat','stand_lon')
        target=bundle/f'2025-{month:02}.parquet';d.write_parquet(target);inputs[str(target.relative_to(root))]=sha(target);n+=d.height
    if n!=2085047:raise ValueError('packed population differs')
    flagless=[]
    for day in DAYS:
        p=root/f'data/adsblol/surface_{day}.parquet'
        if not p.exists():
            if day!='2025-12-31':raise ValueError(f'unregistered sensor gap {day}')
        else:
            inputs[str(p.relative_to(root))]=sha(p)
            schema=pl.read_parquet_schema(p)
            if set(FIELDS)-set(schema)-{'stale'}:raise ValueError(f'unexpected surface schema {day}')
            if 'stale' not in schema:flagless.append(day)
    for p,h in upstream.items():
        if sha(root/p)!=h:raise ValueError(f'upstream drift {p}')
    inputs['scripts/e191_physical_support.py']=sha(__file__)
    save(bundle/'manifest.json',dict(protocol=PROTOCOL,registration_commit=registration,upstream_sha256=upstream,
        runtime_input_sha256=inputs,days=DAYS,rows=n,worker_sha256=sha(__file__),
        missing_sensor_days={'2025-12-31':4715},missing_stale_flag_days=flagless,workers=3,polars_threads_per_worker=2,
        limitation='Observable consistency only; not validated physical truth, not model CV. No targets or predictions used. Days without the stale flag remain unknown, never fresh.'))
    print(f'PACKED {n} target-free departures, 364 sensor days',flush=True)


def day_worker(root,bundle,out,day,inputs):
    root,bundle,out=Path(root),Path(bundle),Path(out);start=time.monotonic()
    path=bundle/f'{day[:7]}.parquet';relative=str(path.relative_to(root))
    if sha(path)!=inputs[relative]:raise ValueError('departure projection drift')
    d=pl.read_parquet(path).filter(pl.col('MVT_TIME_UTC_mvt').dt.strftime('%Y-%m-%d')==day).with_columns(
        cs=pl.col('CALLSIGN_flt').fill_null(pl.col('FLIGHT_mvt')).str.replace_all(' ',''),
        to=pl.col('MVT_TIME_UTC_mvt').dt.epoch('s').cast(pl.Float64))
    surface=root/f'data/adsblol/surface_{day}.parquet';by={};cmap={};point_count=0;has_flag=False
    if surface.exists():
        if sha(surface)!=inputs[str(surface.relative_to(root))]:raise ValueError('surface SHA differs')
        schema=pl.read_parquet_schema(surface)
        if set(FIELDS)-set(schema)-{'stale'}:raise ValueError('unexpected surface schema')
        has_flag='stale' in schema
        if has_flag:
            pts=pl.read_parquet(surface,columns=FIELDS).sort('icao','airport','t');point_count=pts.height
            for cs,ap,icao in pts.filter(pl.col('callsign').is_not_null()).select('callsign','airport','icao').unique().iter_rows():cmap.setdefault((cs,ap),[]).append(icao)
            for (icao,ap),g in pts.group_by(['icao','airport'],maintain_order=True):
                by[(icao,ap)]=[g[c].to_numpy() if c not in ('on_ground','stale') else g[c].fill_null(c=='stale').to_numpy() for c in ['t','lat','lon','gs','on_ground','stale','callsign']]
        else:point_count=pl.scan_parquet(surface).select(pl.len()).collect().item()
    elif day!='2025-12-31' or d.height!=4715:raise ValueError('unexpected absent surface')
    rows=[]
    for mid,ap,cs,to,nm,la,lo in d.select(ID,'ADEP_mvt','cs','to','has_nm','stand_lat','stand_lon').iter_rows():
        matches=cmap.get((cs,ap),[]);track=by.get((matches[0],ap)) if len(matches)==1 else None
        if track is None:
            reason='missing_sensor_day' if not surface.exists() else 'missing_stale_flag' if not has_flag else 'ambiguous_or_missing_identity'
            result=dict(reason=reason,q_mid=None,bracket_s=None,dwell_s=None,return_to_stand=False,n_segment=0)
        else:result=observe(*track[:6],to,(la,lo) if la is not None and lo is not None else None,callsigns=track[6],expected_callsign=cs)
        rows.append(dict(MVT_ID_mvt=mid,airport=ap,month=int(day[5:7]),has_nm=nm,**result))
    schema={ID:pl.Float64,'airport':pl.String,'month':pl.Int8,'has_nm':pl.Boolean,'reason':pl.String,'q_mid':pl.Float64,'bracket_s':pl.Float64,'dwell_s':pl.Float64,'return_to_stand':pl.Boolean,'n_segment':pl.Int64}
    f=pl.DataFrame(rows,schema=schema)
    if f.height!=d.height or f[ID].n_unique()!=d.height:raise ValueError('day ID coverage differs')
    dest=out/'days'/f'{day}.parquet';f.write_parquet(dest)
    if surface.exists() and sha(surface)!=inputs[str(surface.relative_to(root))]:raise ValueError('surface changed during extraction')
    return dict(day=day,rows=f.height,points=point_count,stale_flag_available=has_flag,eligible=f.filter(pl.col('reason')=='eligible').height,seconds=time.monotonic()-start,sha256=sha(dest))


def run(root,bundle,out,registration):
    m=json.loads((bundle/'manifest.json').read_text())
    if m['protocol']!=PROTOCOL or m['registration_commit']!=registration or m['worker_sha256']!=sha(__file__):raise ValueError('registration/source drift')
    if m['days']!=DAYS or m['rows']!=2085047:raise ValueError('incomplete input calendar')
    out.mkdir(parents=True,exist_ok=False);(out/'days').mkdir();records=[]
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        tasks=[pool.submit(day_worker,str(root),str(bundle),str(out),day,m['runtime_input_sha256']) for day in DAYS]
        for future in as_completed(tasks):
            r=future.result();records.append(r)
            if len(records)%10==0:print(json.dumps(dict(completed=len(records),last_day=r['day'],eligible_so_far=sum(x['eligible'] for x in records))),flush=True)
    d=pl.concat([pl.read_parquet(out/'days'/f'{day}.parquet') for day in DAYS])
    ids=pl.concat([pl.read_parquet(bundle/f'2025-{month:02}.parquet',columns=[ID]) for month in range(1,13)])[ID].sort()
    if d.height!=2085047 or d[ID].n_unique()!=d.height or not d[ID].sort().equals(ids):raise ValueError('full-year ID coverage differs')
    groups=d.group_by('airport','month','has_nm').agg(pl.len().alias('rows'),(pl.col('reason')=='eligible').sum().alias('eligible'),pl.col('return_to_stand').sum()).sort('airport','month','has_nm')
    eligible=d.filter(pl.col('reason')=='eligible');eligible.write_parquet(out/'eligible.parquet')
    stats=d.group_by('reason').len().sort('reason')
    save(out/'report.json',dict(protocol=PROTOCOL,registration_commit=registration,rows=d.height,days=sorted(records,key=lambda r:r['day']),
        groups=groups.to_dicts(),reasons=stats.to_dicts(),eligible=eligible.height,
        eligible_sha256=sha(out/'eligible.parquet'),input_manifest_sha256=sha(bundle/'manifest.json'),
        accepted_for_training=False,limitation=m['limitation']))
    print(json.dumps(dict(complete=True,rows=d.height,eligible=eligible.height,accepted_for_training=False)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['pack','run']);p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path);p.add_argument('--registration-commit',required=True);a=p.parse_args()
    if not re.fullmatch('[a-f0-9]{40}',a.registration_commit):p.error('full registration commit required')
    root=a.root.resolve();bundle=a.bundle.resolve()
    if a.action=='pack':pack(root,bundle,a.registration_commit)
    else:run(root,bundle,a.out.resolve(),a.registration_commit)


if __name__=='__main__':main()

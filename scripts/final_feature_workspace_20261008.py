"""Sealed, fresh final-period feature WORK. No network, ADS-B extraction or fit.

CLI actions require separate preregistration; stage/build require Linux cgroups.
Mac seal is bounded metadata. E106 is the existing inline prefix coalesce.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import resource
import shutil
import signal
import subprocess
import sys
import time

os.environ['POLARS_MAX_THREADS']='2'
os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR','load_as_storage')
import numpy as np
import polars as pl

PROTOCOL='final-fresh-FIN1b-E113-motion-20261008-v1'
SELF='scripts/final_feature_workspace_20261008.py'
DOC='analysis/final_feature_workspace_20261008/PROTOCOL.md'
TEST='tests/test_final_feature_workspace_20261008.py'
SOURCE_CONTRACT='analysis/final_feature_workspace_20261008/source_contract.json'
AP=('EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM')
ID,Y='MVT_ID_mvt','TAXITIME_SEC_mvt'
RAW='data/prc-2026-datasets/'
COUNTS=(2085047,670790)
SOURCES=tuple('src/'+n+'.py' for n in ('features','geometry','e28_forward_neighbours',
    'e77_filed_twins','e113_prefix_lookup','e163_build_features','e51_add_route_features',
    'e157_wake_features','e165_build_features','build_final_table','e185_motion_features'))
FINAL_OUTPUTS=('data/features_e106.parquet','data/features_e113.parquet',
               'data/features_fin1b.parquet','data/e185_motion_features.parquet')
STEPS=(
 ('features','features.py',(),{'FEATURES_OUT':'data/features_final.parquet'},'data/features_final.parquet'),
 ('e28','e28_forward_neighbours.py',('build',),{'E28_IN':'data/features_final.parquet','E28_OUT':'data/features_e28.parquet'},'data/features_e28.parquet'),
 ('e77','e77_filed_twins.py',(),{'FEATURES':'data/features_e28.parquet','FEATURES_OUT':'data/features_e77.parquet'},'data/features_e77.parquet'),
 ('e106',None,(),{},'data/features_e106.parquet'),
 ('e113','e113_prefix_lookup.py',(),{},'data/features_e113.parquet'),
 ('e163','e163_build_features.py',(),{},'data/features_e163.parquet'),
 ('e51','e51_add_route_features.py',(),{'FEATURES':'data/features_e106.parquet','FEATURES_OUT':'data/features_e106_e51.parquet'},'data/features_e106_e51.parquet'),
 ('e157','e157_wake_features.py',(),{},'data/features_e157.parquet'),
 ('e165','e165_build_features.py',(),{},'data/features_e165.parquet'),
 ('fin1b','build_final_table.py',('data/features_fin1b.parquet','data/features_e163.parquet',
    'data/features_e106_e51.parquet','data/features_e157.parquet','data/features_e165.parquet'),{},'data/features_fin1b.parquet'),
 ('motion','e185_motion_features.py',(),{},'data/e185_motion_features.parquet'))


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def relative(value):
    p=PurePosixPath(value)
    if (not value or p.is_absolute() or any(x in ('','.','..','.git') for x in value.split('/'))
            or '\\' in value or any(ord(c)<32 for c in value)):
        raise ValueError('unsafe manifest path: '+repr(value))
    return Path(value)


def schema_record(schema):return [[k,str(v)] for k,v in schema.items()]


def id_sha(values,sort=False):
    a=np.asarray(values,dtype='<f8')
    if a.ndim!=1 or not np.isfinite(a).all() or np.any(a!=np.floor(a)) or np.any(np.abs(a)>2**53):
        raise ValueError('IDs must be finite integral float64-safe values')
    return hashlib.sha256(np.sort(a).tobytes() if sort else a.tobytes()).hexdigest()


def label_sha(frame):
    a=frame.select(ID,Y).sort(ID).to_numpy().astype('<f8')
    a[np.isnan(a)]=np.nan
    return hashlib.sha256(a.tobytes()).hexdigest()


def challenge_contract(train,rank,template,expected_counts=COUNTS):
    if train.schema!=rank.schema:raise ValueError('raw training/final schema differs')
    if schema_record(template.schema)!=[[ID,'Float64'],[Y,'Int32']]:raise ValueError('template schema differs')
    if template[Y].is_not_null().any():raise ValueError('template unexpectedly contains labels')
    result={'id_hash_format':'sha256 of contiguous little-endian float64 bytes',
            'weather_days':{},'mvt_days':[]}
    parts=[]
    for label,frame,year,months,count in [('train',train,2025,list(range(1,13)),expected_counts[0]),
                                        ('rank',rank,2026,[1,2,6,7],expected_counts[1])]:
        d=frame.filter(pl.col('PHASE_mvt')=='DEP')
        if len(d)!=count or d[ID].n_unique()!=count:raise ValueError(label+' ID count/uniqueness differs')
        if d['MVT_TIME_UTC_mvt'].null_count() or d['MVT_TIME_UTC_mvt'].dt.year().unique().to_list()!=[year]:
            raise ValueError(label+' period/year differs')
        if sorted(d['MVT_TIME_UTC_mvt'].dt.month().unique().to_list())!=months:raise ValueError(label+' months differ')
        if set(d['ADEP_mvt'].unique().to_list())-set(AP):raise ValueError('unknown airport')
        if label=='rank' and any(d[c].is_not_null().any() for c in (Y,'BLOCK_TIME_UTC_mvt')):
            raise ValueError('final DEP label/BLOCK boundary violated')
        result[label+'_rows']=count
        result[label+'_sorted_id_sha256']=id_sha(d[ID],True)
        result[label+'_raw_order_id_sha256']=id_sha(d[ID])
        parts.append(d)
    if parts[0].select(ID).join(parts[1].select(ID),on=ID).height:raise ValueError('train/final ID overlap')
    if len(template)!=expected_counts[1] or template[ID].n_unique()!=len(template) or id_sha(template[ID],True)!=result['rank_sorted_id_sha256']:
        raise ValueError('template IDs differ from final DEP')
    result['template_order_id_sha256']=id_sha(template[ID])
    dep=pl.concat(parts)
    result['mvt_days']=sorted(dep['MVT_TIME_UTC_mvt'].dt.strftime('%Y-%m-%d').unique().to_list())
    for ap,g in dep.group_by('ADEP_mvt'):
        # All dates used by base weather, MVT wind and movement-only weather.
        dates=set(g['MVT_TIME_UTC_mvt'].dt.strftime('%Y-%m-%d').to_list())
        for expr in [pl.coalesce('AOBT_3_flt',pl.col('MVT_TIME_UTC_mvt')-pl.duration(minutes=15)),
                     pl.col('MVT_TIME_UTC_mvt')-pl.duration(minutes=15)]:
            dates.update(g.select(expr.dt.strftime('%Y-%m-%d')).to_series().drop_nulls().to_list())
        result['weather_days'][ap[0]]=sorted(dates)
    result['all_sorted_id_sha256']=id_sha(dep[ID],True)
    result['training_label_sha256']=label_sha(parts[0])
    return result


def weather_coverage(root,requirements):
    result={}
    for ap,days in sorted(requirements.items()):
        result[ap]={}
        for kind in ('metar','metar_wind'):
            p=root/'data'/kind/(ap+'.csv')
            if not p.is_file():raise FileNotFoundError(str(p))
            d=pl.read_csv(p,infer_schema_length=0,columns=['station','valid'])
            if set(d['station'].drop_nulls().unique())!={ap}:raise ValueError('weather station mismatch: '+str(p))
            available=set(d['valid'].str.slice(0,10).drop_nulls().to_list())
            missing=sorted(set(days)-available)
            result[ap][kind]={'required_days':len(days),'missing_days':missing}
            if missing:raise ValueError(f'missing {kind} coverage {ap}: {missing}')
    return result


def external_paths():
    return ([f'data/{kind}/{ap}.csv' for kind in ('metar','metar_wind') for ap in AP]
        +[f'analysis/sources/samples/osm_overpass/{ap}.json' for ap in AP]
        +[f'data/ext_pru/apt_dly_{y}.csv' for y in (2024,2025,2026)]
        +['data/taxi_route.parquet','data/ext/openflights_airlines.dat'])


def external_coverage(root,contract):
    missing=[p for p in external_paths() if not (root/p).is_file()]
    if missing:raise FileNotFoundError('missing external inputs: '+', '.join(missing))
    for kind,required in [('metar',{'station','valid','tmpc','dwpc','sknt','gust','vsby','skyl1','wxcodes'}),
                          ('metar_wind',{'station','valid','drct','sknt'})]:
        for ap in AP:
            p=root/f'data/{kind}/{ap}.csv'
            with p.open(newline='') as f:header=next(csv.reader(f),[])
            if not required<=set(header):raise ValueError('weather schema incomplete: '+str(p))
    route=pl.read_parquet_schema(root/'data/taxi_route.parquet')
    if not {'ADEP_mvt','stand_key','RUNWAY_mvt','route_m','route_detour'}<=set(route):raise ValueError('route schema incomplete')
    with (root/'data/ext/openflights_airlines.dat').open(newline='',encoding='utf8') as f:
        if not any(len(row)>7 and row[7]=='Y' and re.fullmatch('[A-Z0-9]{2}',row[3] or '') and
                   re.fullmatch('[A-Z]{3}',row[4] or '') for row in csv.reader(f)):
            raise ValueError('OpenFlights active IATA/ICAO mappings absent')
    weather=weather_coverage(root,contract['weather_days'])
    dates=set()
    for year in (2024,2025,2026):
        p=root/f'data/ext_pru/apt_dly_{year}.csv'
        d=pl.read_csv(p,infer_schema_length=0,encoding='utf8-lossy')
        required={'APT_ICAO','FLT_DATE','FLT_ARR_1','DLY_APT_ARR_1','FLT_ARR_1_DLY','DLY_APT_ARR_W_1','DLY_APT_ARR_C_1','DLY_APT_ARR_G_1'}
        if not required<=set(d.columns):raise ValueError('ATFM schema incomplete: '+str(p))
        dates.update(d['FLT_DATE'].str.slice(0,10).drop_nulls().to_list())
    missing=sorted(set(contract['mvt_days'])-dates)
    if missing:raise ValueError('missing ATFM calendar coverage: '+','.join(missing))
    for ap in AP:
        doc=json.loads((root/f'analysis/sources/samples/osm_overpass/{ap}.json').read_text())
        if not doc.get('elements'):raise ValueError('empty OSM elements: '+ap)
    return {'weather':weather,'atfm_missing_days':missing,
            'limitation':'Calendar admission is not per-flight METAR/ATFM/geometry matching coverage.'}


def file_record(path,destination,root):
    path=Path(path);root=Path(root)
    source=str(path.relative_to(root));relative(source);relative(destination)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('input must be an in-root regular file: '+str(path))
    return {'source':source,'path':destination,'bytes':path.stat().st_size,'sha256':sha(path)}


def verify_files(root,spec,staged=False):
    names=set();sources=set()
    for e in spec['files']:
        for key,seen in [('path',names),('source',sources)]:
            relative(e[key])
            if e[key] in seen:raise ValueError('duplicate manifest '+key)
            seen.add(e[key])
        p=root/e['path' if staged else 'source']
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root.resolve()):raise ValueError('unsafe/missing input: '+str(p))
        if p.stat().st_size!=e['bytes'] or sha(p)!=e['sha256']:raise ValueError('input drift: '+str(p))


def materialize(root,work,spec,disk_floor=4*1024**3):
    if work.exists():raise FileExistsError('fresh WORK required: '+str(work))
    verify_files(root,spec)
    if shutil.disk_usage(work.parent).free<disk_floor+sum(e['bytes'] for e in spec['files']):raise ValueError('insufficient staging disk')
    work.mkdir()
    # Copy, never hardlink: read-only modes must not change original inodes.
    for e in spec['files']:
        p=work/e['path'];p.parent.mkdir(parents=True,exist_ok=True)
        with (root/e['source']).open('rb') as src,p.open('xb') as dst:shutil.copyfileobj(src,dst,1024**2)
        p.chmod(0o444)
    verify_files(work,spec,True);verify_files(root,spec)


def e106(work):
    p=work/'data/features_e106.parquet'
    if p.exists():raise FileExistsError(str(p))
    f=pl.read_parquet(work/'data/features_e77.parquet')
    f.with_columns(airline_prefix=pl.coalesce(pl.col('CALLSIGN_flt').str.extract(r'^([A-Z]{3})',1),
         pl.col('FLIGHT_mvt').str.extract(r'^([A-Z]{3})\d',1))).write_parquet(p)


def environment():
    return {**{k:os.environ[k] for k in ('HOME','PATH','LANG','TMPDIR') if k in os.environ},
        'POLARS_MAX_THREADS':'6','OMP_NUM_THREADS':'6','OMP_THREAD_LIMIT':'6',
        'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OMP_WAIT_POLICY':'PASSIVE',
        'GOMP_SPINCOUNT':'0','PYTHONDONTWRITEBYTECODE':'1',
        'POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR':'load_as_storage','X_FEATS':'0','NO_AOBT':'0'}


def execute_steps(work,python,disk_floor=4*1024**3):
    for _,_,_,_,out in STEPS:
        if (work/out).exists():raise FileExistsError('no resume/overwrite: '+out)
    for i,(name,script,args,extra,out) in enumerate(STEPS):
        if shutil.disk_usage(work).free<disk_floor:raise ValueError('build disk floor violated')
        print('FEATURE_STAGE',i,name,flush=True)
        log=work/'logs'/f'{i:02d}_{name}.log'
        with log.open('x') as stream:
            if script is None:e106(work)
            else:subprocess.run([str(python),'-u','src/'+script,*args],cwd=work,
                    env={**environment(),**extra},stdin=subprocess.DEVNULL,stdout=stream,
                    stderr=subprocess.STDOUT,check=True)
        if not (work/out).is_file() or not (work/out).stat().st_size:raise ValueError('producer output absent: '+out)
        save(work/'logs'/f'{i:02d}_{name}.json',{'stage':name,'output':out,'sha256':sha(work/out)})


def check_output(path,expected_schema,contract):
    schema=pl.read_parquet_schema(path)
    if schema_record(schema)!=expected_schema:raise ValueError('output schema/order differs: '+str(path))
    cols=[c for c in (ID,'split',Y,'BLOCK_TIME_UTC_mvt') if c in schema]
    f=pl.read_parquet(path,columns=cols)
    if f[ID].n_unique()!=len(f) or len(f)!=contract['train_rows']+contract['rank_rows'] or id_sha(f[ID],True)!=contract['all_sorted_id_sha256']:
        raise ValueError('output IDs differ: '+str(path))
    result={'rows':len(f),'sha256':sha(path),'schema':schema_record(schema)}
    if 'split' in f:
        if set(f['split'].unique())!={'train','rank'}:raise ValueError('output split differs')
        for part in ('train','rank'):
            d=f.filter(pl.col('split')==part)
            if len(d)!=contract[part+'_rows'] or id_sha(d[ID],True)!=contract[part+'_sorted_id_sha256']:raise ValueError('output split IDs differ')
            result[part+'_rows']=len(d);result[part+'_ordered_id_sha256']=id_sha(d[ID])
            if part=='train' and Y in d and label_sha(d)!=contract['training_label_sha256']:
                raise ValueError('output training label values changed')
            if part=='rank' and any(d[c].is_not_null().any() for c in (Y,'BLOCK_TIME_UTC_mvt') if c in d):
                raise ValueError('output final label boundary violated')
    return result


def resource_guard(action):
    if sys.platform=='darwin' and action=='seal':
        # Bounded, projected metadata only. Peak RSS is checked at each phase;
        # macOS has no Linux-style memory.max, so this is not a kernel RAM cap.
        resource.setrlimit(resource.RLIMIT_CPU,(900,900))
        signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('metadata wall deadline')))
        signal.alarm(900)
        return {'host':'macOS','threads':2,'cpu_seconds_limit':900,'wall_alarm_seconds':900,
                'peak_rss_rejection_bytes':3*1024**3,'memory_limit_kind':'phase peak-RSS checks, not kernel cap'}
    if sys.platform!='linux':raise ValueError('CLI requires bounded Linux user unit; no unbounded Mac build')
    cg=next(l.split(':',2)[2].strip() for l in Path('/proc/self/cgroup').read_text().splitlines() if l.startswith('0:'))
    p=Path('/sys/fs/cgroup')/cg.lstrip('/')
    memory=(p/'memory.max').read_text().strip();swap=(p/'memory.swap.max').read_text().strip();q,period=(p/'cpu.max').read_text().split()
    cpus,ram=(6,24) if action=='build' else (2,3)
    if memory=='max' or int(memory)>ram*1024**3 or swap!='0' or q=='max' or int(q)>cpus*int(period):
        raise ValueError(f'{action} requires cgroup <= {cpus}CPU/{ram}GiB/swap0')
    return {'path':cg,'memory_max':int(memory),'cpu_max':[int(q),int(period)],'swap_max':0}


def metadata_checkpoint():
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    if peak>3*1024**3:raise ValueError('metadata peak RSS exceeded 3GiB; no manifest published')
    return peak


def registered(root,commit):
    if not re.fullmatch('[0-9a-f]{40}',commit or ''):raise ValueError('full preregistration commit required')
    journal=subprocess.check_output(['git','show',commit+':HYPOTHESES_LOG.md'],cwd=root)
    if PROTOCOL.encode() not in journal:raise ValueError('protocol absent from preregistration journal')
    for name in (*SOURCES,SELF,DOC,TEST,SOURCE_CONTRACT):
        raw=subprocess.check_output(['git','show',commit+':'+name],cwd=root)
        if hashlib.sha256(raw).hexdigest()!=sha(root/name):raise ValueError('registered source drift: '+name)


def seal(root,output,commit):
    registered(root,commit)
    if output.exists():raise FileExistsError(str(output))
    training=sorted((root/RAW).glob('training_*.parquet'))
    if len(training)!=12:raise ValueError('exactly 12 training month files required')
    rank=root/RAW/'final_ranking.parquet';template=root/RAW/'final_submitting.parquet'
    mapping=[(str(p.relative_to(root)),str(p.relative_to(root))) for p in training]
    mapping += [(RAW+'final_ranking.parquet',RAW+'ranking.parquet'),(RAW+'final_submitting.parquet',RAW+'submitting.parquet')]
    mapping += [(p,p) for p in (*SOURCES,SELF,DOC,TEST,SOURCE_CONTRACT,*external_paths())]
    files=[file_record(root/src,dst,root) for src,dst in mapping]
    metadata_checkpoint()
    schemas=[pl.read_parquet_schema(p) for p in [*training,rank]]
    if any(s!=schemas[0] for s in schemas):raise ValueError('raw input schemas differ')
    reference=json.loads((root/SOURCE_CONTRACT).read_text())
    if schema_record(schemas[0])!=reference['raw_schema']:raise ValueError('raw canonical schema differs')
    if reference['source_sha256']!={name:sha(root/name) for name in SOURCES}:raise ValueError('feature source contract drift')
    cols=[ID,'PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt','BLOCK_TIME_UTC_mvt',Y]
    train=pl.read_parquet(training,columns=cols);r=pl.read_parquet(rank,columns=cols)
    metadata_checkpoint()
    contract=challenge_contract(train,r,pl.read_parquet(template))
    del train,r
    metadata_checkpoint()
    coverage=external_coverage(root,contract)
    metadata_checkpoint()
    expected=reference['expected_output_schemas']
    spec={'protocol':PROTOCOL,'registration_commit':commit,'files':files,'contract':contract,
          'raw_schema':schema_record(schemas[0]),'expected_output_schemas':expected,
          'external_coverage':coverage,'steps':[x[0] for x in STEPS],
          'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'fits_authorized':0,
          'input_bytes':sum(e['bytes'] for e in files)}
    verify_files(root,spec);registered(root,commit)
    spec['metadata_peak_rss_bytes']=metadata_checkpoint()
    save(output,spec)


def checked_manifest(path,expected):
    if not re.fullmatch('[0-9a-f]{64}',expected or '') or sha(path)!=expected:raise ValueError('trusted manifest SHA required/mismatch')
    spec=json.loads(path.read_text())
    if spec['protocol']!=PROTOCOL or spec['steps']!=[x[0] for x in STEPS] or (spec['contract']['train_rows'],spec['contract']['rank_rows'])!=COUNTS:
        raise ValueError('manifest recipe differs')
    return spec


def completion_identity(spec,manifest_sha,outputs):
    if set(outputs)!=set(FINAL_OUTPUTS):raise ValueError('completion requires exact four output paths')
    fin=outputs['data/features_fin1b.parquet']
    return {'protocol':PROTOCOL,'manifest_sha256':manifest_sha,
        'registration_commit':spec['registration_commit'],'outputs':outputs,
        'feature_path':'data/features_fin1b.parquet','feature_sha256':fin['sha256'],
        'train_rows':fin['train_rows'],'rank_rows':fin['rank_rows'],
        'train_ordered_id_sha256':fin['train_ordered_id_sha256'],'rank_ordered_id_sha256':fin['rank_ordered_id_sha256'],
        'template_sha256':next(e['sha256'] for e in spec['files'] if e['path']==RAW+'submitting.parquet'),
        'source_sha256':{e['path']:e['sha256'] for e in spec['files'] if e['path'].endswith('.py')},'fits_performed':0}


def verify_completion(work,spec,manifest_sha):
    complete=json.loads((work/'feature_complete.json').read_text())
    if not isinstance(complete.get('outputs'),dict) or set(complete['outputs'])!=set(FINAL_OUTPUTS):
        raise ValueError('completion requires exact four output paths')
    outputs={name:check_output(work/name,spec['expected_output_schemas'][name],spec['contract']) for name in FINAL_OUTPUTS}
    expected=completion_identity(spec,manifest_sha,outputs)
    if set(complete)!=set(expected)|{'seconds'}:raise ValueError('completion fields differ')
    for name,value in expected.items():
        if complete[name]!=value:raise ValueError('completion identity/output drift: '+name)
    if not isinstance(complete['seconds'],(int,float)) or not np.isfinite(complete['seconds']) or complete['seconds']<0:
        raise ValueError('invalid completion elapsed seconds')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['seal','stage','build','verify'])
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--work',type=Path)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256')
    p.add_argument('--registration-commit');a=p.parse_args();root=a.root.resolve()
    guard=resource_guard(a.action)
    if a.action=='seal':seal(root,a.manifest.resolve(),a.registration_commit);return
    spec=checked_manifest(a.manifest.resolve(),a.manifest_sha256)
    if sha(__file__)!=next(e['sha256'] for e in spec['files'] if e['path']==SELF):raise ValueError('executing adapter source drift')
    if a.work is None:p.error('--work required')
    work=a.work.resolve()
    if a.action=='stage':
        materialize(root,work,spec);save(work/'manifest.json',spec);(work/'manifest.json').chmod(0o444)
        (work/'logs').mkdir();save(work/'stage_complete.json',{'manifest_sha256':sha(work/'manifest.json'),'cgroup':guard});return
    if sha(work/'manifest.json')!=a.manifest_sha256:raise ValueError('WORK manifest mismatch')
    verify_files(work,spec,True)
    if a.action=='build':
        if shutil.disk_usage(work).free<25*1024**3:raise ValueError('feature build requires 25GiB free; choose Home WORK')
        start=time.monotonic();save(work/'feature_started.json',{'manifest_sha256':a.manifest_sha256,'cgroup':guard})
        execute_steps(work,sys.executable)
        outputs={name:check_output(work/name,spec['expected_output_schemas'][name],spec['contract']) for name in FINAL_OUTPUTS}
        verify_files(work,spec,True)
        save(work/'feature_complete.json',{**completion_identity(spec,a.manifest_sha256,outputs),'seconds':time.monotonic()-start})
    else:verify_completion(work,spec,a.manifest_sha256)
    print('PASS',a.action,work,flush=True)


if __name__=='__main__':main()

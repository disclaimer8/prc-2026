"""Independent target-free CB join replay on ASUS; bounded Windows audit only."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
for name in ('POLARS_MAX_THREADS','OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='2'
import numpy as np
import polars as pl

PROTOCOL='E205-ASUS-independent-current-CB-join-metadata-audit-v1'
SELF='scripts/night_cb_asus_audit_20261007.py'
BASE='analysis/night_cb_asus_audit_20261007'
DOC=BASE+'/protocol.md'
LAUNCHER=BASE+'/run_asus.ps1'
HELPER='scripts/adsb_stage_audit_20261007.py'
SOURCE='scripts/night_cb_20261007.py'
AIRPORTS='EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF LTFM LSZH'.split()
ID='MVT_ID_mvt'
QUERIES=[ID,'split','month','ADEP_mvt','AOBT_3_flt','MVT_TIME_UTC_mvt','wx_thunder','metar_age_min']
NAMES=['cloud_cb','cloud_cb_missed','cloud_cb_base_ft','cloud_cb_amount']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path,value):
    with Path(path).open('x') as output:json.dump(value,output,indent=2,sort_keys=True,allow_nan=False)


def row_digest(ids,values):
    h=hashlib.sha256(np.asarray(ids,dtype='<f8').tobytes())
    for column in np.asarray(values,dtype='<f4').T:
        h.update(np.asarray(np.isnan(column),dtype=np.uint8).tobytes())
        h.update(np.asarray(np.nan_to_num(column,nan=0),dtype='<f4').tobytes())
    return h.hexdigest()


def coverage(frame,values):
    joined=frame.with_columns(*[pl.Series(name,values[:,i]) for i,name in enumerate(NAMES)])
    return joined.group_by('split','ADEP_mvt','month').agg(pl.len().alias('flights'),
        (pl.col('cloud_cb')==1).sum().alias('cb'),(pl.col('cloud_cb_missed')==1).sum().alias('cb_missed'),
        pl.col('cloud_cb').is_nan().sum().alias('cb_unknown')).sort('split','ADEP_mvt','month').to_dicts()


def registered(root,commit):
    if not re.fullmatch('[a-f0-9]{40}',commit or ''):raise ValueError('source commit required')
    for path in (SELF,DOC,LAUNCHER):
        raw=subprocess.check_output(['git','show',commit+':'+path],cwd=root)
        if hashlib.sha256(raw).hexdigest()!=sha(root/path):raise ValueError('committed audit source changed '+path)


def pack(root,bundle,commit):
    registered(root,commit)
    if bundle.exists():raise ValueError('fresh audit bundle required')
    import night_cb_20261007 as W
    reference=json.loads((root/'analysis/night_cb_20261007/input_receipt.json').read_text())
    for path,expected in reference['inputs'].items():
        if path.startswith('data/metar/') or path=='data/features_fin1b.parquet':
            if sha(root/path)!=expected:raise ValueError('original source changed '+path)
    frame,_=W.observable_frame(root);expected=frame.select(NAMES).to_numpy().astype(np.float32)
    if len(frame)!=2429888 or W.coverage(frame)!=reference['coverage']:raise ValueError('original expected coverage differs')
    bundle.mkdir(parents=True);(bundle/'metar').mkdir()
    frame.select(QUERIES).write_parquet(bundle/'queries.parquet')
    np.savez_compressed(bundle/'expected.npz',mvt_id=frame[ID].to_numpy(),values=expected)
    for ap in AIRPORTS:shutil.copy2(root/'data/metar'/f'{ap}.csv',bundle/'metar'/f'{ap}.csv')
    for path in (SELF,HELPER,DOC,LAUNCHER):shutil.copy2(root/path,bundle/Path(path).name)
    spec=dict(protocol=PROTOCOL,source_commit=commit,query_columns=QUERIES,rows=len(frame),
        expected_row_digest=row_digest(frame[ID].to_numpy(),expected),coverage=coverage(frame,expected),
        source_sha256={path:sha(root/path) for path in (SELF,HELPER,DOC,LAUNCHER,SOURCE)},
        source_features_sha256=reference['inputs']['data/features_fin1b.parquet'],
        files={str(path.relative_to(bundle)):sha(path) for path in bundle.rglob('*') if path.is_file()},
        target_free=True,no_fit=True)
    write(bundle/'manifest.json',dict(spec=spec,spec_sha256=digest(spec)))
    print(json.dumps(dict(manifest_sha256=sha(bundle/'manifest.json'),spec_sha256=digest(spec),rows=len(frame))),flush=True)


def check_bundle(bundle,manifest_sha,commit):
    if sha(bundle/'manifest.json')!=manifest_sha:raise ValueError('manifest pin differs')
    manifest=json.loads((bundle/'manifest.json').read_text());spec=manifest['spec']
    if spec['protocol']!=PROTOCOL or digest(spec)!=manifest['spec_sha256'] or spec['source_commit']!=commit or spec['query_columns']!=QUERIES:raise ValueError('protocol mismatch')
    for path,expected in spec['files'].items():
        pure=Path(path)
        if pure.is_absolute() or '..' in pure.parts or sha(bundle/path)!=expected:raise ValueError('portable file drift '+path)
    if sha(__file__)!=spec['source_sha256'][SELF]:raise ValueError('worker source mismatch')
    return manifest


def independent_join(frame,bundle,ac_check):
    anchor=frame.select(pl.col('AOBT_3_flt').fill_null(pl.col('MVT_TIME_UTC_mvt')-pl.duration(seconds=900)).dt.epoch('s')).to_series().to_numpy()
    if not np.isfinite(anchor).all():raise ValueError('invalid anchor')
    airport=frame['ADEP_mvt'].to_numpy();values=np.full((len(frame),4),np.nan,np.float32)
    thunder=np.full(len(frame),np.nan);age=np.full(len(frame),np.nan);reports=[]
    pattern=re.compile(r'\b(FEW|SCT|BKN|OVC)(\d{3})CB\b');amount={'FEW':1,'SCT':2,'BKN':3,'OVC':4}
    for ap in AIRPORTS:
        ac_check();table={};raw_count=0
        with (bundle/'metar'/f'{ap}.csv').open(newline='') as source:
            for row in csv.DictReader(source):
                if row['station']!=ap:raise ValueError('wrong station')
                timestamp=int(datetime.strptime(row['valid'],'%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc).timestamp())
                groups=pattern.findall(row.get('metar') or '');existing=float(bool(re.search('TS|CB',row.get('wxcodes') or '')))
                current=[np.nan]*4
                if row.get('metar'):
                    current=[float(bool(groups)),float(bool(groups) and not existing),np.nan,np.nan]
                    if groups:current[2:]=[float(min(int(height)*100 for _,height in groups)),float(max(amount[name] for name,_ in groups))]
                table[timestamp]=(current,existing);raw_count+=1
        timestamps=np.array(sorted(table),dtype=np.int64);selected=np.flatnonzero(airport==ap)
        indices=np.searchsorted(timestamps,anchor[selected],side='right')-1;safe=np.maximum(indices,0)
        valid=(indices>=0)&(anchor[selected]-timestamps[safe]>=0)&(anchor[selected]-timestamps[safe]<=10800)
        rows=selected[valid];chosen=indices[valid]
        raw_values=np.asarray([table[stamp][0] for stamp in timestamps],dtype=np.float32)
        raw_thunder=np.asarray([table[stamp][1] for stamp in timestamps],dtype=np.float64)
        values[rows]=raw_values[chosen];thunder[rows]=raw_thunder[chosen];age[rows]=(anchor[rows]-timestamps[chosen])//60
        reports.append(dict(airport=ap,queries=len(selected),missing_report=int((~valid).sum()),raw_reports=raw_count,duplicates=raw_count-len(table)))
        print('JOINED',ap,len(selected),flush=True)
    if not np.isin(airport,AIRPORTS).all() or not np.array_equal(thunder,frame['wx_thunder'].to_numpy(),equal_nan=True) or not np.array_equal(age,frame['metar_age_min'].to_numpy(),equal_nan=True):raise ValueError('old weather alignment differs')
    return values,reports


def run(bundle,output,manifest_sha,commit):
    if sys.platform!='win32':raise RuntimeError('Windows ASUS worker required')
    if output.exists():raise ValueError('fresh output required; no retries')
    import psutil
    from adsb_stage_audit_20261007 import windows_limit
    kernel,handle=windows_limit(6*1024**3);process=psutil.Process()
    process.cpu_affinity([0,2]);process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    def ac_check():
        battery=psutil.sensors_battery()
        if not battery or not battery.power_plugged:raise RuntimeError('ASUS AC power required')
    ac_check()
    if psutil.virtual_memory().available<3*1024**3:raise MemoryError('need3GiB available')
    output.mkdir(parents=True);started=time.monotonic()
    def timeout():
        write(output/'timeout.json',dict(protocol=PROTOCOL,timeout_seconds=300));os._exit(124)
    timer=threading.Timer(300,timeout);timer.daemon=True;timer.start()
    awake=kernel.SetThreadExecutionState(0x80000001)
    try:
        if not awake:raise RuntimeError('temporary awake request failed')
        manifest=check_bundle(bundle,manifest_sha,commit)
        write(output/'started.json',dict(pid=os.getpid(),protocol=PROTOCOL,source_commit=commit,
            affinity=process.cpu_affinity(),priority=process.nice(),memory_limit=6*1024**3,timeout_seconds=300))
        frame=pl.read_parquet(bundle/'queries.parquet')
        if frame.columns!=QUERIES or len(frame)!=2429888 or frame[ID].n_unique()!=len(frame):raise ValueError('query schema/coverage differs')
        with np.load(bundle/'expected.npz',allow_pickle=False) as expected:
            expected_ids=expected['mvt_id'];expected_values=expected['values']
        if not np.array_equal(expected_ids,frame[ID].to_numpy()):raise ValueError('expected ID order differs')
        values,reports=independent_join(frame,bundle,ac_check)
        if not np.array_equal(values,expected_values,equal_nan=True):raise ValueError('independent CB array differs')
        obtained=row_digest(expected_ids,values);counts=coverage(frame,values)
        if obtained!=manifest['spec']['expected_row_digest'] or counts!=manifest['spec']['coverage']:raise ValueError('digest/coverage differs')
        ac_check();check_bundle(bundle,manifest_sha,commit)
        write(output/'result.json',dict(protocol=PROTOCOL,source_commit=commit,input_manifest_sha256=manifest_sha,
            spec_sha256=manifest['spec_sha256'],rows=len(frame),row_digest=obtained,all_ids_exact=True,
            all_four_cb_columns_exact=True,current_thunder_and_age_exact=True,coverage=counts,airports=reports,
            seconds=time.monotonic()-started,peak_rss_bytes=process.memory_info().peak_wset,
            affinity=process.cpu_affinity(),memory_limit=6*1024**3,priority=process.nice(),
            runtime=dict(python=sys.version,numpy=np.__version__,polars=pl.__version__,platform=sys.platform),
            target_free=True,no_fit=True,passed=True))
        print('PASS',len(frame),obtained,flush=True)
    finally:
        timer.cancel();kernel.SetThreadExecutionState(0x80000000);kernel.CloseHandle(handle)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['pack','run'])
    parser.add_argument('--root',type=Path,default=Path('.'));parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path);parser.add_argument('--source-commit',required=True);parser.add_argument('--manifest-sha256')
    args=parser.parse_args()
    if args.command=='pack':pack(args.root.resolve(),args.bundle,args.source_commit)
    else:run(args.bundle,args.output,args.manifest_sha256,args.source_commit)


if __name__=='__main__':main()

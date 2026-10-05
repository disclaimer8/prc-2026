"""Recover freshness flags for six fixed pilot days into isolated artifacts.

No original surface file is replaced, and no recovered data is used by a model.
The frozen official preferred-release list determines the source replicas.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

os.environ['POLARS_MAX_THREADS'] = '2'
import polars as pl
import e191_physical_support as P

DAYS = [f'2025-{m:02}-01' for m in range(7,13)]
PROTOCOL = 'E191-freshness-recovery-six-fixed-days-v1'


def urls_for_day(text, day):
    tag = f'/v{day.replace("-", ".")}-planes-readsb-'
    lines = [line.strip() for line in text.splitlines() if tag in line]
    if len(lines) != 1:
        raise ValueError('one preferred replica per fixed day required')
    urls = lines[0].split(',')
    prefix = 'https://github.com/adsblol/globe_history_2025/releases/download/'
    if not urls or any(not u.startswith(prefix) or tag not in u or not re.fullmatch(r'https://[A-Za-z0-9_./-]+', u) for u in urls):
        raise ValueError('untrusted or mismatched archive URL')
    return urls


def point_parity(old, new):
    if 'stale' not in new.columns or 'stale' in old.columns:
        raise ValueError('expected old flagless and new flagged data')
    if set(new.columns) != set(old.columns) | {'stale'} or old.height != new.height:
        return False
    cols = old.columns
    return old.sort(cols, nulls_last=True).equals(new.select(cols).sort(cols, nulls_last=True))


def bounded_process(command, timeout):
    child=subprocess.Popen(command,start_new_session=True)
    try:
        code=child.wait(timeout=timeout)
        if code:raise subprocess.CalledProcessError(code,command)
    finally:
        # The downloader and extractor inherit this isolated process group.
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait()


def prepare(root, out, bundle, registration):
    out.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), root/'scripts/e191_physical_support.py', root/'src/adsb_extract.py',
             root/'data/adsblol/PREFERRED_2025.txt', bundle/'manifest.json']
    files += [root/f'data/adsblol/surface_{d}.parquet' for d in DAYS]
    files += [bundle/f'{d[:7]}.parquet' for d in DAYS]
    for d in DAYS:
        if 'stale' in pl.read_parquet_schema(root/f'data/adsblol/surface_{d}.parquet'):
            raise ValueError('pilot day already has flag')
    preferred=(root/'data/adsblol/PREFERRED_2025.txt').read_text()
    P.save(out/'manifest.json', dict(protocol=PROTOCOL, registration_commit=registration,
        files={str(p.relative_to(root)):P.sha(p) for p in files},
        bundle=str(bundle.relative_to(root)), days={d:urls_for_day(preferred,d) for d in DAYS},
        workers=3, limit_bytes_per_day=12*1024**3, accepted_for_training=False))


def one_day(root, out, m, day):
    start=time.monotonic(); dest=out/day; dest.mkdir(); surface=dest/'surface.parquet'
    for rel,h in m['files'].items():
        if P.sha(root/rel)!=h: raise ValueError(f'input drift: {rel}')
    archive_sha=hashlib.sha256(); total=0
    with (dest/'extraction.log').open('x') as log:
        env={**os.environ,'SURFACE_OUT':str(surface),'POLARS_MAX_THREADS':'2','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        child=subprocess.Popen([sys.executable,str(root/'src/adsb_extract.py'),day],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env)
        try:
            for url in m['days'][day]:
                print(f'{day}: streaming {url.rsplit("/",1)[-1]}',flush=True)
                with urllib.request.urlopen(url,timeout=120) as response:
                    while chunk:=response.read(4*1024**2):
                        total+=len(chunk)
                        if total>m['limit_bytes_per_day'] or time.monotonic()-start>7200:
                            raise ValueError('pilot byte/time budget exceeded')
                        archive_sha.update(chunk); child.stdin.write(chunk)
            child.stdin.close()
            if child.wait(timeout=600): raise RuntimeError('extractor failed; partial output retained')
        except BaseException:
            child.terminate()
            try: child.wait(timeout=20)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
            raise
    old_path=root/f'data/adsblol/surface_{day}.parquet'
    old=pl.read_parquet(old_path);new=pl.read_parquet(surface)
    parity=point_parity(old,new)
    record=dict(day=day,seconds=time.monotonic()-start,bytes=total,stream_sha256=archive_sha.hexdigest(),
        output_sha256=P.sha(surface),original_sha256=P.sha(old_path),rows=new.height,
        common_points_equal=parity,stale_nulls=new['stale'].null_count(),
        stale_true=int(new['stale'].fill_null(False).sum()),accepted_for_training=False)
    if parity:
        # An isolated view lets the unchanged support selector consume this day.
        audit=dest/'audit';(audit/'data/adsblol').mkdir(parents=True);(audit/'days').mkdir()
        (audit/'data/adsblol'/f'surface_{day}.parquet').symlink_to(surface)
        b=audit/'bundle';b.mkdir()
        dp=b/f'{day[:7]}.parquet';shutil.copyfile(root/m['bundle']/dp.name,dp)
        sp=audit/'data/adsblol'/f'surface_{day}.parquet'
        r=P.day_worker(audit,b,audit,day,{str(p.relative_to(audit)):P.sha(p) for p in (dp,sp)})
        record['support']=r
    for rel,h in m['files'].items():
        if P.sha(root/rel)!=h: raise ValueError(f'input changed during extraction: {rel}')
    P.save(dest/'report.json',record);print(json.dumps(record),flush=True)
    return record


def run(root,out,registration):
    m=json.loads((out/'manifest.json').read_text())
    if m['protocol']!=PROTOCOL or m['registration_commit']!=registration or list(m['days'])!=DAYS:
        raise ValueError('protocol drift')
    if any(p.name!='manifest.json' for p in out.iterdir()): raise ValueError('fresh prepared output required')
    records=[]
    def supervise(day):
        bounded_process([sys.executable,str(Path(__file__).resolve()),'day','--root',str(root),'--out',str(out),
            '--registration-commit',registration,'--day',day],timeout=7200)
        return json.loads((out/day/'report.json').read_text())
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(supervise,day):day for day in DAYS}
        for f in as_completed(futures):
            try: records.append(f.result())
            except Exception as exc:
                record=dict(day=futures[f],failed=True,error=str(exc),accepted_for_training=False)
                records.append(record);print(json.dumps(record),flush=True)
    P.save(out/'report.json',dict(protocol=PROTOCOL,registration_commit=registration,days=records,accepted_for_training=False))
    if any(r.get('failed') or not r.get('common_points_equal') for r in records):
        raise SystemExit('Pilot incomplete or point parity differs; no source promotion')
    print('COMPLETE: six days recovered; support audit only, no source promotion',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','day'])
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--out',type=Path,required=True)
    p.add_argument('--bundle',type=Path);p.add_argument('--registration-commit',required=True);p.add_argument('--day',choices=DAYS);a=p.parse_args()
    if not re.fullmatch('[a-f0-9]{40}',a.registration_commit):p.error('full registration commit required')
    if a.action=='prepare':prepare(a.root.resolve(),a.out.resolve(),a.bundle.resolve(),a.registration_commit)
    elif a.action=='day':
        m=json.loads((a.out/'manifest.json').read_text())
        if m['protocol']!=PROTOCOL or m['registration_commit']!=a.registration_commit:raise ValueError('protocol drift')
        one_day(a.root.resolve(),a.out.resolve(),m,a.day)
    else:run(a.root.resolve(),a.out.resolve(),a.registration_commit)


if __name__=='__main__':main()

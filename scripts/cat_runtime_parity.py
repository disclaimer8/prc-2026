"""Qualify one CatBoost fit on Hetzner; never alter or merge the Home queue."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
import cat_fin1b_oof as C

PROTOCOL='cat-fin1b-hetzner-six-thread-exact-parity-v1'
HOME_LINEAGE='9ef4316be06938cedd0f072895ddea1850ee1e4d8aff234f2c5541cdd7d4f2f0'
HOME_REG='55ba99180752bdd65e83b1abdf30d00dca69836b'
NAME='fold_1_7'


def compare(reference_ids,reference,ids,prediction):
    if not np.array_equal(ids,reference_ids) or reference.shape!=prediction.shape or ids.shape!=prediction.shape:
        raise ValueError('ID order or prediction shape differs')
    if not np.isfinite(reference).all() or not np.isfinite(prediction).all():raise ValueError('nonfinite prediction')
    return dict(rows=len(ids),exact=bool(np.array_equal(reference,prediction)),max_abs=float(np.max(np.abs(reference-prediction))))


def snapshot(path):
    m=C.read_json(path/'manifest.json')
    if m['lineage']!=HOME_LINEAGE or C.digest({k:v for k,v in m.items() if k!='lineage'})!=HOME_LINEAGE:
        raise ValueError('Home lineage differs')
    if m['registration']!=HOME_REG or m['source_sha256']!=C.SOURCES or m['environment']!=C.SCIENTIFIC:
        raise ValueError('Home protocol differs')
    if m['runner_sha256']!=C.sha(path/'runner.py') or m['runner_sha256']!=C.sha(C.__file__):
        raise ValueError('Home wrapper source differs')
    C.verify_sources(path/'sources')
    for name in ('features.parquet','metadata.npz'):
        if C.sha(path/'inputs'/NAME/name)!=m['inputs'][NAME][name]:raise ValueError('fold bytes changed')
    with np.load(path/'inputs'/NAME/'metadata.npz',allow_pickle=False) as z:ids=z['mvt_id']
    if len(ids)!=344419 or not C.integral_ids(ids) or len(np.unique(ids))!=len(ids):raise ValueError('heldout population differs')
    ref=C.verify_seed(path/NAME/'seed_00',ids,dict(seed=0,lineage=HOME_LINEAGE,fold=1))
    return m,ids,ref


def prepare(inp,out,registration):
    snapshot(inp);out.mkdir(parents=True,exist_ok=False)
    files=[inp/'manifest.json',inp/'runner.py',*sorted((inp/'sources').glob('*.py')),
        inp/'inputs'/NAME/'features.parquet',inp/'inputs'/NAME/'metadata.npz',
        inp/NAME/'seed_00'/'prediction.npz',inp/NAME/'seed_00'/'manifest.json']
    C.write_json(out/'manifest.json',dict(protocol=PROTOCOL,registration_commit=registration,
        home_lineage=HOME_LINEAGE,files={str(p.relative_to(inp)):C.sha(p) for p in files},
        worker_sha256=C.sha(__file__),helper_sha256=C.sha(C.__file__),fold=1,seed=0,iterations=5600,
        threshold='np.array_equal on all predictions; max_abs=0; no y metric',accepted_for_merge=False))


def resources():
    if set(os.sched_getaffinity(0))!={0,2,4,6,8,10} or os.getpriority(os.PRIO_PROCESS,0)<10:
        raise ValueError('Hetzner affinity/nice differs')
    cg=next(line.split('::',1)[1] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    path=Path('/sys/fs/cgroup')/cg.lstrip('/')
    quota,period=(path/'cpu.max').read_text().split()
    if quota=='max' or int(quota)>6*int(period):raise ValueError('CPU cgroup bound missing')
    if int((path/'memory.max').read_text())>26*1024**3 or (path/'memory.swap.max').read_text().strip()!='0':
        raise ValueError('memory/swap bound missing')
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    if available<28*1024**3:raise ValueError('insufficient available memory for shared production server')
    return dict(affinity=sorted(os.sched_getaffinity(0)),nice=os.getpriority(os.PRIO_PROCESS,0),memory_available=available,cgroup=cg,
        cpu_quota=int(quota),cpu_period=int(period),memory_max=int((path/'memory.max').read_text()),memory_swap_max=0)


def run(inp,out,registration):
    m=C.read_json(out/'manifest.json')
    if m['protocol']!=PROTOCOL or m['registration_commit']!=registration or m['worker_sha256']!=C.sha(__file__) or m['helper_sha256']!=C.sha(C.__file__):
        raise ValueError('registration/source drift')
    def verify():
        for name,h in m['files'].items():
            if C.sha(inp/name)!=h:raise ValueError(f'snapshot drift {name}')
        return snapshot(inp)
    home,ids,ref=verify();runtime=C.packages()
    if runtime['packages']!=home['runtime']['packages']:raise ValueError('scientific package versions differ')
    if any(p.name!='manifest.json' for p in out.iterdir()):raise ValueError('fresh prepared output required')
    limits=resources();attempt=out/'attempt';attempt.mkdir()
    env=C.child_environment(inp/'inputs'/NAME/'features.parquet',attempt)
    env.update(CATBOOST_THREADS='6',OMP_NUM_THREADS='6',OMP_THREAD_LIMIT='6')
    command=[sys.executable,str(inp/'sources/catboost_seeds.py'),'5600','0']
    C.write_json(out/'invocation.json',dict(runtime=runtime,home_runtime=home['runtime'],limits=limits,environment=env,command=command))
    start=time.monotonic()
    with (out/'fit.log').open('x') as log:
        try:child=subprocess.run(command,cwd=attempt,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=6*3600)
        except subprocess.TimeoutExpired:
            C.write_json(out/'exit.json',dict(timeout=True,seconds=time.monotonic()-start));raise
    C.write_json(out/'exit.json',dict(returncode=child.returncode,seconds=time.monotonic()-start))
    if child.returncode:raise ValueError(f'fit failed {child.returncode}; retained attempt')
    verify();pred=C.prediction(attempt/'raw_seed0.npz',ids)
    report=compare(ids,ref,ids,pred)
    report.update(protocol=PROTOCOL,registration_commit=registration,seconds=time.monotonic()-start,runtime=runtime,
        prediction_sha256=C.sha(attempt/'raw_seed0.npz'),accepted_for_merge=False,
        limitation='One input/runtime qualification only; separate exclusive assignment and merge protocol required.')
    C.write_json(out/'report.json',report);print(report,flush=True)
    if not report['exact']:raise SystemExit('Qualification failed: predictions differ')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--registration-commit',required=True);a=p.parse_args()
    if not re.fullmatch('[a-f0-9]{40}',a.registration_commit):p.error('full registration SHA required')
    if a.action=='prepare':prepare(a.input.resolve(),a.out.resolve(),a.registration_commit)
    else:run(a.input.resolve(),a.out.resolve(),a.registration_commit)


if __name__=='__main__':main()

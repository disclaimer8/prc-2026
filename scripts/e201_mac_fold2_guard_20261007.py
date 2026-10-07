"""Run only E201's unstarted Macfold2 after reboot with local resource guards.

No changes to the frozen model worker, no resumed partial fit, no auto-retry.
All signals target only the child process group created by this wrapper.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback

REG='d664e153b3f0732b4047011cc062f58f42581a6c'
MANIFEST='f8696b7326f196ba72f51e44be5a6b658c8f063fcac09e1bd54274c80218a2aa'
WORKER='9da861964ddec3dade4c2a1438c20214ee21236388b9337b3fffb16cab594cbd'
MAC_COMPLETE='6a8cc558ec874206e0e2ff12cd06273ef71558da040dce23389dedc10ec324b3'
MAC_AUDIT='38915251e1d492efd8df3064c1574565ddce00f0ae4c04cd8d165c3d3b03841e'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False)
def pressure():
    s=subprocess.check_output(['/usr/bin/memory_pressure','-Q'],text=True,timeout=15,env={**os.environ,'LC_ALL':'C'})
    m=re.search(r'System-wide memory free percentage:\s*(\d+)%',s)
    if not m:raise ValueError('unrecognized memory_pressure output')
    return int(m.group(1))
def reclaimable():
    s=subprocess.check_output(['/usr/bin/vm_stat'],text=True,timeout=15,env={**os.environ,'LC_ALL':'C'})
    page=int(re.search(r'page size of (\d+) bytes',s).group(1));values={}
    for line in s.splitlines()[1:]:
        if ':' in line:
            key,value=line.split(':',1);value=value.strip().rstrip('.')
            if value.isdigit():values[key]=int(value)
    return page*sum(values.get(k,0) for k in ('Pages free','Pages inactive','Pages speculative'))
def stop_own(child):
    if child.poll() is not None:return
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=15)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--probe',action='store_true');a=p.parse_args();root=a.root.resolve()
    if a.probe:return print(json.dumps(dict(memory_free_percent=pressure(),reclaimable_bytes=reclaimable(),disk_free_bytes=shutil.disk_usage(root).free)),flush=True)
    state=root/'data/e201_mac_fold2_guard_20261007';state.mkdir(exist_ok=False)
    base=root/'data/e201_mac_20261007';bundle=root/'data/e201_inputs_20261007';worker=bundle/'scripts/e201_donor_oof.py';part=base/'v60_fold2';child=None
    start=time.time();save(state/'started.json',dict(pid=os.getpid(),registration_commit=REG,queue_sha256=sha(__file__),started_unix=start))
    try:
        if sha(bundle/'manifest.json')!=MANIFEST or sha(worker)!=WORKER:raise ValueError('frozen E201 source/input drift')
        if sha(base/'v60_fold1/complete.json')!=MAC_COMPLETE or sha(base/'v60_fold1/transfer_receipt.json')!=MAC_AUDIT:raise ValueError('completed first fold changed')
        if part.exists() or list(base.glob('.v60_fold2.partial-*')) or (base/'fold2_ready.json').exists():raise ValueError('Macfold2 already attempted; no resume/retry')
        free=reclaimable();disk=shutil.disk_usage(root).free
        if free<16*1024**3 or disk<4*1024**3:raise ValueError('startup requires16GiB reclaimable RAM and4GiB free disk')
        os.nice(10)
        env={**os.environ,'OMP_NUM_THREADS':'6','OMP_THREAD_LIMIT':'6','POLARS_MAX_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OMP_WAIT_POLICY':'PASSIVE','GOMP_SPINCOUNT':'0','PYTHONUNBUFFERED':'1'}
        save(state/'admitted.json',dict(reclaimable_bytes=free,disk_free_bytes=disk,memory_free_percent=pressure(),threads=6,input_manifest_sha256=MANIFEST,worker_sha256=WORKER))
        common=['--input',str(bundle),'--output',str(part),'--registration-commit',REG]
        for action,extra in [('run-fold',['--donor','v60','--fold','2','--threads','6']),('audit',[])]:
            low=0
            with (base/f'v60_fold2_{action}.log').open('xb') as log:
                child=subprocess.Popen([sys.executable,str(worker),action,*common,*extra],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(state/f'{action}_started.json',dict(pid=child.pid,started_unix=time.time()))
                with (state/f'{action}_resources.jsonl').open('x') as resources:
                    while child.poll() is None:
                        pct=pressure();disk=shutil.disk_usage(root).free;low=low+1 if pct<10 else 0
                        resources.write(json.dumps(dict(unix=time.time(),pid=child.pid,memory_free_percent=pct,disk_free_bytes=disk,consecutive_low=low))+'\n');resources.flush()
                        if low>=3:raise RuntimeError('memory_pressure below10% in3consecutive10s samples')
                        if disk<4*1024**3:raise RuntimeError('disk headroom below4GiB')
                        if time.time()-start>48*3600:raise TimeoutError('bounded48h exceeded')
                        time.sleep(10)
                if child.returncode:raise subprocess.CalledProcessError(child.returncode,child.args)
            child=None
        proof=read(part/'transfer_receipt.json');packet=base/'v60_fold2_small.tar.gz'
        with tarfile.open(packet,'x:gz') as archive:
            for name in [*sorted(proof['small_files']),'complete.json','transfer_receipt.json']:archive.add(part/name,arcname='v60_fold2/'+name,recursive=False)
        ready=dict(donor='v60',fold=2,receipt_sha256=sha(part/'complete.json'),transfer_sha256=sha(part/'transfer_receipt.json'),packet=packet.name,packet_sha256=sha(packet))
        save(base/'fold2_ready.json',ready);save(state/'complete.json',dict(registration_commit=REG,input_manifest_sha256=MANIFEST,fold=ready,seconds=time.time()-start));print('COMPLETE Macfold2; no fit repeated',flush=True)
    except BaseException:
        if child is not None:stop_own(child)
        save(state/'failure.json',dict(error=traceback.format_exc(),auto_retry=False,partial_preserved=True));raise

if __name__=='__main__':main()

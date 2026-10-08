"""Bounded Mac feature build; unchanged producers, discard only new intermediates."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import shutil
import final_feature_workspace_20261008 as W

PROTOCOL='final-feature-Mac12threads-24GiB-4GiBdisk-2h-v1'
SELF='scripts/final_feature_mac_20261008.py'
DOC='analysis/final_feature_workspace_20261008/MAC_PROTOCOL.md'
AREA='data/final_feature_mac_execution_20261008'
WORK='data/final_feature_work_mac_20261008'
MANIFEST='data/final_feature_manifest_mac_20261008.json'
# Only outputs created by this fresh WORK; keep all four declared final tables.
DISCARD={'e28':['data/features_final.parquet'],'e77':['data/features_e28.parquet'],
         'e106':['data/features_e77.parquet'],'fin1b':['data/features_e163.parquet',
         'data/features_e106_e51.parquet','data/features_e157.parquet','data/features_e165.parquet']}


def registered(root,reg):
    W.registered(root,reg)
    for name in (SELF,DOC):
        b=subprocess.check_output(['git','show',reg+':'+name],cwd=root)
        if hashlib.sha256(b).hexdigest()!=W.sha(root/name):raise ValueError('Mac adapter source drift')
    if PROTOCOL.encode() not in subprocess.check_output(['git','show',reg+':HYPOTHESES_LOG.md'],cwd=root):raise ValueError('Mac protocol not registered')


def worker(root,reg,specsha):
    registered(root,reg);spec=W.checked_manifest(root/MANIFEST,specsha);work=root/WORK
    W.materialize(root,work,spec);W.save(work/'manifest.json',spec);(work/'logs').mkdir()
    W.save(work/'stage_complete.json',{'manifest_sha256':specsha,'host':'Mac','adapter':PROTOCOL})
    W.save(work/'feature_started.json',{'manifest_sha256':specsha,'adapter':PROTOCOL})
    start=time.monotonic()
    for i,(name,script,args,extra,out) in enumerate(W.STEPS):
        if shutil.disk_usage(work).free<4*1024**3:raise ValueError('4GiB disk floor')
        if (work/out).exists():raise FileExistsError(out)
        print('FEATURE_STAGE',i,name,flush=True)
        with (work/'logs'/f'{i:02d}_{name}.log').open('x') as log:
            if script is None:W.e106(work)
            else:
                env={**W.environment(),**extra,'POLARS_MAX_THREADS':'12','OMP_NUM_THREADS':'12','OMP_THREAD_LIMIT':'12'}
                subprocess.run([sys.executable,'-u','src/'+script,*args],cwd=work,env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not (work/out).is_file() or not (work/out).stat().st_size:raise ValueError('producer missing output '+out)
        W.save(work/'logs'/f'{i:02d}_{name}.json',{'stage':name,'output':out,'sha256':W.sha(work/out)})
        for old in DISCARD.get(name,[]):
            p=work/old
            if old in W.FINAL_OUTPUTS or p.is_symlink() or not p.resolve().is_relative_to(work.resolve()):raise ValueError('unsafe intermediate cleanup')
            p.unlink()
    outputs={name:W.check_output(work/name,spec['expected_output_schemas'][name],spec['contract']) for name in W.FINAL_OUTPUTS}
    W.verify_files(work,spec,True)
    W.save(work/'feature_complete.json',{**W.completion_identity(spec,specsha,outputs),'seconds':time.monotonic()-start})
    print('FEATURE_COMPLETE',work,flush=True)


def rss_group(pgid):
    out=subprocess.check_output(['ps','-axo','pgid=,rss='],text=True)
    return sum(int(line.split()[1])*1024 for line in out.splitlines() if len(line.split())==2 and int(line.split()[0])==pgid)


def terminate_group(child):
    if child is None:return
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:return
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        child.poll()
        try:os.killpg(child.pid,0)
        except ProcessLookupError:return
        time.sleep(.1)
    try:os.killpg(child.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    child.wait(timeout=10)


def interrupted(signum,frame):
    raise InterruptedError('supervisor signal '+str(signum))


def supervise(root,reg):
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    registered(root,reg);area=root/AREA;area.mkdir(exist_ok=False)
    start=time.monotonic();peak=0;child=None;failure=None
    try:
        with (area/'seal.log').open('x') as log:
            child=subprocess.Popen([sys.executable,str(root/W.SELF),'seal','--root',str(root),'--registration-commit',reg,
                '--manifest',str(root/MANIFEST)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            if child.wait(timeout=900):raise RuntimeError('metadata seal failed; see seal.log')
        terminate_group(child);child=None
        specsha=W.sha(root/MANIFEST)
        with (area/'worker.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',str(root/SELF),'worker','--root',str(root),
                '--registration-commit',reg,'--manifest-sha256',specsha],cwd=root,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
            W.save(area/'started.json',{'protocol':PROTOCOL,'registration_commit':reg,'pid':child.pid,
                'pgid':child.pid,'manifest_sha256':specsha,'started_unix':time.time()})
            while child.poll() is None:
                peak=max(peak,rss_group(child.pid))
                if peak>24*1024**3:raise MemoryError('24GiB group RSS ceiling')
                if shutil.disk_usage(root).free<4*1024**3:raise RuntimeError('4GiB disk floor')
                if time.monotonic()-start>7200:raise TimeoutError('2h wall ceiling')
                time.sleep(1)
            if child.returncode:raise RuntimeError('feature worker exit '+str(child.returncode))
        terminate_group(child);child=None
        if not (root/WORK/'feature_complete.json').is_file():raise RuntimeError('missing feature completion')
        W.save(area/'complete.json',{'protocol':PROTOCOL,'registration_commit':reg,'status':'PASS',
            'seconds':time.monotonic()-start,'peak_group_rss_bytes':peak,
            'feature_complete_sha256':W.sha(root/WORK/'feature_complete.json')})
    except BaseException as exc:
        failure=repr(exc)
        terminate_group(child)
        W.save(area/'failed.json',{'protocol':PROTOCOL,'reason':failure,'seconds':time.monotonic()-start,'peak_group_rss_bytes':peak})
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['supervise','worker'])
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--registration-commit',required=True)
    p.add_argument('--manifest-sha256');a=p.parse_args();root=a.root.resolve()
    if a.action=='worker':worker(root,a.registration_commit,a.manifest_sha256)
    else:supervise(root,a.registration_commit)


if __name__=='__main__':main()

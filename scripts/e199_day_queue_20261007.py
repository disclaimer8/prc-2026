"""Durable E199 continuation, exact frozen worker, no retries or recipe changes.

Wait for already-running laptop extraction; collect/merge; run Home/Hetzner
screen; only a verified strict PASS admits four fixed confirmation pairs.
This controller is committed and independently reviewed after worker reg344.
It never edits shared sources, environments, journal, submissions or GPU jobs.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

PROTOCOL='E199-durable-two-laptop-extraction-fixed-CPU-continuation-20261007-v1'
SELF='scripts/e199_day_queue_20261007.py'
REG='f21fa8493dad3e7f539d23cb2751a3cc5be97131'
WORKER='scripts/e199_airport_surface_state.py'
WORKER_SHA='4d7c7755790c133444e5e8980e82e27dd3e1928de3c1b9da805cc1709699dd6b'
WORKER_PROTOCOL='E199-other-ground-spatial-state-v-counts-retrospective-2pair-3seed-v1'
BUNDLE='data/e199_inputs_20261007'
BUNDLE_SHA='6b057a9aaaf0c26710a19f564f8a0b1a203cb7c6073f2590e528d1ff3251633f'
INPUT='data/e199_screen_inputs_20261007'
OUTPUT='data/e199_screen_20261007'
STATE='data/e199_queue_20261007'
ASSETS='data/e199_queue_assets_20261007'
NODES={
    '171':dict(host='a1@192.168.1.171',root='/home/a1/prc-2026',python='.venv/bin/python',threads=6,memory='10G'),
    '175':dict(host='a1@192.168.1.175',root='/home/a1/prc-2026',python='.venv-cat-parity/bin/python',threads=6,memory='10G'),
    'home':dict(host='home-pc-lan',root='/home/a1/prc-2026',python='.venv/bin/python',threads=12,memory='16G'),
    'hetzner':dict(host='hetzner',root='/root/prc-2026',python='.venv/bin/python',threads=6,memory='16G'),
}
EXTRACTION={
    0:dict(node='171',unit='prc-e199-surface-s0-20261007.service',invocation='49b18fb54c2846a1a4aa8766094220c0'),
    1:dict(node='175',unit='prc-e199-surface-s1-20261007.service',invocation='022d0b1fbab445bbbd3178dfc12ab27d'),
}
ASSIGN={1:'home',4:'hetzner',2:'171',3:'175',5:'hetzner',6:'home'}
KEYS=('body_sse','decided_sse','trim3_sse','trim20_sse')
COMPARISONS=('spatial_vs_counts','spatial_vs_parent')


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):return json.loads(Path(p).read_text())


def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False)


def now():return dt.datetime.now(dt.timezone.utc).isoformat()


class Queue:
    def __init__(self,root,registration):
        self.root=root;self.registration=registration;self.state=root/STATE
        self.started=time.monotonic();self.installed=set();self.phase='startup'

    def event(self,event,**info):
        record=dict(time=now(),event=event,phase=self.phase,**info)
        with (self.state/'events.jsonl').open('a') as f:
            f.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(record,sort_keys=True),flush=True)

    def command(self,args,timeout=120,allowed=(0,)):
        p=subprocess.run(list(map(str,args)),cwd=self.root,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
        if p.returncode not in allowed:
            raise RuntimeError(json.dumps(dict(command=list(map(str,args)),returncode=p.returncode,stdout=p.stdout[-3000:],stderr=p.stderr[-3000:])))
        return p

    def ssh(self,node,args,timeout=120,allowed=(0,)):
        return self.command(['ssh','-oBatchMode=yes','-oConnectTimeout=10','-oServerAliveInterval=10','-oServerAliveCountMax=2',NODES[node]['host'],shlex.join(list(map(str,args)))],timeout,allowed)

    def remote_path(self,node,relative):return NODES[node]['root']+'/'+relative

    def exists(self,node,relative):
        return self.ssh(node,['test','-e',self.remote_path(node,relative)],allowed=(0,1)).returncode==0

    def status(self,node,unit):
        p=self.ssh(node,['systemctl','--user','show',unit,'-p','ActiveState','-p','SubState','-p','MainPID','-p','InvocationID','-p','Result','-p','ExecMainStatus','-p','CPUQuotaPerSecUSec','-p','MemoryMax','-p','MemorySwapMax'])
        return dict(line.split('=',1) for line in p.stdout.splitlines() if '=' in line)

    @staticmethod
    def active(s):return s.get('ActiveState') in ('active','activating','deactivating') or int(s.get('MainPID','0'))>0

    def pause(self):
        if time.monotonic()-self.started>20*3600:raise TimeoutError('E199 queue exceeded20h')
        time.sleep(20)

    def transfer(self,node,relative,receive=False):
        local=self.root/relative;remote=NODES[node]['host']+':'+self.remote_path(node,relative)
        if receive:
            if local.exists():raise FileExistsError('refusing to overwrite collected output '+str(local))
            local.mkdir(parents=True)
            self.command(['rsync','-a',remote+'/',str(local)+'/'],timeout=900)
        else:
            if self.exists(node,relative):raise FileExistsError('refusing remote input overwrite '+remote)
            self.ssh(node,['mkdir','-p',self.remote_path(node,relative)])
            self.command(['rsync','-a',str(local)+'/',remote+'/'],timeout=900)

    def local_worker(self,args):
        env={**os.environ,'POLARS_MAX_THREADS':'2','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','OMP_THREAD_LIMIT':'2'}
        cmd=[str(self.root/'.venv/bin/python'),'-u',str(self.root/BUNDLE/'scripts/e199_airport_surface_state.py'),*map(str,args)]
        with (self.state/(self.phase+'.log')).open('x') as f:
            subprocess.run(cmd,cwd=self.root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800)

    def extraction(self):
        self.phase='await_extraction';pending=set(EXTRACTION)
        while pending:
            for shard in sorted(pending.copy()):
                job=EXTRACTION[shard];node=job['node'];s=self.status(node,job['unit'])
                if self.active(s):
                    if s['InvocationID']!=job['invocation']:raise ValueError('extraction invocation changed')
                    continue
                if s.get('Result') not in ('','success',None) or s.get('ExecMainStatus','0')!='0':raise RuntimeError('extraction failed '+json.dumps(s))
                relative=f'data/e199_surface_20261007_s{shard}'
                if not self.exists(node,relative+'/complete.json'):raise RuntimeError('extraction stopped without durable completion')
                self.transfer(node,relative,receive=True)
                r=read(self.root/relative/'complete.json')
                if r['protocol']!=WORKER_PROTOCOL or r['registration_commit']!=REG or r['input_manifest_sha256']!=BUNDLE_SHA or r['shard']!=shard:raise ValueError('extraction receipt lineage')
                expected=sorted(d for d,x in read(self.root/BUNDLE/'manifest.json')['days'].items() if x['shard']==shard)
                if r['days']!=expected or sorted(x['day'] for x in r['receipts'])!=expected:raise ValueError('incomplete extraction dates')
                for item in r['receipts']:
                    if sha(self.root/relative/(item['day']+'.parquet'))!=item['sha256']:raise ValueError('collected day hash')
                self.event('extraction_verified',node=node,shard=shard,days=len(expected),rows=r['rows'],receipt_sha256=sha(self.root/relative/'complete.json'))
                pending.remove(shard)
            if pending:self.pause()

    def merge(self):
        self.phase='merge'
        self.local_worker(['merge','--root',self.root,'--input',self.root/BUNDLE,'--parts',self.root/'data/e199_surface_20261007_s0',self.root/'data/e199_surface_20261007_s1','--output',self.root/INPUT,'--registration-commit',REG])
        m=read(self.root/INPUT/'manifest.json')
        self.event('merged',coverage_admission=m['coverage_admission'],eligible_train=m['eligible_train'],input_manifest_sha256=sha(self.root/INPUT/'manifest.json'))
        return m['coverage_admission']

    def install(self,node):
        if node in self.installed:return
        n=NODES[node]
        # Only new private inputs; existing interpreter and GPU environment untouched.
        self.transfer(node,INPUT)
        if node not in ('171','175'):self.transfer(node,BUNDLE)
        if self.exists(node,ASSETS):raise FileExistsError('remote controller assets already exist')
        self.ssh(node,['mkdir','-p',self.remote_path(node,ASSETS),self.remote_path(node,OUTPUT)])
        p=self.ssh(node,[self.remote_path(node,n['python']),'-c','import sys,numpy,polars,lightgbm; print(sys.version); print(numpy.__version__,polars.__version__,lightgbm.__version__)'])
        self.event('runtime_preflight',node=node,runtime=p.stdout.strip());self.installed.add(node)

    def wait_cpu(self,node):
        start=time.monotonic()
        while True:
            p=self.ssh(node,['systemctl','--user','list-units','--type=service','--state=running','--no-legend','--plain'])
            units=[line.split()[0] for line in p.stdout.splitlines() if line.split()]
            if node=='hetzner':
                system=self.ssh(node,['systemctl','list-units','--type=service','--state=running','--no-legend','--plain'])
                units.extend(line.split()[0] for line in system.stdout.splitlines() if line.split())
            # Fixed known CPU queues; GPU E197/E198 and E200 rank are deliberately excluded.
            busy=[u for u in units if u.startswith(('prc-cat-','prc-e195-','prc-e184-','prc-e185-','prc-e199-','prc-e200-capture','prc-e200-stage','prc-e200-chain'))]
            if not busy:return
            if time.monotonic()-start>4*3600:raise TimeoutError('CPU reservation wait exceeded4h '+node)
            self.event('wait_cpu',node=node,units=busy);self.pause()

    def launch(self,fold):
        node=ASSIGN[fold];self.install(node);self.wait_cpu(node);n=NODES[node]
        relative=f'{OUTPUT}/fold_{fold}_{fold+6}';unit=f'prc-e199-fit-f{fold}-20261007.service'
        if self.exists(node,relative):raise FileExistsError('fit output already exists')
        if self.active(self.status(node,unit)):raise RuntimeError('fit unit already active')
        argv=[self.remote_path(node,n['python']),'-u',self.remote_path(node,BUNDLE+'/scripts/e199_airport_surface_state.py'),'fit','--input',self.remote_path(node,INPUT),'--output',self.remote_path(node,relative),'--registration-commit',REG,'--fold',str(fold),'--threads',str(n['threads'])]
        if fold not in (1,4):
            screen=OUTPUT+'/evaluation.json'
            self.command(['scp',str(self.root/screen),n['host']+':'+self.remote_path(node,ASSETS+'/screen_evaluation.json')],timeout=120)
            argv+=['--screen-evaluation',self.remote_path(node,ASSETS+'/screen_evaluation.json')]
        if node=='hetzner':argv=['taskset','-c','0,2,4,6,8,10',*argv]
        if node in ('171','175'):argv=['systemd-inhibit','--what=sleep:idle','--mode=block','--who=PRC-E199','--why=E199-fixed-CPU-confirmation',*argv]
        log=self.remote_path(node,OUTPUT+f'/fold_{fold}_{fold+6}.log')
        script='#!/bin/bash\nset -euo pipefail\nset -o noclobber\n'+f'exec >{shlex.quote(log)} 2>&1\n'+'exec '+shlex.join(['env',f'OMP_NUM_THREADS={n["threads"]}',f'OMP_THREAD_LIMIT={n["threads"]}','POLARS_MAX_THREADS=2','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','OMP_WAIT_POLICY=PASSIVE','GOMP_SPINCOUNT=0',*argv])+'\n'
        local=self.state/f'fit_{fold}.sh';local.write_text(script)
        remote=self.remote_path(node,ASSETS+f'/fit_{fold}.sh')
        self.command(['scp',str(local),n['host']+':'+remote],timeout=120)
        cmd=['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+n['root'],'--property=MemoryMax='+n['memory'],'--property=MemorySwapMax=0','--property=CPUQuota='+str(n['threads']*100)+'%','--property=Nice=10','--property=RuntimeMaxSec=14400','--property=KillMode=control-group','/bin/bash',remote]
        self.ssh(node,cmd);s=self.status(node,unit)
        if not self.active(s) or not s.get('InvocationID'):raise RuntimeError('fit did not become active '+json.dumps(s))
        job=dict(fold=fold,node=node,relative=relative,unit=unit,invocation=s['InvocationID'])
        save(self.state/f'launch_{fold}.json',dict(**job,status=s,worker_registration=REG,script_sha256=sha(local)))
        self.event('fit_launched',**job,status=s);return job

    def collect_fit(self,job):
        node=job['node'];relative=job['relative'];s=self.status(node,job['unit'])
        if self.active(s):
            if s['InvocationID']!=job['invocation']:raise ValueError('fit invocation changed')
            return False
        if s.get('Result') not in ('','success',None) or s.get('ExecMainStatus','0')!='0':raise RuntimeError('fit failed '+json.dumps(s))
        if not self.exists(node,relative+'/complete.json'):raise RuntimeError('fit stopped without complete receipt')
        self.transfer(node,relative,receive=True);r=read(self.root/relative/'complete.json')
        if r['protocol']!=WORKER_PROTOCOL or r['registration_commit']!=REG or r['fold']!=job['fold'] or r['input_manifest_sha256']!=sha(self.root/INPUT/'manifest.json'):raise ValueError('fit receipt lineage')
        if job['fold'] not in (1,4) and r['screen_evaluation_sha256']!=sha(self.root/OUTPUT/'evaluation.json'):raise ValueError('continuation screen differs')
        for name,h in r['files'].items():
            if sha(self.root/relative/name)!=h:raise ValueError('collected fit artifact hash')
        self.event('fit_verified',fold=job['fold'],node=node,seconds=r['seconds'],receipt_sha256=sha(self.root/relative/'complete.json'));return True

    def fit_group(self,folds):
        self.phase='fit_'+('_'.join(map(str,folds)))
        # Independent hosts must launch without one host's transfer delaying another.
        with ThreadPoolExecutor(max_workers=len(folds)) as pool:jobs=list(pool.map(self.launch,folds))
        while jobs:
            for job in jobs.copy():
                if self.collect_fit(job):jobs.remove(job)
            if jobs:self.pause()

    def evaluation(self,folds,name):
        self.phase='evaluate_'+name
        self.local_worker(['evaluate','--input',self.root/INPUT,'--parts',*[self.root/OUTPUT/f'fold_{f}_{f+6}' for f in folds],'--output',self.root/OUTPUT/name,'--registration-commit',REG])
        p=self.root/OUTPUT/name;r=read(p)
        if r['protocol']!=WORKER_PROTOCOL or r['registration_commit']!=REG or r['input_manifest_sha256']!=sha(self.root/INPUT/'manifest.json') or sorted(x['fold'] for x in r['reports'])!=sorted(folds):raise ValueError('evaluation lineage')
        strict=all(all(row[c][k]<0 for c in COMPARISONS for k in KEYS) for row in r['reports'] if row['fold'] in (1,4))
        if strict!=r['spatial_screen_pass']:raise ValueError('screen gate recomputation differs')
        self.event('evaluation_complete',name=name,spatial_screen_pass=strict,six_pair_gate_pass=r['six_pair_gate_pass'],sha256=sha(p));return r

    def run(self):
        source=subprocess.check_output(['git','show',self.registration+':'+SELF],cwd=self.root)
        if hashlib.sha256(source).hexdigest()!=sha(self.root/SELF):raise ValueError('controller differs from reviewed commit')
        if sha(self.root/BUNDLE/'manifest.json')!=BUNDLE_SHA or sha(self.root/WORKER)!=WORKER_SHA or sha(self.root/BUNDLE/'scripts/e199_airport_surface_state.py')!=WORKER_SHA:raise ValueError('fixed extraction source/bundle changed')
        if self.state.exists():raise FileExistsError('fresh controller state; no automatic resume')
        self.state.mkdir(parents=True);save(self.state/'started.json',dict(protocol=PROTOCOL,controller_registration=self.registration,worker_registration=REG,pid=os.getpid(),started=now(),assignment=ASSIGN,extraction=EXTRACTION))
        try:
            self.extraction()
            if not self.merge():
                save(self.state/'complete.json',dict(protocol=PROTOCOL,status='coverage_failed_no_fits',finished=now()));return
            (self.root/OUTPUT).mkdir(exist_ok=False)
            self.fit_group([1,4]);r=self.evaluation([1,4],'evaluation.json')
            if r['spatial_screen_pass']:
                self.fit_group([2,3,5,6]);r=self.evaluation([1,2,3,4,5,6],'evaluation_all6.json')
            save(self.state/'complete.json',dict(protocol=PROTOCOL,controller_registration=self.registration,worker_registration=REG,finished=now(),status='six_pairs_complete' if len(r['reports'])==6 else 'screen_failed_no_extension',spatial_screen_pass=r['spatial_screen_pass'],six_pair_gate_pass=r['six_pair_gate_pass'],accepted_for_submission=False))
        except BaseException as e:
            save(self.state/'failure.json',dict(protocol=PROTOCOL,phase=self.phase,time=now(),error=repr(e),automatic_retry=False,launched_jobs=[p.name for p in self.state.glob('launch_*.json')],note='Already launched bounded services may finish; no further launch/retry by controller.'))
            self.event('failure',error=repr(e));raise


def selftest():
    assert len(set(ASSIGN[f] for f in (1,4)))==2 and len(set(ASSIGN[f] for f in (2,3,5,6)))==4
    assert Queue.active({'ActiveState':'running','MainPID':'5'}) and not Queue.active({'ActiveState':'inactive','MainPID':'0'})
    assert set(ASSIGN)==set(range(1,7)) and NODES['hetzner']['threads']==6
    print('controller static contracts PASS; no network, files, fits or scores')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','selftest']);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--registration-commit');a=p.parse_args()
    if a.action=='selftest':return selftest()
    if not a.registration_commit or len(a.registration_commit)!=40:raise ValueError('full reviewed controller commit required')
    root=a.root.resolve();lock=root/'data/e199_day_queue_20261007.lock'
    with lock.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);Queue(root,a.registration_commit).run()


if __name__=='__main__':main()

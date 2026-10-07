"""One-shot E207 transport, existing laptop preparation and Home GPU units.

No retry or new environment. Only this controller's recorded unit invocations
are stopped on failure; all partial artifacts remain for inspection.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import e207_full_token_sampler as W

REMOTE='/home/a1/prc-2026'
HOME='home-pc-lan'
NODES={171:('a1@192.168.1.171','.venv',0),175:('a1@192.168.1.175','.venv-cat-parity',1)}
GPU='prc-e207-trajectory-20261008.service'


def active(s):return s.get('ActiveState') in ('active','activating','deactivating','reloading') or int(s.get('MainPID','0'))>0


def raise_stop(signum,frame):
    raise InterruptedError('E207 controller received signal '+str(signum))


def stopping_handler(cleanup,cleanup_started,stopped):
    def handler(signum,frame):
        # Do not interrupt cleanup again, including cleanup already in progress
        # on this thread. Its SSH calls and process waits have finite deadlines.
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,signal.SIG_IGN)
        if cleanup_started():return
        stopped.set()
        cleanup();raise_stop(signum,frame)
    return handler


def terminate_owned_process(process):
    if process.poll() is not None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait(timeout=5)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--registration-commit',required=True);p.add_argument('--detach',action='store_true');a=p.parse_args()
    root=a.root.resolve();reg=a.registration_commit;W.registration(root,reg)
    if shutil.disk_usage(root).free<2*1024**3:raise ValueError('Mac needs2GiB free for bundle/small receipts')
    if a.detach:
        path=root/'analysis/e207_full_token_20261008/launch.json'
        with path.open('x') as marker:
            with (path.parent/'controller.log').open('x') as log:
                child=subprocess.Popen(['/usr/bin/caffeinate','-i',sys.executable,'-u',__file__,'--root',str(root),'--registration-commit',reg],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            json.dump(dict(pid=child.pid,registration_commit=reg,worker_sha256=W.C.sha(root/W.SELF),queue_sha256=W.C.sha(__file__)),marker);marker.write('\n')
        print('STARTED DETACHED CONTROLLER',child.pid,flush=True);return
    state=root/'analysis/e207_full_token_20261008/execution';state.mkdir(exist_ok=False)
    deadline=time.monotonic()+21600;lock=threading.RLock();admission_lock=threading.Lock();owned=[];stopped=threading.Event()
    transports=set();cleanup_lock=threading.RLock();cleanup_started=False;owner_token='E207-'+reg+'-'+uuid.uuid4().hex
    def record(event,**extra):
        row=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),event=event,**extra)
        with lock:
            with (state/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(row),flush=True)
    def call(cmd,timeout=1800,capture=False):
        remaining=deadline-time.monotonic()
        if stopped.is_set() or remaining<=0:raise TimeoutError('E207 total six-hour deadline')
        process=subprocess.Popen(cmd,text=True,stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,start_new_session=True)
        with lock:transports.add(process)
        try:
            if stopped.is_set():raise InterruptedError('controller cancelled')
            stdout,stderr=process.communicate(timeout=min(timeout,remaining))
            if process.returncode:raise subprocess.CalledProcessError(process.returncode,cmd,stdout,stderr)
            return subprocess.CompletedProcess(cmd,process.returncode,stdout,stderr)
        except BaseException:
            terminate_owned_process(process);raise
        finally:
            with lock:transports.discard(process)
    def remote(host,args,timeout=60):return call(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join(args)],timeout,True).stdout
    def live(host,unit):return dict(x.split('=',1) for x in remote(host,['systemctl','--user','show',unit,'-p','ActiveState','-p','MainPID','-p','InvocationID','-p','Result','-p','ExecMainStatus']).splitlines() if '=' in x)
    def worker(env,action,*args):return [f'{REMOTE}/{env}/bin/python','-u',f'{REMOTE}/{W.INPUT}/scripts/{Path(W.SELF).name}',action,'--root',REMOTE,'--registration-commit',reg,*args]
    def launch(host,unit,cpu,mem,command):
        before=live(host,unit)
        if active(before) or before.get('InvocationID'):raise ValueError('unit previously attempted: '+unit)
        log=REMOTE+'/data/'+unit.removesuffix('.service')+'.log'
        owned.append((host,unit,None))
        remote(host,['systemd-run','--user','--unit='+unit,'--working-directory='+REMOTE,
            '--description='+owner_token,
            '--property=CPUQuota='+str(cpu)+'%','--property=MemoryMax='+mem,'--property=MemorySwapMax=0',
            '--property=Nice=10','--property=KillMode=control-group','--property=TimeoutStopSec=30',
            '--property=RuntimeMaxSec='+str(max(1,int(deadline-time.monotonic()))),
            '--property=StandardOutput=append:'+log,'--property=StandardError=inherit',*command])
        s=live(host,unit)
        if s.get('InvocationID'):owned.append((host,unit,s['InvocationID']))
        if not s.get('InvocationID') or (not active(s) and (s.get('Result')!='success' or s.get('ExecMainStatus')!='0')):
            raise ValueError('unit failed at launch: '+repr(s))
        record('started',host=host,unit=unit,properties=s,log=log);return s
    def wait_unit(host,unit,initial):
        while True:
            s=live(host,unit)
            if s.get('InvocationID')!=initial['InvocationID']:raise ValueError('owned unit invocation changed')
            if not active(s):
                if s.get('Result')!='success' or s.get('ExecMainStatus')!='0':raise ValueError('unit failed: '+repr(s))
                record('exited',host=host,unit=unit,properties=s);return
            record('progress',host=host,unit=unit,MainPID=s['MainPID'])
            time.sleep(min(30,max(0,deadline-time.monotonic())))
    def copy(source,dest,recursive=False):
        call(['scp','-3','-o','BatchMode=yes',*(['-r'] if recursive else []),source,dest],1800)
    def file_ready(host,path):
        return remote(host,['sh','-c','if test -f '+shlex.quote(path)+'; then echo READY; else echo WAIT; fi']).strip()=='READY'
    def cleanup():
        nonlocal cleanup_started
        with cleanup_lock:
            if cleanup_started:return
            cleanup_started=True
        stopped.set()
        with lock:running=list(transports)
        for process in running:terminate_owned_process(process)
        seen=set()
        for host,unit,invocation in reversed(owned):
            if (host,unit) in seen:continue
            seen.add((host,unit))
            try:
                args=['systemctl','--user','show',unit,'-p','InvocationID','-p','MainPID','-p','Description']
                r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join(args)],capture_output=True,text=True,timeout=20)
                s=dict(x.split('=',1) for x in r.stdout.splitlines() if '=' in x)
                owned_match=s.get('InvocationID')==invocation if invocation else s.get('Description')==owner_token
                if owned_match and int(s.get('MainPID','0'))>0:
                    subprocess.run(['ssh','-o','BatchMode=yes',host,shlex.join(['systemctl','--user','stop',unit])],timeout=50,check=True)
                    record('stopped_owned_invocation',host=host,unit=unit,invocation=invocation)
            except BaseException as exc:record('cleanup_error',host=host,unit=unit,error=repr(exc))
    signal_stop=stopping_handler(cleanup,lambda:cleanup_started,stopped)
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,signal_stop)
    def safe_task(function,value):
        try:return function(value)
        except BaseException:
            cleanup();raise
    try:
        record('controller_started',pid=os.getpid(),registration_commit=reg)
        for host in [HOME,*[n[0] for n in NODES.values()]]:
            units=remote(host,['systemctl','--user','list-units','--type=service','--state=running','--plain','--no-legend'])
            if any(line.strip().startswith('prc-') for line in units.splitlines()):raise ValueError('another PRC unit running on '+host+': '+units)
            processes=remote(host,['ps','-eo','pid,args'])
            busy=[line for line in processes.splitlines() if 'python' in line.lower() and any(t in line for t in ('/prc-2026/','scripts/e19','scripts/e20','src/train.py','cat_laptop'))]
            if busy:raise ValueError('another PRC process on '+host+': '+repr(busy))
            free=int(remote(host,['df','-Pk',REMOTE]).splitlines()[-1].split()[3])*1024
            if free<(70 if host==HOME else 40)*1024**3:raise ValueError('insufficient free disk on '+host)
            record('idle_and_disk_checked',host=host,free_bytes=free)
        W.pack(root,reg)
        def distribute(host):
            remote(host,['test','!','-e',REMOTE+'/'+W.INPUT])
            call(['rsync','-a',str(root/W.INPUT),host+':'+REMOTE+'/data/'])
        with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(lambda host:safe_task(distribute,host),[HOME,*[n[0] for n in NODES.values()]]))
        rawunits={}
        for node,(host,env,shard) in NODES.items():
            unit=f'prc-e207-raw-{node}-20261008.service'
            rawunits[node]=(unit,launch(host,unit,600,'10G',worker(env,'extract','--shard',str(shard))))
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda node:safe_task(lambda n:wait_unit(NODES[n][0],*rawunits[n]),node),NODES))
        # Each shard is extracted exactly once. Peer exchange streams through
        # scp without storing any large raw/table data on Mac.
        for node,(host,env,shard) in NODES.items():
            other=NODES[175 if node==171 else 171][0]
            remote(other,['test','!','-e',f'{REMOTE}/{W.RAW}/s{shard}'])
            copy(f'{host}:{REMOTE}/{W.RAW}/s{shard}',f'{other}:{REMOTE}/{W.RAW}/',True)
            remote(HOME,['mkdir','-p',f'{REMOTE}/{W.RAW}/s{shard}'])
            copy(f'{host}:{REMOTE}/{W.RAW}/s{shard}/complete.json',f'{HOME}:{REMOTE}/{W.RAW}/s{shard}/complete.json')
            copy(f'{host}:{REMOTE}/{W.RAW}/s{shard}/complete.json',str(state/f'raw_s{shard}_complete.json'))
        remote(NODES[171][0],worker(NODES[171][1],'materialize'),600)
        for host in (HOME,NODES[175][0]):copy(f'{NODES[171][0]}:{REMOTE}/{W.MATERIALIZED}',f'{host}:{REMOTE}/{W.MATERIALIZED}')
        copy(f'{NODES[171][0]}:{REMOTE}/{W.MATERIALIZED}',str(state/'materialized.json'))
        materialized=json.loads((state/'materialized.json').read_text());spec=materialized['spec_sha256']
        record('raw_complete_materialized',input_spec_sha256=spec)
        remote(HOME,['mkdir','-p',REMOTE+'/'+W.INCOMING]);prep={}
        for node,(host,env,_) in NODES.items():
            unit=f'prc-e207-prep-{node}-20261008.service'
            command=['systemd-inhibit','--what=sleep:idle','--mode=block','--who=PRC-E207','--why=E207-preparation',*worker(env,'prepare-node','--node',str(node))]
            prep[node]=(unit,launch(host,unit,600,'10G',command))
        gpu=launch(HOME,GPU,200,'12G',worker('.venv-numemb','gpu-queue'))
        def stream(node):
            host,env,_=NODES[node];unit,initial=prep[node]
            try:
                for fold in W.NODE_FOLDS[node]:
                    name=f'fold_{fold}_{fold+6}';folder=f'{REMOTE}/{W.PREPARED}/{name}'
                    while not file_ready(host,folder+'/complete.json'):
                        s=live(host,unit)
                        if not active(s) or s.get('InvocationID')!=initial['InvocationID']:raise ValueError('prep stopped before fold publication')
                        time.sleep(10)
                    record('streaming',fold=fold,node=node)
                    copy(host+':'+folder,HOME+':'+REMOTE+'/'+W.INCOMING+'/',True)
                    with admission_lock:
                        au=f'prc-e207-admit-f{fold}-20261008.service'
                        ai=launch(HOME,au,100,'1G',worker('.venv-numemb','admit','--fold',str(fold)));wait_unit(HOME,au,ai)
                        receipt=json.loads(remote(HOME,['cat',f'{REMOTE}/{W.PREPARED}/{name}/admission.json']))
                        if receipt['registration_commit']!=reg or receipt['input_spec_sha256']!=spec or not receipt['passed']:raise ValueError('admission lineage differs')
                        W.C.write_json(state/(name+'_admission.json'),receipt);record('admitted',fold=fold)
                wait_unit(host,unit,initial)
            except BaseException:
                cleanup();raise
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(stream,NODES))
        wait_unit(HOME,GPU,gpu)
        for filename in ('complete.json','evaluation.json'):
            copy(f'{HOME}:{REMOTE}/{W.OUTPUT}/{filename}',str(state/filename))
        copy(f'{HOME}:{REMOTE}/{W.SMOKE}',str(state/'smoke.json'))
        result=json.loads((state/'evaluation.json').read_text())
        if result['protocol']!=W.PROTOCOL or result['registration_commit']!=reg or result['accepted_for_submission']:raise ValueError('final receipt differs')
        record('complete_no_ranking',gate=result['six_pair_gate_pass'],evaluation_sha256=W.C.sha(state/'evaluation.json'))
    except BaseException as exc:
        record('failed_no_retry',error=repr(exc));cleanup();raise


if __name__=='__main__':main()

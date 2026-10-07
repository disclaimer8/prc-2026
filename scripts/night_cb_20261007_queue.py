"""Durable one-shot E205 pack, distributed launch and six-fold collection."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import night_cb_20261007 as W

STATE='analysis/night_cb_20261007/execution'
BUNDLE='analysis/night_cb_20261007/inputs'
REMOTE_BUNDLE='data/night_cb_inputs_20261007'
NODES=[
    dict(name='hetzner',host='hetzner',root='/root/prc-2026',python='.venv/bin/python',folds='1,4',memory='12G',user=False),
    dict(name='laptop171',host='a1@192.168.1.171',root='/home/a1/prc-2026',python='.venv/bin/python',folds='2,5',memory='9G',user=True),
    dict(name='laptop175',host='a1@192.168.1.175',root='/home/a1/prc-2026',python='.venv-cat-parity/bin/python',folds='3,6',memory='9G',user=True)]


def command(args,timeout=120):
    return subprocess.check_output(args,stderr=subprocess.STDOUT,text=True,timeout=timeout)


def remote(node,args,timeout=120):
    return command(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',node['host'],shlex.join(args)],timeout)


def unit(node):return 'prc-e205-cb-'+node['name']+'-20261007'


def outpath(node):return 'data/night_cb_'+node['name']+'_20261007'


def status(node):
    args=['systemctl']+(['--user'] if node['user'] else [])+['show',unit(node),
        '-p','LoadState','-p','ActiveState','-p','SubState','-p','Result','-p','ExecMainStatus',
        '-p','MainPID','-p','InvocationID','-p','MemoryMax','-p','MemorySwapMax','-p','Nice','-p','CPUQuotaPerSecUSec','-p','AllowedCPUs','-p','RuntimeMaxUSec']
    return dict(line.split('=',1) for line in remote(node,args).splitlines() if '=' in line)


def check_limits(node,observed):
    expected=int(node['memory'][:-1])*1024**3
    if observed.get('MemoryMax')!=str(expected) or observed.get('MemorySwapMax')!='0' or observed.get('Nice')!='10' or observed.get('CPUQuotaPerSecUSec')!='6s':raise ValueError('resource limits differ '+node['name'])
    if observed.get('RuntimeMaxUSec')!='6h':raise ValueError('runtime bound differs '+node['name'])
    if not node['user'] and observed.get('AllowedCPUs','').replace(',',' ').split()!=['0','2','4','6','8','10']:raise ValueError('physical CPU assignment differs')
    if not observed.get('InvocationID'):raise ValueError('missing invocation '+node['name'])


def stage_node(node,bundle,state):
    root=node['root'];destination=root+'/'+REMOTE_BUNDLE;output=root+'/'+outpath(node)
    # Fail on preexisting artifacts or a named unit; no silent overwrite/resume.
    previous=status(node)
    if previous.get('LoadState') not in ('not-found',None):raise ValueError('named unit already exists '+node['name'])
    remote(node,['test','!','-e',destination]);remote(node,['test','!','-e',output])
    command(['scp','-q','-r',str(bundle),node['host']+':'+destination],timeout=1200)
    verification=remote(node,[root+'/'+node['python'],destination+'/night_cb_20261007.py','verify','--input',destination],timeout=180)
    print('VERIFIED',node['name'],verification.strip(),flush=True)
    args=['systemd-run']+(['--user'] if node['user'] else [])+['--unit='+unit(node),
        '--property=Type=exec','--property=RemainAfterExit=yes','--property=RuntimeMaxSec=6h',
        '--property=WorkingDirectory='+root,'--property=CPUQuota=600%',
        '--property=MemoryMax='+node['memory'],'--property=MemorySwapMax=0','--property=Nice=10',
        '--property=StandardOutput=append:'+root+'/'+outpath(node)+'.log',
        '--property=StandardError=append:'+root+'/'+outpath(node)+'.log']
    if not node['user']:args+=['--property=AllowedCPUs=0,2,4,6,8,10']
    for name,value in [('OMP_NUM_THREADS','6'),('OMP_THREAD_LIMIT','6'),('OMP_WAIT_POLICY','PASSIVE'),
        ('GOMP_SPINCOUNT','0'),('OPENBLAS_NUM_THREADS','1'),('MKL_NUM_THREADS','1'),('POLARS_MAX_THREADS','2')]:args+=['--setenv='+name+'='+value]
    args+=[root+'/'+node['python'],destination+'/night_cb_20261007.py','run','--input',destination,
        '--output',output,'--folds',node['folds'],'--threads','6']
    launch=remote(node,args);observed=status(node);check_limits(node,observed)
    W.write_json(state/(node['name']+'_launch.json'),dict(node=node,status=observed,launch=launch,time=time.time()))
    print('LAUNCHED',node['name'],json.dumps(observed),flush=True)
    return observed


def execute(root,commit):
    W.registration(root,commit);state=root/STATE
    if (state/'started.json').exists():raise ValueError('controller already attempted; no automatic retry')
    state.mkdir(parents=True,exist_ok=True);W.write_json(state/'started.json',dict(protocol=W.PROTOCOL,commit=commit,pid=os.getpid(),time=time.time()))
    try:
        bundle=root/BUNDLE;W.pack(root,bundle,commit)
        # Disjoint node copies/launches; no cross-node dataset or output mutation.
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures={node['name']:pool.submit(stage_node,node,bundle,state) for node in NODES}
            launched={name:future.result() for name,future in futures.items()}
        pending=list(NODES);deadline=time.monotonic()+6*3600+900
        while pending:
            if time.monotonic()>deadline:raise TimeoutError('bounded E205 collection wait exceeded')
            for node in list(pending):
                observed=status(node);check_limits(node,observed)
                if observed['InvocationID']!=launched[node['name']]['InvocationID']:raise ValueError('invocation changed '+node['name'])
                W.write_json(state/(node['name']+'_status.json'),observed)
                terminal=observed.get('SubState') in ('exited','failed','dead') or observed.get('ActiveState')=='failed'
                if not terminal:continue
                if observed.get('Result')!='success' or observed.get('ExecMainStatus')!='0' or observed.get('MainPID')!='0':raise ValueError('worker failed '+node['name']+' '+json.dumps(observed))
                remote(node,['test','-f',node['root']+'/'+outpath(node)+'/complete.json'])
                local=state/node['name']
                command(['scp','-q','-r',node['host']+':'+node['root']+'/'+outpath(node),str(local)],timeout=900)
                command(['scp','-q',node['host']+':'+node['root']+'/'+outpath(node)+'.log',str(state/(node['name']+'.log'))],timeout=120)
                W.write_json(state/(node['name']+'_terminal.json'),dict(status=observed,time=time.time()))
                pending.remove(node);print('COLLECTED',node['name'],flush=True)
            if pending:time.sleep(30)
        W.evaluate(bundle,[state/node['name'] for node in NODES],state/'evaluation.json')
        W.write_json(state/'complete.json',dict(protocol=W.PROTOCOL,commit=commit,evaluation_sha256=W.sha(state/'evaluation.json'),time=time.time(),accepted_for_submission=False))
    except BaseException as error:
        W.write_json(state/'failed.json',dict(protocol=W.PROTOCOL,commit=commit,time=time.time(),error=repr(error),traceback=traceback.format_exc()))
        raise


def launch(root,commit):
    W.registration(root,commit);state=root/STATE
    if state.exists():raise ValueError('fresh controller state required')
    state.mkdir(parents=True)
    with (state/'controller.log').open('x') as log:
        process=subprocess.Popen(['/usr/bin/caffeinate','-i',str(root/'.venv/bin/python'),
            str(root/'scripts/night_cb_20261007_queue.py'),'execute','--root',str(root),'--registration-commit',commit],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
    W.write_json(state/'launch.json',dict(pid=process.pid,commit=commit,time=time.time(),protocol=W.PROTOCOL))
    print(json.dumps(dict(controller_pid=process.pid,state=str(state))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['launch','execute'])
    parser.add_argument('--root',type=Path,default=Path('.'));parser.add_argument('--registration-commit',required=True)
    args=parser.parse_args();globals()[args.command](args.root.resolve(),args.registration_commit)

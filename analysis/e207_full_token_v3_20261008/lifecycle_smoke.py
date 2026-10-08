"""Bounded remote systemd lifecycle smoke; no project data or model execution."""
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import e207_full_token_queue_v3_20261008 as Q


def remote(host,args,check=True):
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join(args)],capture_output=True,text=True,timeout=20,check=check)


def state(host,unit):
    names=('ActiveState','SubState','MainPID','InvocationID','Result','ExecMainStatus','Description')
    args=['systemctl','--user','show',unit]
    for name in names:args.extend(['-p',name])
    return dict(line.split('=',1) for line in remote(host,args).stdout.splitlines() if '=' in line)


def node(host):
    result=[]
    for code in (0,7):
        owner='E207-v3-lifecycle-'+uuid.uuid4().hex;unit='prc-'+owner.lower()+'.service';initial=None
        item=dict(host=host,unit=unit,exit_code=code,owner=owner)
        try:
            remote(host,['systemd-run','--user','--unit='+unit,'--description='+owner,
                '--property=RemainAfterExit=yes','--property=CPUQuota=10%',
                '--property=MemoryMax=32M','--property=MemorySwapMax=0',
                '--property=KillMode=control-group','--property=TimeoutStopSec=3',
                '--property=RuntimeMaxSec=15','/bin/sh','-c',f'sleep 2; exit {code}'])
            initial=state(host,unit);item['initial']=initial
            if not initial['InvocationID']:raise AssertionError('missing initial invocation')
            deadline=time.monotonic()+20
            while True:
                last=state(host,unit)
                if last['InvocationID']!=initial['InvocationID']:raise AssertionError('real unit identity changed')
                if not Q.active(last):break
                if time.monotonic()>deadline:raise TimeoutError('synthetic unit did not finish')
                time.sleep(.25)
            item['terminal']=last
            if code==0:
                assert last['ActiveState']=='active' and last['SubState']=='exited' and last['MainPID']=='0'
                assert Q.unit_finished(last,initial)
                item['recognized']='terminal_success'
            else:
                assert last['Result']=='exit-code' and last['ExecMainStatus']=='7'
                try:Q.unit_finished(last,initial)
                except ValueError:item['recognized']='terminal_failure_rejected'
                else:raise AssertionError('nonzero terminal exit accepted')
            repeated=state(host,unit);item['retained_after_second_query']=repeated
            assert repeated['InvocationID']==initial['InvocationID']
        finally:
            live=state(host,unit)
            if live.get('Description')!=owner or (initial and live.get('InvocationID')!=initial['InvocationID']):
                raise AssertionError('cannot clean unrelated unit')
            remote(host,['systemctl','--user','stop',unit])
            remote(host,['systemctl','--user','reset-failed',unit],check=False)
            after=state(host,unit);item['after_cleanup']=after
            assert after['MainPID']=='0' and after['ActiveState']=='inactive'
        result.append(item)
    return result


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        reports=list(pool.map(node,['a1@192.168.1.171','a1@192.168.1.175','home-pc-lan']))
    receipt=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),synthetic_only=True,
        target_fits=0,passed=True,queue_sha256=Q.W.C.sha(Q.__file__),reports=reports)
    Q.W.C.write_json(Path(__file__).with_name('lifecycle_smoke.json'),receipt)
    print(json.dumps(receipt,indent=2))

"""Durable E203 current-chain check and conditional successor to E202.

Preparation may run while E202 trains; no GPU before completed E202 plus both
scientific gates. Late arming accepts exact durable completed output evidence.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import e203_trajectory_production as W
HOME='home-pc-lan'
REMOTE='/home/a1/prc-2026'
E202_UNIT='prc-e202-trajectory-20261007.service'
PREP_UNIT='prc-e203-rank-prep-20261007.service'
GPU_UNIT='prc-e203-ordered-rank-20261007.service'

def remote(args):
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOME,shlex.join(args)],text=True,timeout=60)

def state(unit):
    return dict(x.split('=',1) for x in remote(['systemctl','--user','show',unit,'-p','ActiveState','-p','MainPID','-p','InvocationID','-p','Result','-p','ExecMainStatus']).splitlines() if '=' in x)

def active(s):return s.get('ActiveState') in ('active','activating','deactivating','reloading') or int(s.get('MainPID','0'))>0

def exists(path):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOME,shlex.join(['test','-f',path])],capture_output=True,text=True,timeout=30)
    if p.returncode not in (0,1):raise RuntimeError('remote existence check failed '+p.stderr)
    return p.returncode==0

def terminal(unit,initial,deadline):
    while True:
        s=state(unit)
        if active(s):
            if s['InvocationID']!=initial['InvocationID']:raise ValueError('invocation changed '+unit)
        else:
            if s.get('Result') not in ('','success',None) or s.get('ExecMainStatus') not in ('','0',None):raise ValueError('unit failed '+repr(s))
            return s
        if time.monotonic()>deadline:raise TimeoutError('unit deadline '+unit)
        time.sleep(20)

def mac_available_bytes():
    # Reclaimable inactive/file-backed pages count; active E201 is never stopped.
    s=subprocess.check_output(['vm_stat'],text=True);page=int(re.search(r'page size of (\d+) bytes',s).group(1));values={}
    for line in s.splitlines()[1:]:
        if ':' in line:
            k,v=line.split(':',1);v=v.strip().rstrip('.')
            if v.isdigit():values[k]=int(v)
    return page*sum(values.get(k,0) for k in ('Pages free','Pages inactive','Pages speculative'))

def wait_mac_memory(minimum,deadline):
    while mac_available_bytes()<minimum:
        if time.monotonic()>deadline:raise TimeoutError('Mac memory reservation unavailable')
        time.sleep(20)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--registration-commit',required=True);a=p.parse_args();root=a.root.resolve();reg=a.registration_commit
    W.registration(root,reg);sp=root/'data/e203_execution_20261007.json';ep=root/'data/e203_execution_20261007.jsonl'
    if sp.exists() or ep.exists():raise ValueError('fresh controller required; no retry')
    current=dict(protocol=W.PROTOCOL,registration_commit=reg,pid=os.getpid(),started=datetime.datetime.now(datetime.timezone.utc).isoformat())
    def save(status,**data):
        e=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=status,**data);current.update(e);tmp=sp.with_suffix('.tmp');tmp.write_text(json.dumps(current,indent=2)+'\n');tmp.replace(sp)
        with ep.open('a') as f:f.write(json.dumps(e)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(e),flush=True)
    def local(action):
        wait_mac_memory(4*1024**3,time.monotonic()+8*3600)
        with (root/'data'/('e203_'+action+'_20261007.log')).open('x') as f:subprocess.run([sys.executable,'-u',str(root/W.SELF),action,'--root',str(root),'--registration-commit',reg,'--threads','2'],cwd=root,env={**os.environ,'OMP_THREAD_LIMIT':'2','OMP_NUM_THREADS':'2','POLARS_MAX_THREADS':'2'},stdout=f,stderr=subprocess.STDOUT,check=True,timeout=3600)
        save('local_action_complete',action=action)
    def worker(action):return [REMOTE+'/.venv-numemb/bin/python','-u',REMOTE+'/'+W.PACK+'/scripts/'+Path(W.SELF).name,action,'--root',REMOTE,'--registration-commit',reg]
    def launch(unit,cpu,mem,limit,command,log):
        if active(state(unit)):raise ValueError('duplicate active unit '+unit)
        remote(['systemd-run','--user','--unit='+unit,'--working-directory='+REMOTE,'--property=CPUQuota='+str(cpu)+'%','--property=MemoryMax='+mem,'--property=MemorySwapMax=0','--property=Nice=10','--property=RuntimeMaxSec='+str(limit),'--property=StandardOutput=append:'+REMOTE+'/data/'+log,'--property=StandardError=inherit',*command]);s=state(unit)
        if not active(s) or not s.get('InvocationID') or int(s.get('MainPID',0))<=0:raise ValueError('launch did not produce live process')
        save('unit_started',unit=unit,properties=s);return s
    try:
        save('pack_and_selected_cv')
        local('pack-rank')
        remote(['test','!','-e',REMOTE+'/'+W.PACK]);subprocess.run(['rsync','-a',str(root/W.PACK),HOME+':'+REMOTE+'/data/'],check=True,timeout=1800)
        # Only2.6GiB compressed existing raw sequences are transferred. No large
        # prepared tensor returns to Mac. Existing shared rawfiles are not overwritten.
        for shard in (0,1):
            relative=f'data/e194_sequence_20261007_s{shard}'
            subprocess.run(['rsync','-a','--ignore-existing',str(root/relative)+'/',HOME+':'+REMOTE+'/'+relative+'/'],check=True,timeout=1800)
        # Wait until transient E202 admissions have finished and reserve real
        # physical headroom before adding the bounded two-GiB preparation.
        prep_deadline=time.monotonic()+8*3600
        while True:
            admitted=all(exists(REMOTE+f'/data/e202_prepared_20261007/fold_{f}_{f+6}/admission.json') for f in (2,3,5,6))
            meminfo=remote(['cat','/proc/meminfo'])
            available=int(re.search(r'^MemAvailable:\s+(\d+) kB$',meminfo,re.M).group(1))*1024
            if admitted and available>=8*1024**3:break
            if time.monotonic()>prep_deadline:raise TimeoutError('Home preparation headroom/admission deadline')
            time.sleep(20)
        save('rank_prep_resource_admitted',available_bytes=available,all_e202_folds_admitted=True)
        prep=launch(PREP_UNIT,100,'2G',7200,worker('prepare-rank'),'e203_rank_prep_20261007.log')
        local('cv-selected')
        save('await_e202_remaining24fits',prep_invocation=prep['InvocationID'])
        deadline=time.monotonic()+8*3600;observed=None
        while True:
            s=state(E202_UNIT)
            if active(s):
                if observed is not None and s['InvocationID']!=observed:raise ValueError('E202 invocation changed while waiting')
                observed=s['InvocationID']
            elif exists(REMOTE+'/'+W.EXT_OUTPUT+'/complete.json') and exists(REMOTE+'/'+W.EXT_OUTPUT+'/evaluation.json'):
                if s.get('Result') not in ('','success',None) or s.get('ExecMainStatus') not in ('','0',None):raise ValueError('E202 terminal failure')
                # Wait for the E202 owner's independent collection, retaining
                # exact fixed source registration despite operational relaunch.
                independent=root/'data/e202_independent_review_20261007/evaluation.json'
                if independent.exists() and (root/W.EXT_OUTPUT/'evaluation.json').exists():
                    if W.A.read(independent)!=W.A.read(root/W.EXT_OUTPUT/'evaluation.json'):raise ValueError('E202 independent evaluation differs')
                    proof=W.extension(root)
                    if remote(['sha256sum',REMOTE+'/'+W.EXT_OUTPUT+'/complete.json']).split()[0]!=proof['complete_sha256']:raise ValueError('Home/Mac E202complete differs')
                    save('e202_completed_proof',observed_live_invocation=observed,late_armed=observed is None,proof=proof);break
            elif s.get('ActiveState')=='failed':raise ValueError('E202 failed without complete')
            if time.monotonic()>deadline:raise TimeoutError('E202 completion/collection exceeded8h')
            time.sleep(20)
        local('cv-confirmation');local('combine-cv');gate=W.A.read(root/W.CV_OUT/'result.json')
        if not gate['gate_pass']:
            save('complete_gate_failed_no_rank',current_chain_gate_pass=gate['current_chain_gate_pass'],extension_pass=gate['extension']['passed'],result_sha256=W.A.sha(root/W.CV_OUT/'result.json'));return
        terminal(PREP_UNIT,prep,time.monotonic()+7200)
        # CPU table verification and GPU smoke run under the same exclusive GPU
        # successor unit, before any real ranking fit. Each is a fresh process.
        local('authorize-rank');auth=root/W.PACK/'gate_authorization.json';subprocess.run(['scp',str(auth),HOME+':'+REMOTE+'/'+W.PACK+'/gate_authorization.json'],check=True,timeout=300)
        if remote(['sha256sum',REMOTE+'/'+W.PACK+'/gate_authorization.json']).split()[0]!=W.A.sha(auth):raise ValueError('gate transport differs')
        assets=root/'data/e203_queue_assets_20261007';assets.mkdir();script=assets/'gpu.sh'
        script.write_text('#!/bin/bash\nset -euo pipefail\n'+shlex.join(worker('smoke'))+'\n'+shlex.join(worker('rank'))+'\n')
        remote(['mkdir','-p',REMOTE+'/data/e203_queue_assets_20261007']);subprocess.run(['scp',str(script),HOME+':'+REMOTE+'/data/e203_queue_assets_20261007/gpu.sh'],check=True,timeout=300)
        if active(state(E202_UNIT)):raise ValueError('E202 became active before successor')
        gpu=launch(GPU_UNIT,200,'12G',7200,['/bin/bash',REMOTE+'/data/e203_queue_assets_20261007/gpu.sh'],'e203_rank_gpu_20261007.log')
        terminal(GPU_UNIT,gpu,time.monotonic()+7200)
        if not exists(REMOTE+'/'+W.RANK_OUT+'/complete.json'):raise ValueError('ranking ended without durablecomplete')
        subprocess.run(['rsync','-a',HOME+':'+REMOTE+'/'+W.RANK_OUT,str(root/'data/')+'/'],check=True,timeout=1800);W.verify_rank_output(root,reg)
        save('ranking_complete',complete_sha256=W.A.sha(root/W.RANK_OUT/'complete.json'))
        # Original main-chain LightGBM is a separate fresh process. Never
        # overlap the E201 Mac donor job or consume <2GiB disk headroom.
        deadline=time.monotonic()+12*3600
        while True:
            rows=subprocess.check_output(['ps','-axo','command='],text=True)
            live_e201=any('e201_donor_oof.py' in x and 'run-fold' in x for x in rows.splitlines())
            if not live_e201 and mac_available_bytes()>=8*1024**3:break
            if time.monotonic()>deadline:raise TimeoutError('Mac E201/production memory release deadline')
            time.sleep(30)
        local('capture-control');local('build')
        save('complete_candidate_no_upload',manifest_sha256=W.A.sha(root/W.FINAL/'manifest.json'),candidate_sha256=W.A.sha(root/W.FINAL/'candidate.parquet'))
    except BaseException as e:save('failed_no_retry',error=repr(e));raise

if __name__=='__main__':
    if '--launch' in sys.argv:
        args=[v for v in sys.argv[1:] if v!='--launch']
        root=Path(args[args.index('--root')+1]).resolve()
        reg=args[args.index('--registration-commit')+1]
        W.registration(root,reg)
        with (root/'data/e203_controller_20261007.log').open('x') as f:
            child=subprocess.Popen(['/usr/bin/caffeinate','-i',sys.executable,'-u',str(Path(__file__).resolve()),*args],cwd=root,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print(child.pid,flush=True)
    else:main()

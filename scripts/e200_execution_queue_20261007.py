"""One-shot durable Mac controller for the registered E200 fixed sequence."""
from pathlib import Path
import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
import time
import e200_mdn_production as Q

UNIT='prc-e200-mdn-rank-20261007.service'
HOST='home-pc-lan'
REMOTE='/home/a1/prc-2026'

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--registration-commit',required=True);a=p.parse_args()
    root=a.root.resolve();reg=a.registration_commit;statepath=root/'data/e200_execution_20261007.json'
    if statepath.exists():raise ValueError('fresh one-shot controller only; no automatic retry')
    Q.registration(root,reg)
    state=dict(protocol=Q.PROTOCOL,registration_commit=reg,pid=os.getpid(),started=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='starting')
    def save(status,**extra):
        state.update(status=status,updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
        tmp=statepath.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(statepath);print(status,flush=True)
    def local(action):return [sys.executable,'-u',str(root/Q.SELF),action,'--root',str(root),'--registration-commit',reg]
    def remote(args):
        return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,shlex.join(args)],text=True)
    def run_logged(action):
        with (root/f'data/e200_{action}_20261007.log').open('x') as f:subprocess.run(local(action),cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
    capture=None
    try:
        save('cv_running');run_logged('cv');gate=Q.A.read(root/Q.CV_OUT/'result.json')
        if not gate['gate_pass']:save('negative_gate_no_ranking',gate_sha256=Q.A.sha(root/Q.CV_OUT/'result.json'));return
        save('gate_pass_pack_rank');run_logged('pack-rank')
        subprocess.run(['rsync','-a',str(root/Q.PACK),HOST+':'+REMOTE+'/data/'],check=True)
        save('arming_gpu_dependency')
        print(remote(['systemd-run','--user','--unit='+UNIT,'--working-directory='+REMOTE,
            '--property=CPUQuota=200%','--property=MemoryMax=12G','--property=MemorySwapMax=0','--property=Nice=10','--property=RuntimeMaxSec=45000',
            '--property=StandardOutput=append:'+REMOTE+'/data/e200_rank_20261007.log','--property=StandardError=inherit',
            REMOTE+'/.venv-numemb/bin/python','-u',REMOTE+'/'+Q.PACK+'/scripts/'+Path(Q.SELF).name,
            'queue-rank','--root',REMOTE,'--registration-commit',reg]),flush=True)
        gpu=remote(['systemctl','--user','show',UNIT,'-p','ActiveState','-p','MainPID','-p','InvocationID','-p','Result'])
        initial=dict(x.split('=',1) for x in gpu.splitlines() if '=' in x)
        if initial.get('ActiveState')!='active' or int(initial.get('MainPID','0'))<=0:raise ValueError('GPU dependency queue did not start')
        with (root/'data/e200_capture-control_20261007.log').open('x') as f:
            capture=subprocess.Popen(local('capture-control'),cwd=root,stdout=f,stderr=subprocess.STDOUT)
        save('gpu_queued_and_cpu_control_running',gpu_unit=UNIT,gpu_invocation=initial['InvocationID'],gpu_pid=int(initial['MainPID']),cpu_pid=capture.pid)
        deadline=time.monotonic()+12*3600
        while True:
            state['cpu_returncode']=capture.poll()
            raw=remote(['systemctl','--user','show',UNIT,'-p','ActiveState','-p','MainPID','-p','InvocationID','-p','Result'])
            live=dict(x.split('=',1) for x in raw.splitlines() if '=' in x)
            active=live.get('ActiveState') in ('active','activating','deactivating') or int(live.get('MainPID','0'))>0
            if active and live.get('InvocationID')!=initial['InvocationID']:raise ValueError('GPU queue invocation changed')
            if not active:
                if live.get('Result') not in (None,'','success'):raise ValueError('GPU queue failed; no retry')
                remote(['test','-f',REMOTE+'/'+Q.RANK_OUT+'/complete.json']);break
            if time.monotonic()>deadline:raise TimeoutError('GPU dependency/ranking exceeded12h')
            save('waiting_gpu',cpu_returncode=capture.poll());time.sleep(30)
        save('collecting_rank')
        subprocess.run(['rsync','-a',HOST+':'+REMOTE+'/'+Q.RANK_OUT,str(root/'data/')+'/'],check=True)
        subprocess.run(['scp',HOST+':'+REMOTE+'/data/e200_rank_20261007.log',str(root/'data/e200_rank_20261007.log')],check=True)
        subprocess.run(['scp',HOST+':'+REMOTE+'/data/e200_gpu_dependency_20261007.json',str(root/'data/e200_gpu_dependency_20261007.json')],check=True)
        Q.verify_rank_output(root,reg);save('ranking_verified_waiting_cpu',cpu_returncode=capture.poll())
        if capture.wait()!=0:raise ValueError('CPU control capture failed; member retained, no candidate build')
        save('building_full_candidate');run_logged('build')
        r=Q.A.read(root/Q.FINAL/'manifest.json');save('candidate_complete_no_upload',candidate=str(root/Q.FINAL/'candidate.parquet'),candidate_sha256=r['sha256'])
    except BaseException as exc:
        save('failed_no_retry',error=repr(exc),cpu_pid=capture.pid if capture else None,cpu_returncode=capture.poll() if capture else None)
        raise

if __name__=='__main__':main()

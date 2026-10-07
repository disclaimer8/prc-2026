"""One-shot E202 pack, laptop preparation, streaming admission and GPU collection.

All scientific computation lives in the frozen E202/E197 bundle. Laptop CPU
release is independent of transfer completion. No fit or preparation retries.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
import e202_trajectory_extension as Q

HOME='home-pc-lan'
REMOTE='/home/a1/prc-2026'
GPU_UNIT='prc-e202-trajectory-20261007.service'
OLD_UNIT='prc-e197-trajectory-20261007.service'
NODES={171:dict(host='a1@192.168.1.171',env='.venv',folds=(2,5)),
       175:dict(host='a1@192.168.1.175',env='.venv-cat-parity',folds=(3,6))}


def remote(host,args):
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join(args)],text=True)


def live(host,unit):
    s=remote(host,['systemctl','--user','show',unit,'-p','ActiveState','-p','MainPID','-p','InvocationID','-p','Result','-p','ExecMainStatus'])
    return dict(x.split('=',1) for x in s.splitlines() if '=' in x)


def active(s):return s.get('ActiveState') in ('active','activating','deactivating') or int(s.get('MainPID','0'))>0


def successful_exit(s):
    if active(s) or s.get('Result')!='success' or s.get('ExecMainStatus')!='0':raise ValueError('unit failed or did not finish successfully: '+repr(s))


def file_exists(host,path):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join(['test','-f',path])],capture_output=True,text=True)
    if p.returncode not in (0,1):raise RuntimeError('SSH read failed: '+p.stderr)
    return p.returncode==0


def free_space(host,minimum):
    p=remote(host,['df','-Pk',REMOTE]);available=int(p.splitlines()[-1].split()[3])*1024
    if available<minimum:raise ValueError('insufficient free disk on '+host)
    return available


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--registration-commit',required=True);a=p.parse_args()
    root=a.root.resolve();reg=a.registration_commit;statepath=root/'data/e202_execution_20261007.json';eventpath=root/'data/e202_execution_20261007.jsonl'
    if statepath.exists() or eventpath.exists():raise ValueError('fresh one-shot controller required; no automatic retry')
    Q.registration(root,reg);lock=threading.Lock();admission_lock=threading.Lock()
    state=dict(protocol=Q.PROTOCOL,registration_commit=reg,pid=os.getpid(),started=datetime.datetime.now(datetime.timezone.utc).isoformat(),nodes={})
    def save(status,**extra):
        with lock:
            event=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=status,**extra)
            state.update(status=status,updated=event['time'],**extra)
            tmp=statepath.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(statepath)
            with eventpath.open('a') as f:f.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
    def local(action):return [sys.executable,'-u',str(root/Q.SELF),action,'--root',str(root),'--registration-commit',reg]
    def worker(env,action,*args):return [REMOTE+'/'+env+'/bin/python','-u',REMOTE+'/'+Q.INPUT+'/scripts/'+Path(Q.SELF).name,action,'--root',REMOTE,'--registration-commit',reg,*args]
    def launch(host,unit,cpu,mem,limit,command,log):
        if active(live(host,unit)):raise ValueError('unit already active; duplicate launch refused')
        result=remote(host,['systemd-run','--user','--unit='+unit,'--working-directory='+REMOTE,
            '--property=CPUQuota='+str(cpu)+'%','--property=MemoryMax='+mem,'--property=MemorySwapMax=0',
            '--property=Nice=10','--property=RuntimeMaxSec='+str(limit),
            '--property=StandardOutput=append:'+REMOTE+'/data/'+log,'--property=StandardError=inherit',*command])
        s=live(host,unit)
        if not active(s) or int(s['MainPID'])<=0 or not s.get('InvocationID'):raise ValueError('unit did not start: '+result+repr(s))
        save('unit_started',host=host,unit=unit,unit_state=s)
        return s
    def wait_unit(host,unit,initial,deadline):
        while True:
            s=live(host,unit)
            if active(s):
                if s.get('InvocationID')!=initial['InvocationID']:raise ValueError('unit invocation changed')
            else:successful_exit(s);return s
            if time.monotonic()>deadline:raise TimeoutError('unit exceeded controller deadline')
            time.sleep(15)
    try:
        save('packing')
        with (root/'data/e202_pack_20261007.log').open('x') as f:subprocess.run(local('pack'),cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
        m=Q.C.read_json(root/Q.INPUT/'manifest.json');spec=m['spec_sha256'];save('packed',input_spec_sha256=spec)
        # Historical E197 is complete. Pin its published receipt and require no
        # active old unit; this is not a fragile "must observe predecessor live".
        if active(live(HOME,OLD_UNIT)):raise ValueError('E197 still active')
        oldsha=remote(HOME,['sha256sum',REMOTE+'/'+Q.OLD_OUTPUT+'/complete.json']).split()[0]
        if oldsha!=m['spec']['screen_complete_sha256']:raise ValueError('Home original completed screen differs')
        disks={HOME:free_space(HOME,45*1024**3)}
        for node,n in NODES.items():disks[str(node)]=free_space(n['host'],24*1024**3)
        save('disk_and_predecessor_verified',free_bytes=disks,old_complete_sha256=oldsha)
        def transfer_bundle(host):
            remote(host,['test','!', '-e',REMOTE+'/'+Q.INPUT])
            subprocess.run(['rsync','-a',str(root/Q.INPUT),host+':'+REMOTE+'/data/'],check=True)
        with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(transfer_bundle,[HOME,*[n['host'] for n in NODES.values()]]))
        remote(HOME,['mkdir','-p',REMOTE+'/'+Q.INCOMING])
        initial={}
        for node,n in NODES.items():
            unit=f'prc-e202-prep-{node}-20261007.service'
            command=['systemd-inhibit','--what=sleep:idle','--mode=block','--who=PRC-E202','--why=E202-sequence-preparation',*worker(n['env'],'prepare-node','--node',str(node))]
            initial[node]=launch(n['host'],unit,600,'10G',14400,command,f'e202_prep_{node}_20261007.log')
        gpu=launch(HOME,GPU_UNIT,200,'12G',18000,worker('.venv-numemb','run'),'e202_gpu_20261007.log')
        save('preparation_and_gpu_wait_running',gpu_unit=GPU_UNIT,gpu_pid=int(gpu['MainPID']),gpu_invocation=gpu['InvocationID'],prep_units=initial)
        # Per-host pipelines run independently. The training order is fixed
        # 2,3,5,6; fold2 can fit while later folds are still in transit.
        def stream_node(node):
            n=NODES[node];host=n['host'];prepunit=f'prc-e202-prep-{node}-20261007.service';deadline=time.monotonic()+4*3600
            for fold in n['folds']:
                name=f'fold_{fold}_{fold+6}';folder=REMOTE+'/'+Q.PREPARED+'/'+name
                while not file_exists(host,folder+'/complete.json'):
                    s=live(host,prepunit)
                    if not active(s):raise ValueError('preparation ended before requested fold published: '+repr(s))
                    if s.get('InvocationID')!=initial[node]['InvocationID']:raise ValueError('preparation invocation changed')
                    if time.monotonic()>deadline:raise TimeoutError('preparation/transfer timeout')
                    time.sleep(10)
                save('streaming_fold',node=node,fold=fold)
                subprocess.run(['scp','-3','-r','-o','BatchMode=yes',host+':'+folder,HOME+':'+REMOTE+'/'+Q.INCOMING+'/'],check=True)
                with admission_lock:
                    unit=f'prc-e202-admit-f{fold}-20261007.service'
                    admission=launch(HOME,unit,100,'1G',1800,worker('.venv-numemb','admit','--fold',str(fold)),f'e202_admit_f{fold}_20261007.log')
                    wait_unit(HOME,unit,admission,time.monotonic()+1800)
                    final=REMOTE+'/'+Q.PREPARED+'/'+name
                    receipt=json.loads(remote(HOME,['cat',final+'/admission.json']))
                    if receipt['registration_commit']!=reg or receipt['input_spec_sha256']!=spec or receipt['fold']!=fold or not receipt['passed']:raise ValueError('Home admission receipt differs')
                    dest=root/'data/e202_preparation_receipts_20261007'/name;dest.mkdir(parents=True)
                    for filename in ('complete.json','normalization.json','admission.json'):
                        subprocess.run(['scp','-o','BatchMode=yes',HOME+':'+final+'/'+filename,str(dest/filename)],check=True)
                    save('fold_admitted',node=node,fold=fold,admission_sha256=Q.C.sha(dest/'admission.json'))
            wait_unit(host,prepunit,initial[node],deadline)
            marker=json.loads(remote(host,['cat',REMOTE+f'/data/e202_handoff_{node}_20261007.json']))
            if marker['status']!='two_prepared_complete' or marker['registration_commit']!=reg or marker['input_spec_sha256']!=spec or marker['folds']!=list(n['folds']):raise ValueError('CPU release marker differs')
            # Marker was written at CPU completion, before these transfers.
            save('node_streaming_complete',node=node,cpu_release_marker=marker)
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(stream_node,NODES))
        save('all_four_folds_admitted_waiting_gpu')
        terminal=wait_unit(HOME,GPU_UNIT,gpu,time.monotonic()+5*3600)
        remote(HOME,['test','-f',REMOTE+'/'+Q.OUTPUT+'/evaluation.json'])
        save('collecting_completed24fits',gpu_terminal=terminal)
        subprocess.run(['rsync','-a',HOME+':'+REMOTE+'/'+Q.OUTPUT,str(root/'data/')+'/'],check=True)
        subprocess.run(['scp','-o','BatchMode=yes',HOME+':'+REMOTE+'/data/e202_gpu_20261007.log',str(root/'data/e202_gpu_20261007.log')],check=True)
        # Re-evaluate in an isolated hardlinked copy, preserving original
        # published output and evaluating all24fits/IDs/seed arithmetic again.
        import shutil
        review=root/'data/e202_independent_review_20261007'
        shutil.copytree(root/Q.OUTPUT,review,copy_function=os.link,ignore=shutil.ignore_patterns('evaluation.json'))
        Q.evaluate(root,root/Q.INPUT,review,reg)
        if Q.C.read_json(review/'evaluation.json')!=Q.C.read_json(root/Q.OUTPUT/'evaluation.json'):raise ValueError('independent evaluation differs')
        save('complete_verified_no_ranking',complete_sha256=Q.C.sha(root/Q.OUTPUT/'complete.json'),evaluation_sha256=Q.C.sha(root/Q.OUTPUT/'evaluation.json'))
    except BaseException as exc:
        save('failed_no_retry',error=repr(exc));raise


if __name__=='__main__':main()

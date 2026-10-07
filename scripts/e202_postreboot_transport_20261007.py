"""Recover only E202 transport after Mac reboot; original GPU fit keeps running.

No preparation/model retry. Preserve incomplete incoming files before replacing
them. Admit immutable tables using the original registered worker and collect
the original 24-fit result for an independent unchanged evaluation.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import e202_trajectory_extension as Q

REG='55fe9b68f7cc8405681f6c85d576979e067b55c3'
SPEC='6d604eaaca830fa83c18ebb78986b028d338869123d5f8c10b2450dde1ec4309'
GPU_UNIT='prc-e202-trajectory-20261007.service'
GPU_INV='30d9322da32d4cfdb55ef5c7c3d5ecbf'
HOME='home-pc-lan';REMOTE='/home/a1/prc-2026'
NODES={171:('a1@192.168.1.171',(2,5)),175:('a1@192.168.1.175',(3,6))}
PROTOCOL='E202-postreboot-transport-only-original-gpu-no-refit-v1'
SELF='scripts/e202_postreboot_transport_20261007.py'
DOC='analysis/e202_postreboot_transport_protocol_20261007.md'
FILES={'values.npy','known.npy','present.npy','support.npy','query_ids.npy','normalization.json'}

def remote(host,args,timeout=120):
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join(args)],text=True,timeout=timeout)
def exists(host,path):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join(['test','-f',path])],capture_output=True,text=True,timeout=30)
    if r.returncode not in (0,1):raise RuntimeError('remote read failed '+r.stderr)
    return r.returncode==0
def rsha(host,path):return remote(host,['sha256sum',path],timeout=300).split()[0]
def live(unit):
    s=remote(HOME,['systemctl','--user','show',unit,'-p','ActiveState','-p','MainPID','-p','InvocationID','-p','Result','-p','ExecMainStatus'])
    return dict(x.split('=',1) for x in s.splitlines() if '=' in x)
def active(s):return s.get('ActiveState') in ('active','activating','deactivating') or int(s.get('MainPID','0'))>0
def success(s):
    if active(s) or s.get('Result') not in ('','success') or s.get('ExecMainStatus') not in ('','0'):raise ValueError('unit did not exit successfully '+str(s))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--recovery-commit',required=True);a=p.parse_args();root=a.root.resolve()
    if not re.fullmatch('[a-f0-9]{40}',a.recovery_commit):raise ValueError('full recovery registration required')
    for path in (SELF,DOC):
        raw=subprocess.check_output(['git','show',a.recovery_commit+':'+path],cwd=root)
        if hashlib.sha256(raw).hexdigest()!=Q.C.sha(root/path):raise ValueError('recovery code/protocol drift')
    if PROTOCOL.encode() not in subprocess.check_output(['git','show',a.recovery_commit+':HYPOTHESES_LOG.md'],cwd=root):raise ValueError('recovery not registered')
    Q.registration(root,REG);bundle=root/Q.INPUT;m=Q.verify_bundle(root,bundle,REG,queries=True)
    if m['spec_sha256']!=SPEC:raise ValueError('frozen E202 input changed')
    statepath=root/'data/e202_postreboot_transport_20261007.json';eventpath=statepath.with_suffix('.jsonl')
    if statepath.exists() or eventpath.exists():raise ValueError('fresh one-shot recovery required')
    lock=threading.Lock();admission_lock=threading.Lock();started=time.monotonic()
    current=dict(protocol=PROTOCOL,recovery_commit=a.recovery_commit,registration_commit=REG,input_spec_sha256=SPEC,pid=os.getpid(),gpu_invocation=GPU_INV)
    def save(status,**extra):
        with lock:
            event=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=status,**extra);current.update(event)
            tmp=statepath.with_suffix('.tmp');tmp.write_text(json.dumps(current,indent=2)+'\n');tmp.replace(statepath)
            with eventpath.open('a') as f:f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
            print(json.dumps(event),flush=True)
    def gpu_state():
        s=live(GPU_UNIT)
        if active(s):
            if s['InvocationID']!=GPU_INV:raise ValueError('original GPU invocation changed')
        else:
            success(s)
            if not exists(HOME,REMOTE+'/'+Q.OUTPUT+'/complete.json'):raise ValueError('original GPU ended without complete; no refit allowed')
        return s
    def verify_admission(path,fold,complete_sha,files):
        r=json.loads(remote(HOME,['cat',path+'/admission.json']))
        if r['registration_commit']!=REG or r['input_spec_sha256']!=SPEC or r['fold']!=fold or not r['passed'] or r['prepared_complete_sha256']!=complete_sha or r['files']!=files:raise ValueError('admission proof differs')
        if rsha(HOME,path+'/complete.json')!=complete_sha:raise ValueError('admitted complete drift')
        return r
    def preserve(path,fold,name):
        archive=REMOTE+f'/data/e202_postreboot_archive_20261007/fold_{fold}_{fold+6}'
        remote(HOME,['mkdir','-p',archive]);dest=archive+'/'+name
        remote(HOME,['test','!','-e',dest]);remote(HOME,['mv','--',path,dest])
        save('partial_preserved',fold=fold,file=name,archive=dest)
    def stream(node):
        host,folds=NODES[node];marker=json.loads(remote(host,['cat',REMOTE+f'/data/e202_handoff_{node}_20261007.json']))
        if marker['status']!='two_prepared_complete' or marker['registration_commit']!=REG or marker['input_spec_sha256']!=SPEC or marker['folds']!=list(folds) or str(marker['node'])!=str(node):raise ValueError('prepared CPU handoff differs')
        if sorted(x['fold'] for x in marker['prepared_receipts'])!=list(folds):raise ValueError('prepared fold population differs')
        for fold in folds:
            gpu_state();name=f'fold_{fold}_{fold+6}';source=REMOTE+'/'+Q.PREPARED+'/'+name;incoming=REMOTE+'/'+Q.INCOMING+'/'+name;final=REMOTE+'/'+Q.PREPARED+'/'+name
            expected=next(x for x in marker['prepared_receipts'] if x['fold']==fold)
            complete=json.loads(remote(host,['cat',source+'/complete.json']))
            if rsha(host,source+'/complete.json')!=expected['complete_sha256'] or complete['files']!=expected['files'] or set(complete['files'])!=FILES or complete['registration_commit']!=REG or complete['input_spec_sha256']!=SPEC or complete['fold']!=fold or complete['source_sha256']!=Q.C.sha(root/Q.SELF):raise ValueError('immutable source preparation differs')
            if exists(HOME,final+'/admission.json'):
                verify_admission(final,fold,expected['complete_sha256'],complete['files']);save('already_admitted',fold=fold);continue
            remote(HOME,['test','!','-e',final]);remote(HOME,['mkdir','-p',incoming])
            # Leave complete.json until last; admission can only use fully checked bytes.
            hashes={**complete['files'],'complete.json':expected['complete_sha256']}
            for filename,expected_sha in hashes.items():
                target=incoming+'/'+filename
                if exists(HOME,target):
                    if rsha(HOME,target)==expected_sha:
                        save('verified_existing_file',fold=fold,file=filename);continue
                    preserve(target,fold,filename)
                if rsha(host,source+'/'+filename)!=expected_sha:raise ValueError('source bytes changed before copy')
                save('copying_missing_file',fold=fold,file=filename,node=node)
                subprocess.run(['scp','-3','-o','BatchMode=yes',host+':'+source+'/'+filename,HOME+':'+target],check=True,timeout=1800)
                if rsha(HOME,target)!=expected_sha:raise ValueError('transferred file checksum differs')
                save('file_verified',fold=fold,file=filename)
            if exists(HOME,incoming+'/admission.json'):preserve(incoming+'/admission.json',fold,'admission.json')
            with admission_lock:
                gpu_state();unit=f'prc-e202-postreboot-admit-f{fold}-20261007.service'
                if active(live(unit)):raise ValueError('duplicate admission unit')
                command=[REMOTE+'/.venv-numemb/bin/python','-u',REMOTE+'/'+Q.INPUT+'/scripts/'+Path(Q.SELF).name,'admit','--root',REMOTE,'--registration-commit',REG,'--fold',str(fold)]
                remote(HOME,['systemd-run','--user','--unit='+unit,'--working-directory='+REMOTE,'--property=CPUQuota=100%','--property=MemoryMax=1G','--property=MemorySwapMax=0','--property=Nice=10','--property=RuntimeMaxSec=1800','--property=StandardOutput=append:'+REMOTE+f'/data/e202_postreboot_admit_f{fold}_20261007.log','--property=StandardError=inherit',*command])
                first=live(unit)
                if not active(first) or not first.get('InvocationID'):raise ValueError('admission did not start')
                save('admission_started',fold=fold,unit=unit,properties=first);deadline=time.monotonic()+1800
                while True:
                    s=live(unit)
                    if not active(s):success(s);break
                    if s['InvocationID']!=first['InvocationID']:raise ValueError('admission invocation changed')
                    if time.monotonic()>deadline:raise TimeoutError('admission timeout')
                    time.sleep(10)
                verify_admission(final,fold,expected['complete_sha256'],complete['files'])
                receipt_dir=root/'data/e202_preparation_receipts_20261007'/name;receipt_dir.mkdir(parents=True,exist_ok=True)
                for filename in ('complete.json','normalization.json','admission.json'):
                    target=receipt_dir/filename;h=rsha(HOME,final+'/'+filename)
                    if target.exists():
                        if Q.C.sha(target)!=h:raise ValueError('existing local preparation receipt differs')
                    else:subprocess.run(['scp','-o','BatchMode=yes',HOME+':'+final+'/'+filename,str(target)],check=True,timeout=120)
                save('fold_admitted',fold=fold,admission_sha256=Q.C.sha(receipt_dir/'admission.json'))
    try:
        save('verified_reboot_recovery',gpu_state=gpu_state())
        if rsha(HOME,REMOTE+'/'+Q.INPUT+'/manifest.json')!=Q.C.sha(bundle/'manifest.json'):raise ValueError('Home frozen input differs')
        remote(HOME,['mkdir','-p',REMOTE+'/'+Q.INCOMING])
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(stream,NODES))
        save('four_folds_admitted_original_gpu_continues')
        deadline=time.monotonic()+5*3600
        while active(gpu_state()):
            if time.monotonic()>deadline:raise TimeoutError('original GPU completion timeout')
            time.sleep(20)
        remote(HOME,['test','-f',REMOTE+'/'+Q.OUTPUT+'/evaluation.json'])
        if (root/Q.OUTPUT).exists():raise ValueError('local GPU output already exists; preserve rather than overwrite')
        save('collecting_original24fits')
        subprocess.run(['rsync','-a',HOME+':'+REMOTE+'/'+Q.OUTPUT,str(root/'data/')+'/'],check=True,timeout=1800)
        for filename in ('complete.json','evaluation.json'):
            if rsha(HOME,REMOTE+'/'+Q.OUTPUT+'/'+filename)!=Q.C.sha(root/Q.OUTPUT/filename):raise ValueError('collected completion differs')
        review=root/'data/e202_independent_review_20261007'
        shutil.copytree(root/Q.OUTPUT,review,copy_function=os.link,ignore=shutil.ignore_patterns('evaluation.json'))
        Q.evaluate(root,bundle,review,REG)
        if Q.C.read_json(review/'evaluation.json')!=Q.C.read_json(root/Q.OUTPUT/'evaluation.json'):raise ValueError('independent evaluation differs')
        save('complete_verified_no_ranking',complete_sha256=Q.C.sha(root/Q.OUTPUT/'complete.json'),evaluation_sha256=Q.C.sha(root/Q.OUTPUT/'evaluation.json'),seconds=time.monotonic()-started)
    except BaseException as exc:save('failed_no_retry',error=repr(exc));raise

if __name__=='__main__':main()

"""Exact qualification and conditional transfer of three untouched Cat90 jobs.

The frozen four-node queue is never edited. Laptop171 is drained after its
seven retained fits only if laptop175 qualifies and activates the transfer.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
import numpy as np
import cat_night_queue as Q
C,R,L=Q.C,Q.R,Q.L
BASE_REG='41129d5003aa54619259e67dd65af974860e2a1e'
BASE_LINEAGE='6cd125245ccb67db4d4add728aa4c7ada68c7bb1e3a8a1b1d17e4281082acfa7'
QUAL_REG='55304792ee0bc0e39868c9138904723522f5fee9'
UNIT='prc-cat-night-laptop-20261006.service'
WATCH_UNIT='prc-cat-extra175-drain-20261006.service'
PROTOCOL='cat90-extra175-exact-conditional-tail3-v1'


def assignments():
    a=Q.assignments();a['laptop175']=a['laptop'][-3:];a['laptop']=a['laptop'][:-3]
    return a


def validate_assignments(a):
    jobs=[tuple(j) for v in a.values() for j in v]
    if a!=assignments() or len(jobs)!=90 or len(set(jobs))!=90 or set(jobs)!=set(C.schedule()):
        raise ValueError('overlay must cover all90 exactly once')


def drained(completed,retained):
    done=[tuple(j) for j in completed];wanted=set(map(tuple,retained))
    if len(set(done))!=len(done) or not set(done)<=wanted:raise ValueError('donor has duplicate or transferred completed jobs')
    return set(done)==wanted


def require_qualification(r,runtime,pred_sha):
    expected=dict(exact=True,max_abs=0.,rows=344419,runtime=runtime,registration_commit=QUAL_REG,prediction_sha256=pred_sha)
    if any(r[k]!=v for k,v in expected.items()):raise ValueError('extra laptop exact qualification failed')


def register(bundle,qualification,out,registration):
    b=Q.verify(bundle,BASE_REG)
    if b['lineage']!=BASE_LINEAGE:raise ValueError('wrong base execution')
    q=C.read_json(qualification/'manifest.json')
    if q['registration_commit']!=QUAL_REG or q['worker_sha256']!=C.sha(L.__file__):raise ValueError('wrong qualification source')
    for name,h in q['files'].items():
        if C.sha(qualification/name)!=h:raise ValueError('qualification input changed')
    out.mkdir(parents=True,exist_ok=False)
    m=dict(protocol=PROTOCOL,registration=registration,assignments=assignments(),base_lineage=BASE_LINEAGE,
        base_manifest_sha256=C.sha(bundle/'manifest.json'),pool_seal_sha256=C.sha(bundle/'pools_complete.json'),
        qualification_manifest_sha256=C.sha(qualification/'manifest.json'),worker_sha256=C.sha(__file__),helpers=Q.helpers(),
        packages=b['runtime_packages'],cpu_threads=6,accepted_for_submission=False,
        activation='after exact qualification; donor tail must be unstarted; verified watchdog armed before new production')
    validate_assignments(m['assignments']);m['lineage']=C.digest(m);C.write_json(out/'manifest.json',m)


def verify(bundle,overlay,registration):
    b=Q.verify(bundle,BASE_REG);m=C.read_json(overlay/'manifest.json')
    if m['protocol']!=PROTOCOL or m['registration']!=registration or m['lineage']!=C.digest({k:v for k,v in m.items() if k!='lineage'}):raise ValueError('overlay registration drift')
    if m['worker_sha256']!=C.sha(__file__) or m['helpers']!=Q.helpers() or b['lineage']!=m['base_lineage'] or C.sha(bundle/'manifest.json')!=m['base_manifest_sha256'] or C.sha(bundle/'pools_complete.json')!=m['pool_seal_sha256']:raise ValueError('overlay input/source drift')
    validate_assignments(m['assignments']);return b,m


def donor_results(snapshot,basework,b,m):
    inv=C.read_json(basework/'invocation.json')
    if inv['lineage']!=BASE_LINEAGE or inv['node']!='laptop' or inv['tasks']!=b['assignments']['laptop']:raise ValueError('wrong donor invocation')
    done=[];hashes={}
    for p in basework.glob('fold_*/seed_*/receipt.json'):
        r=C.read_json(p);seed,fold=r['seed'],r['fold']
        Q.check_result(p.parent,b,'laptop',seed,fold,Q.metadata(snapshot,fold)['mvt_id'])
        if r['runtime']!=inv['runtime']:raise ValueError('donor runtime differs')
        done.append([seed,fold]);hashes[f'{seed}:{fold}']=C.sha(p)
    drained(done,m['assignments']['laptop'])
    return done,hashes


def active():return subprocess.run(['systemctl','--user','is-active','--quiet',UNIT]).returncode==0


def publish_armed(path,data):
    if path.exists():raise ValueError('activation already published')
    temporary=path.parent/('.armed-'+uuid.uuid4().hex)
    C.write_json(temporary,data);os.rename(temporary,path)


def reserved_started(basework,jobs):
    return any((basework/'attempts'/f'{Q.name(fold)}-s{seed:02}').exists() or
               (basework/f'{Q.name(fold)}-s{seed:02}.log').exists() for seed,fold in jobs)


def activate(snapshot,bundle,overlay,work,basework,registration):
    requested=os.environ.get('SSH_ORIGINAL_COMMAND','')
    if requested not in ('','preflight'):raise ValueError('unsupported restricted SSH request')
    b,m=verify(bundle,overlay,registration)
    if work.exists():raise ValueError('activation exists; explicit recovery required')
    donor_results(snapshot,basework,b,m)
    if reserved_started(basework,m['assignments']['laptop175']):raise ValueError('reserved tail already started')
    if not active():raise ValueError('donor is not active')
    if requested=='preflight':
        print(json.dumps(dict(lineage=m['lineage'],preflight=True,transferred=m['assignments']['laptop175'])),flush=True)
        return
    args=[sys.executable,str(Path(__file__).resolve()),'watch','--snapshot',str(snapshot),'--bundle',str(bundle),
          '--overlay',str(overlay),'--work',str(work),'--basework',str(basework),'--registration-commit',registration]
    subprocess.run(['systemd-run','--user',f'--unit={WATCH_UNIT}','--property=Nice=10','--property=CPUQuota=100%',
                    '--property=MemoryMax=1G','--property=MemorySwapMax=0',*args],check=True,stdout=subprocess.DEVNULL)
    end=time.monotonic()+30
    while not (work/'armed.json').exists():
        if time.monotonic()>end:raise ValueError('watchdog did not arm')
        time.sleep(.5)
    armed=C.read_json(work/'armed.json')
    if armed['lineage']!=m['lineage']:raise ValueError('wrong watchdog activation')
    state=subprocess.run(['systemctl','--user','is-active','--quiet',WATCH_UNIT]).returncode==0
    if not state and not (work/'drained.json').exists():raise ValueError('watchdog failed to stay active')
    print(json.dumps(armed),flush=True)


def watch(snapshot,bundle,overlay,work,basework,registration):
    b,m=verify(bundle,overlay,registration);Q.original(snapshot)
    work.mkdir(parents=True,exist_ok=False)
    publish_armed(work/'armed.json',dict(lineage=m['lineage'],unit=UNIT,retained=m['assignments']['laptop'],transferred=m['assignments']['laptop175']))
    deadline=time.monotonic()+36*3600
    while True:
        done,hashes=donor_results(snapshot,basework,b,m)
        if drained(done,m['assignments']['laptop']):break
        if not active() or time.monotonic()>deadline:raise ValueError('donor stopped early or watchdog timed out')
        time.sleep(10)
    subprocess.run(['systemctl','--user','stop',UNIT],check=True,timeout=60)
    if active() or subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MainPID','--value'],text=True).strip()!='0':raise ValueError('donor process remains')
    done,hashes=donor_results(snapshot,basework,b,m)
    if not drained(done,m['assignments']['laptop']):raise ValueError('donor incomplete after stop')
    C.write_json(work/'drained.json',dict(lineage=m['lineage'],completed=done,receipt_sha256=hashes,unit=UNIT,mainpid=0,partial_preserved=True))


def run(snapshot,bundle,overlay,qualification,work,identity,known_hosts,registration):
    b,m=verify(bundle,overlay,registration);Q.original(snapshot);L.limits();runtime=C.packages()
    if runtime['packages']!=m['packages'] or C.sha(qualification/'manifest.json')!=m['qualification_manifest_sha256']:raise ValueError('runtime or qualification bundle differs')
    work.mkdir(parents=True,exist_ok=False);C.write_json(work/'invocation.json',dict(lineage=m['lineage'],runtime=runtime,tasks=m['assignments']['laptop175']))
    with (work/'qualification.log').open('x') as log:
        subprocess.run([sys.executable,'-u',str(Path(L.__file__).resolve()),'run','--bundle',str(qualification),
                        '--out',str(work/'qualification'),'--registration-commit',QUAL_REG],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=6*3600)
    q=C.read_json(work/'qualification/report.json');raw=work/'qualification/prediction.npz'
    require_qualification(q,runtime,C.sha(raw))
    with np.load(bundle/'qualification_reference.npz',allow_pickle=False) as z:
        actual=C.prediction(raw,z['mvt_id']);cmp=R.compare(z['mvt_id'],z['reg'],z['mvt_id'],actual)
    if not cmp['exact']:raise ValueError('qualification reference mismatch')
    print('EXACT QUALIFICATION PASSED; activating donor transfer',flush=True)
    response=subprocess.check_output(['ssh','-i',str(identity),'-o','IdentitiesOnly=yes','-o','BatchMode=yes',
        '-o','StrictHostKeyChecking=yes','-o',f'UserKnownHostsFile={known_hosts}','-o','ConnectTimeout=10',
        'a1@192.168.1.171'],text=True,timeout=60)
    armed=json.loads(response)
    if armed['lineage']!=m['lineage'] or armed['transferred']!=m['assignments']['laptop175']:raise ValueError('activation response differs')
    C.write_json(work/'activation.json',armed)
    for seed,fold in m['assignments']['laptop175']:
        verify(bundle,overlay,registration);Q.original(snapshot,fold)
        attempt=work/'attempts'/f'{Q.name(fold)}-s{seed:02}';env=C.child_environment('__prepared_pools__',attempt)
        env.update(CATBOOST_THREADS='6',OMP_NUM_THREADS='6',OMP_THREAD_LIMIT='6')
        print(f'START laptop175 {Q.name(fold)} seed={seed}',flush=True)
        with (work/f'{Q.name(fold)}-s{seed:02}.log').open('x') as log:
            subprocess.run([sys.executable,'-u',str(Path(Q.__file__).resolve()),'pool-fit','--bundle',str(bundle),'--work',str(attempt),
                '--registration-commit',BASE_REG,'--seed',str(seed),'--fold',str(fold)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=6*3600)
        verify(bundle,overlay,registration);Q.original(snapshot,fold)
        origin=dict(qualification_sha256=C.sha(work/'qualification/report.json'),activation_sha256=C.sha(work/'activation.json'),
                    pool_manifest_sha256=C.sha(bundle/'pools'/Q.name(fold)/'manifest.json'))
        Q.publish(attempt/f'raw_seed{seed}.npz',work/Q.name(fold)/f'seed_{seed:02}',m,'laptop175',seed,fold,Q.metadata(snapshot,fold)['mvt_id'],origin)
        print(f'COMPLETE laptop175 {Q.name(fold)} seed={seed}',flush=True)
    C.write_json(work/'complete.json',dict(lineage=m['lineage'],tasks=m['assignments']['laptop175']))


def merge(snapshot,bundle,overlay,work,basework,drainwork,registration):
    b,m=verify(bundle,overlay,registration)
    if C.read_json(work/'complete.json')!={'lineage':m['lineage'],'tasks':m['assignments']['laptop175']}:raise ValueError('extra node incomplete')
    inv=C.read_json(work/'invocation.json');q=C.read_json(work/'qualification/report.json')
    require_qualification(q,inv['runtime'],C.sha(work/'qualification/prediction.npz'))
    if inv['runtime']['packages']!=m['packages'] or inv['lineage']!=m['lineage']:raise ValueError('extra runtime drift')
    activation=C.read_json(work/'activation.json');drain=C.read_json(drainwork/'drained.json')
    if activation!=C.read_json(drainwork/'armed.json') or activation['lineage']!=m['lineage'] or drain['lineage']!=m['lineage'] or drain['mainpid']!=0:raise ValueError('donor handoff differs')
    done,hashes=donor_results(snapshot,basework/'laptop',b,m)
    if not drained(done,m['assignments']['laptop']) or hashes!=drain['receipt_sha256']:raise ValueError('donor handoff receipts differ')
    runtimes={'laptop175':inv['runtime']};qualifications={'laptop175':q}
    for node in Q.NODES:
        root=basework/node;old=C.read_json(root/'invocation.json');runtimes[node]=old['runtime']
        if old['lineage']!=BASE_LINEAGE or old['tasks']!=b['assignments'][node] or old['registration']!=BASE_REG or old['runtime']['packages']!=m['packages']:raise ValueError('original node invocation differs')
        if node!='laptop' and C.read_json(root/'complete.json')!={'lineage':BASE_LINEAGE,'node':node,'tasks':b['assignments'][node]}:raise ValueError('original node incomplete')
        if node!='home':
            report=C.read_json(root/'qualification.json');qualifications[node]=report
            tol=1e-6 if node=='mac' else 0.
            if not report['passed'] or report['lineage']!=BASE_LINEAGE or report['runtime']!=old['runtime'] or report['rows']!=344419 or report['absolute_tolerance']!=tol or report['max_abs']>tol:raise ValueError('original node qualification differs')
        elif old['runtime']!=b['home_runtime']:raise ValueError('Home runtime differs')
    final=work/'merged'
    if final.exists():raise ValueError('merged output exists')
    staging=work/('.merge-'+uuid.uuid4().hex);staging.mkdir()
    for fold in range(1,7):
        Q.original(snapshot,fold);meta=Q.metadata(snapshot,fold);pred=[];proof=[]
        for seed in range(15):
            node=next(n for n,v in m['assignments'].items() if [seed,fold] in v)
            extra=node=='laptop175';root=work if extra else basework/node;dest=root/Q.name(fold)/f'seed_{seed:02}'
            rec=C.read_json(dest/'receipt.json')
            if rec['runtime']!=runtimes[node]:raise ValueError('fit runtime differs')
            if extra and (rec['origin']['qualification_sha256']!=C.sha(work/'qualification/report.json') or rec['origin']['activation_sha256']!=C.sha(work/'activation.json') or rec['origin']['pool_manifest_sha256']!=C.sha(bundle/'pools'/Q.name(fold)/'manifest.json')):raise ValueError('extra fit provenance differs')
            pred.append(Q.check_result(dest,m if extra else b,node,seed,fold,meta['mvt_id']));proof.append(C.sha(dest/'receipt.json'))
        stack=np.stack(pred);np.savez(staging/f'{Q.name(fold)}.npz',**meta,per_seed=stack,reg2=stack.mean(axis=0))
        C.write_json(staging/f'{Q.name(fold)}.json',dict(lineage=m['lineage'],receipts=proof,seeds=list(range(15))))
    C.write_json(staging/'complete.json',dict(lineage=m['lineage'],fits=90,folds=6,runtimes=runtimes,qualifications=qualifications,accepted_for_submission=False))
    os.rename(staging,final)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['register','run','activate','watch','merge'])
    for name in ('snapshot','bundle','overlay','qualification','work','basework','drainwork','identity','known-hosts'):p.add_argument('--'+name,type=Path)
    p.add_argument('--registration-commit',required=True);a=p.parse_args()
    if not re.fullmatch('[a-f0-9]{40}',a.registration_commit):p.error('full registration SHA required')
    for name in ('snapshot','bundle','overlay','qualification','work','basework','drainwork','identity','known_hosts'):
        if getattr(a,name):setattr(a,name,getattr(a,name).resolve())
    common=(a.snapshot,a.bundle,a.overlay,a.work,a.basework,a.registration_commit)
    if a.action=='register':register(a.bundle,a.qualification,a.overlay,a.registration_commit)
    elif a.action=='activate':activate(*common)
    elif a.action=='watch':watch(*common)
    elif a.action=='run':run(a.snapshot,a.bundle,a.overlay,a.qualification,a.work,a.identity,a.known_hosts,a.registration_commit)
    else:merge(a.snapshot,a.bundle,a.overlay,a.work,a.basework,a.drainwork,a.registration_commit)


if __name__=='__main__':main()

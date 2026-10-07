"""Transport/execution wrapper only: registered E197 after complete E198."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/home/a1/prc-2026')
UNIT='prc-e198-density-20261007.service'
REG='bfe793c62f7cecf633c7cc31c747f48eb0676f2d'
PROTOCOL='E198-mdn4-remaining4pair-confirmation-mean-control-3seed8epoch-v1'
EXPECTED_PARENT_SPEC='37a13dbb83e852c5ef4a6bde9b84dfd9a2882affc5fc44d8c2654658602c691f'
RECEIPT=ROOT/'data/e197_queue_20261007.jsonl'
WORKER=ROOT/'data/e197_inputs_20261007/scripts/e197_trajectory_encoder.py'


def record(event,**data):
    item=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event,**data)
    with RECEIPT.open('a') as f:
        f.write(json.dumps(item,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(item),flush=True)


def predecessor_complete(reg):
    p=ROOT/'data/e198_density_20261007/complete.json'
    if not p.exists(): return None
    c=json.loads(p.read_text())
    m=json.loads((ROOT/'data/e198_density_inputs_20261007/manifest.json').read_text())
    digest=hashlib.sha256(json.dumps(m['spec'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if (m['spec_sha256']!=digest or digest!=EXPECTED_PARENT_SPEC or m['spec']['registration_commit']!=reg
        or c['protocol']!=PROTOCOL or c['registration_commit']!=reg
        or c['input_spec_sha256']!=digest or len(c['arms'])!=8):
        raise RuntimeError('E198 completion lineage differs')
    expected={(fold,arm) for fold in (2,3,5,6) for arm in ('mean','mdn4')}
    if {(a['fold'],a['arm']) for a in c['arms']}!=expected: raise RuntimeError('E198 arm inventory differs')
    if any(a['seeds']!=[0,1,2] or len(a['fits'])!=3 or {f['seed'] for f in a['fits']}!={0,1,2}
           or any(f['epochs']!=8 for f in a['fits']) for a in c['arms']):
        raise RuntimeError('E198 24-fit receipt differs')
    for arm in c['arms']:
        folder=p.parent/f'fold_{arm["fold"]}_{arm["fold"]+6}'/arm['arm']
        if json.loads((folder/'manifest.json').read_text())!=arm:
            raise RuntimeError('E198 arm manifest differs from complete')
        for name,expected_hash in arm['files'].items():
            if Path(name).name!=name: raise RuntimeError('unexpected E198 artifact path')
            h=hashlib.sha256()
            with (folder/name).open('rb') as f:
                for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
            if h.hexdigest()!=expected_hash:raise RuntimeError('E198 model/prediction artifact drift')
    ep=p.with_name('evaluation.json');ev=json.loads(ep.read_text())
    if (ev['protocol']!=PROTOCOL or ev['registration_commit']!=reg or len(ev['reports'])!=6
        or {r['fold'] for r in ev['reports']}!=set(range(1,7)) or ev['accepted_for_submission']):
        raise RuntimeError('E198 evaluation receipt differs')
    return dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),registration=reg,fits=24,
                evaluation_sha256=hashlib.sha256(ep.read_bytes()).hexdigest(),
                six_pair_gate_pass=ev['six_pair_gate_pass'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--predecessor-registration',required=True);a=p.parse_args()
    if RECEIPT.exists(): raise RuntimeError('Queue already attempted; no automatic retry')
    with RECEIPT.open('x'):pass
    start=time.monotonic();prior=None;seen=False
    record('queue_started',registration=REG,predecessor_registration=a.predecessor_registration,unit=UNIT)
    while time.monotonic()-start<6*3600:
        s=subprocess.run(['systemctl','--user','show',UNIT,'-p','LoadState','-p','ActiveState',
                          '-p','MainPID','-p','ExecMainStatus','-p','InvocationID'],
                          capture_output=True,text=True,check=True,timeout=15)
        state=dict(x.split('=',1) for x in s.stdout.splitlines() if '=' in x)
        if state!=prior:record('predecessor_state',state=state);prior=state
        active=state.get('ActiveState')
        if active in ('active','activating','deactivating','reloading'):
            seen=True;time.sleep(15);continue
        if state.get('MainPID','0')!='0': raise RuntimeError('predecessor main process still alive')
        if active=='failed' or state.get('ExecMainStatus','0')!='0': raise RuntimeError('E198 failed; queue stopped')
        c=predecessor_complete(a.predecessor_registration)
        if c is None:
            if seen: raise RuntimeError('E198 exited without complete receipt; queue stopped')
            time.sleep(15);continue
        if active!='inactive': raise RuntimeError('unexpected predecessor terminal state')
        record('predecessor_verified',completion=c,observed_live=seen)
        break
    else: raise TimeoutError('E198 exceeded6h before verified completion')
    os.chdir(ROOT)
    common=['--root',str(ROOT),'--input',str(ROOT/'data/e197_inputs_20261007'),
            '--prepared',str(ROOT/'data/e197_prepared_20261007'),'--registration-commit',REG]
    record('verify_e197_inputs')
    subprocess.run([sys.executable,'-u',str(WORKER),'verify',*common],check=True,timeout=900)
    record('launch_e197_fit',registered_protocol='E197-own-previous-local-order-vs-flat-K64-2pair-3seed-8epoch-v1')
    subprocess.run([sys.executable,'-u',str(WORKER),'run',*common,
        '--output',str(ROOT/'data/e197_trajectory_20261007'),
        '--smoke',str(ROOT/'data/e197_smoke_20261007.json')],check=True,timeout=4*3600)
    record('fit_complete_begin_evaluation')
    subprocess.run([sys.executable,'-u',str(WORKER),'evaluate',*common,
                   '--output',str(ROOT/'data/e197_trajectory_20261007')],check=True,timeout=1800)
    result=json.loads((ROOT/'data/e197_trajectory_20261007/evaluation.json').read_text())
    record('complete',local_order_screen_pass=result['local_order_screen_pass'],
           context_capacity_screen_pass=result['context_capacity_screen_pass'],accepted_for_submission=False)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if RECEIPT.exists():record('failed',exception=type(exc).__name__,message=str(exc))
        raise

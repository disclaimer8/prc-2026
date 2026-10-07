"""CPU-only laptop handoff: two verified E202 preparations then fixed E201."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

E202_REG='55fe9b68f7cc8405681f6c85d576979e067b55c3'
E202_SPEC='6d604eaaca830fa83c18ebb78986b028d338869123d5f8c10b2450dde1ec4309'
RUNNER_SHA='bdebef7add20daf76a819132261efde86ffda7d03e13344028ed368d8819da18'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--node',required=True,choices=['171','175']);p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    root=a.root.resolve();marker=root/f'data/e202_handoff_{a.node}_20261007.json';runner=root/'scripts/e201_node_queue_20261007.py'
    state=root/f'data/e201_waiter_{a.node}_20261007';state.mkdir(exist_ok=False)
    def save(name,x):(state/name).write_text(json.dumps(x,indent=2,sort_keys=True))
    if sha(runner)!=RUNNER_SHA:raise ValueError('E201 queue hash drift')
    start=time.time();save('started.json',dict(node=a.node,pid=os.getpid(),e202_registration=E202_REG,e202_spec_sha256=E202_SPEC,started_unix=start))
    try:
        while True:
            if time.time()-start>6*3600:raise TimeoutError('E202 preparation handoff not complete within6h')
            if marker.exists():
                m=json.loads(marker.read_text());folds=[2,5] if a.node=='171' else [3,6]
                if m['status']!='two_prepared_complete' or m['registration_commit']!=E202_REG or m['input_spec_sha256']!=E202_SPEC or str(m['node'])!=a.node or m['folds']!=folds:
                    raise ValueError('E202 handoff lineage/status differs')
                if sorted(x['fold'] for x in m['prepared_receipts'])!=folds:raise ValueError('need exactly two E202 prepared receipts')
                for r in m['prepared_receipts']:
                    if len(r['complete_sha256'])!=64 or not r['files']:raise ValueError('incomplete prepared proof')
                unit=f'prc-e202-prep-{a.node}-20261007.service'
                result=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','-p','MainPID','-p','Result'],capture_output=True,text=True,check=True)
                props=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
                if props.get('ActiveState')=='failed' or props.get('Result') not in ('success',''):
                    raise ValueError('E202 prep service failed '+str(props))
                if props.get('MainPID')=='0' and props.get('ActiveState')=='inactive' and props.get('Result')=='success':
                    save('released.json',dict(marker_sha256=sha(marker),e202_handoff=m,unit=unit,unit_exit=props,released_unix=time.time()))
                    break
            time.sleep(10)
        os.execv(sys.executable,[sys.executable,'-u',str(runner),'--node',a.node,'--input',str(root/'data/e201_inputs_20261007'),'--output',str(root/f'data/e201_{a.node}_20261007')])
    except BaseException:
        save('failure.json',dict(error=traceback.format_exc(),auto_retry=False));raise

if __name__=='__main__':main()

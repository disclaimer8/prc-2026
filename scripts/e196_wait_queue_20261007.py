"""Run the independently registered E196 only after the actual E193 service exits."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/home/a1/prc-2026')
UNIT='prc-e193-weather-20261007.service'
REGISTRATION='9078e3508e81bb7c1e6cee432226063b0067b44e'
RECEIPT=ROOT/'data/e196_queue_20261007.jsonl'
WORKER=ROOT/'data/e196_density_inputs_20261007/scripts/e196_conditional_density.py'


def record(event,**fields):
    entry=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event,**fields)
    with RECEIPT.open('a') as f:
        f.write(json.dumps(entry,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(entry),flush=True)


def main():
    if RECEIPT.exists():raise RuntimeError('Queue already attempted; no automatic retry')
    start=time.monotonic();seen=False;prior=None
    while time.monotonic()-start<6*3600:
        p=subprocess.run(['systemctl','--user','show',UNIT,'-p','LoadState','-p','ActiveState','-p','MainPID','-p','ExecMainStatus'],capture_output=True,text=True,timeout=15,check=True)
        state=dict(x.split('=',1) for x in p.stdout.splitlines() if '=' in x)
        if state!=prior:record('predecessor_state',unit=UNIT,state=state);prior=state
        active=state.get('ActiveState')
        if active in ('active','activating','deactivating','reloading'):
            seen=True;time.sleep(15);continue
        complete=ROOT/'data/e193_weather_20261007/complete.json'
        if not seen and not complete.exists():raise RuntimeError('Predecessor was never observed running and has no completion receipt')
        if state.get('MainPID','0')!='0':raise RuntimeError('Predecessor still has a live main process')
        if active not in ('inactive','failed'):raise RuntimeError('Unknown predecessor state')
        record('launch_e196',registration=REGISTRATION,predecessor_complete=complete.exists(),predecessor_terminal=state)
        os.chdir(ROOT)
        os.execv(sys.executable,[sys.executable,'-u',str(WORKER),'run','--root',str(ROOT),'--input',str(ROOT/'data/e196_density_inputs_20261007'),'--output',str(ROOT/'data/e196_density_20261007'),'--registration-commit',REGISTRATION])
    raise TimeoutError('Predecessor exceeded6h; independent queue stopped without a fit')


if __name__=='__main__':main()

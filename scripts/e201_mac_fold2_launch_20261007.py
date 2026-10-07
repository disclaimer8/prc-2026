"""Detach the registered guarded Mac calculation in its own process session."""
from pathlib import Path
import hashlib
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
GUARD=ROOT/'scripts/e201_mac_fold2_guard_20261007.py'
EXPECTED='db667f10c36e77cfb703d50c0bf8e2f7185056eefd73aec4ba2c2575dac3f786'
if hashlib.sha256(GUARD.read_bytes()).hexdigest()!=EXPECTED:raise ValueError('registered guard changed')
if (ROOT/'data/e201_mac_fold2_guard_20261007').exists():raise ValueError('existing attempt; never retry automatically')
with (ROOT/'data/e201_mac_fold2_guard_20261007_detached.log').open('x') as log:
    child=subprocess.Popen(['/usr/bin/caffeinate','-i',sys.executable,'-u',str(GUARD),'--root',str(ROOT)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(child.pid,flush=True)

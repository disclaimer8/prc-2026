"""Registered one-shot controllers after the Mac reboot; never retry model fits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROTOCOL = 'PRC-postreboot-day-controllers-20261007-v1'
SELF = 'scripts/prc_day_recovery_20261007.py'
DOC = 'analysis/prc_day_recovery_protocol_20261007.md'
RECOVERY = '5ef410c1974e5cb0f6e64e3e5bfdbf9a27254e38'
E203 = '6e95e3a65936a3401e41499bb340f2d54a1a2121'
E201 = 'd664e153b3f0732b4047011cc062f58f42581a6c'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--registration', required=True)
    p.add_argument('--launch', action='store_true')
    a = p.parse_args()
    root = a.root.resolve()
    for name in (SELF, DOC, 'scripts/e201_collect_queue_20261007.py'):
        saved = subprocess.check_output(['git', 'show', a.registration + ':' + name], cwd=root)
        if hashlib.sha256(saved).hexdigest() != digest(root / name):
            raise ValueError('registered operational source changed: ' + name)
    if PROTOCOL.encode() not in subprocess.check_output(['git', 'show', a.registration + ':HYPOTHESES_LOG.md'], cwd=root):
        raise ValueError('day controllers not registered')
    if a.launch:
        with (root / 'data/prc_day_recovery_20261007.log').open('x') as log:
            child = subprocess.Popen(
                ['/usr/bin/caffeinate', '-i', sys.executable, '-u', str(root / SELF),
                 '--root', str(root), '--registration', a.registration],
                cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        print(child.pid, flush=True)
        return

    state = root / 'data/prc_day_recovery_20261007'
    state.mkdir(exist_ok=False)
    os.nice(10)
    children = {}
    logs = []

    def save(name, value):
        with (state / name).open('x') as f:
            json.dump(value, f, indent=2, sort_keys=True)

    def launch(name, args):
        log = (state / (name + '.log')).open('x')
        logs.append(log)
        child = subprocess.Popen([sys.executable, '-u', *args], cwd=root,
                                 stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        children[name] = child
        save(name + '_started.json', dict(pid=child.pid, unix=time.time(), args=args))
        print(name, 'started', child.pid, flush=True)

    save('started.json', dict(protocol=PROTOCOL, registration=a.registration,
                             pid=os.getpid(), unix=time.time()))
    launch('e202_transport', ['scripts/e202_postreboot_transport_20261007.py',
                             '--root', str(root), '--recovery-commit', RECOVERY])
    launch('e201_collector', ['scripts/e201_collect_queue_20261007.py', '--root', str(root),
                             '--state', str(root / 'data/e201_collection_20261007_postreboot'),
                             '--home-memoryfix'])
    deadline = time.monotonic() + 18 * 3600
    recorded = set()
    decided = False
    while time.monotonic() < deadline:
        for name, child in list(children.items()):
            if child.poll() is not None and name not in recorded:
                save(name + '_exit.json', dict(exit_code=child.returncode, unix=time.time(), auto_retry=False))
                recorded.add(name)
        if not decided:
            guard = root / 'data/e201_mac_fold2_guard_20261007'
            if (guard / 'failure.json').exists():
                save('e203_not_started.json', dict(reason='Mac guarded fold failed; no concurrent local fit or automatic retry'))
                decided = True
            elif children['e202_transport'].poll() not in (None, 0):
                save('e203_not_started.json', dict(reason='E202 transport failed; original remote workers untouched'))
                decided = True
            elif (guard / 'complete.json').exists():
                proof = json.loads((guard / 'complete.json').read_text())
                part = root / 'data/e201_mac_20261007/v60_fold2'
                if proof['registration_commit'] != E201 or proof['fold']['receipt_sha256'] != digest(part / 'complete.json') or proof['fold']['transfer_sha256'] != digest(part / 'transfer_receipt.json'):
                    raise ValueError('Mac completion/audit proof differs')
                launch('e203', ['scripts/e203_execution_queue_20261007.py', '--root', str(root),
                               '--registration-commit', E203])
                decided = True
        if decided and len(recorded) == len(children):
            save('complete.json', dict(exit_codes={n: c.returncode for n, c in children.items()},
                                      candidate_upload_performed=False, unix=time.time()))
            return
        time.sleep(20)
    save('deadline.json', dict(reason='18h controller bound; remote/model workers untouched', auto_retry=False))


if __name__ == '__main__':
    main()

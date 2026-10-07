"""One-shot, identity-checked E201 stop and unchanged E203 launch.

Preserves every artifact and the independent E202 transfer/GPU controllers.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback

PROTOCOL = 'PRC-prize-priority-stop-E201-start-registered-E203-v1'
SELF = 'scripts/prc_prize_priority_20261007.py'
DOC = 'analysis/prc_prize_priority_20261007.md'
E203 = '6e95e3a65936a3401e41499bb340f2d54a1a2121'
UNITS = [
    ('home-pc-lan', True, 'prc-e201-donors-home-memoryfix-20261007.service',
     'deadafd189ab4f3889aba956a35127c6'),
    ('hetzner', False, 'prc-e201-donors-hetzner-20261007.service',
     '473da44c223e4d85afb5cecdb9b4aff3'),
    ('a1@192.168.1.171', True, 'prc-e201-donors-171-20261007.service',
     '207a8dc9527e43b78ba4a6711cbd273e'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remote(host, args):
    return subprocess.check_output(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, shlex.join(args)],
        text=True, timeout=60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--registration', required=True)
    a = p.parse_args()
    root = a.root.resolve()
    for name in (SELF, DOC):
        frozen = subprocess.check_output(['git', 'show', a.registration + ':' + name], cwd=root)
        if hashlib.sha256(frozen).hexdigest() != sha(root / name):
            raise ValueError('priority protocol/source drift')
    if PROTOCOL.encode() not in subprocess.check_output(
            ['git', 'show', a.registration + ':HYPOTHESES_LOG.md'], cwd=root):
        raise ValueError('priority decision not registered')
    import e203_trajectory_production as W
    W.registration(root, E203)
    state = root / 'data/prc_prize_priority_20261007'
    state.mkdir(exist_ok=False)

    def save(name, data):
        with (state / name).open('x') as f:
            json.dump(data, f, indent=2, sort_keys=True)

    save('started.json', dict(registration=a.registration, protocol=PROTOCOL,
                             unix=time.time(), pid=os.getpid()))
    try:
        for host, user, unit, invocation in UNITS:
            ctl = ['systemctl', *(['--user'] if user else [])]

            def status():
                return dict(line.split('=', 1) for line in remote(host, [*ctl, 'show', unit,
                    '-p', 'ActiveState', '-p', 'SubState', '-p', 'InvocationID',
                    '-p', 'MainPID', '-p', 'Result']).splitlines() if '=' in line)

            before = status()
            if before.get('ActiveState') in ('active', 'activating', 'deactivating'):
                if before.get('InvocationID') != invocation:
                    raise ValueError('E201 invocation changed: ' + unit)
                remote(host, [*ctl, 'stop', unit])
            after = status()
            if int(after.get('MainPID', '0')) or after.get('ActiveState') in ('active', 'activating', 'deactivating'):
                raise ValueError('E201 unit has not stopped: ' + unit)
            save(unit + '.json', dict(host=host, before=before, after=after,
                                     reason='prize documentation/reproduction priority', unix=time.time()))
            print('E201 stopped:', host, unit, flush=True)

        guard = root / 'data/e201_mac_fold2_guard_20261007'
        if not (guard / 'complete.json').exists() and not (guard / 'failure.json').exists():
            record = json.loads((guard / 'run-fold_started.json').read_text())
            pid = record['pid']
            row = subprocess.run(['ps', '-p', str(pid), '-o', 'command='],
                                 capture_output=True, text=True, check=True).stdout.strip()
            for token in ('e201_donor_oof.py', 'run-fold', '--fold 2',
                          str(root / 'data/e201_mac_20261007/v60_fold2')):
                if token not in row:
                    raise ValueError('Mac E201 process identity differs')
            if pid != 27607 or os.getpgid(pid) != pid:
                raise ValueError('Mac E201 child group differs')
            save('mac_stop_requested.json', dict(pid=pid, unix=time.time(), signal='SIGTERM'))
            os.killpg(pid, signal.SIGTERM)
            deadline = time.monotonic() + 55
            while not (guard / 'failure.json').exists() and not (guard / 'complete.json').exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('guard has not recorded child exit; no E203 launch')
                time.sleep(1)
        save('mac_stopped.json', dict(guard_complete=(guard / 'complete.json').exists(),
             guard_failure=(guard / 'failure.json').exists(), partials_preserved=True, unix=time.time()))

        # Obtain a durable decision from the previous launch owner first. Its
        # successful-fold path owns E203; only its explicit refusal transfers
        # launch ownership here. This closes the check/launch race.
        day = root / 'data/prc_day_recovery_20261007'
        existing = root / 'data/e203_execution_20261007.json'
        deadline = time.monotonic() + 65
        while True:
            if (day / 'e203_started.json').exists() and existing.exists():
                started = json.loads(existing.read_text())
                if started['registration_commit'] != E203:
                    raise ValueError('unexpected existing E203')
                save('e203_existing.json', started)
                break
            if (day / 'e203_not_started.json').exists():
                if not (guard / 'failure.json').exists():
                    raise ValueError('previous owner refused despite complete fold; inspect before launching')
                decision = json.loads((day / 'e203_not_started.json').read_text())
                if decision.get('reason') != 'Mac guarded fold failed; no concurrent local fit or automatic retry':
                    raise ValueError('previous owner refused for another reason')
                save('e203_launch_ownership.json', decision)
                break
            if time.monotonic() > deadline:
                raise TimeoutError('previous E203 launch owner has not relinquished ownership')
            time.sleep(1)
        if not (state / 'e203_existing.json').exists():
            if existing.exists():
                raise ValueError('unowned E203 state appeared; no duplicate launch')
            answer = subprocess.check_output([sys.executable,
                str(root / 'scripts/e203_execution_queue_20261007.py'), '--root', str(root),
                '--registration-commit', E203, '--launch'], cwd=root, text=True, timeout=60)
            save('e203_launched.json', dict(pid=int(answer.strip()), registration=E203,
                                          unix=time.time()))
            print('E203 launched:', answer.strip(), flush=True)
        save('complete.json', dict(protocol=PROTOCOL, registration=a.registration,
                                  e202_untouched=True, no_upload=True, unix=time.time()))
    except BaseException:
        save('failure.json', dict(error=traceback.format_exc(), auto_retry=False,
                                 partials_preserved=True))
        raise


if __name__ == '__main__':
    main()

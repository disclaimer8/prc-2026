"""One bounded successor to the already-running E203 controller; never retries."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

import reproduce_best as R

SELF = 'scripts/reproduce_best_queue_20261007.py'
FILES = (SELF, R.SELF, R.SPEC, 'docs/reproduction.md')
E203_REG = '6e95e3a65936a3401e41499bb340f2d54a1a2121'
E203_PID = 51200
E203_STARTED = '2026-10-07T14:59:25.379082+00:00'
STATE = 'data/reproduce_best_queue_20261007'
LIMIT = 18 * 3600


def registration(root, commit):
    import hashlib
    if not re.fullmatch('[a-f0-9]{40}', commit or ''):
        raise ValueError('full reviewed registration required')
    for name in FILES:
        frozen = subprocess.check_output(['git', 'show', commit + ':' + name], cwd=root)
        if hashlib.sha256(frozen).hexdigest() != R.sha(root / name):
            raise ValueError('registered reproduction source/documentation changed: ' + name)
    if R.PROTOCOL.encode() not in subprocess.check_output(['git', 'show', commit + ':HYPOTHESES_LOG.md'], cwd=root):
        raise ValueError('reproduction protocol is not registered')


def memory():
    env = {**os.environ, 'LC_ALL':'C'}
    raw = subprocess.check_output(['/usr/bin/memory_pressure', '-Q'], text=True, timeout=15, env=env)
    matched = re.search(r'System-wide memory free percentage:\s*(\d+)%', raw)
    if matched is None:
        raise ValueError('unrecognized memory pressure output')
    raw = subprocess.check_output(['/usr/bin/vm_stat'], text=True, timeout=15, env=env)
    page = int(re.search(r'page size of (\d+) bytes', raw).group(1))
    values = {}
    for line in raw.splitlines()[1:]:
        if ':' in line:
            key, value = line.split(':', 1)
            value = value.strip().rstrip('.')
            if value.isdigit():
                values[key] = int(value)
    return dict(memory_free_percent=int(matched.group(1)),
        reclaimable_bytes=page*sum(values.get(k, 0) for k in ('Pages free', 'Pages inactive', 'Pages speculative')))


def process_alive(pid):
    # Metadata only; never dumps command arguments or signals the dependency.
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='], capture_output=True, text=True, timeout=15)
    if result.returncode == 1 and not result.stdout.strip():
        return False
    if result.returncode:
        raise RuntimeError('cannot establish dependency process exit: ' + result.stderr)
    return bool(result.stdout.strip()) and not result.stdout.strip().startswith('Z')


def dependency(root):
    state = R.read(root / 'data/e203_execution_20261007.json')
    if (state['registration_commit'] != E203_REG or state['pid'] != E203_PID
        or state['started'] != E203_STARTED):
        raise ValueError('E203 controller identity changed')
    return state


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def terminate_owned_group(child, pgid, term_timeout=30, kill_timeout=15):
    """The PGID belongs to our start_new_session child, even after its exit."""
    if child is None or pgid != child.pid or pgid in (os.getpgrp(), E203_PID):
        raise ValueError('refusing to signal a group not owned by this queue')

    def send(sig):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass  # Group can exit between the existence check and signal.

    def wait_empty(timeout):
        until = time.monotonic() + timeout
        while True:
            child.poll()  # Reap the direct child; descendants may remain.
            if not group_exists(pgid):
                return True
            if time.monotonic() >= until:
                return False
            time.sleep(.1)

    if not group_exists(pgid):
        child.poll()
        return
    send(signal.SIGTERM)
    if wait_empty(term_timeout):
        return
    send(signal.SIGKILL)
    if not wait_empty(kill_timeout):
        raise TimeoutError('owned child group still exists after bounded TERM/KILL')


def termination_requested(signum, _frame):
    raise RuntimeError('queue termination requested by signal ' + str(signum))


def main(root, commit):
    registration(root, commit)
    directory = root / STATE
    directory.mkdir(exist_ok=False)
    deadline = time.monotonic() + LIMIT
    child = None
    owned_pgid = None
    spawning = False
    pending_termination = None

    def on_termination(signum, frame):
        nonlocal pending_termination
        if spawning:
            # Let Popen finish so the owned PGID can be cleaned up. Unlike
            # blocking signals, this does not pass a blocked mask to the child.
            pending_termination = signum
        else:
            termination_requested(signum, frame)

    def event(status, **extra):
        payload = dict(unix=time.time(), status=status, **extra)
        with (directory / 'events.jsonl').open('a') as f:
            f.write(json.dumps(payload, sort_keys=True) + '\n')
            f.flush()
            os.fsync(f.fileno())
        print(json.dumps(payload), flush=True)

    def stop_own():
        if owned_pgid is not None:
            terminate_owned_group(child, owned_pgid)

    def run(action, output, cache):
        nonlocal child, owned_pgid, spawning
        registration(root, commit)
        while True:
            stats = memory()
            disk = shutil.disk_usage(root).free
            if disk < 4*1024**3:
                raise ValueError('less than4GiB disk before ' + action)
            if stats['reclaimable_bytes'] >= 16*1024**3 and stats['memory_free_percent'] >= 10:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('memory admission deadline')
            time.sleep(20)
        args = [sys.executable, '-u', str(root / R.SELF), action, '--root', str(root),
                '--output', str(output), '--model-cache', str(cache)]
        if action == 'capture':
            args += ['--registration', commit]
        with (directory / (action + '.log')).open('x') as log:
            # Do not allow termination between spawning and recording ownership.
            spawning = True
            try:
                child = subprocess.Popen(args, cwd=root, env=R.environment(), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                owned_pgid = child.pid
            finally:
                spawning = False
            if pending_termination is not None:
                termination_requested(pending_termination, None)
            R.save(directory / (action + '_started.json'), dict(pid=child.pid, pgid=owned_pgid, args=args, unix=time.time(),
                memory=stats, disk_free_bytes=disk, worker_sha256=R.sha(root / R.SELF)))
            event(action + '_started', pid=child.pid)
            low = 0
            while child.poll() is None:
                stats = memory()
                disk = shutil.disk_usage(root).free
                low = low + 1 if stats['memory_free_percent'] < 10 else 0
                if low >= 3 or disk < 2500*1024**2:
                    raise RuntimeError('reproduction memory/disk floor violated; stop own group without retry')
                if time.monotonic() >= deadline:
                    raise TimeoutError('18h reproduction queue deadline')
                time.sleep(10)
            R.save(directory / (action + '_exit.json'), dict(pid=child.pid, returncode=child.returncode, unix=time.time()))
            if child.returncode:
                raise RuntimeError(action + ' failed; no retry')
            if group_exists(owned_pgid):
                raise RuntimeError(action + ' leader exited with live descendants; cleanup before failure')
            child = None
            owned_pgid = None
        proof = R.read(output / 'complete.json')
        if proof['protocol'] != R.PROTOCOL or not proof['exact_predictions'] or proof['rows'] != 344841:
            raise ValueError(action + ' exact-output proof missing')
        event(action + '_complete', receipt_sha256=R.sha(output / 'complete.json'))

    old_handlers = {sig:signal.signal(sig, on_termination) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        os.nice(10)
        first = dependency(root)
        R.save(directory / 'started.json', dict(protocol=R.PROTOCOL, registration=commit, pid=os.getpid(),
            dependency_pid=E203_PID, dependency_started=E203_STARTED, unix=time.time(), deadline_hours=18,
            source_sha256={n:R.sha(root/n) for n in FILES}))
        event('waiting_for_e203', observed_status=first['status'])
        while True:
            state = dependency(root)
            terminal = state['status'] in ('complete_gate_failed_no_rank', 'complete_candidate_no_upload', 'failed_no_retry')
            if terminal and not process_alive(E203_PID):
                break
            if not terminal and not process_alive(E203_PID):
                raise RuntimeError('E203 exited without a terminal receipt; blocked without auto-recovery')
            if time.monotonic() >= deadline:
                raise TimeoutError('E203 terminal-exit deadline')
            time.sleep(20)
        R.save(directory / 'e203_terminal.json', state)
        if state['status'] == 'failed_no_retry':
            R.save(directory / 'blocked.json', dict(reason='E203 technical failure requires review; no automatic capture',
                dependency=state, fits_performed=0, unix=time.time()))
            event('blocked_e203_technical_failure')
            return
        spec, setup = R.check(root)
        cache = root / 'data/e203_chain_20261007/models'
        if (cache.parent / 'control_verified.json').exists():
            R.verify_cache(cache, spec, setup)
            event('reuse_exact_e203_cache', control_receipt_sha256=R.sha(cache.parent / 'control_verified.json'))
        else:
            if state['status'] != 'complete_gate_failed_no_rank' or cache.parent.exists():
                raise ValueError('unexpected incomplete E203 capture; no duplicate fit')
            cache = root / 'data/v12074_replay_models'
            event('independent_capture_after_negative_e203_gate')
            run('capture', root / 'data/v12074_capture', cache)
        run('replay', root / 'data/reproduced_v12074', cache)
        R.save(directory / 'complete.json', dict(protocol=R.PROTOCOL, registration=commit,
            replay_receipt_sha256=R.sha(root/'data/reproduced_v12074/complete.json'),
            model_cache=str(cache), exact_predictions=True, uploaded=False, unix=time.time()))
        event('complete_reproduction_no_upload')
    except BaseException:
        original_error = traceback.format_exc()
        # Further TERM/INT must not interrupt the bounded cleanup/failure receipt.
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        cleanup_error = None
        try:
            stop_own()
        except BaseException:
            cleanup_error = traceback.format_exc()
        R.save(directory / 'failure.json', dict(error=original_error, cleanup_error=cleanup_error,
            owned_pgid=owned_pgid, no_retry=True, partials_preserved=True,
            dependency_not_signalled=True, unix=time.time()))
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--registration', required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.launch:
        registration(root, args.registration)
        if (root/STATE).exists():
            raise ValueError('fresh queue state required; no retry')
        with (root/'data/reproduce_best_queue_20261007.log').open('x') as log:
            process = subprocess.Popen(['/usr/bin/caffeinate', '-i', sys.executable, '-u', str(root/SELF),
                '--root', str(root), '--registration', args.registration], cwd=root, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(process.pid, flush=True)
    else:
        main(root, args.registration)

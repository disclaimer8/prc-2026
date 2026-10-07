"""Durable one-shot v12075 replay supervisor: 6 threads, 16 GiB monitored RSS, 4 h."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import reproduce_v12075 as V
import reproduce_best_queue_20261007 as Q

STATE = 'data/reproduce_v12075_launch_20261007'
LOG = 'data/reproduce_v12075_launch_20261007.log'


def group_resources(raw, pgid):
    rss = cpu = processes = 0
    for line in raw.splitlines():
        _pid, group, memory, percent = line.split()
        if int(group) == pgid:
            rss += int(memory) * 1024
            cpu += float(percent)
            processes += 1
    return dict(rss_bytes=rss, cpu_percent=cpu, processes=processes)


def enforce_limits(resources, elapsed, disk_free):
    if resources['rss_bytes'] > 16 * 1024**3:
        raise RuntimeError('owned process group exceeds 16 GiB RSS; stop without retry')
    if elapsed > 4 * 3600:
        raise RuntimeError('4 h replay deadline exceeded; stop without retry')
    if disk_free < 2500 * 1024**2:
        raise RuntimeError('replay disk floor below 2.5 GiB; stop without retry')


def supervise(root, commit):
    V.registration(root, commit)
    state = root / STATE
    state.mkdir(exist_ok=False)
    child = None
    pgid = None
    spawning = False
    pending_signal = None
    start = time.monotonic()
    peak_rss = 0

    def event(status, **extra):
        record = dict(status=status, unix=time.time(), **extra)
        with (state / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(record, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(record), flush=True)

    def signal_received(signum, _frame):
        nonlocal pending_signal
        if spawning:
            pending_signal = signum
        else:
            raise RuntimeError('supervisor termination requested: ' + str(signum))

    handlers = {s: signal.signal(s, signal_received) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        os.nice(10)
        V.R.save(state / 'started.json', dict(protocol=V.PROTOCOL, registration=commit,
            pid=os.getpid(), unix=time.time(), thread_limit=6, rss_limit_gib=16,
            rss_guard_interval_seconds=2, wall_limit_hours=4, automatic_retry=False))
        while True:
            memory = Q.memory()
            disk = shutil.disk_usage(root).free
            enforce_limits({'rss_bytes': 0}, time.monotonic() - start, disk)
            if disk < 4 * 1024**3:
                raise RuntimeError('need 4 GiB disk before replay')
            if memory['reclaimable_bytes'] >= 16 * 1024**3 and memory['memory_free_percent'] >= 10:
                break
            event('waiting_for_memory', **memory)
            time.sleep(20)
        command = [sys.executable, '-u', str(root / V.SELF), 'replay', '--root', str(root), '--registration', commit]
        with (state / 'replay.log').open('x') as log:
            spawning = True
            try:
                child = subprocess.Popen(command, cwd=root, env=V.R.environment(),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                pgid = child.pid
            finally:
                spawning = False
            if pending_signal is not None:
                raise RuntimeError('termination requested while spawning')
            V.R.save(state / 'child.json', dict(pid=child.pid, pgid=pgid, command=command, unix=time.time()))
            event('replay_started', pid=child.pid, pgid=pgid)
            last_event = 0
            while child.poll() is None:
                raw = subprocess.check_output(['ps', '-axo', 'pid=,pgid=,rss=,%cpu='], text=True, timeout=15)
                resources = group_resources(raw, pgid)
                peak_rss = max(peak_rss, resources['rss_bytes'])
                elapsed = time.monotonic() - start
                disk = shutil.disk_usage(root).free
                enforce_limits(resources, elapsed, disk)
                if elapsed - last_event >= 30:
                    event('replay_running', **resources, peak_rss_bytes=peak_rss, disk_free_bytes=disk, elapsed_seconds=elapsed)
                    last_event = elapsed
                time.sleep(2)
        V.R.save(state / 'exit.json', dict(returncode=child.returncode, unix=time.time(), peak_rss_bytes=peak_rss))
        if child.returncode:
            raise RuntimeError('replay failed; preserve partials and do not retry')
        if Q.group_exists(pgid):
            raise RuntimeError('replay leader exited with live descendants')
        proof = V.R.read(root / V.OUTPUT / 'complete.json')
        if (proof['protocol'] != V.PROTOCOL or proof['registration'] != commit
            or proof['rows'] != 344841 or not proof['byte_identical'] or not proof['exact_predictions']
            or proof['fits_performed'] != 0 or proof['output_sha256'] != V.BEST_SHA):
            raise ValueError('successful exact replay proof missing')
        V.R.save(state / 'complete.json', dict(protocol=V.PROTOCOL, registration=commit,
            replay_receipt_sha256=V.R.sha(root / V.OUTPUT / 'complete.json'),
            exact_predictions=True, byte_identical=True, fits_performed=0, uploaded=False,
            elapsed_seconds=time.monotonic()-start, peak_rss_bytes=peak_rss, unix=time.time()))
        event('complete_exact_v12075_no_upload', peak_rss_bytes=peak_rss)
    except BaseException:
        failure = traceback.format_exc()
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        cleanup = None
        try:
            if pgid is not None:
                Q.terminate_owned_group(child, pgid, term_timeout=15, kill_timeout=5)
        except BaseException:
            cleanup = traceback.format_exc()
        V.R.save(state / 'failure.json', dict(error=failure, cleanup_error=cleanup,
            owned_pgid=pgid, no_retry=True, partials_preserved=True, peak_rss_bytes=peak_rss, unix=time.time()))
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--registration', required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.launch:
        V.registration(root, args.registration)
        if (root / STATE).exists() or (root / V.OUTPUT).exists():
            raise ValueError('one-shot launch requires fresh state and output')
        with (root / LOG).open('x') as log:
            process = subprocess.Popen(['/usr/bin/caffeinate', '-i', sys.executable, '-u', str(root / V.LAUNCHER),
                '--root', str(root), '--registration', args.registration], cwd=root, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(supervisor_pid=process.pid, log=LOG)), flush=True)
    else:
        supervise(root, args.registration)


if __name__ == '__main__':
    main()

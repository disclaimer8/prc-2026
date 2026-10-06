"""Windows-only resource qualification of the frozen Cat fold1/seed0.

No queue ownership or merge. The original prepared bundle is read-only.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import traceback

PROTOCOL = 'cat-ally-windows-exact-qualification-v1'
PARAMETERS = dict(iterations=5600, learning_rate=.08, depth=8,
                  loss_function='RMSE', thread_count=6, verbose=500, random_seed=0)
PACKAGES = dict(catboost='1.2.10', lightgbm='4.7.0', numpy='2.5.3',
                polars='1.44.2', pandas='3.0.5', pyarrow='25.0.1')
INPUT_REGISTRATION = '55304792ee0bc0e39868c9138904723522f5fee9'
HOME_LINEAGE = '9ef4316be06938cedd0f072895ddea1850ee1e4d8aff234f2c5541cdd7d4f2f0'
MEMORY_LIMIT = 6 * 1024**3


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def verify_bundle(bundle):
    m = json.loads((bundle / 'manifest.json').read_text())
    if (m['protocol'] != 'cat-fin1b-laptop-prepared-pools-exact-parity-v1'
            or m['registration_commit'] != INPUT_REGISTRATION
            or m['home_lineage'] != HOME_LINEAGE
            or m['parameters'] != PARAMETERS
            or m['home_runtime']['packages'] != PACKAGES
            or m['train_rows'] != 1622159 or m['rank_rows'] != 344419
            or len(m['columns']) != 170):
        raise ValueError('Original scientific protocol differs')
    for name, expected in m['files'].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha(path) != expected:
            raise ValueError(f'Input bytes differ: {name}')
    return m


def compare(ids, reference, got_ids, prediction):
    import numpy as np
    if (ids.ndim != 1 or ids.shape != reference.shape
            or ids.shape != prediction.shape or not np.array_equal(ids, got_ids)
            or not np.isfinite(ids).all() or len(np.unique(ids)) != len(ids)
            or not np.isfinite(reference).all() or not np.isfinite(prediction).all()):
        raise ValueError('Invalid prediction population')
    return dict(rows=len(ids), exact=bool(np.array_equal(reference, prediction)),
                max_abs=float(np.max(np.abs(reference - prediction))))


def windows_limit(memory_limit=MEMORY_LIMIT):
    """Kernel-enforced committed-memory limit for this process, no global edits."""
    if sys.platform != 'win32':
        raise RuntimeError('Windows worker required')
    from ctypes import wintypes as W

    class Basic(ctypes.Structure):
        _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                    ('flags', W.DWORD), ('min_ws', ctypes.c_size_t),
                    ('max_ws', ctypes.c_size_t), ('active', W.DWORD),
                    ('affinity', ctypes.c_size_t), ('priority', W.DWORD),
                    ('scheduling', W.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ('read_ops', 'write_ops', 'other_ops', 'read', 'write', 'other')]

    class Extended(ctypes.Structure):
        _fields_ = [('basic', Basic), ('io', IO),
                    ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                    ('peak_process', ctypes.c_size_t), ('peak_job', ctypes.c_size_t)]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, W.LPCWSTR]
    kernel.CreateJobObjectW.restype = W.HANDLE
    kernel.SetInformationJobObject.argtypes = [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD]
    kernel.SetInformationJobObject.restype = W.BOOL
    kernel.QueryInformationJobObject.argtypes = [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD, ctypes.c_void_p]
    kernel.QueryInformationJobObject.restype = W.BOOL
    kernel.GetCurrentProcess.restype = W.HANDLE
    kernel.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
    kernel.AssignProcessToJobObject.restype = W.BOOL
    kernel.CloseHandle.argtypes = [W.HANDLE]
    kernel.SetThreadExecutionState.argtypes = [W.DWORD]
    kernel.SetThreadExecutionState.restype = W.DWORD
    handle = kernel.CreateJobObjectW(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    info = Extended()
    info.basic.flags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.process_memory = memory_limit
    if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())
    observed = Extended()
    if not kernel.QueryInformationJobObject(handle, 9, ctypes.byref(observed), ctypes.sizeof(observed), None):
        raise ctypes.WinError(ctypes.get_last_error())
    if observed.process_memory != memory_limit or not observed.basic.flags & 0x100:
        raise RuntimeError('Windows memory limit was not applied')
    return kernel, handle


def prepare(bundle, out, registration):
    verify_bundle(bundle)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'manifest.json', dict(protocol=PROTOCOL, registration=registration,
        worker_sha256=sha(__file__), bundle_manifest_sha256=sha(bundle / 'manifest.json'),
        parameters=PARAMETERS, packages=PACKAGES, memory_limit=MEMORY_LIMIT,
        threshold='344419 IDs; finite; np.array_equal; max_abs=0; no score',
        accepted_for_merge=False))


def run(bundle, out, registration):
    manifest = json.loads((out / 'manifest.json').read_text())
    if (manifest['protocol'] != PROTOCOL or manifest['registration'] != registration
            or manifest['worker_sha256'] != sha(__file__)
            or manifest['bundle_manifest_sha256'] != sha(bundle / 'manifest.json')
            or manifest['parameters'] != PARAMETERS or manifest['memory_limit'] != MEMORY_LIMIT
            or manifest['packages'] != PACKAGES):
        raise ValueError('Execution protocol drift')
    if {p.name for p in out.iterdir()} != {'manifest.json'}:
        raise ValueError('Fresh prepared output required; no auto-resume')
    kernel, handle = windows_limit()
    import psutil
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity([0, 2, 4, 6, 8, 10])
    battery = psutil.sensors_battery()
    if not battery or not battery.power_plugged:
        raise RuntimeError('Connect AC power before qualification')
    if psutil.virtual_memory().available < 2 * 1024**3:
        raise RuntimeError('Less than 2 GiB available before starting')
    runtime = dict(python=sys.version, packages={p: version(p) for p in PACKAGES})
    if runtime['packages'] != PACKAGES:
        raise ValueError('Scientific packages differ')
    write_json(out / 'invocation.json', dict(runtime=runtime, pid=os.getpid(),
        worker_sha256=sha(__file__), registration=registration,
        affinity=process.cpu_affinity(), priority=process.nice(), memory_limit=MEMORY_LIMIT,
        available_memory=psutil.virtual_memory().available, timeout_seconds=21600))
    for key in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT'):
        os.environ[key] = '6'
    os.environ.update(POLARS_MAX_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    os.chdir(out)
    log = (out / 'fit.log').open('x', encoding='utf-8', buffering=1)
    sys.stdout = sys.stderr = log
    start = time.monotonic()
    done = threading.Event()

    def monitor():
        while not done.wait(10):
            battery = psutil.sensors_battery()
            reason = ('timeout' if time.monotonic() - start > 21600 else
                      'AC disconnected' if not battery or not battery.power_plugged else
                      'available RAM below 256 MiB' if psutil.virtual_memory().available < 256 * 1024**2 else None)
            if reason:
                write_json(out / 'guard_stop.json', dict(reason=reason, seconds=time.monotonic()-start))
                os._exit(2)

    threading.Thread(target=monitor, daemon=True).start()
    try:
        if not kernel.SetThreadExecutionState(0x80000001):
            raise ctypes.WinError(ctypes.get_last_error())
        m = verify_bundle(bundle)
        import numpy as np
        import polars as pl
        from catboost import CatBoostRegressor, Pool
        with np.load(bundle / 'arrays.npz', allow_pickle=False) as z:
            arrays = dict(z)
        if len(arrays['target']) != m['train_rows'] or len(arrays['mvt_id']) != m['rank_rows']:
            raise ValueError('Pool row count differs')

        def pool(name, y=None):
            frame = pl.read_parquet(bundle / f'{name}.parquet')
            if frame.columns != m['columns']:
                raise ValueError('Feature order differs')
            pdf = frame.to_pandas()
            del frame
            for column in m['categoricals']:
                pdf[column] = pdf[column].fillna('NA').astype(str)
            value = Pool(pdf, y, cat_features=m['categoricals'])
            del pdf
            gc.collect()
            if value.get_feature_names() != m['columns']:
                raise ValueError('CatBoost feature order differs')
            return value

        print('BUILDING frozen train Pool', flush=True)
        train_pool = pool('train', arrays['target'])
        model = CatBoostRegressor(**PARAMETERS)
        print('TRAINING 1622159 rows; seed0; fold1+7; 5600 iterations; six threads', flush=True)
        model.fit(train_pool)
        del train_pool
        gc.collect()
        prediction = model.predict(pool('rank')) + arrays['base_p']
        np.savez(out / 'prediction.npz', mvt_id=arrays['mvt_id'], reg=prediction)
        model.save_model(str(out / 'model.cbm'))
        verify_bundle(bundle)
        report = compare(arrays['mvt_id'], arrays['reference'], arrays['mvt_id'], prediction)
        report.update(protocol=PROTOCOL, registration=registration, runtime=runtime,
                      seconds=time.monotonic()-start, prediction_sha256=sha(out/'prediction.npz'),
                      model_sha256=sha(out/'model.cbm'), accepted_for_merge=False,
                      limitation='One Windows technical repeat only; no queue reassignment or submission.')
        write_json(out / 'report.json', report)
        print(json.dumps(report), flush=True)
        if not report['exact']:
            raise ValueError('Exact qualification failed; keep artifacts, do not relax threshold')
    except BaseException as exc:
        write_json(out / 'failed.json', dict(error=repr(exc), seconds=time.monotonic()-start))
        traceback.print_exc()
        raise
    finally:
        done.set()
        kernel.SetThreadExecutionState(0x80000000)
        kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'check-limit'])
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--registration-commit')
    args = parser.parse_args()
    if args.action == 'check-limit':
        kernel, handle = windows_limit(256 * 1024**2)
        try:
            allocation = bytearray(384 * 1024**2)
        except MemoryError:
            print('WINDOWS_MEMORY_LIMIT_OK', flush=True)
        else:
            raise RuntimeError('Memory allocation exceeded supposed limit')
        finally:
            kernel.CloseHandle(handle)
        return
    if not args.bundle or not args.out or not re.fullmatch('[a-f0-9]{40}', args.registration_commit or ''):
        parser.error('bundle, out and full registration SHA required')
    globals()[args.action](args.bundle.resolve(), args.out.resolve(), args.registration_commit)


if __name__ == '__main__':
    main()

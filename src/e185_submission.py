"""Build the registered E185-only v12008 and attempt one guarded Oct 5 upload.

Run on the node holding v1139's input artifacts, after code+forecast registration:
  .venv/bin/python src/e185_submission.py build --registration-commit HASH \
      --gate-json data/e185_strict_20261004.json
  .venv/bin/python src/e185_submission.py preflight
  .venv/bin/python src/e185_submission.py upload --wait

Versions 12000..12019 are exclusively reserved; final is 12008. No retry/resume
over a previous reservation: investigate an interrupted build and register fresh
numbers. Upload never retries an intent, including a lost connection.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time

os.environ.setdefault('POLARS_MAX_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import numpy as np
import polars as pl

from submission_guard import (GuardError, TARGET, ID,
                              exclusive_lock, fetch_leaderboard, require_window,
                              sha256, single_attempt, validate_rank,
                              validate_submission, verify_hashes)

VERSION = 12008
WEIGHT = 0.35 * 2 / 3
PROTOCOL = 'E185-v1139-fixed-body035-times-two-thirds-2026-10-05'
FINAL = f'submissions/gentle-igloo_v{VERSION}.parquet'
MANIFEST = f'submissions/gentle-igloo_v{VERSION}.manifest.json'
RANK = 'data/e185_motion_rank.npz'
RANK_LOG = 'data/e185_rank.log'
TEMPLATE = 'data/prc-2026-datasets/submitting.parquet'
REFERENCE = 'submissions/gentle-igloo_v1139.parquet'
DONORS = [f'submissions/gentle-igloo_v{v}.parquet' for v in (1121, 1139, 291, 361, 44, 60)] + ['submissions/parts_v1121.npz']
D7 = 'n_pts,sf_n_seg,vi_n_pts,vi_prev_pts,vi_alt_first,vm_n_pts,vm_prev_pts'
D3 = 'vi_n_pts,vi_prev_pts,vi_alt_first'


def utcnow():
    return datetime.now(timezone.utc)


def write_exclusive_json(path, payload):
    with open(path, 'x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def validate_gate(evidence):
    if (evidence.get('valid') is not True or evidence.get('audit_only') is not True
            or evidence.get('status') != 'audit_valid' or evidence.get('fold_count') != 6
            or evidence.get('rows') != 2085047 or evidence.get('target_verified_against_features') is not True):
        raise GuardError('E185 requires a complete strict audit-only result')
    for key, expected in [('control', 'cv_preds_e172_raw_lobt_sched_eobt_aobt_mlpa_f1b'),
                          ('candidate', 'e185_cv_sub'), ('features', 'features_fin1b.parquet')]:
        if Path(evidence.get(key, '')).name != expected:
            raise GuardError('E185 gate lineage differs from registration 255')
    metrics = evidence.get('metrics', {})
    if not all(isinstance(metrics.get(k), (float, int)) and np.isfinite(metrics[k]) for k in ('decided', 'body', 'trimmed', 'lower_decided')):
        raise GuardError('E185 gate metrics are missing/nonfinite')
    # Registration 255 did not introduce a separate global-decided criterion.
    if not (type(metrics['lower_decided']) is int and 4 <= metrics['lower_decided'] <= 6
            and metrics['body'] < 0 and metrics['trimmed'] < 0):
        raise GuardError('E185 failed its original registered gate')
    return metrics


def reserve_versions(root, commit):
    if not re.fullmatch(r'[a-f0-9]{7,40}', commit):
        raise GuardError('registration commit hash required')
    subs = Path(root)/'submissions'
    for p in subs.iterdir():
        if any(12000 <= int(n) <= 12019 for n in re.findall(r'_v(\d+)(?:\D|$)', p.name)):
            raise GuardError(f'reserved version already exists: {p.name}')
    write_exclusive_json(subs/'e185_v12000_12019.reservation.json',
                         {'protocol': PROTOCOL, 'versions': list(range(12000, 12020)),
                          'registration_commit': commit, 'created_utc': utcnow().isoformat()})


def prepare_workspace(root, build, inputs, donors, sources):
    root, build = Path(root), Path(build)
    build.mkdir(exist_ok=False)
    for directory in ('data', 'src', 'submissions', 'logs'):
        (build/directory).mkdir()
    for name in sorted(set(inputs + donors)):
        destination = build/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to((root/name).resolve())
    for name in sources:
        destination = build/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/name, destination)


def stage_commands():
    """Same arguments and arithmetic as 95_builds_4oct T4, with E185 before E167."""
    stage = {'STAGE_SEEDS': '42,1,2,3,4', 'ADSB_DIR': 'data/adsblol_kt1',
             'PARTS': 'submissions/parts_v12001.npz'}
    return [
        (['e185_apply.py', '1121', '12000'], {'E185_W': str(WEIGHT), 'E185_RANK': RANK}),
        (['e167_nm_window_rule.py', '12000', '12001'], {}),
        (['@tabpfn', '12001', '12002'], {}),
        (['e141_apply.py', '12002', '12003'], {**stage, 'STAGE_KEEP_ALL': '1', 'DROP_FEATS': D7,
                                            'WALK_DIR': 'data/adsblol_sw_kt1', 'E36_BASE_DIR': 'data/cv_preds_e116prod'}),
        (['e64_apply.py', '12003', '12004'], {**stage, 'E64_KEEP_ALL': '1', 'DROP_FEATS': D3}),
        (['e152_stage_gate.py', '12002', '12004', '12001', '12005'], {'E152_GATE': '0.95'}),
        (['blend_submissions.py', '12006'], {'BLEND_BASE': '12005', 'BLEND_PARTS': '44:0.1214,60:0.0270'}),
        (['lobt_edge_lift.py', '12006', '12007'], {'LOBT_CLIP': '1'}),
        (['protect_tabpfn_rows.py', '12007', '12002', '12008'], {}),
    ]


def tabpfn_rows(build, source, out):
    read = lambda v: pl.read_parquet(build/f'submissions/gentle-igloo_v{v}.parquet')
    donor = read(361)
    ids = donor.join(read(291), on=ID, suffix='_b').filter(pl.col(TARGET) != pl.col(TARGET+'_b')).select(ID)
    if ids.height != 42:
        raise GuardError('TabPFN mask must contain exactly 42 donor rows')
    values = donor.join(ids, on=ID).select(ID, pl.col(TARGET).alias('tab'))
    base = read(source)
    base.join(values, on=ID, how='left').with_columns(pl.coalesce('tab', TARGET).alias(TARGET)).select(base.columns).write_parquet(
        build/f'submissions/gentle-igloo_v{out}.parquet')


def collect_inputs(root, gate_name):
    inputs = {RANK, RANK_LOG, TEMPLATE, 'data/prc-2026-datasets/ranking.parquet',
              'data/features_e106.parquet', 'data/features_fin1b.parquet', gate_name}
    patterns = ['data/prc-2026-datasets/training_*.parquet',
                'data/adsblol/offblock2_2025-*.parquet',
                'data/adsblol/visit_202[56]-*.parquet', 'data/adsblol/visit_m_202[56]-*.parquet',
                'data/adsblol/head_202[56]-*.parquet',
                'data/adsblol_kt1/offblock2_202[56]-*.parquet',
                'data/adsblol_kt1/offblock3_202[56]-*.parquet',
                'data/adsblol_kt1/offblock5_202[56]-*.parquet',
                'data/adsblol_sw_kt1/offblock2_202[56]-*.parquet',
                'data/cv_preds_nomonth/fold_*.npz', 'data/cv_preds_e116prod/fold_*.npz']
    for pattern in patterns:
        paths = sorted(root.glob(pattern))
        if not paths:
            raise GuardError(f'missing stage inputs: {pattern}')
        if 'cv_preds_' in pattern and len(paths) != 6:
            raise GuardError(f'incomplete stage OOF: {pattern}')
        inputs.update(str(p.relative_to(root)) for p in paths)
    for name in sorted(inputs | set(DONORS)):
        if not (root/name).is_file():
            raise GuardError(f'missing prerequisite: {name}')
    return sorted(inputs)


def clean_environment():
    # No inherited experimental knobs or credentials enter a stage process.
    env = {k: os.environ[k] for k in ('PATH', 'HOME', 'LANG', 'LC_ALL', 'TMPDIR', 'SYSTEMROOT') if k in os.environ}
    env.update(OMP_NUM_THREADS='6', POLARS_MAX_THREADS='6', OPENBLAS_NUM_THREADS='6', MKL_NUM_THREADS='6',
               POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR='load_as_storage',
               FEATURES='data/features_e106.parquet', PROXY_SCALE='1', CLF_COST_WEIGHT='1', CLF_COST_MIN_SPREAD='3600',
               DROP_PREFIX='day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config')
    return env


def validate_final(root):
    read = lambda name: pl.read_parquet(root/name)
    return validate_submission(read(FINAL), read(TEMPLATE), read(REFERENCE),
                               read('submissions/gentle-igloo_v291.parquet'), read('submissions/gentle-igloo_v361.parquet'))


def build_submission(root, registration_commit, gate_json):
    root = root.resolve()
    gate_path = (root/gate_json).resolve()
    if not gate_path.is_relative_to(root):
        raise GuardError('gate evidence must be inside project root')
    gate_name = str(gate_path.relative_to(root))
    gate = json.loads(gate_path.read_text())
    metrics = validate_gate(gate)
    inputs = collect_inputs(root, gate_name)
    sources = sorted(str(p.relative_to(root)) for p in (root/'src').glob('*.py'))
    ids = pl.read_parquet(root/TEMPLATE)[ID].to_numpy()
    ranking = validate_rank(root/RANK, root/RANK_LOG, ids)
    build = root/'data/e185_build_v12000'
    with exclusive_lock(root/'data/e185_build.lock'):
        reserve_versions(root, registration_commit)
        print('hashing immutable source/input inventory', flush=True)
        input_hashes = {name: sha256(root/name) for name in inputs + DONORS}
        source_hashes = {name: sha256(root/name) for name in sources}
        prepare_workspace(root, build, inputs, DONORS, sources)
        verify_hashes(build, source_hashes)
        verify_hashes(build, input_hashes)
        env = clean_environment()
        prefix = ['nice', '-n', '10']
        if sys.platform.startswith('linux'):
            prefix = ['taskset', '-c', '0-5'] + prefix
        for index, (args, extra) in enumerate(stage_commands()):
            print(f'stage {index + 1}/9 {args[0]}', flush=True)
            if args[0] == '@tabpfn':
                tabpfn_rows(build, int(args[1]), int(args[2]))
            else:
                with open(build/f'logs/{index:02d}_{args[0]}.log', 'x') as log:
                    subprocess.run(prefix + [sys.executable, 'src/'+args[0], *args[1:]], cwd=build,
                                   env={**env, **extra}, stdout=log, stderr=subprocess.STDOUT, check=True)
        validation = validate_final(build)
        # Source copies are fixed; also reject a concurrent change in root inputs.
        verify_hashes(root, input_hashes)
        verify_hashes(root, source_hashes)
        outputs = sorted(p for p in (build/'submissions').iterdir() if not p.is_symlink())
        expected = {f'gentle-igloo_v{v}.parquet' for v in range(12000, 12009)} | {'parts_v12000.npz', 'parts_v12001.npz'}
        if {p.name for p in outputs} != expected:
            raise GuardError('unexpected/missing assembly outputs')
        output_hashes = {'submissions/'+p.name: sha256(p) for p in outputs}
        manifest = {'protocol': PROTOCOL, 'version': VERSION, 'created_utc': utcnow().isoformat(),
                    'registration_commit': registration_commit, 'weight': WEIGHT,
                    'forecast_delta': -0.06, 'forecast_interval': [-0.15, 0.04],
                    'reference': REFERENCE, 'gate_file': gate_name, 'gate_metrics': metrics, 'ranking': ranking,
                    'sources': source_hashes, 'inputs': input_hashes, 'outputs': output_hashes,
                    'validation': validation, 'build_directory': str(build), 'stage_commands': stage_commands()}
        # Hard links fail if a name exists; final/manifest are never overwritten.
        for p in outputs:
            os.link(p, root/'submissions'/p.name)
        write_exclusive_json(root/MANIFEST, manifest)
        print(f'BUILD READY v{VERSION}; manifest {MANIFEST}', flush=True)
    return manifest


def preflight(root):
    root = root.resolve()
    manifest = json.loads((root/MANIFEST).read_text())
    if (manifest.get('protocol') != PROTOCOL or manifest.get('version') != VERSION
            or manifest.get('weight') != WEIGHT or manifest.get('reference') != REFERENCE):
        raise GuardError('manifest does not describe the fixed registered E185 recipe')
    required_inputs = set(DONORS) | {RANK, RANK_LOG, TEMPLATE, 'data/features_e106.parquet',
                                    'data/features_fin1b.parquet', 'data/prc-2026-datasets/ranking.parquet',
                                    manifest.get('gate_file', '')}
    required_sources = {'src/e185_submission.py', 'src/submission_guard.py', 'src/upload.py'} | {
        'src/'+args[0] for args, _ in stage_commands() if args[0] != '@tabpfn'}
    if (not required_inputs <= set(manifest.get('inputs', {}))
            or not required_sources <= set(manifest.get('sources', {}))):
        raise GuardError('manifest omits a required input or source')
    for group in ('sources', 'inputs', 'outputs'):
        verify_hashes(root, manifest[group])
    if FINAL not in manifest['outputs'] or RANK not in manifest['inputs'] or RANK_LOG not in manifest['inputs']:
        raise GuardError('manifest omits required E185 output/ranking')
    ids = pl.read_parquet(root/TEMPLATE)[ID].to_numpy()
    validate_gate(json.loads((root/manifest['gate_file']).read_text()))
    validate_rank(root/RANK, root/RANK_LOG, ids)
    validate_final(root)
    return manifest


def load_upload_credentials(root):
    """Use only the two competition keys; never print or persist their values."""
    credential_file = root/'.prc-upload-credentials.json'
    if credential_file.exists():
        info = credential_file.stat()
        if (credential_file.is_symlink() or not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077):
            raise GuardError('competition credential file must be private and owned by this user')
        try:
            credentials = json.loads(credential_file.read_text())
        except (ValueError, OSError):
            raise GuardError('competition credential file is unreadable') from None
        expected = {'PRC_ACCESS_KEY', 'PRC_SECRET_KEY'}
        if (not isinstance(credentials, dict) or set(credentials) != expected
                or not all(isinstance(v, str) and v for v in credentials.values())):
            raise GuardError('competition credential file must contain exactly two nonempty keys')
        os.environ.update(credentials)
    else:
        from dotenv import load_dotenv
        load_dotenv(root/'.env')
    if not all(os.environ.get(key) for key in ('PRC_ACCESS_KEY', 'PRC_SECRET_KEY')):
        raise GuardError('competition upload credentials are unavailable on this node')


def bucket_inventory(root):
    from minio import Minio
    load_upload_credentials(root)
    client = Minio('s3.opensky-network.org', access_key=os.environ['PRC_ACCESS_KEY'],
                   secret_key=os.environ['PRC_SECRET_KEY'], secure=True)
    objects = []
    for obj in client.list_objects('prc-2026-gentle-igloo', recursive=True):
        if obj.object_name.endswith('.parquet'):
            if not re.fullmatch(r'gentle-igloo_v\d+\.parquet', obj.object_name) or obj.last_modified is None:
                raise GuardError('ambiguous team bucket object')
            objects.append({'filename': obj.object_name, 'last_modified': obj.last_modified.isoformat()})
    return objects


def upload_submission(root, wait=False):
    root = root.resolve()
    start = datetime(2026, 10, 5, tzinfo=timezone.utc)
    if wait:
        while True:
            delay = (start-utcnow()).total_seconds()
            if delay <= 0:
                break
            time.sleep(min(30, delay))
    require_window(utcnow())
    manifest = preflight(root)
    def inventory():
        # Bucket times cover unprocessed uploads; leaderboard covers processed
        # attempts. Pagination, provider errors and unknown dates fail closed.
        return fetch_leaderboard(), bucket_inventory(root)
    def upload():
        require_window(utcnow())
        if (datetime(2026, 10, 6, tzinfo=timezone.utc) - utcnow()).total_seconds() < 120:
            raise GuardError('too close to UTC day end for a bounded upload')
        verify_hashes(root, {FINAL: manifest['outputs'][FINAL], REFERENCE: manifest['inputs'][REFERENCE],
                             'src/upload.py': manifest['sources']['src/upload.py']})
        result = subprocess.run([sys.executable, 'src/upload.py', str(VERSION)], cwd=root,
                                env={**os.environ, 'UPLOAD_REF': REFERENCE, 'UPLOAD_MAX_RMS': '400'},
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90)
        if result.returncode:
            raise GuardError('uploader did not report success')
    used = single_attempt(root/'data/submission_upload_ledger.jsonl', Path(FINAL).name,
                          manifest['outputs'][FINAL], now=utcnow, inventory=inventory, upload=upload)
    print(f'UPLOADED v{VERSION}; accounted {used}/5 UTC-day slots; check leaderboard separately', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['build', 'preflight', 'upload'])
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--registration-commit')
    parser.add_argument('--gate-json', default='data/e185_strict_20261004.json')
    parser.add_argument('--wait', action='store_true', help='upload only: wait until Oct 5 UTC')
    args = parser.parse_args()
    try:
        if args.action == 'build':
            if not args.registration_commit:
                raise GuardError('--registration-commit required before build')
            build_submission(args.root, args.registration_commit, args.gate_json)
        elif args.action == 'preflight':
            preflight(args.root)
            print(f'PREFLIGHT OK v{VERSION}')
        else:
            upload_submission(args.root, args.wait)
    except Exception as exc:
        # Credential-bearing third-party exception bodies are deliberately not printed.
        print(f'STOP: {exc}' if isinstance(exc, GuardError) else f'STOP: {type(exc).__name__}; inspect local artifacts', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

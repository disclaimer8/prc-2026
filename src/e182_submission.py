"""Fixed E182-only v12028 assembly and guarded single-PUT upload.

Register code and forecast together before build. Never fits E182 rank members,
never resumes a reserved version block, never changes the shared source tree.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import e185_submission as E
import numpy as np
import polars as pl
from submission_guard import GuardError, ID, TARGET, ROWS, exact_ids, exclusive_lock, sha256, verify_hashes

prepare_workspace = E.prepare_workspace
VERSION = 12028
PROTOCOL = 'E182-only-FIN1b-five-members-fixed-T4-2026-10-05'
FINAL = f'submissions/gentle-igloo_v{VERSION}.parquet'
MANIFEST = f'submissions/gentle-igloo_v{VERSION}.manifest.json'
BUILD = 'data/e182_build_v12020'
FEATURES = 'data/features_fin1b.parquet'
FEATURE_SHA = 'f061a441f6c97c232d9a403d612471e45020eb4ad56f501c2acfd7220a7d3052'
REFERENCE = 'submissions/gentle-igloo_v1139.parquet'
BEST = 'submissions/gentle-igloo_v12008.parquet'
CAT = 'submissions/cat_fin1b_seeds15.npz'
LGB = 'submissions/lgb_reg_seeds_v870.npz'
MLP = 'submissions/e130_mlpa_rank.npz'
FROZEN_HASHES = {
    CAT: 'ed525302b6e77ac186c37da34cbf3d643923e92652d7750712b6c38ff13fc1a3',
    MLP: 'd191f9a9ad566f7128646783a961b9a3d0f24ccf0a67143ef4a717e17f8ac2a5',
    FEATURES: FEATURE_SHA,
}
MEMBERS = [('raw', 2565, '0.0333333333333333'), ('lobt', 3065, '0.10'),
           ('sched', 2958, '0.10'), ('eobt', 2736, '0.10'), ('aobt', 2738, '0.10')]
DONORS = [f'submissions/gentle-igloo_v{v}.parquet' for v in (870, 1139, 12008, 291, 361, 44, 60)]
RESOURCE_PATCHES = {'src/train.py': 10, 'src/e36_adsb_stand_geometry.py': 8,
                    'src/e64_visit_stage.py': 8}
RANK_PROVENANCE = json.loads(Path(__file__).with_name('e182_rank_lineage.json').read_text())
CV_UNIT = 'prc-e184-cv-6t-20261005.service'
FORECAST = {'reference': REFERENCE, 'delta_seconds': -0.07, 'interval_seconds': [-0.20, 0.10],
            'reference_score': 232.6200, 'current_best': BEST, 'current_best_score': 232.3586,
            'delta_vs_current_best': 0.1914, 'interval_vs_current_best': [0.0614, 0.3614],
            'basis': 'new board forecast: approximately 30% of accepted E182 body delta -0.213973',
            'original_cv_forecast': {'registration': 248, 'delta_seconds': -0.15,
                                     'interval_seconds': [-0.40, 0.10]},
            'purpose': 'E182 component comparison; no E185 combination in this recipe'}


def validate_gate(evidence):
    if (evidence.get('valid') is not True or evidence.get('audit_only') is not True
            or evidence.get('status') != 'audit_valid' or evidence.get('fold_count') != 6
            or evidence.get('rows') != 2085047 or evidence.get('target_verified_against_features') is not True):
        raise GuardError('E182 requires the complete strict audit accepted in journal 265')
    for key, expected in [('control', 'cv_preds_e172_raw_lobt_sched_eobt_aobt_mlpa_f1b'),
                          ('candidate', 'cv_preds_e172_rawf_lobtf_schedf_eobtf_aobtf_mlpa_f1b'),
                          ('features', 'features_fin1b.parquet')]:
        if Path(evidence.get(key, '')).name != expected:
            raise GuardError('E182 gate lineage differs from the registered comparison')
    metrics = evidence.get('metrics', {})
    if not all(type(metrics.get(k)) in (float, int) and np.isfinite(metrics[k])
               for k in ('decided', 'body', 'trimmed', 'lower_decided')):
        raise GuardError('E182 gate metrics missing or nonfinite')
    if not (type(metrics['lower_decided']) is int and 4 <= metrics['lower_decided'] <= 6
            and metrics['trimmed'] < 0):
        raise GuardError('E182 failed its original four-fold/trimmed gate')
    return metrics


def scalar(z, key, expected):
    value = z[key]
    if value.ndim != 0 or type(value.item()) is not type(expected) or value.item() != expected:
        raise GuardError(f'wrong ranking metadata: {key}')


def prediction_arrays(z, ids, rows, seeds=None):
    exact_ids(z['mvt_id'], ids, rows)
    reg = z['reg']
    if reg.shape != (rows,) or reg.dtype.kind not in 'iuf' or not np.isfinite(reg).all():
        raise GuardError('ranking predictions missing or nonfinite')
    if seeds is not None:
        per_seed = z['per_seed']
        if (per_seed.shape != (seeds, rows) or per_seed.dtype.kind not in 'iuf'
                or not np.isfinite(per_seed).all() or not np.array_equal(reg, per_seed.mean(axis=0))):
            raise GuardError('ranking does not equal its complete seed average')


def validate_member(path, ids, mode, rounds, features_sha, rows=ROWS):
    try:
        with np.load(path, allow_pickle=False) as z:
            prediction_arrays(z, ids, rows, 5)
            for key, value in dict(complete=True, status='complete', expected_seeds=5, completed_seeds=5,
                                   expected_rows=rows, mode=mode, rounds=rounds,
                                   features_sha256=features_sha).items():
                scalar(z, key, value)
            if Path(z['features_path'].item()).name != 'features_fin1b.parquet' or features_sha != FEATURE_SHA:
                raise GuardError('ranking must use the registered FIN1b feature table')
            provenance = json.loads(z['provenance_json'].item())
            threads = provenance['params'].pop('num_threads')
            omp = provenance['config'].pop('OMP_NUM_THREADS')
            if type(threads) is not int or threads not in (12, 24) or omp != str(threads):
                raise GuardError('ranking thread provenance differs from the registered home/Mac run')
            if provenance != RANK_PROVENANCE:
                raise GuardError('ranking source, model, feature columns or seed lineage changed')
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        if isinstance(exc, GuardError):
            raise
        raise GuardError(f'incomplete/unreadable E182 member: {Path(path).name}') from exc
    return {'mode': mode, 'rounds': rounds, 'seeds': 5, 'rows': rows, 'sha256': sha256(path),
            'features_sha256': features_sha, 'num_threads': threads}


def validate_rankings(root, ids, features_sha):
    result = {mode: validate_member(root/f'submissions/e130_{mode}f_rank.npz', ids, mode, rounds, features_sha)
              for mode, rounds, _ in MEMBERS}
    verify_hashes(root, FROZEN_HASHES)
    for name, seeds in [(CAT, 15), (MLP, None), (LGB, None)]:
        with np.load(root/name, allow_pickle=False) as z:
            if name == LGB:
                if z['per_seed'].ndim != 2 or z['per_seed'].shape[0] < 1:
                    raise GuardError('base LightGBM seed artifact is incomplete')
                seeds = z['per_seed'].shape[0]
            prediction_arrays(z, ids, ROWS, seeds)
    return result


def reserve_versions(root, commit):
    if not re.fullmatch('[a-f0-9]{40}', commit):
        raise GuardError('full registration commit required')
    for path in (root/'submissions').iterdir():
        if any(12020 <= int(n) <= 12039 for n in re.findall(r'_v(\d+)(?:\D|$)', path.name)):
            raise GuardError(f'E182 reserved version already exists: {path.name}')
    E.write_exclusive_json(root/'submissions/e182_v12020_12039.reservation.json',
                           {'protocol': PROTOCOL, 'versions': list(range(12020, 12040)),
                            'registration_commit': commit, 'forecast': FORECAST, 'created_utc': E.utcnow().isoformat()})


def stage_commands():
    members = ','.join(f'submissions/e130_{mode}f_rank.npz:{weight}' for mode, _, weight in MEMBERS)
    stage = {'STAGE_SEEDS': '42,1,2,3,4', 'ADSB_DIR': 'data/adsblol_kt1',
             'PARTS': 'submissions/parts_v12021.npz'}
    return [
        (['blend.py', '870', '12020', '0.65'], {'FEATURES': FEATURES, 'LGB_SEEDS': LGB,
                                             'E130_MEMBERS': members + ',' + MLP + ':0.25'}),
        (['e167_nm_window_rule.py', '12020', '12021'], {}),
        (['@tabpfn', '12021', '12022'], {}),
        (['e141_apply.py', '12022', '12023'], {**stage, 'STAGE_KEEP_ALL': '1', 'DROP_FEATS': E.D7,
                                           'WALK_DIR': 'data/adsblol_sw_kt1', 'E36_BASE_DIR': 'data/cv_preds_e116prod'}),
        (['e64_apply.py', '12023', '12024'], {**stage, 'E64_KEEP_ALL': '1', 'DROP_FEATS': E.D3}),
        (['e152_stage_gate.py', '12022', '12024', '12021', '12025'], {'E152_GATE': '0.95'}),
        (['blend_submissions.py', '12026'], {'BLEND_BASE': '12025', 'BLEND_PARTS': '44:0.1214,60:0.0270'}),
        (['lobt_edge_lift.py', '12026', '12027'], {'LOBT_CLIP': '1'}),
        (['protect_tabpfn_rows.py', '12027', '12022', '12028'], {}),
    ]


def collect_inputs(root, gate_name):
    inputs = {E.TEMPLATE, FEATURES, 'data/features_e106.parquet', gate_name, CAT, LGB, MLP,
              'data/prc-2026-datasets/ranking.parquet', 'submissions/clf_v870.txt', 'submissions/reg_v870.txt'}
    inputs.update(f'submissions/e130_{mode}f_rank.npz' for mode, _, _ in MEMBERS)
    inputs.update(str(p.relative_to(root)) for p in root.glob('submissions/clf_v870_s*.txt'))
    patterns = ['data/prc-2026-datasets/training_*.parquet',
                'data/adsblol/offblock2_2025-*.parquet', 'data/adsblol/visit_202[56]-*.parquet',
                'data/adsblol/visit_m_202[56]-*.parquet', 'data/adsblol/head_202[56]-*.parquet',
                'data/adsblol_kt1/offblock2_202[56]-*.parquet', 'data/adsblol_kt1/offblock3_202[56]-*.parquet',
                'data/adsblol_kt1/offblock5_202[56]-*.parquet', 'data/adsblol_sw_kt1/offblock2_202[56]-*.parquet',
                'data/cv_preds_nomonth/fold_*.npz', 'data/cv_preds_e116prod/fold_*.npz']
    for pattern in patterns:
        paths = sorted(root.glob(pattern))
        if not paths or ('cv_preds_' in pattern and len(paths) != 6):
            raise GuardError(f'missing/incomplete stage input: {pattern}')
        inputs.update(str(p.relative_to(root)) for p in paths)
    for name in inputs | set(DONORS):
        if not (root/name).is_file():
            raise GuardError(f'missing prerequisite: {name}')
    return sorted(inputs)


def clean_environment():
    env = E.clean_environment()
    env.update(OMP_THREAD_LIMIT='6', OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    return env


def prepare_resources(build):
    for name, original in RESOURCE_PATCHES.items():
        path = build/name
        text = path.read_text()
        updated, count = re.subn(r'\bnum_threads=' + str(original) + r'\b', 'num_threads=6', text)
        if count != 1:
            raise GuardError(f'unknown stage threading source: {name}')
        path.write_text(updated)
    # The legacy blend reads this fixed filename. Only the private alias is made.
    (build/'submissions/cat_reg_rank.npz').symlink_to((build/CAT).resolve())


def validate_cv_state(text):
    state = dict(line.split('=', 1) for line in text.splitlines() if '=' in line)
    if (state.get('LoadState') != 'loaded' or not (state.get('ActiveState') == 'inactive'
            or (state.get('ActiveState') == 'active' and state.get('FreezerState') == 'frozen'))):
        raise GuardError('E184 must be frozen or finished; no concurrent LightGBM CV')


def require_cv_quiet():
    if not sys.platform.startswith('linux') or os.sched_getaffinity(0) != {0, 2, 4, 6, 8, 10}:
        raise GuardError('build requires the registered six-physical-core Hetzner affinity')
    result = subprocess.run(['systemctl', 'show', CV_UNIT, '--property=LoadState,ActiveState,FreezerState'],
                            capture_output=True, text=True, check=True, timeout=5)
    validate_cv_state(result.stdout)


def run_stage(build, index, args, env):
    require_cv_quiet()
    if args[0] == '@tabpfn':
        E.tabpfn_rows(build, int(args[1]), int(args[2]))
        return
    with open(build/f'logs/{index:02d}_{args[0]}.log', 'x') as log:
        child = subprocess.Popen(['nice', '-n', '10', sys.executable, '-u', 'src/'+args[0], *args[1:]],
                                 cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                require_cv_quiet()
                time.sleep(1)
            if child.returncode:
                raise GuardError(f'assembly stage failed: {args[0]}; inspect its scratch log')
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise


def validate_final(root):
    read = lambda name: pl.read_parquet(root/name)
    candidate, template = read(FINAL), read(E.TEMPLATE)
    result = E.validate_submission(candidate, template, read(REFERENCE),
                                    read('submissions/gentle-igloo_v291.parquet'),
                                    read('submissions/gentle-igloo_v361.parquet'))
    best = read(BEST)
    if best.schema != template.schema or best.columns != template.columns:
        raise GuardError('current best schema differs from template')
    exact_ids(best[ID].to_numpy(), template[ID].to_numpy(), ROWS)
    delta = candidate.sort(ID)[TARGET].to_numpy().astype(float) - best.sort(ID)[TARGET].to_numpy().astype(float)
    if not np.isfinite(delta).all():
        raise GuardError('nonfinite current best comparison')
    result['vs_v12008'] = {'changed': int(np.count_nonzero(delta)), 'rms': float(np.sqrt(np.mean(delta**2))),
                           'mean_delta': float(delta.mean()), 'max_abs': float(np.abs(delta).max())}
    return result


def expected_outputs():
    return {f'submissions/gentle-igloo_v{v}.parquet' for v in range(12020, 12029)} | {
        'submissions/parts_v12020.npz', 'submissions/parts_v12021.npz'}


def build_submission(root, registration_commit, gate_json):
    root = root.resolve()
    gate_path = (root/gate_json).resolve()
    if not gate_path.is_relative_to(root):
        raise GuardError('gate evidence must be inside project root')
    gate_name = str(gate_path.relative_to(root))
    metrics = validate_gate(json.loads(gate_path.read_text()))
    inputs = collect_inputs(root, gate_name)
    sources = sorted(str(p.relative_to(root)) for p in (root/'src').glob('*.py')) + ['src/e182_rank_lineage.json']
    ids = pl.read_parquet(root/E.TEMPLATE)[ID].to_numpy()
    rankings = validate_rankings(root, ids, sha256(root/FEATURES))
    require_cv_quiet()
    build = root/BUILD
    with exclusive_lock(root/'data/e182_build.lock'):
        reserve_versions(root, registration_commit)
        input_hashes = {name: sha256(root/name) for name in inputs + DONORS}
        source_hashes = {name: sha256(root/name) for name in sources}
        prepare_workspace(root, build, inputs, DONORS, sources)
        verify_hashes(build, source_hashes)
        verify_hashes(build, input_hashes)
        prepare_resources(build)
        scratch_sources = {name: sha256(build/name) for name in sources}
        env = clean_environment()
        for index, (args, extra) in enumerate(stage_commands()):
            print(f'stage {index+1}/9 {args[0]}', flush=True)
            run_stage(build, index, args, {**env, **extra})
        validation = validate_final(build)
        verify_hashes(root, input_hashes)
        verify_hashes(root, source_hashes)
        verify_hashes(build, scratch_sources)
        outputs = sorted(p for p in (build/'submissions').iterdir() if not p.is_symlink())
        if {'submissions/'+p.name for p in outputs} != expected_outputs():
            raise GuardError('unexpected/missing E182 assembly outputs')
        output_hashes = {'submissions/'+p.name: sha256(p) for p in outputs}
        manifest = {'protocol': PROTOCOL, 'version': VERSION, 'registration_commit': registration_commit,
                    'created_utc': E.utcnow().isoformat(), 'forecast': FORECAST, 'reference': REFERENCE,
                    'gate_file': gate_name, 'gate_metrics': metrics, 'rankings': rankings,
                    'sources': source_hashes, 'inputs': input_hashes, 'outputs': output_hashes,
                    'scratch_sources': scratch_sources, 'resource_patches': RESOURCE_PATCHES,
                    'environment': env, 'cv_unit': CV_UNIT, 'validation': validation,
                    'build_directory': BUILD, 'stage_commands': stage_commands()}
        # Exclusive, durable manifest inside scratch precedes publication; no overwrites.
        E.write_exclusive_json(build/'manifest.json', manifest)
        for path in outputs:
            os.link(path, root/'submissions'/path.name)
        os.link(build/'manifest.json', root/MANIFEST)
        print(f'BUILD READY v{VERSION}; immutable manifest {MANIFEST}', flush=True)
    return manifest


def preflight(root):
    root = root.resolve()
    manifest = json.loads((root/MANIFEST).read_text())
    if (manifest.get('protocol') != PROTOCOL or manifest.get('version') != VERSION
            or manifest.get('forecast') != FORECAST or manifest.get('reference') != REFERENCE
            or manifest.get('build_directory') != BUILD
            or manifest.get('stage_commands') != json.loads(json.dumps(stage_commands()))
            or manifest.get('resource_patches') != RESOURCE_PATCHES
            or not re.fullmatch('[a-f0-9]{40}', manifest.get('registration_commit', ''))):
        raise GuardError('manifest differs from the fixed E182 registration')
    required = set(collect_inputs(root, manifest['gate_file'])) | set(DONORS)
    required_sources = {'src/e182_submission.py', 'src/e182_rank_lineage.json', 'src/e185_submission.py',
                        'src/e185_single_put_upload.py', 'src/submission_guard.py'} | set(RESOURCE_PATCHES) | {
        'src/'+args[0] for args, _ in stage_commands() if args[0] != '@tabpfn'}
    if (set(manifest.get('inputs', {})) != required or set(manifest.get('outputs', {})) != expected_outputs()
            or not required_sources <= set(manifest.get('sources', {}))
            or set(manifest.get('scratch_sources', {})) != set(manifest['sources'])):
        raise GuardError('manifest omits required E182 source/input/output')
    for key in ('sources', 'inputs', 'outputs'):
        verify_hashes(root, manifest[key])
    verify_hashes(root/BUILD, manifest['scratch_sources'])
    validate_gate(json.loads((root/manifest['gate_file']).read_text()))
    validate_rankings(root, pl.read_parquet(root/E.TEMPLATE)[ID].to_numpy(), manifest['inputs'][FEATURES])
    validate_final(root)
    return manifest


def upload_submission(root):
    # Lazy imports keep build/test machines independent of the upload environment.
    from e185_single_put_upload import OnePutHTTP
    from minio import Minio
    E.require_window(E.utcnow())
    manifest = preflight(root)
    path = root/FINAL
    if not 0 < path.stat().st_size <= 5*1024*1024:
        raise GuardError('candidate must fit one S3 PUT')
    E.load_upload_credentials(root)
    http = OnePutHTTP()
    client = Minio('s3.opensky-network.org', access_key=os.environ['PRC_ACCESS_KEY'],
                   secret_key=os.environ['PRC_SECRET_KEY'], secure=True, http_client=http)

    def inventory():
        return E.fetch_leaderboard(), E.bucket_inventory(root)

    def timeout(_signum, _frame):
        raise TimeoutError('bounded single PUT expired')

    def put():
        E.require_window(E.utcnow())
        if (datetime(2026, 10, 6, tzinfo=timezone.utc)-E.utcnow()).total_seconds() < 120:
            raise GuardError('too close to UTC day end')
        verify_hashes(root, {FINAL: manifest['outputs'][FINAL]})
        previous = signal.signal(signal.SIGALRM, timeout)
        try:
            signal.alarm(90)
            client.fput_object('prc-2026-gentle-igloo', path.name, str(path), num_parallel_uploads=1)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    try:
        used = E.single_attempt(root/'data/submission_upload_ledger.jsonl', path.name,
                                manifest['outputs'][FINAL], now=E.utcnow, inventory=inventory, upload=put)
    finally:
        http.clear()
    print(f'UPLOADED v{VERSION}; accounted {used}/5 UTC-day slots', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'preflight', 'upload'])
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--registration-commit')
    parser.add_argument('--gate-json', default='data/e182_strict_20261005.json')
    args = parser.parse_args()
    try:
        if args.action == 'build':
            if not args.registration_commit:
                raise GuardError('--registration-commit required')
            build_submission(args.root, args.registration_commit, args.gate_json)
        elif args.action == 'preflight':
            preflight(args.root)
            print(f'PREFLIGHT OK v{VERSION}')
        else:
            upload_submission(args.root.resolve())
    except Exception as exc:
        print(f'STOP: {exc}' if isinstance(exc, GuardError) else f'STOP: {type(exc).__name__}; inspect local artifacts', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

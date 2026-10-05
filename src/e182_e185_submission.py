"""Separately registered E182+E185 fixed chain, final v12048.

Requires the complete, unchanged E182 build and the accepted fixed-weight
additivity evidence. Existing builders and shared stage sources stay untouched.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import sys

import e182_submission as B
import e185_submission as E
import numpy as np
import polars as pl
from submission_guard import GuardError, ID, TARGET, ROWS, exact_ids, exclusive_lock, sha256, verify_hashes

VERSION = 12048
PROTOCOL = 'E182-plus-E185-fixed-body035-times-two-thirds-T4-2026-10-05'
WEIGHT = .35*2/3
FINAL = f'submissions/gentle-igloo_v{VERSION}.parquet'
MANIFEST = f'submissions/gentle-igloo_v{VERSION}.manifest.json'
BUILD = 'data/e182_e185_build_v12040'
BASE = 'submissions/gentle-igloo_v12020.parquet'
PARTS = 'submissions/parts_v12020.npz'
REFERENCE = 'submissions/gentle-igloo_v12008.parquet'
PARENT_REGISTRATION = 'a5ab305e4edf94f0d57e3962acb1e193d2f12110'
ADDITIVITY_REGISTRATION = '6fb22b43c69465729d0ee3463c43c205771274c5'
ADDITIVITY_DIRECTORY = 'data/e185_on_e182_w035_20261005'
ADDITIVITY_MANIFEST = ADDITIVITY_DIRECTORY + '/manifest.json'
VERDICT = ADDITIVITY_DIRECTORY + '/verdict.json'
EVIDENCE_HASHES = {
    ADDITIVITY_MANIFEST: 'e73a34324dd937bea1f1f6a8392079db675fba3024311376df87684855510e21',
    VERDICT: 'a5f951e8e65c08e4debebbf33edb67fd92e3004aae1525475b774e63399b1390',
}
E185_RANK_SHA = 'ed13473f97202fd21589257f5cb98c81c5ddbdf4c1fc68380eac544ed36d3cd2'
FORECAST = {'reference': REFERENCE, 'reference_score': 232.3586,
            'delta_seconds': -0.07, 'interval_seconds': [-0.20, 0.10],
            'basis': 'retain verified E185 gain and expect the registered E182 transfer; no board weight tuning',
            'scope': 'separately registered combination after accepted additivity verdict 271'}


def validate_additivity(manifest, verdict):
    folds = {f'fold_{a}_{a+6}.npz' for a in range(1, 7)}
    flags = {'four_of_six': True, 'trimmed_lower': True, 'body_lower': True}
    try:
        if (manifest['registration_commit'] != ADDITIVITY_REGISTRATION
                or manifest['weight'] != .35 or manifest['ranking_weight'] != WEIGHT
                or manifest['subset'] != 'LOBT_flt IS NULL AND ADEP_mvt != LIRF'
                or Path(manifest['control_source']).name != 'cv_preds_e172_rawf_lobtf_schedf_eobtf_aobtf_mlpa_f1b'
                or Path(manifest['member_source']).name != 'cv_e130motion'
                or manifest['gate']['passed'] is not True or manifest['gate']['flags'] != flags
                or manifest['input_sha256'][B.FEATURES] != B.FEATURE_SHA
                or manifest['output_sha256']['verdict.json'] != EVIDENCE_HASHES[VERDICT]
                or len(manifest['folds']) != 6 or {f['name'] for f in manifest['folds']} != folds
                or any(f['max_abs_rebuild'] != {'reg': 0., 'pred': 0.} for f in manifest['folds'])):
            raise GuardError('additivity manifest differs from registered fixed E185-on-E182 experiment')
        for key, expected in {'status': 'accepted', 'valid': True, 'gate_passed': True,
                              'audit_only': False, 'require_body': True, 'fold_count': 6,
                              'rows': 2085047, 'target_verified_against_features': True}.items():
            if type(verdict[key]) is not type(expected) or verdict[key] != expected:
                raise GuardError(f'additivity verdict failed: {key}')
        for key in ('control', 'candidate'):
            path = Path(verdict[key])
            if path.name != key or path.parent.name != Path(ADDITIVITY_DIRECTORY).name:
                raise GuardError('additivity verdict lineage changed')
            checks = verdict['values_consistency'][key]
            if set(checks) != folds or any(v['checked'] is not True or v['consistent'] is not True
                                           or v['max_abs_seconds'] != 0. for v in checks.values()):
                raise GuardError('additivity values were not faithfully rebuilt in all six folds')
        metrics = verdict['metrics']
        if (Path(verdict['features']).name != 'features_fin1b.parquet' or verdict['gate_flags'] != flags
                or not all(type(metrics[k]) in (int, float) and np.isfinite(metrics[k])
                           for k in ('decided', 'body', 'trimmed', 'lower_decided'))
                or type(metrics['lower_decided']) is not int or not 4 <= metrics['lower_decided'] <= 6
                or metrics['body'] >= 0 or metrics['trimmed'] >= 0):
            raise GuardError('additivity failed its original four-fold/body/trimmed gate')
    except (KeyError, TypeError, AttributeError) as exc:
        raise GuardError('additivity evidence is incomplete') from exc
    return metrics


def read_additivity(root):
    verify_hashes(root, EVIDENCE_HASHES)
    return validate_additivity(json.loads((root/ADDITIVITY_MANIFEST).read_text()),
                               json.loads((root/VERDICT).read_text()))


def parent_inputs(parent):
    if parent.get('registration_commit') != PARENT_REGISTRATION:
        raise GuardError('parent E182 build has a different registration')
    try:
        return {**parent['inputs'], **{name: parent['outputs'][name] for name in (BASE, PARTS, B.FINAL)}}
    except (KeyError, TypeError) as exc:
        raise GuardError('parent E182 manifest omits its pre-E167 base or complete final') from exc


def prerequisites(root):
    # Parent gate, all five complete ranking members, all source/input/output SHA,
    # final schema and protected rows must pass before reserving any new version.
    parent = B.preflight(root)
    inputs = parent_inputs(parent)
    metrics = read_additivity(root)
    verify_hashes(root, {E.RANK: E185_RANK_SHA})
    ids = pl.read_parquet(root/E.TEMPLATE)[ID].to_numpy()
    ranking = E.validate_rank(root/E.RANK, root/E.RANK_LOG, ids)
    inputs.update(EVIDENCE_HASHES)
    inputs.update({E.RANK: E185_RANK_SHA, E.RANK_LOG: sha256(root/E.RANK_LOG), B.MANIFEST: sha256(root/B.MANIFEST)})
    verify_hashes(root, inputs)
    return parent, inputs, metrics, ranking


def reserve_versions(root, commit):
    if not re.fullmatch('[a-f0-9]{40}', commit):
        raise GuardError('full combination registration commit required')
    for path in (root/'submissions').iterdir():
        if any(12040 <= int(n) <= 12059 for n in re.findall(r'_v(\d+)(?:\D|$)', path.name)):
            raise GuardError(f'combination reserved version already exists: {path.name}')
    E.write_exclusive_json(root/'submissions/e182_e185_v12040_12059.reservation.json',
                           {'protocol': PROTOCOL, 'versions': list(range(12040, 12060)),
                            'registration_commit': commit, 'forecast': FORECAST, 'created_utc': E.utcnow().isoformat()})


def stage_commands():
    stage = {'STAGE_SEEDS': '42,1,2,3,4', 'ADSB_DIR': 'data/adsblol_kt1',
             'PARTS': 'submissions/parts_v12041.npz'}
    return [
        (['e185_apply.py', '12020', '12040'], {'E185_W': str(WEIGHT), 'E185_RANK': E.RANK}),
        (['e167_nm_window_rule.py', '12040', '12041'], {}),
        (['@tabpfn', '12041', '12042'], {}),
        (['e141_apply.py', '12042', '12043'], {**stage, 'STAGE_KEEP_ALL': '1', 'DROP_FEATS': E.D7,
                                           'WALK_DIR': 'data/adsblol_sw_kt1', 'E36_BASE_DIR': 'data/cv_preds_e116prod'}),
        (['e64_apply.py', '12043', '12044'], {**stage, 'E64_KEEP_ALL': '1', 'DROP_FEATS': E.D3}),
        (['e152_stage_gate.py', '12042', '12044', '12041', '12045'], {'E152_GATE': '0.95'}),
        (['blend_submissions.py', '12046'], {'BLEND_BASE': '12045', 'BLEND_PARTS': '44:0.1214,60:0.0270'}),
        (['lobt_edge_lift.py', '12046', '12047'], {'LOBT_CLIP': '1'}),
        (['protect_tabpfn_rows.py', '12047', '12042', '12048'], {}),
    ]


def expected_outputs():
    return {f'submissions/gentle-igloo_v{v}.parquet' for v in range(12040, 12049)} | {
        'submissions/parts_v12040.npz', 'submissions/parts_v12041.npz'}


def validate_final(root):
    read = lambda name: pl.read_parquet(root/name)
    candidate, template = read(FINAL), read(E.TEMPLATE)
    result = E.validate_submission(candidate, template, read(REFERENCE),
                                    read('submissions/gentle-igloo_v291.parquet'),
                                    read('submissions/gentle-igloo_v361.parquet'))
    result['rms_vs_v12008'] = result.pop('rms_vs_v1139')
    for name in (B.FINAL, B.REFERENCE):
        reference = read(name)
        if reference.schema != template.schema or reference.columns != template.columns:
            raise GuardError('comparison schema differs from template')
        exact_ids(reference[ID].to_numpy(), template[ID].to_numpy(), ROWS)
        delta = candidate.sort(ID)[TARGET].to_numpy().astype(float) - reference.sort(ID)[TARGET].to_numpy().astype(float)
        if not np.isfinite(delta).all():
            raise GuardError('nonfinite comparison predictions')
        result[Path(name).stem] = {'changed': int(np.count_nonzero(delta)), 'rms': float(np.sqrt(np.mean(delta**2))),
                                 'mean_delta': float(delta.mean()), 'max_abs': float(np.abs(delta).max())}
    return result


def build_submission(root, registration_commit):
    root = root.resolve()
    parent, input_hashes, metrics, ranking = prerequisites(root)
    B.require_cv_quiet()
    sources = sorted(str(p.relative_to(root)) for p in (root/'src').glob('*.py')) + ['src/e182_rank_lineage.json']
    build = root/BUILD
    with exclusive_lock(root/'data/e182_e185_build.lock'):
        reserve_versions(root, registration_commit)
        verify_hashes(root, parent['sources'])
        source_hashes = {name: sha256(root/name) for name in sources}
        E.prepare_workspace(root, build, sorted(input_hashes), [], sources)
        verify_hashes(build, source_hashes)
        verify_hashes(build, input_hashes)
        B.prepare_resources(build)
        scratch_sources = {name: sha256(build/name) for name in sources}
        env = B.clean_environment()
        for index, (args, extra) in enumerate(stage_commands()):
            print(f'stage {index+1}/9 {args[0]}', flush=True)
            B.run_stage(build, index, args, {**env, **extra})
        validation = validate_final(build)
        verify_hashes(root, input_hashes)
        verify_hashes(root, source_hashes)
        verify_hashes(build, scratch_sources)
        outputs = sorted(p for p in (build/'submissions').iterdir() if not p.is_symlink())
        if {'submissions/'+p.name for p in outputs} != expected_outputs():
            raise GuardError('unexpected/missing combination assembly outputs')
        output_hashes = {'submissions/'+p.name: sha256(p) for p in outputs}
        manifest = {'protocol': PROTOCOL, 'version': VERSION, 'registration_commit': registration_commit,
                    'created_utc': E.utcnow().isoformat(), 'forecast': FORECAST, 'weight': WEIGHT,
                    'reference': REFERENCE, 'parent_manifest': B.MANIFEST, 'parent_registration': PARENT_REGISTRATION,
                    'additivity_registration': ADDITIVITY_REGISTRATION, 'additivity_metrics': metrics,
                    'ranking': ranking, 'sources': source_hashes, 'inputs': input_hashes, 'outputs': output_hashes,
                    'scratch_sources': scratch_sources, 'resource_patches': B.RESOURCE_PATCHES,
                    'environment': env, 'cv_unit': B.CV_UNIT, 'validation': validation,
                    'build_directory': BUILD, 'stage_commands': stage_commands()}
        E.write_exclusive_json(build/'manifest.json', manifest)
        for path in outputs:
            os.link(path, root/'submissions'/path.name)
        os.link(build/'manifest.json', root/MANIFEST)
        print(f'BUILD READY v{VERSION}; immutable manifest {MANIFEST}', flush=True)
    return manifest


def preflight(root):
    root = root.resolve()
    _, required_inputs, _, _ = prerequisites(root)
    manifest = json.loads((root/MANIFEST).read_text())
    required_sources = {'src/e182_e185_submission.py', 'src/e182_submission.py', 'src/e182_rank_lineage.json',
                        'src/e185_submission.py', 'src/e185_single_put_upload.py', 'src/submission_guard.py'} | set(B.RESOURCE_PATCHES) | {
        'src/'+args[0] for args, _ in stage_commands() if args[0] != '@tabpfn'}
    fixed = dict(protocol=PROTOCOL, version=VERSION, forecast=FORECAST, weight=WEIGHT, reference=REFERENCE,
                 parent_manifest=B.MANIFEST, parent_registration=PARENT_REGISTRATION,
                 additivity_registration=ADDITIVITY_REGISTRATION, resource_patches=B.RESOURCE_PATCHES,
                 cv_unit=B.CV_UNIT, build_directory=BUILD,
                 stage_commands=json.loads(json.dumps(stage_commands())))
    if (any(manifest.get(k) != v for k, v in fixed.items())
            or not re.fullmatch('[a-f0-9]{40}', manifest.get('registration_commit', ''))
            or manifest.get('inputs') != required_inputs
            or set(manifest.get('outputs', {})) != expected_outputs()
            or not required_sources <= set(manifest.get('sources', {}))
            or set(manifest.get('scratch_sources', {})) != set(manifest.get('sources', {}))):
        raise GuardError('manifest differs from the fixed E182+E185 registration')
    for key in ('sources', 'inputs', 'outputs'):
        verify_hashes(root, manifest[key])
    verify_hashes(root/BUILD, manifest['scratch_sources'])
    validate_final(root)
    return manifest


def upload_submission(root):
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
    args = parser.parse_args()
    try:
        if args.action == 'build':
            if not args.registration_commit:
                raise GuardError('--registration-commit required')
            build_submission(args.root, args.registration_commit)
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

"""Final transport (09.10): two explicit modes, exactly one PUT each, pins in a sealed manifest.

control  Jan/Jul rows of the assembled four-month final file, extracted on the ordinary template
         (344,841 IDs, values/dtypes/column and row order as every regular submission), written as
         submissions/gentle-igloo_v<N>.parquet with an explicit unused N and uploaded as a regular
         submission (counts against the 5-per-UTC-day quota). Requires a pre-registered forecast
         line in HYPOTHESES_LOG.md that names v<N> and a corridor.
final    gentle-igloo_final.parquet uploaded unchanged under exactly that object name. Refused when
         any ledger line records the final name, after 2026-10-14T09:30Z, when fewer than one regular
         slot is free (the organiser never said whether the final counts), or when the validator
         report, the assembly complete.json or MANIFEST.json differ from the file. After the PUT the
         ledger records that regular uploads are closed; `control` refuses from then on.

Pins live in a sealed JSON manifest (`seal`, network-free, no constants to edit later). The manifest
is committed together with the journal line; `proof` exports Git object proofs of that commit (this
script, the manifest, HYPOTHESES_LOG.md); `preflight`/`upload` verify them without a checkout, as the
08.10 canary does. Frozen helpers are imported, pinned by SHA-256, never copied. No retry, no
overwrite, STOP on any mismatch.

    PY=.venv/bin/python                     # Mac; Hetzner: /root/prc-2026/.venv-upload/bin/python
    $PY scripts/final_transport_20261009.py seal --mode control --final-file <dir>/gentle-igloo_final.parquet \
        --complete <work>/complete.json --version N --forecast-line L --day 2026-10-13
    $PY scripts/final_transport_20261009.py seal --mode final --final-file <dir>/gentle-igloo_final.parquet \
        --complete <work>/complete.json --validator-report <dir>/validate.json \
        --manifest-json <dir>/MANIFEST.json --day 2026-10-13
    git add analysis/final_transport_20261009/manifest_<mode>.json HYPOTHESES_LOG.md; git commit; git push
    $PY scripts/final_transport_20261009.py proof --manifest <m> --registration-commit C --source-sha256 S > proof.json
    $PY scripts/final_transport_20261009.py inspect   --manifest <m>                         # local, no registration
    $PY scripts/final_transport_20261009.py preflight --manifest <m> --registration-proof proof.json \
        --registration-commit C --source-sha256 S
    $PY scripts/final_transport_20261009.py upload    (same arguments; only /root/prc-2026 + .venv-upload)

MANIFEST.json (runbook §2) must carry: file_sha256, size_bytes, template_sha256, ranking_sha256,
git_commit, journal, builder, machine, created_utc.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import signal
import sys

os.environ.setdefault('POLARS_MAX_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR', 'load_as_storage')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'src'))
sys.path.insert(0, str(HERE))
import numpy as np
import polars as pl
import submission_guard as G
import e185_submission as E
import submission_transport_20261008 as C
import validate_final_submission as V

PROTOCOL = 'final-transport-control-janjul-and-final-single-PUT-20261009-v1'
SELF = 'scripts/final_transport_20261009.py'
TEAM = 'gentle-igloo'
FINAL_NAME = f'{TEAM}_final.parquet'
BUCKET = 'prc-2026-gentle-igloo'
DEADLINE = datetime(2026, 10, 14, 9, 30, tzinfo=timezone.utc)   # 30 min before the 10:00Z deadline
SCIENTIFIC_PROTOCOL = 'final-uniform-tail-E156fresh-noE81-fourmonths-20261008-v1'
BUILDER = 'scripts/final_tail_assembly_20261008.py'
LEDGER = 'data/submission_upload_ledger.jsonl'
CANONICAL_ROOT = Path('/root/prc-2026')
MAX_SECONDS = V.MAX_SECONDS
ROWS = {'control': G.ROWS, 'final': 670790}
TEMPLATES = {'control': 'data/prc-2026-datasets/submitting.parquet',
             'final': 'data/prc-2026-datasets/final_submitting.parquet'}
RANKING = 'data/prc-2026-datasets/final_ranking.parquet'
MANIFEST_DIR = 'analysis/final_transport_20261009'
SOURCES = {
    'src/submission_guard.py': 'e2815dbbc2e16edbbe6140de3fbbb6cd4fcd937c7277bcf508ce4837595d0c57',
    'src/e185_submission.py': 'b1031e156dcd5ab65f808def63664b71eadfa99574d92112cd18b80c9dfef46c',
    'src/e185_single_put_upload.py': 'd7cc42d6ff5a5b16a3eb8be5858308596feb3268d00b022cfae4d1339c8354a8',
    'scripts/submission_transport_20261008.py': '79808a22e127d21709b50ab50576e023057afef61bae766e775e8fc1b26790f0',
    'scripts/validate_final_submission.py': '215a4c04f0f890058bed3bcf282762f8578b47268b80678b72e93103b6c577c2',
}
RUNTIME = {'src/submission_guard.py': G, 'src/e185_submission.py': E,
           'scripts/submission_transport_20261008.py': C, 'scripts/validate_final_submission.py': V}
MANIFEST_KEYS = ('file_sha256', 'size_bytes', 'template_sha256', 'ranking_sha256', 'git_commit', 'journal',
                 'builder', 'machine', 'created_utc')
NAME = re.compile(rf'{re.escape(TEAM)}_(v\d+|final)\.parquet')
CORRIDOR = re.compile(r'\[\s*\d+(?:[.,]\d+)?\s*;\s*\d+(?:[.,]\d+)?\s*\]')
EVENTS = {'intent', 'uploaded', 'ambiguous', 'closed'}
CREDENTIALS_LOADED = False


def now():
    return datetime.now(timezone.utc)


def require_window(day, current=None):
    current = now() if current is None else current
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', day or ''):
        raise G.GuardError('manifest day must be a UTC date')
    if current >= DEADLINE:
        raise G.GuardError('past 2026-10-14T09:30Z (deadline margin); nothing is uploaded any more')
    end = min(datetime.fromisoformat(day).replace(tzinfo=timezone.utc) + timedelta(days=1), DEADLINE)
    if G.timestamp(current).date().isoformat() != day or (end - current).total_seconds() < 120:
        raise G.GuardError(f'upload allowed only on {day} UTC with at least 120 seconds before the window ends')


def require_canonical_host(root):
    if root != CANONICAL_ROOT or Path(sys.prefix).resolve() != root/'.venv-upload':
        raise G.GuardError('canonical server root and existing .venv-upload required')


def relative(root, path):
    path = Path(path)
    path = path if path.is_absolute() else root/path
    try:
        rel = path.relative_to(root)
    except ValueError:
        raise G.GuardError('pinned files must live inside the project root') from None
    if not (root/rel).is_file():
        raise G.GuardError(f'missing file: {rel.as_posix()}')
    return rel.as_posix()


def load_json(root, name):
    return json.loads((root/name).read_text())


def read_frame(root, name):
    return pl.read_parquet(root/name)


def check_frame(frame, template, rows, label):
    if frame.schema != template.schema or frame.columns != template.columns:
        raise G.GuardError(f'{label}: schema differs from template')
    if template.height != rows:
        raise G.GuardError(f'{label}: template has {template.height} rows, expected {rows}')
    G.exact_ids(frame[G.ID].to_numpy(), template[G.ID].to_numpy(), rows)
    values = frame[G.TARGET].to_numpy()
    if frame[G.TARGET].null_count() or values.dtype.kind not in 'iu' or values.min() < 0 or values.max() > MAX_SECONDS:
        raise G.GuardError(f'{label}: predictions must be integer seconds in [0, {MAX_SECONDS}]')


def check_complete(receipt, final_sha):
    fixed = dict(output_sha256=final_sha, protocol=SCIENTIFIC_PROTOCOL, rows=ROWS['final'], uploaded=False,
                 e81_applied=False, donor_rows_copied=0)
    if any(receipt.get(k) != v for k, v in fixed.items()) or Path(str(receipt.get('output', ''))).name != FINAL_NAME:
        raise G.GuardError('assembly complete.json does not describe the unchanged four-month final file')


def derive_control(final, template):
    """Jan/Jul rows of the final file in template row order; values and dtypes untouched."""
    if final.schema != template.schema:
        raise G.GuardError('final file schema differs from the ordinary template')
    if final[G.ID].n_unique() != final.height:
        raise G.GuardError('final file has duplicated IDs')
    order = template.select(G.ID).with_row_index('_pos')
    control = order.join(final, on=G.ID, how='left').sort('_pos').select(template.columns)
    if control[G.TARGET].null_count():
        raise G.GuardError('ordinary template IDs are missing from the final file')
    return control


def check_report(report, sha):
    checks = report.get('checks')
    if report.get('passed') is not True or not isinstance(checks, list) or not checks:
        raise G.GuardError('validator report does not record a passed run')
    hard = [c for c in checks if c.get('level') == 'HARD']
    if len(hard) < 8 or any(c.get('ok') is not True for c in hard):
        raise G.GuardError('validator report has a failed or missing hard check')
    named = {c.get('name'): c for c in checks}
    if named.get('sha256', {}).get('sha256') != sha:
        raise G.GuardError('validator report describes another file')
    overlap = named.get('reference overlap')
    if overlap is not None and not (isinstance(overlap.get('rms_diff'), (int, float)) and overlap['rms_diff'] > 0):
        raise G.GuardError('Jan/Jul rows equal the reference exactly (copied rows) or RMS missing')


def check_manifest_json(mj, m):
    if not isinstance(mj, dict) or any(k not in mj for k in MANIFEST_KEYS):
        raise G.GuardError('MANIFEST.json lacks a runbook §2 field: ' + ', '.join(MANIFEST_KEYS))
    if (mj['file_sha256'] != m['candidate_sha256'] or mj['size_bytes'] != m['size_bytes']
            or mj['template_sha256'] != m['template_sha256'] or mj['ranking_sha256'] != m['final']['ranking_sha256']
            or mj['builder'] != BUILDER):
        raise G.GuardError('MANIFEST.json differs from the sealed pins')
    if (not re.fullmatch('[a-f0-9]{40}', str(mj['git_commit'])) or not mj['journal']
            or not isinstance(mj['machine'], str) or not mj['machine']):
        raise G.GuardError('MANIFEST.json needs a full pipeline commit, journal number and machine')
    G.timestamp(mj['created_utc'])


def line_pin(lines, number):
    if not isinstance(number, int) or not 1 <= number <= len(lines):
        raise G.GuardError('journal line missing')
    return hashlib.sha256(lines[number-1].encode()).hexdigest(), lines[number-1]


def forecast_pin(lines, number, version):
    sha, line = line_pin(lines, number)
    if f'v{version}' not in line or not CORRIDOR.search(line):
        raise G.GuardError('pre-registered forecast line (version + corridor) is missing from the journal')
    return sha


def check_journal(lines, m):
    sha, line = line_pin(lines, m['registration_line'])
    if sha != m['registration_line_sha256'] or PROTOCOL not in line:
        raise G.GuardError('transport registration line differs in the journal')
    if m['mode'] == 'control':
        c = m['control']
        if forecast_pin(lines, c['forecast_journal_line'], c['version']) != c['forecast_line_sha256']:
            raise G.GuardError('pre-registered forecast line (version + corridor) is missing from the journal')


def read_ledger(path):
    if not Path(path).exists():
        return []
    try:
        records = [json.loads(line) for line in Path(path).read_text().splitlines()]
        for record in records:
            if not isinstance(record, dict) or record.get('event') not in EVENTS or not NAME.fullmatch(record['filename']):
                raise ValueError('invalid record')
            G.timestamp(record['at'])
        return records
    except (ValueError, KeyError, TypeError) as exc:
        raise G.GuardError('malformed upload ledger; manual reconciliation required') from exc


def refuse_by_ledger(records, mode, filename):
    if any(r['filename'] == FINAL_NAME for r in records):
        raise G.GuardError('ledger already records the final file; regular uploads are closed' if mode == 'control'
                           else 'ledger already records a final PUT; no second final upload, no overwrite')
    if any(r['filename'] == filename for r in records):
        raise G.GuardError('version already in the ledger; no retry')


def check_version_unused(root, version, filename):
    for p in (root/'submissions').iterdir():
        if any(int(n) == version for n in re.findall(r'_v(\d+)(?:\D|$)', p.name)):
            raise G.GuardError(f'version already used: {p.name}')
    refuse_by_ledger(read_ledger(root/LEDGER), 'control', filename)


def write_once(path, frame):
    tmp = path.with_name(path.name + '.tmp')
    if path.exists() or tmp.exists():
        raise G.GuardError(f'refusing to overwrite {path.name}')
    frame.write_parquet(tmp)
    try:
        os.link(tmp, path)
    except FileExistsError:
        raise G.GuardError(f'refusing to overwrite {path.name}') from None
    finally:
        tmp.unlink()


def journal_pins(root):
    lines = (root/'HYPOTHESES_LOG.md').read_text().splitlines()
    numbers = [i+1 for i, line in enumerate(lines) if PROTOCOL in line]
    if not numbers:
        raise G.GuardError('transport protocol is not registered in HYPOTHESES_LOG.md')
    return lines, numbers[0]


def common_pins(root, a, mode, candidate):
    final_rel = relative(root, a.final_file)
    complete_rel = relative(root, a.complete)
    final_sha = G.sha256(root/final_rel)
    check_complete(load_json(root, complete_rel), final_sha)
    lines, registration_line = journal_pins(root)
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', a.day or ''):
        raise G.GuardError('--day YYYY-MM-DD (UTC day of the PUT) is required')
    size = (root/candidate).stat().st_size
    if not 0 < size <= 5*1024**2:
        raise G.GuardError('candidate must fit a single S3 PUT')
    return dict(protocol=PROTOCOL, mode=mode, sealed_utc=now().isoformat(), day=a.day,
                filename=Path(candidate).name, candidate=candidate, candidate_sha256=G.sha256(root/candidate),
                size_bytes=size, rows=ROWS[mode], template=TEMPLATES[mode], template_sha256=G.sha256(root/TEMPLATES[mode]),
                source_final=final_rel, source_final_sha256=final_sha,
                assembly_complete=complete_rel, assembly_complete_sha256=G.sha256(root/complete_rel),
                builder=BUILDER, builder_sha256=G.sha256(root/BUILDER),
                registration_line=registration_line, registration_line_sha256=line_pin(lines, registration_line)[0]), lines


def seal_control(root, a):
    if not isinstance(a.version, int) or a.version <= 0 or not isinstance(a.forecast_line, int):
        raise G.GuardError('--version N and --forecast-line L are required for control')
    filename = f'{TEAM}_v{a.version}.parquet'
    candidate = f'submissions/{filename}'
    check_version_unused(root, a.version, filename)
    forecast_sha = forecast_pin(journal_pins(root)[0], a.forecast_line, a.version)  # before the version is consumed
    final = read_frame(root, relative(root, a.final_file))
    check_frame(final, read_frame(root, TEMPLATES['final']), ROWS['final'], 'final file')
    template = read_frame(root, TEMPLATES['control'])
    control = derive_control(final, template)
    check_frame(control, template, ROWS['control'], 'control file')
    write_once(root/candidate, control)
    if not read_frame(root, candidate).equals(control):
        raise G.GuardError('written control file does not read back identically')
    m, lines = common_pins(root, a, 'control', candidate)
    m['control'] = dict(version=a.version, forecast_journal_line=a.forecast_line, forecast_line_sha256=forecast_sha)
    check_journal(lines, m)
    return m


def seal_final(root, a):
    if not a.validator_report or not a.manifest_json:
        raise G.GuardError('--validator-report and --manifest-json are required for final')
    candidate = relative(root, a.final_file)
    if Path(candidate).name != FINAL_NAME:
        raise G.GuardError(f'final candidate must be named {FINAL_NAME}')
    check_frame(read_frame(root, candidate), read_frame(root, TEMPLATES['final']), ROWS['final'], 'final file')
    m, lines = common_pins(root, a, 'final', candidate)
    report_rel, mj_rel = relative(root, a.validator_report), relative(root, a.manifest_json)
    check_report(load_json(root, report_rel), m['candidate_sha256'])
    m['final'] = dict(validator_report=report_rel, validator_report_sha256=G.sha256(root/report_rel),
                      manifest_json=mj_rel, manifest_json_sha256=G.sha256(root/mj_rel),
                      ranking_sha256=G.sha256(root/relative(root, RANKING)), deadline_utc=DEADLINE.isoformat())
    check_manifest_json(load_json(root, mj_rel), m)
    check_journal(lines, m)
    return m


def seal(root, a):
    if a.mode not in ROWS:
        raise G.GuardError('--mode control|final is required')
    out = root/(a.manifest or f'{MANIFEST_DIR}/manifest_{a.mode}.json')
    if out.exists():
        raise G.GuardError(f'manifest exists: {out}; seal a new one under another name')
    m = seal_control(root, a) if a.mode == 'control' else seal_final(root, a)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'x', encoding='utf-8') as stream:
        json.dump(m, stream, indent=1, sort_keys=True, allow_nan=False)
        stream.write('\n')
    validate(root, m)
    return out, m


def validate(root, m):
    """Local, network-free checks of the unchanged candidate against the sealed pins."""
    mode = m.get('mode')
    if m.get('protocol') != PROTOCOL or mode not in ROWS or m.get('rows') != ROWS[mode]:
        raise G.GuardError('manifest protocol/mode/rows differ')
    if m['filename'] != Path(m['candidate']).name or m['template'] != TEMPLATES[mode] or m['builder'] != BUILDER:
        raise G.GuardError('manifest names differ from the registered layout')
    pins = {**SOURCES, m['candidate']: m['candidate_sha256'], m['template']: m['template_sha256'],
            m['assembly_complete']: m['assembly_complete_sha256'], m['builder']: m['builder_sha256']}
    if mode == 'control':
        pins[m['source_final']] = m['source_final_sha256']
    else:
        pins[m['final']['validator_report']] = m['final']['validator_report_sha256']
        pins[m['final']['manifest_json']] = m['final']['manifest_json_sha256']
    G.verify_hashes(root, pins)
    for path, module in RUNTIME.items():
        if G.sha256(Path(module.__file__).resolve()) != SOURCES[path]:
            raise G.GuardError('imported runtime differs: '+path)
    path = root/m['candidate']
    if path.stat().st_size != m['size_bytes'] or not 0 < m['size_bytes'] <= 5*1024**2:
        raise G.GuardError('candidate size differs or exceeds a single S3 PUT')
    template = read_frame(root, m['template'])
    candidate = read_frame(root, m['candidate'])
    check_frame(candidate, template, ROWS[mode], mode)
    check_complete(load_json(root, m['assembly_complete']), m['source_final_sha256'])
    if mode == 'control':
        if m['filename'] != f"{TEAM}_v{m['control']['version']}.parquet" or m['candidate'] != 'submissions/'+m['filename']:
            raise G.GuardError('control file name differs from its version')
        final = read_frame(root, m['source_final'])
        if final.height != ROWS['final']:
            raise G.GuardError('source final file row count differs')
        if not derive_control(final, template).equals(candidate):
            raise G.GuardError('control file differs from the Jan/Jul rows of the final file')
    else:
        if m['filename'] != FINAL_NAME or m['source_final_sha256'] != m['candidate_sha256']:
            raise G.GuardError(f'final mode uploads {FINAL_NAME} itself, unchanged')
        check_report(load_json(root, m['final']['validator_report']), m['candidate_sha256'])
        check_manifest_json(load_json(root, m['final']['manifest_json']), m)
        ranking = root/RANKING
        fresh = V.run(argparse.Namespace(file=str(path), template=str(root/m['template']),
                                         ranking=str(ranking) if ranking.is_file() else None, reference=None,
                                         reference_rmse=V.DEFAULT_REFERENCE_RMSE, json=None))
        if not fresh.passed:
            raise G.GuardError('validator hard checks fail on the exact bytes: '
                               + '; '.join(c['name'] for c in fresh.checks if c['level'] == 'HARD' and not c['ok']))
    refuse_by_ledger(read_ledger(root/LEDGER), mode, m['filename'])
    return dict(mode=mode, filename=m['filename'], sha256=m['candidate_sha256'], rows=ROWS[mode],
                size_bytes=m['size_bytes'], day=m['day'], deadline_utc=DEADLINE.isoformat(),
                version=m.get('control', {}).get('version'))


def registration(proof, commit, source_sha, manifest_rel, manifest_bytes):
    if not re.fullmatch('[a-f0-9]{40}', commit or '') or not re.fullmatch('[a-f0-9]{64}', source_sha or ''):
        raise G.GuardError('full transport registration and source SHA256 required')
    if proof.get('protocol') != PROTOCOL or proof.get('registration') != commit or proof.get('manifest') != manifest_rel:
        raise G.GuardError('transport registration proof differs')
    blobs = C.verify_commit(proof['transport'], commit,
                            {SELF: source_sha, manifest_rel: hashlib.sha256(manifest_bytes).hexdigest(),
                             'HYPOTHESES_LOG.md': None})
    m = json.loads(manifest_bytes)
    check_journal(blobs['HYPOTHESES_LOG.md'].decode().splitlines(), m)
    if G.sha256(Path(__file__).resolve()) != source_sha:
        raise G.GuardError('active transport differs from registration')
    return m


def export_proof(root, commit, manifest_rel):
    return dict(protocol=PROTOCOL, registration=commit, manifest=manifest_rel,
                transport=C.export_commit(root, commit, [SELF, manifest_rel, 'HYPOTHESES_LOG.md']))


def preflight(root, proof_path, commit, source_sha, manifest_path):
    manifest_rel = relative(root, manifest_path)
    manifest_bytes = (root/manifest_rel).read_bytes()
    m = registration(json.loads(Path(proof_path).read_text()), commit, source_sha, manifest_rel, manifest_bytes)
    G.verify_hashes(root, {SELF: source_sha})
    if now() >= DEADLINE:
        raise G.GuardError('past 2026-10-14T09:30Z (deadline margin); nothing is uploaded any more')
    return m, validate(root, m)


def make_client(root):
    global CREDENTIALS_LOADED
    from e185_single_put_upload import OnePutHTTP
    from minio import Minio
    if G.sha256(Path(inspect.getfile(OnePutHTTP)).resolve()) != SOURCES['src/e185_single_put_upload.py']:
        raise G.GuardError('imported single PUT transport differs')
    E.load_upload_credentials(root)
    CREDENTIALS_LOADED = True
    http = OnePutHTTP()
    client = Minio('s3.opensky-network.org', access_key=os.environ['PRC_ACCESS_KEY'],
                   secret_key=os.environ['PRC_SECRET_KEY'], secure=True, http_client=http)
    return http, client


def bucket_inventory(root):
    """Like e185_submission.bucket_inventory, but the final object name is a valid team object."""
    global CREDENTIALS_LOADED
    from minio import Minio
    E.load_upload_credentials(root)
    CREDENTIALS_LOADED = True
    client = Minio('s3.opensky-network.org', access_key=os.environ['PRC_ACCESS_KEY'],
                   secret_key=os.environ['PRC_SECRET_KEY'], secure=True)
    objects = []
    for obj in client.list_objects(BUCKET, recursive=True):
        if obj.object_name.endswith('.parquet'):
            if not NAME.fullmatch(obj.object_name) or obj.last_modified is None:
                raise G.GuardError('ambiguous team bucket object')
            objects.append({'filename': obj.object_name, 'last_modified': obj.last_modified.isoformat()})
    return objects


def inventory(root):
    return G.fetch_leaderboard(), bucket_inventory(root)


def single_attempt(ledger, filename, digest, mode, day, *, now, inventory, upload):
    """submission_guard.single_attempt with a ledger that knows the final name and a `closed` event."""
    ledger = Path(ledger)
    with G.exclusive_lock(ledger.with_suffix(ledger.suffix + '.lock')):
        require_window(day, now())
        records = read_ledger(ledger)
        refuse_by_ledger(records, mode, filename)
        items, objects = inventory()  # Any network failure occurs before intent/write.
        current = now()
        used = G.check_quota(items, objects, records, current, filename)
        common = {'filename': filename, 'sha256': digest, 'at': G.timestamp(current).isoformat(),
                  'used_before': used, 'protocol': PROTOCOL, 'mode': mode}
        G.append_event(ledger, {**common, 'event': 'intent'})
        try:
            require_window(day, now())
            upload()
        except BaseException as exc:
            G.append_event(ledger, {**common, 'event': 'ambiguous', 'error_type': type(exc).__name__})
            raise G.GuardError('upload outcome ambiguous; intent retained; automatic retry forbidden') from None
        G.append_event(ledger, {**common, 'event': 'uploaded'})
        if mode == 'final':
            G.append_event(ledger, {**common, 'event': 'closed',
                                    'note': 'regular uploads closed after the final PUT'})
    return used + 1


def upload(root, proof_path, commit, source_sha, manifest_path):
    require_canonical_host(root)
    m, checks = preflight(root, proof_path, commit, source_sha, manifest_path)
    require_window(m['day'])
    G.DAY = m['day']  # This process only; the shared guard source remains frozen.
    http, client = make_client(root)
    path = root/m['candidate']

    def timeout(_signum, _frame):
        raise TimeoutError('single PUT expired')

    def put():
        require_window(m['day'])
        G.verify_hashes(root, {SELF: source_sha, m['candidate']: m['candidate_sha256'], **SOURCES})
        previous = signal.signal(signal.SIGALRM, timeout)
        try:
            signal.alarm(90)
            client.fput_object(BUCKET, m['filename'], str(path), num_parallel_uploads=1)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    print(f"TRANSPORT {PROTOCOL}; mode {m['mode']}; registration {commit}; source {source_sha}", flush=True)
    print(json.dumps(dict(preflight=checks), sort_keys=True), flush=True)
    started = now()
    try:
        used = single_attempt(root/LEDGER, m['filename'], m['candidate_sha256'], m['mode'], m['day'],
                              now=now, inventory=lambda: inventory(root), upload=put)
    finally:
        http.clear()
    receipt = dict(uploaded=True, mode=m['mode'], filename=m['filename'], sha256=m['candidate_sha256'],
                   version=m.get('control', {}).get('version'), started_utc=started.isoformat(),
                   finished_utc=now().isoformat(), slots_used=used, network_put_attempts=1,
                   regular_uploads_closed=m['mode'] == 'final')
    print(json.dumps(receipt, sort_keys=True), flush=True)
    print(f"UPLOADED {m['filename']}; accounted {used}/5 UTC-day slots", flush=True)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['seal', 'proof', 'inspect', 'preflight', 'upload'])
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--mode', choices=sorted(ROWS))
    p.add_argument('--final-file', help='seal: the assembled four-month gentle-igloo_final.parquet')
    p.add_argument('--complete', help='seal: assembly complete.json of that file')
    p.add_argument('--version', type=int, help='seal control: explicit unused regular version N')
    p.add_argument('--forecast-line', type=int, help='seal control: HYPOTHESES_LOG.md line with v<N> and the corridor')
    p.add_argument('--validator-report', help='seal final: validate_final_submission.py --json output')
    p.add_argument('--manifest-json', help='seal final: MANIFEST.json per runbook §2')
    p.add_argument('--day', help='seal: UTC day of the PUT (YYYY-MM-DD)')
    p.add_argument('--manifest', help='sealed manifest path (default analysis/final_transport_20261009/manifest_<mode>.json)')
    p.add_argument('--registration-commit')
    p.add_argument('--source-sha256')
    p.add_argument('--registration-proof', type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    try:
        if a.action == 'seal':
            out, m = seal(root, a)
            print(json.dumps(dict(sealed=str(out.relative_to(root)), mode=m['mode'], filename=m['filename'],
                                  sha256=m['candidate_sha256'], day=m['day'], network_used=False), sort_keys=True))
            print(f'SEALED {out.relative_to(root)}; commit it with HYPOTHESES_LOG.md before proof/preflight', file=sys.stderr)
            return 0
        if not a.manifest:
            p.error('--manifest is required')
        if a.action == 'inspect':
            print(json.dumps(dict(protocol=PROTOCOL, registered_transport=False, network_used=False,
                                  checks=validate(root, load_json(root, relative(root, a.manifest)))), sort_keys=True))
        elif a.action == 'proof':
            if not re.fullmatch('[a-f0-9]{40}', a.registration_commit or ''):
                raise G.GuardError('full registration commit required')
            manifest_rel = relative(root, a.manifest)
            proof = export_proof(root, a.registration_commit, manifest_rel)
            registration(proof, a.registration_commit, a.source_sha256, manifest_rel, (root/manifest_rel).read_bytes())
            print(json.dumps(proof, sort_keys=True))
        elif a.registration_proof is None:
            p.error('--registration-proof is required')
        elif a.action == 'preflight':
            m, checks = preflight(root, a.registration_proof, a.registration_commit, a.source_sha256, a.manifest)
            print(json.dumps(dict(protocol=PROTOCOL, registration=a.registration_commit, network_used=False,
                                  checks=checks), sort_keys=True))
        else:
            upload(root, a.registration_proof, a.registration_commit, a.source_sha256, a.manifest)
    except Exception as exc:
        # After credentials are loaded, third-party exception bodies are not printed.
        detail = str(exc) if isinstance(exc, G.GuardError) else (
            type(exc).__name__ if CREDENTIALS_LOADED else f'{type(exc).__name__}: {exc}')
        print('STOP: '+detail, file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

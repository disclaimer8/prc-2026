"""Synthetic tests for scripts/final_transport_20261009.py: small parquet files, a temp Git repo,
a fake single-PUT client. No competition rows, no network, no credentials.

Run from the repo root:
    .venv/bin/python -m unittest discover -s tests -p 'test_final_transport_20261009.py'
"""
import argparse
import contextlib
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR', 'load_as_storage')
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'src'))
import final_transport_20261009 as T  # noqa: E402
import submission_guard as G  # noqa: E402
import validate_final_submission as V  # noqa: E402

ID, Y = G.ID, G.TARGET
UTC = timezone.utc
N_FINAL, N_CONTROL, VERSION = 12, 6, 12101
DAY = '2026-10-13'
NOW = datetime(2026, 10, 13, 18, 0, tzinfo=UTC)
FORECAST_LINE = f'| 415 | 13.10 09:00Z | **Контроль Jan/Jul финала: v{VERSION}** | прогноз 232,2 с, коридор [231,8; 232,7] с | 🟡 |'


class FakeHTTP:
    def __init__(self):
        self.cleared = 0

    def clear(self):
        self.cleared += 1


class FakeClient:
    """Records PUTs; a second PUT in the same process raises like OnePutHTTP; may fail once."""
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def fput_object(self, bucket, name, path, num_parallel_uploads=None):
        if self.calls:
            raise RuntimeError('a PUT was already attempted; network retry forbidden')
        self.calls.append((bucket, name, Path(path).read_bytes(), num_parallel_uploads))
        if self.fail is not None:
            raise self.fail


def synthetic_ranking(ids):
    rows = []
    months = (1, 7, 2, 6)
    for k, mvt in enumerate(ids):
        takeoff = datetime(2026, months[k % 4], 10, 6 + k % 12, tzinfo=UTC)
        sched = takeoff - timedelta(seconds=900 + 60*k)
        rows.append({ID: float(mvt), 'PHASE_mvt': 'DEP', 'ADEP_mvt': 'EDDF' if k % 2 else 'EGLL',
                     'MVT_TIME_UTC_mvt': takeoff, 'SCHED_TIME_UTC_mvt': sched, 'IOBT_flt': sched, 'LOBT_flt': sched,
                     'EOBT_1_flt': sched, 'AOBT_3_flt': sched})
    tz = pl.Datetime('us', 'UTC')
    return pl.DataFrame(rows).with_columns([pl.col(c).cast(tz) for c in
                                           ('MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'IOBT_flt', 'LOBT_flt', 'EOBT_1_flt', 'AOBT_3_flt')])


def git(root, *args):
    return subprocess.check_output(['git', '-c', 'user.name=t', '-c', 'user.email=t@example.org', *args],
                                   cwd=root, text=True).strip()


class Fixture(unittest.TestCase):
    def setUp(self):
        self.saved = {k: getattr(T, k) for k in ('ROWS', 'now', 'require_canonical_host', 'make_client', 'inventory')}
        T.ROWS = {'control': N_CONTROL, 'final': N_FINAL}
        T.now = lambda: NOW
        T.require_canonical_host = lambda root: None
        self.client = FakeClient()
        self.http = FakeHTTP()
        T.make_client = lambda root: (self.http, self.client)
        self.inventory_calls = 0

        def inventory(root):
            self.inventory_calls += 1
            return [], []
        T.inventory = inventory
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.build_root()

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(T, k, v)
        self.tmp.cleanup()

    # ------------------------------------------------------------------ synthetic project
    def build_root(self):
        root = self.root
        for rel in [T.SELF, T.BUILDER, *T.SOURCES]:
            (root/rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/rel, root/rel)
        (root/'submissions').mkdir()
        (root/'data/prc-2026-datasets').mkdir(parents=True)
        rng = np.random.default_rng(7)
        ids = 202600001.0 + np.arange(N_FINAL, dtype=np.float64)
        order = rng.permutation(N_FINAL)
        final_ids = ids[order]
        pl.DataFrame({ID: final_ids, Y: pl.Series([None]*N_FINAL, dtype=pl.Int32)}).write_parquet(root/T.TEMPLATES['final'])
        jan_jul = ids[[k for k in range(N_FINAL) if k % 4 in (0, 1)]]  # synthetic_ranking: k%4 in (0,1) -> Jan/Jul
        control_ids = jan_jul[rng.permutation(len(jan_jul))]
        assert len(control_ids) == N_CONTROL
        pl.DataFrame({ID: control_ids, Y: pl.Series([None]*N_CONTROL, dtype=pl.Int32)}).write_parquet(root/T.TEMPLATES['control'])
        synthetic_ranking(ids).write_parquet(root/T.RANKING)
        self.values = rng.integers(300, 2000, N_FINAL).astype(np.int32)
        self.final = pl.DataFrame({ID: final_ids, Y: self.values})
        self.final_dir = root/'analysis/final_submission_test'
        self.final_dir.mkdir(parents=True)
        self.final_path = self.final_dir/T.FINAL_NAME
        self.write_final(self.final)
        self.write_journal()

    def write_final(self, frame, complete=True):
        frame.write_parquet(self.final_path)
        sha = G.sha256(self.final_path)
        if complete:
            (self.final_dir/'complete.json').write_text(json.dumps(dict(
                output=str(self.final_path), output_sha256=sha, protocol=T.SCIENTIFIC_PROTOCOL, rows=N_FINAL,
                uploaded=False, e81_applied=False, donor_rows_copied=0, fits_performed=0)))
        with contextlib.redirect_stdout(io.StringIO()):
            V.main(['--file', str(self.final_path), '--template', str(self.root/T.TEMPLATES['final']),
                    '--ranking', str(self.root/T.RANKING), '--json', str(self.final_dir/'validate.json')])
        (self.final_dir/'MANIFEST.json').write_text(json.dumps(dict(
            file_sha256=sha, size_bytes=self.final_path.stat().st_size,
            template_sha256=G.sha256(self.root/T.TEMPLATES['final']), ranking_sha256=G.sha256(self.root/T.RANKING),
            git_commit='a'*40, journal=414, builder=T.BUILDER, machine='mac-test', created_utc=NOW.isoformat())))

    def write_journal(self, forecast=FORECAST_LINE):
        lines = ['# Журнал гипотез', '', '| # | Время | Гипотеза | Как проверено | Итог |', '|---|---|---|---|---|']
        lines += [f'| {n} | 08.10 | строка {n} | — | ✅ |' for n in range(5, 414)]
        lines.append(f'| 414 | 08.10 17:00Z | **Финальный транспорт** | {T.PROTOCOL}; {T.SELF} | ➡ |')
        lines.append(forecast)
        (self.root/'HYPOTHESES_LOG.md').write_text('\n'.join(lines) + '\n')
        self.forecast_line = len(lines)

    def args(self, mode, **over):
        base = dict(mode=mode, final_file=str(self.final_path), complete=str(self.final_dir/'complete.json'),
                    version=VERSION, forecast_line=self.forecast_line, validator_report=str(self.final_dir/'validate.json'),
                    manifest_json=str(self.final_dir/'MANIFEST.json'), day=DAY, manifest=None)
        base.update(over)
        return argparse.Namespace(**base)

    def seal(self, mode, **over):
        out, m = T.seal(self.root, self.args(mode, **over))
        return out.relative_to(self.root).as_posix(), m

    def register(self, manifest_rel):
        if not (self.root/'.git').exists():
            git(self.root, 'init', '-q')
        git(self.root, 'add', '-A')
        git(self.root, 'commit', '-q', '-m', 'register')
        commit = git(self.root, 'rev-parse', 'HEAD')
        source_sha = G.sha256(ROOT/T.SELF)
        proof = T.export_proof(self.root, commit, manifest_rel)
        proof_path = self.root/f'proof_{Path(manifest_rel).stem}.json'
        proof_path.write_text(json.dumps(proof))
        return dict(proof_path=proof_path, commit=commit, source_sha=source_sha, manifest_path=manifest_rel)

    def ledger(self):
        return T.read_ledger(self.root/T.LEDGER)

    def upload(self, reg):
        with contextlib.redirect_stdout(io.StringIO()):
            return T.upload(self.root, **reg)

    def inject_ledger(self, filename, event='uploaded', at='2026-10-12T12:00:00+00:00'):
        G.append_event(self.root/T.LEDGER, {'filename': filename, 'event': event, 'at': at, 'sha256': 'f'*64})


class ControlMode(Fixture):
    def test_seal_derives_janjul_rows_unchanged_and_uploads_once(self):
        manifest_rel, m = self.seal('control')
        self.assertEqual(m['filename'], f'gentle-igloo_v{VERSION}.parquet')
        control = pl.read_parquet(self.root/m['candidate'])
        template = pl.read_parquet(self.root/T.TEMPLATES['control'])
        self.assertEqual(control.schema, template.schema)
        self.assertTrue(np.array_equal(control[ID].to_numpy(), template[ID].to_numpy()), 'template row order kept')
        joined = control.join(self.final, on=ID, suffix='_final')
        self.assertEqual(joined.height, N_CONTROL)
        self.assertTrue((joined[Y] == joined[Y+'_final']).all(), 'values unchanged')
        reg = self.register(manifest_rel)
        _, checks = T.preflight(self.root, **reg)
        self.assertEqual(checks['version'], VERSION)
        receipt = self.upload(reg)
        self.assertEqual(len(self.client.calls), 1)
        bucket, name, body, parallel = self.client.calls[0]
        self.assertEqual((bucket, name, parallel), (T.BUCKET, m['filename'], 1))
        self.assertEqual(body, (self.root/m['candidate']).read_bytes())
        self.assertEqual([r['event'] for r in self.ledger()], ['intent', 'uploaded'])
        self.assertEqual(receipt['slots_used'], 1)
        self.assertEqual(self.http.cleared, 1)
        # the same process or a new run: refused by the ledger before any network action
        calls_before, inventory_before = len(self.client.calls), self.inventory_calls
        with self.assertRaisesRegex(G.GuardError, 'already in the ledger'):
            self.upload(reg)
        self.assertEqual((len(self.client.calls), self.inventory_calls), (calls_before, inventory_before))

    def test_version_reuse_refused(self):
        (self.root/f'submissions/gentle-igloo_v{VERSION}.parquet').write_bytes(b'x')
        with self.assertRaisesRegex(G.GuardError, 'version already used'):
            self.seal('control')
        (self.root/f'submissions/gentle-igloo_v{VERSION}.parquet').unlink()
        self.inject_ledger(f'gentle-igloo_v{VERSION}.parquet')
        with self.assertRaisesRegex(G.GuardError, 'already in the ledger'):
            self.seal('control')

    def test_control_after_final_refused(self):
        self.inject_ledger(T.FINAL_NAME)
        with self.assertRaisesRegex(G.GuardError, 'regular uploads are closed'):
            self.seal('control')
        self.assertFalse((self.root/f'submissions/gentle-igloo_v{VERSION}.parquet').exists())

    def test_forecast_line_must_name_version_and_corridor(self):
        self.write_journal(forecast=f'| 415 | 13.10 | v{VERSION} без коридора | — | 🟡 |')
        with self.assertRaisesRegex(G.GuardError, 'forecast line'):
            self.seal('control')
        self.write_journal(forecast='| 415 | 13.10 | v99999 коридор [231,8; 232,7] | — | 🟡 |')
        with self.assertRaisesRegex(G.GuardError, 'forecast line'):
            self.seal('control')
        self.assertFalse((self.root/f'submissions/gentle-igloo_v{VERSION}.parquet').exists(), 'version not consumed')
        self.write_journal()
        self.seal('control')

    def test_dtype_change_in_control_file_is_a_hash_mismatch(self):
        manifest_rel, m = self.seal('control')
        reg = self.register(manifest_rel)
        tampered = pl.read_parquet(self.root/m['candidate']).with_columns(pl.col(Y).cast(pl.Int64))
        tampered.write_parquet(self.root/m['candidate'])
        with self.assertRaisesRegex(G.GuardError, 'hash mismatch'):
            T.preflight(self.root, **reg)
        with self.assertRaises(G.GuardError):
            self.upload(reg)
        self.assertEqual(self.client.calls, [])


class FinalMode(Fixture):
    def test_seal_upload_once_and_close_regular_uploads(self):
        manifest_rel, m = self.seal('final')
        self.assertEqual((m['filename'], m['rows']), (T.FINAL_NAME, N_FINAL))
        reg = self.register(manifest_rel)
        _, checks = T.preflight(self.root, **reg)
        self.assertEqual(checks['filename'], T.FINAL_NAME)
        receipt = self.upload(reg)
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.client.calls[0][:2], (T.BUCKET, T.FINAL_NAME))
        self.assertEqual(self.client.calls[0][2], self.final_path.read_bytes())
        self.assertEqual([r['event'] for r in self.ledger()], ['intent', 'uploaded', 'closed'])
        self.assertTrue(receipt['regular_uploads_closed'])
        with self.assertRaisesRegex(G.GuardError, 'no second final upload'):
            self.upload(reg)
        self.assertEqual(len(self.client.calls), 1)
        # a later control is refused, and the frozen guard stops on this ledger too
        with self.assertRaisesRegex(G.GuardError, 'regular uploads are closed'):
            self.seal('control')
        with self.assertRaises(G.GuardError):
            G.read_ledger(self.root/T.LEDGER)

    def test_ledger_already_has_final(self):
        self.inject_ledger(T.FINAL_NAME, event='intent')
        with self.assertRaisesRegex(G.GuardError, 'already records a final PUT'):
            self.seal('final')

    def test_wrong_row_count(self):
        self.write_final(self.final.head(N_FINAL - 1))
        with self.assertRaisesRegex(G.GuardError, 'row count'):
            self.seal('final')
        with self.assertRaisesRegex(G.GuardError, 'row count'):
            self.seal('control')

    def test_id_set_mismatch(self):
        ids = self.final[ID].to_numpy().copy()
        ids[0] = 999999999.0
        self.write_final(self.final.with_columns(pl.Series(ID, ids)))
        with self.assertRaisesRegex(G.GuardError, 'differ from template'):
            self.seal('final')
        with self.assertRaisesRegex(G.GuardError, 'differ from template'):
            self.seal('control')

    def test_dtype_change(self):
        self.write_final(self.final.with_columns(pl.col(Y).cast(pl.Int64)))
        with self.assertRaisesRegex(G.GuardError, 'schema differs'):
            self.seal('final')
        with self.assertRaisesRegex(G.GuardError, 'schema differs'):
            self.seal('control')

    def test_validator_report_and_manifest_json_pins(self):
        report = json.loads((self.final_dir/'validate.json').read_text())
        for c in report['checks']:
            if c['name'] == 'sha256':
                c['sha256'] = '0'*64
        (self.final_dir/'validate.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(G.GuardError, 'another file'):
            self.seal('final')
        self.write_final(self.final)
        mj = json.loads((self.final_dir/'MANIFEST.json').read_text())
        del mj['ranking_sha256']
        (self.final_dir/'MANIFEST.json').write_text(json.dumps(mj))
        with self.assertRaisesRegex(G.GuardError, 'MANIFEST.json'):
            self.seal('final')

    def test_quota_requires_a_free_regular_slot(self):
        manifest_rel, _ = self.seal('final')
        reg = self.register(manifest_rel)
        items = [dict(teamName=T.TEAM, filename=f'gentle-igloo_v{v}.parquet', submissionId=str(v),
                      processedAt=f'{DAY}T0{v % 5}:00:00Z') for v in range(13000, 13005)]
        T.inventory = lambda root: (items, [])
        with self.assertRaisesRegex(G.GuardError, 'daily upload limit'):
            self.upload(reg)
        self.assertEqual((self.client.calls, self.ledger()), ([], []))


class WindowAndRegistration(Fixture):
    def test_deadline_refusal(self):
        manifest_rel, _ = self.seal('final', day='2026-10-14')
        reg = self.register(manifest_rel)
        T.now = lambda: datetime(2026, 10, 14, 9, 31, tzinfo=UTC)
        with self.assertRaisesRegex(G.GuardError, 'past 2026-10-14T09:30Z'):
            self.upload(reg)
        T.now = lambda: datetime(2026, 10, 14, 9, 29, tzinfo=UTC)   # inside the day, < 120 s before the margin
        with self.assertRaisesRegex(G.GuardError, '120 seconds'):
            self.upload(reg)
        T.now = lambda: datetime(2026, 10, 13, 23, 0, tzinfo=UTC)   # wrong UTC day for this manifest
        with self.assertRaisesRegex(G.GuardError, 'allowed only on 2026-10-14'):
            self.upload(reg)
        self.assertEqual((self.client.calls, self.inventory_calls, self.ledger()), ([], 0, []))

    def test_manifest_tamper_after_commit(self):
        manifest_rel, m = self.seal('control')
        reg = self.register(manifest_rel)
        path = self.root/manifest_rel
        tampered = dict(m, candidate_sha256='0'*64)
        path.write_text(json.dumps(tampered, indent=1, sort_keys=True) + '\n')
        with self.assertRaisesRegex(G.GuardError, 'registered source or journal differs'):
            T.preflight(self.root, **reg)
        path.write_text(json.dumps(m, indent=1, sort_keys=True) + '\n')  # byte-identical to the commit again
        T.preflight(self.root, **reg)
        with self.assertRaisesRegex(G.GuardError, 'registration|differs'):
            T.preflight(self.root, **dict(reg, commit='1'*40))
        with self.assertRaisesRegex(G.GuardError, 'registration|differs'):
            T.preflight(self.root, **dict(reg, source_sha='2'*64))
        self.assertEqual(self.client.calls, [])

    def test_failed_put_is_ambiguous_and_never_retried(self):
        manifest_rel, _ = self.seal('control')
        reg = self.register(manifest_rel)
        self.client = FakeClient(fail=TimeoutError('single PUT expired'))
        T.make_client = lambda root: (self.http, self.client)
        with self.assertRaisesRegex(G.GuardError, 'ambiguous'):
            self.upload(reg)
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual([r['event'] for r in self.ledger()], ['intent', 'ambiguous'])
        with self.assertRaises(G.GuardError):
            self.upload(reg)
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.http.cleared, 1)


if __name__ == '__main__':
    unittest.main()

"""Quota, artifact, and transaction safety on synthetic files; no network/training."""
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import submission_guard as G
except ModuleNotFoundError:
    G = None

NOW = datetime(2026, 10, 5, 0, 1, tzinfo=timezone.utc)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(G, 'submission safety guard is not implemented')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def item(self, version, day='2026-10-05', team='gentle-igloo', **kwargs):
        return dict(submissionId=f'id-{version}', teamName=team,
                    filename=f'{team}_v{version}.parquet',
                    processedAt=f'{day}T00:00:10Z', **kwargs)

    def test_window_rejects_before_reset_after_day_and_naive_time(self):
        for now in [datetime(2026, 10, 4, 23, 59, 59, tzinfo=timezone.utc),
                    datetime(2026, 10, 6, tzinfo=timezone.utc), datetime(2026, 10, 5)]:
            with self.subTest(now=now), self.assertRaises(G.GuardError):
                G.require_window(now)
        G.require_window(datetime(2026, 10, 5, tzinfo=timezone.utc))
        G.require_window(datetime(2026, 10, 5, 23, 59, 59, tzinfo=timezone.utc))

    def test_quota_unions_bucket_ledger_and_all_leaderboard_pages(self):
        items = [self.item(1), self.item(2), self.item(99, team='another-team'), self.item(10, day='2026-10-04')]
        objects = [{'filename': 'gentle-igloo_v2.parquet', 'last_modified': NOW.isoformat()},
                   {'filename': 'gentle-igloo_v3.parquet', 'last_modified': NOW.isoformat()}]
        ledger = [{'event': 'intent', 'filename': 'gentle-igloo_v3.parquet', 'at': NOW.isoformat()},
                  {'event': 'uploaded', 'filename': 'gentle-igloo_v3.parquet', 'at': NOW.isoformat()},
                  {'event': 'intent', 'filename': 'gentle-igloo_v4.parquet', 'at': NOW.isoformat()},
                  {'event': 'uploaded', 'filename': 'gentle-igloo_v4.parquet', 'at': NOW.isoformat()}]
        self.assertEqual(G.check_quota(items, objects, ledger, NOW, 'gentle-igloo_v12008.parquet'), 4)
        with self.assertRaisesRegex(G.GuardError, 'limit'):
            G.check_quota(items + [self.item(5)], objects, ledger, NOW, 'gentle-igloo_v12008.parquet')

    def test_duplicate_remote_version_rejects_even_from_previous_day(self):
        with self.assertRaisesRegex(G.GuardError, 'already'):
            G.check_quota([self.item(12008, day='2026-10-04')], [], [], NOW, 'gentle-igloo_v12008.parquet')

    def test_bad_pagination_cannot_silently_under_count(self):
        pages = iter([{'items': [self.item(1)], 'nextCursor': 'x'},
                      {'items': [self.item(2)], 'nextCursor': None}])
        result = G.fetch_leaderboard(lambda url: next(pages))
        self.assertEqual(len(result), 2)
        for page in [{'items': [], 'nextCursor': 'x'}, {'items': [self.item(1)], 'nextCursor': 'x'}]:
            with self.subTest(page=page), self.assertRaises(G.GuardError):
                G.fetch_leaderboard(lambda url: page)
        with self.assertRaises(G.GuardError):
            G.fetch_leaderboard(lambda url: {'items': 'wrong', 'nextCursor': None})

    def test_partial_or_mismatched_rank_never_passes(self):
        rank, log = self.root/'rank.npz', self.root/'rank.log'
        ids = np.array([1, 2, 3])
        log.write_text('\n'.join(f'seed {i} done (1 min)' for i in range(5)) + '\ndone\n')
        good = np.full((5, 3), 120.)
        np.savez(rank, mvt_id=ids, per_seed=good, reg=good.mean(axis=0))
        G.validate_rank(rank, log, ids, expected_rows=3)
        for pred, rank_ids, reg in [(good[:4], ids, good[0]), (good, [1, 1, 3], good[0]),
                                    (good, ids, [120, 121, 120]), (good*np.nan, ids, good[0])]:
            np.savez(rank, mvt_id=rank_ids, per_seed=pred, reg=reg)
            with self.subTest(shape=pred.shape), self.assertRaises(G.GuardError):
                G.validate_rank(rank, log, ids, expected_rows=3)
        np.savez(rank, mvt_id=ids, per_seed=good, reg=good[0])
        log.write_text('seed 4 done\n')
        with self.assertRaises(G.GuardError):
            G.validate_rank(rank, log, ids, expected_rows=3)

    def test_submission_preserves_exact_ids_schema_and_42_donor_rows(self):
        ids = np.arange(50, dtype=np.float64)  # Official template uses Float64 IDs.
        def frame(values):
            return pl.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': np.array(values, dtype=np.int32)})
        template = frame(np.zeros(50))
        old = frame(np.full(50, 100))
        donor = frame(np.r_[np.full(42, 200), np.full(8, 100)])
        final = frame(np.r_[np.full(42, 200), np.full(8, 101)])
        check = lambda candidate: G.validate_submission(candidate, template, old, old, donor, expected_rows=50)
        result = check(final.reverse())
        self.assertEqual(result['protected_rows'], 42)
        self.assertEqual(result['changed'], 50)
        for candidate in [final.with_columns(pl.col('TAXITIME_SEC_mvt').cast(pl.Float64)),
                          pl.concat([final.head(49), final.head(1)]),
                          frame(np.r_[np.full(41, 200), 199, np.full(8, 101)])]:
            with self.subTest(schema=candidate.schema), self.assertRaises(G.GuardError):
                check(candidate)

    def test_exact_integer_float_ids_match_official_template_without_rounding(self):
        template = np.array([202180234., 213809059., 210000000.])
        G.exact_ids(template[::-1], template, 3)
        G.exact_ids(template.astype(np.int64), template, 3)
        for bad in [np.array([202180234.5, 213809059., 210000000.]),
                    np.array([np.nan, 213809059., 210000000.]),
                    np.array([np.inf, 213809059., 210000000.]),
                    np.array([float(2**54), 213809059., 210000000.]),
                    np.array([2**64-1, 213809059, 210000000], dtype=np.uint64)]:
            with self.subTest(bad=bad), self.assertRaises(G.GuardError):
                G.exact_ids(bad, template, 3)

    def test_hash_manifest_refuses_changes_and_path_escape(self):
        p = self.root/'input'; p.write_bytes(b'unchanged')
        hashes = {'input': G.sha256(p)}
        G.verify_hashes(self.root, hashes)
        p.write_bytes(b'changed')
        with self.assertRaisesRegex(G.GuardError, 'hash'):
            G.verify_hashes(self.root, hashes)
        with self.assertRaises(G.GuardError):
            G.verify_hashes(self.root, {'../outside': '0'*64})

    def test_ambiguous_upload_is_durable_and_never_retried(self):
        ledger = self.root/'ledger.jsonl'
        calls = []
        def network():
            calls.append('put')
            raise TimeoutError('connection lost after request')
        for _ in range(2):
            with self.assertRaises(G.GuardError):
                G.single_attempt(ledger, 'gentle-igloo_v12008.parquet', 'a'*64,
                                 now=lambda: NOW, inventory=lambda: ([], []), upload=network)
        self.assertEqual(calls, ['put'])
        events = [json.loads(x) for x in ledger.read_text().splitlines()]
        self.assertEqual([x['event'] for x in events], ['intent', 'ambiguous'])

    def test_midnight_crossing_after_inventory_makes_no_network_write(self):
        ledger = self.root/'ledger.jsonl'
        moments = iter([NOW, datetime(2026, 10, 6, tzinfo=timezone.utc)])
        calls = []
        with self.assertRaises(G.GuardError):
            G.single_attempt(ledger, 'gentle-igloo_v12008.parquet', 'a'*64,
                             now=lambda: next(moments), inventory=lambda: ([], []), upload=lambda: calls.append('put'))
        self.assertFalse(calls)
        self.assertFalse(ledger.exists())

    def test_fifth_success_is_recorded_and_sixth_rejected(self):
        ledger = self.root/'ledger.jsonl'
        calls = []
        G.single_attempt(ledger, 'gentle-igloo_v12008.parquet', 'a'*64,
                         now=lambda: NOW, inventory=lambda: ([self.item(i) for i in range(4)], []),
                         upload=lambda: calls.append('put'))
        self.assertEqual(calls, ['put'])
        with self.assertRaises(G.GuardError):
            G.single_attempt(ledger, 'gentle-igloo_v12009.parquet', 'b'*64,
                             now=lambda: NOW, inventory=lambda: ([self.item(i) for i in range(4)], []),
                             upload=lambda: calls.append('put'))
        self.assertEqual(calls, ['put'])


if __name__ == '__main__':
    unittest.main()

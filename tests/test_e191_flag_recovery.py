import sys
import unittest
import subprocess
import time
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import e191_flag_recovery as R


class RecoveryTests(unittest.TestCase):
    def test_blocked_extractor_pipe_is_stopped_by_independent_deadline(self):
        code = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(600)'],stdin=subprocess.PIPE); p.stdin.write(b'x'*(16*1024**2))"
        start=time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            R.bounded_process([sys.executable,'-c',code],timeout=.3)
        self.assertLess(time.monotonic()-start,3)

    def test_urls_are_same_fixed_day_and_trusted_archive(self):
        url = 'https://github.com/adsblol/globe_history_2025/releases/download/v2025.07.01-planes-readsb-prod-0/part.tar.aa'
        self.assertEqual(R.urls_for_day(url, '2025-07-01'), [url])
        for bad in [url.replace('github.com/', 'github.com.evil/'), url+'\n'+url, url.replace('2025.07.01', '2025.07.02')]:
            with self.assertRaises(ValueError): R.urls_for_day(bad, '2025-07-01')

    def test_parity_requires_same_points_and_freshness_is_separate(self):
        old = pl.DataFrame({'icao':['a','b'], 't':[1.,2.], 'gs':[None,5.]})
        new = old.reverse().with_columns(pl.Series('stale',[False,True]))
        self.assertTrue(R.point_parity(old,new))
        self.assertFalse(R.point_parity(old,new.with_columns(gs=pl.lit(7.)) ))
        self.assertFalse(R.point_parity(old,new.head(1)))
        with self.assertRaises(ValueError): R.point_parity(old,old)


if __name__=='__main__': unittest.main()

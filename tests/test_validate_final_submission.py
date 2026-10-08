"""Unit tests for scripts/validate_final_submission.py on small synthetic parquet files.

Run from the repo root:
    .venv/bin/python -m unittest discover -s tests -p 'test_validate_final_submission.py'
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_final_submission as V  # noqa: E402

AIRPORTS = ("EDDF", "EGLL")
MONTHS = (datetime(2026, 1, 10, tzinfo=timezone.utc), datetime(2026, 2, 10, tzinfo=timezone.utc),
          datetime(2026, 6, 10, tzinfo=timezone.utc), datetime(2026, 7, 10, tzinfo=timezone.utc))
N_PER_CELL = 5
N = len(AIRPORTS) * len(MONTHS) * N_PER_CELL  # 40 departures


def synthetic_ranking():
    """DEP rows for 2 airports x 4 months plus a few ARR rows that must be ignored."""
    rows = []
    mvt_id = 202600001.0
    for month in MONTHS:
        for apt in AIRPORTS:
            for k in range(N_PER_CELL):
                takeoff = month + timedelta(hours=6 + k, minutes=7 * k)
                sched = takeoff - timedelta(seconds=900 + 60 * k)
                rows.append({"MVT_ID_mvt": mvt_id, "PHASE_mvt": "DEP", "ADEP_mvt": apt, "MVT_TIME_UTC_mvt": takeoff,
                             "SCHED_TIME_UTC_mvt": sched, "IOBT_flt": sched + timedelta(minutes=1),
                             "LOBT_flt": sched + timedelta(minutes=1), "EOBT_1_flt": sched - timedelta(minutes=2),
                             "AOBT_3_flt": None if k == 0 else sched + timedelta(minutes=3)})
                mvt_id += 1
            rows.append({"MVT_ID_mvt": mvt_id, "PHASE_mvt": "ARR", "ADEP_mvt": "LFPG", "MVT_TIME_UTC_mvt": month,
                         "SCHED_TIME_UTC_mvt": month, "IOBT_flt": month, "LOBT_flt": month, "EOBT_1_flt": month,
                         "AOBT_3_flt": month})
            mvt_id += 1
    df = pl.DataFrame(rows)
    tz = pl.Datetime("us", "UTC")
    return df.with_columns([pl.col(c).cast(tz) for c in
                            ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "IOBT_flt", "LOBT_flt", "EOBT_1_flt", "AOBT_3_flt")])


def valid_values(ids):
    rng = np.random.default_rng(0)
    return pl.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": rng.integers(300, 2000, len(ids)).astype(np.int32)})


class ValidatorFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        ranking = synthetic_ranking()
        ranking.write_parquet(cls.dir / "final_ranking.parquet")
        dep_ids = ranking.filter(pl.col("PHASE_mvt") == "DEP")["MVT_ID_mvt"]
        cls.ids = dep_ids.to_numpy()
        template = pl.DataFrame({"MVT_ID_mvt": dep_ids,
                                 "TAXITIME_SEC_mvt": pl.Series([None] * len(dep_ids), dtype=pl.Int32)})
        template.write_parquet(cls.dir / "final_submitting.parquet")
        cls.valid = valid_values(cls.ids)
        # reference covers only the Jan/Jul rows, like v12075 does
        jan_jul = ranking.filter((pl.col("PHASE_mvt") == "DEP")
                                 & pl.col("MVT_TIME_UTC_mvt").dt.month().is_in([1, 7]))["MVT_ID_mvt"]
        ref = cls.valid.filter(pl.col("MVT_ID_mvt").is_in(jan_jul.implode()))
        ref = ref.with_columns((pl.col("TAXITIME_SEC_mvt") + 10).cast(pl.Int32))
        cls.reference = cls.dir / "gentle-igloo_v12075.parquet"
        ref.write_parquet(cls.reference)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def write(self, df, name="gentle-igloo_final.parquet", sub="case"):
        d = self.dir / sub
        d.mkdir(exist_ok=True)
        p = d / name
        df.write_parquet(p)
        return p

    def run_validator(self, path, extra=()):
        argv = ["--file", str(path), "--template", str(self.dir / "final_submitting.parquet"),
                "--ranking", str(self.dir / "final_ranking.parquet"), *extra]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = V.main(argv)
        return code, out.getvalue() + err.getvalue()


class TestHardChecks(ValidatorFixture):
    def test_valid_file_passes(self):
        p = self.write(self.valid, sub="ok")
        code, out = self.run_validator(p, ["--reference", str(self.reference), "--reference-rmse", "100"])
        self.assertEqual(code, 0, out)
        self.assertIn("RESULT: ALL HARD CHECKS PASS", out)
        self.assertIn("[PASS] dtypes", out)
        self.assertIn("per month x airport distribution", out)
        self.assertIn("fallback regime share", out)
        # reference diff: every overlapping row differs by exactly 10 s
        self.assertIn("RMS diff=10.000 s max |diff|=10 s", out)
        self.assertIn("Gram-identity RMSE range on the overlap: [90.0000, 110.0000] s", out)

    def test_wrong_dtype_fails_and_is_not_cast(self):
        bad = self.valid.with_columns(pl.col("TAXITIME_SEC_mvt").cast(pl.Int64))
        code, out = self.run_validator(self.write(bad, sub="dtype"))
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] dtypes", out)
        self.assertIn("file Int64 vs template Int32", out)

    def test_float_values_fail_dtype(self):
        bad = self.valid.with_columns(pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
        code, out = self.run_validator(self.write(bad, sub="float"))
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] dtypes", out)

    def test_missing_row_fails(self):
        code, out = self.run_validator(self.write(self.valid.head(N - 1), sub="missing"))
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] row count", out)
        self.assertIn("missing=1 extra=0", out)

    def test_extra_row_fails(self):
        extra = pl.DataFrame({"MVT_ID_mvt": [999999999.0], "TAXITIME_SEC_mvt": pl.Series([800], dtype=pl.Int32)})
        code, out = self.run_validator(self.write(pl.concat([self.valid, extra]), sub="extra"))
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] row count", out)
        self.assertIn("missing=0 extra=1", out)

    def test_duplicate_id_fails(self):
        dup = pl.concat([self.valid.head(N - 1), self.valid.head(1)])  # same row count, one ID twice, one missing
        code, out = self.run_validator(self.write(dup, sub="dup"))
        self.assertEqual(code, 1)
        self.assertIn("[PASS] row count", out)
        self.assertIn("[FAIL] ID duplicates: 1 duplicated IDs", out)
        self.assertIn("[FAIL] ID set equality", out)

    def test_nan_id_fails(self):
        ids = self.ids.copy()
        ids[3] = np.nan
        bad = self.valid.with_columns(pl.Series("MVT_ID_mvt", ids))
        code, out = self.run_validator(self.write(bad, sub="nanid"))
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] ID null/NaN: null=0, NaN=1", out)

    def test_null_value_fails(self):
        vals = self.valid["TAXITIME_SEC_mvt"].to_list()
        vals[0] = None
        bad = self.valid.with_columns(pl.Series("TAXITIME_SEC_mvt", vals, dtype=pl.Int32))
        code, out = self.run_validator(self.write(bad, sub="nullval"))
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] no null/NaN/inf: null=1", out)

    def test_negative_and_huge_values_fail(self):
        vals = self.valid["TAXITIME_SEC_mvt"].to_list()
        vals[0], vals[1] = -5, V.MAX_SECONDS + 1
        bad = self.valid.with_columns(pl.Series("TAXITIME_SEC_mvt", vals, dtype=pl.Int32))
        code, out = self.run_validator(self.write(bad, sub="range"))
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] value range", out)
        self.assertIn("negative=1", out)

    def test_wrong_name_fails(self):
        code, out = self.run_validator(self.write(self.valid, name="gentle-igloo_v99.parquet", sub="name"))
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] file name", out)
        # everything else still passes: the name is the only defect
        self.assertIn("[PASS] ID set equality", out)

    def test_wrong_column_order_fails(self):
        bad = self.valid.select("TAXITIME_SEC_mvt", "MVT_ID_mvt")
        code, out = self.run_validator(self.write(bad, sub="order"))
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] columns", out)

    def test_json_report(self):
        p = self.write(self.valid, sub="json")
        js = self.dir / "json" / "report.json"
        code, _ = self.run_validator(p, ["--json", str(js)])
        self.assertEqual(code, 0)
        rep = json.loads(js.read_text())
        self.assertTrue(rep["passed"])
        names = {c["name"] for c in rep["checks"]}
        self.assertTrue({"sha256", "dtypes", "ID set equality", "value range"} <= names)
        sha = next(c for c in rep["checks"] if c["name"] == "sha256")["sha256"]
        self.assertEqual(sha, V.sha256_file(p))

    def test_missing_file_exit_2(self):
        code, _ = self.run_validator(self.dir / "nowhere" / "gentle-igloo_final.parquet")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()

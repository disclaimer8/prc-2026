"""Synthetic contracts for the optional E211 stage (src/e211_neighbour_level_apply.py). No competition rows are opened.

    .venv/bin/python -m unittest discover -s tests -p 'test_e211_neighbour_level_apply.py'
"""
import json
import os
import subprocess
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
sys.path.insert(0, str(ROOT / "src"))
import e211_neighbour_level_cv as E  # noqa: E402
import e211_neighbour_level_apply as S  # noqa: E402

ID, Y = E.ID, E.Y
T0 = datetime(2026, 2, 10, 6, 0, tzinfo=timezone.utc)


def raw_frame(rng, n_dep=400, n_arr=300, days=3):
    """DEP + ARR rows in the dataset's column layout: two airports, two runways, a LIRF-without-NM block, some AOBT_3 nulls."""
    rows = []
    aps = ["EDDF", "LTFM", "LIRF"]
    for i in range(n_dep):
        ap = aps[i % 3]
        t = T0 + timedelta(seconds=float(rng.uniform(0, days * 86400)))
        has_nm = not (ap == "LIRF" and i % 2 == 0)
        aobt = None if rng.random() < 0.1 else t - timedelta(seconds=float(rng.uniform(300, 2400)))
        rows.append(dict(MVT_ID_mvt=float(1000 + i), PHASE_mvt="DEP", ADEP_mvt=ap, ADES_mvt="XXXX", RUNWAY_mvt=f"R{i % 2}",
                         STAND_mvt=f"S{i % 5}", FLIGHT_ID_mvt=(float(i) if has_nm else None), MVT_TIME_UTC_mvt=t, AOBT_3_flt=aobt,
                         TAXITIME_SEC_mvt=None))
    for j in range(n_arr):
        ap = aps[j % 3]
        t = T0 + timedelta(seconds=float(rng.uniform(0, days * 86400)))
        rows.append(dict(MVT_ID_mvt=float(5000 + j), PHASE_mvt="ARR", ADEP_mvt="XXXX", ADES_mvt=ap, RUNWAY_mvt=f"R{j % 2}",
                         STAND_mvt=f"S{j % 5}", FLIGHT_ID_mvt=float(j), MVT_TIME_UTC_mvt=t, AOBT_3_flt=None,
                         TAXITIME_SEC_mvt=int(rng.integers(200, 1200))))
    df = pl.DataFrame(rows, schema_overrides={"FLIGHT_ID_mvt": pl.Float64, "TAXITIME_SEC_mvt": pl.Int32,
                                              "MVT_TIME_UTC_mvt": pl.Datetime("us", "UTC"), "AOBT_3_flt": pl.Datetime("us", "UTC")})
    return df


class E211Stage(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(211)
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "submissions").mkdir()
        self.rank = raw_frame(self.rng)
        self.rank.write_parquet(root / "ranking.parquet")
        train = raw_frame(self.rng, n_dep=50, n_arr=600, days=30)
        train.write_parquet(root / "training_2025-01-01_2025-02-01.parquet")
        dep = self.rank.filter(pl.col("PHASE_mvt") == "DEP")
        pred = self.rng.integers(600, 2000, dep.height).astype(float)
        pred[:5] = 90000  # copies of the regime mixture: never touched
        self.sub = pl.DataFrame({ID: dep[ID].to_numpy()[::-1], Y: pred[::-1]}).with_columns(pl.col(Y).cast(pl.Int64))
        self.sub.write_parquet(root / "submissions/gentle-igloo_v1.parquet")
        beta = {"EDDF": [0.3, -0.2, 0.1], "LTFM": [-0.4, 0.5, -0.1], "LIRF": [0.2, 0.2, 0.2]}
        (root / "beta.json").write_text(json.dumps(dict(beta=beta)))
        self.beta = beta
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def run_stage(self, scale, new_v):
        env = {**os.environ, "E211_SUBDIR": str(self.root / "submissions"), "E211_BETA": str(self.root / "beta.json"),
               "E211_SCALE": str(scale), "E211_RANKING": str(self.root / "ranking.parquet"),
               "E211_TRAINING_GLOB": str(self.root / "training_*.parquet"), "OMP_NUM_THREADS": "1", "POLARS_MAX_THREADS": "1"}
        subprocess.run([sys.executable, str(ROOT / "src/e211_neighbour_level_apply.py"), "1", str(new_v)], env=env, check=True,
                       capture_output=True, cwd=self.root)
        return pl.read_parquet(self.root / f"submissions/gentle-igloo_v{new_v}.parquet")

    def test_scale_zero_is_identity(self):
        out = self.run_stage(0.0, 2)
        self.assertEqual(out.schema, self.sub.schema)
        self.assertTrue(out.equals(self.sub))

    def test_population_order_and_values(self):
        out = self.run_stage(1.0, 3)
        self.assertEqual(out.schema, self.sub.schema)
        self.assertTrue(np.array_equal(out[ID].to_numpy(), self.sub[ID].to_numpy()), "ID set/order changed")
        dep, arr_rank = E.load_raw([str(self.root / "ranking.parquet")])
        _, arr_train = E.load_raw([str(self.root / "training_2025-01-01_2025-02-01.parquet")])
        d = self.sub.with_columns(pl.col(ID).cast(pl.Int64).alias("_id")).join(dep.rename({ID: "_id"}), on="_id", how="left",
                                                                                 maintain_order="left")
        s = self.sub[Y].cast(pl.Float64).to_numpy()
        corr = S.correction(d, s, arr_rank, arr_train, self.beta, 1.0)
        expected = np.clip(np.rint(s + corr), 0, None)
        self.assertTrue(np.array_equal(out[Y].cast(pl.Float64).to_numpy(), expected))
        untouched = ((s > 7200) | ((d["ADEP_mvt"] == "LIRF") & ~d["has_nm"]).to_numpy())
        self.assertTrue(np.array_equal(out[Y].to_numpy()[untouched], self.sub[Y].to_numpy()[untouched]), "protected rows moved")
        self.assertTrue((corr[untouched] == 0).all())
        self.assertGreater(int((out[Y] != self.sub[Y]).sum()), 10, "the stage corrected nothing on the applied rows")
        self.assertTrue(np.all(np.abs(corr) <= sum(abs(b) for v in self.beta.values() for b in v) * E.CLIP))

    def test_signals_centred_within_airport_day_and_label_free(self):
        dep, arr_rank = E.load_raw([str(self.root / "ranking.parquet")])
        _, arr_train = E.load_raw([str(self.root / "training_2025-01-01_2025-02-01.parquet")])
        s = self.rng.integers(600, 2000, dep.height).astype(float)
        sig = E.build_signals(dep.select(ID, "ADEP_mvt", "RUNWAY_mvt", "t", "own_p"), s, E.arrival_residuals(arr_rank, normal_from=arr_train))
        for c, m in (("S1", "m1"), ("S2", "m2"), ("S3", "m3")):
            g = sig.filter(pl.col(m).is_not_nan()).group_by("ADEP_mvt", "day").agg(pl.col(c).sum())
            self.assertTrue(np.allclose(g[c].to_numpy(), 0.0, atol=1e-6), c + " not centred")
            self.assertTrue((sig.filter(pl.col(m).is_nan())[c] == 0.0).all())
        # label-free: the departures' TAXITIME column is all null in the ranking frame and never read by the signal builder
        self.assertEqual(self.rank.filter(pl.col("PHASE_mvt") == "DEP")[Y].null_count(), (self.rank["PHASE_mvt"] == "DEP").sum())
        # shifting the chain prediction of one row changes only its neighbours' S1/S3 (own residual proxy), never S2
        s2 = s.copy()
        s2[7] += 500
        sig2 = E.build_signals(dep.select(ID, "ADEP_mvt", "RUNWAY_mvt", "t", "own_p"), s2, E.arrival_residuals(arr_rank, normal_from=arr_train))
        self.assertTrue(np.array_equal(sig["S2"].to_numpy(), sig2["S2"].to_numpy()))


if __name__ == "__main__":
    unittest.main()

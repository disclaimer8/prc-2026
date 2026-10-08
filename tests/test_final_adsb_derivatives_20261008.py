"""Synthetic tests of scripts/final_adsb_derivatives_20261008.py: no real producer, no real ADS-B data.

Stub producers with the real producers' file names are written into a temporary repository root; they read the
same hard-coded paths (data/prc-2026-datasets/ranking.parquet, data/adsblol/surface_<day>.parquet) and the same
env knobs, so the WORK layout, the recipe env, the receipts and the refusals are exercised end to end.
"""
import datetime as dt
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
M = importlib.import_module("final_adsb_derivatives_20261008")

STUB = r'''
import json, os, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl
day = sys.argv[1]
name = os.path.basename(__file__)
prefix = {"adsb_offblock_v2.py": "offblock2", "adsb_offblock_v3.py": "offblock3", "adsb_offblock_v5.py": "offblock5",
          "adsb_visit.py": "visit", "e69_visit_merge.py": "visit_m", "e134_head_features.py": "head"}[name]
out_dir = "data/adsblol" if prefix == "head" else os.environ.get("ADSB_OUT_DIR", "data/adsblol")
ctl = json.load(open(f"data/adsblol/surface_{day}.parquet"))
if ctl.get("fail") == name:
    sys.exit(3)
y, m, d = map(int, day.split("-"))
src = pl.read_parquet("data/prc-2026-datasets/ranking.parquet")
dep = src.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("MVT_TIME_UTC_mvt").dt.date() == pl.date(y, m, d)))
k = ctl.get("matched", 0)
ids = dep["MVT_ID_mvt"].to_list()[:dep.height - ctl.get("drop_rows", 0)]
rows = [(i, 3 if j < k else 0, 1 if j < k else 0, *([1.0] * 5 if j < k else [None] * 5)) for j, i in enumerate(ids)]
schema = {"MVT_ID_mvt": pl.Float64, "coverage": pl.Int64, "n_cand": pl.Int64, "sf_n_seg": pl.Float64,
          "gs_stand_known": pl.Float64, "vi_first_seen": pl.Float64, "vm_n_pts": pl.Float64, "hd_gs0": pl.Float64}
df = pl.DataFrame(rows, schema=schema, orient="row")
if prefix == "head":
    df = df.filter(pl.col("hd_gs0").is_not_null())
os.makedirs(out_dir, exist_ok=True)
json.dump({k2: os.environ.get(k2) for k2 in ("MOVING_KT", "STAND_WALK", "VISIT_PREV_DAY", "ADSB_OUT_DIR")},
          open(f"{out_dir}/{prefix}_{day}.env.json", "w"))
df.write_parquet(f"{out_dir}/{prefix}_{day}.parquet")
'''
DAYS = ["2026-01-10", "2026-02-10", "2026-06-10", "2026-07-10"]
NDEP = {"2026-01-10": 3, "2026-02-10": 2, "2026-06-10": 2, "2026-07-10": 1}


def ranking() -> pl.DataFrame:
    rows, i = [], 1000
    for day, n in NDEP.items():
        y, m, d = map(int, day.split("-"))
        for _ in range(n):
            rows.append((float(i), "DEP", "EDDF", dt.datetime(y, m, d, 12, tzinfo=dt.timezone.utc))); i += 1
        rows.append((float(i), "ARR", "EDDF", dt.datetime(y, m, d, 13, tzinfo=dt.timezone.utc))); i += 1
    return pl.DataFrame(rows, schema={"MVT_ID_mvt": pl.Float64, "PHASE_mvt": pl.String, "ADEP_mvt": pl.String,
                                      "MVT_TIME_UTC_mvt": pl.Datetime("us", "UTC")}, orient="row")


class FinalAdsbDerivativeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name) / "repo"
        (self.root / "src").mkdir(parents=True); (self.root / "analysis/sources").mkdir(parents=True)
        (self.root / "data/prc-2026-datasets").mkdir(parents=True); (self.root / "data/adsblol").mkdir()
        names = {p.script for p in M.PRODUCTS}
        self.assertEqual(len(names), 6)
        for s in names:
            (self.root / s).write_text(STUB)
        ranking().write_parquet(self.root / M.FINAL_RANKING)
        self.rsha = M.sha(self.root / M.FINAL_RANKING)
        self.surf = self.root / "data/adsblol"; self.out = self.root / "data/out"

    def tearDown(self):
        self.tmp.cleanup()

    def surface(self, day, **ctl):
        (self.surf / f"surface_{day}.parquet").write_text(json.dumps(ctl))

    def run_days(self, days, resume=False):
        return M.run(self.root, self.out, self.surf, workers=2, threads=1, resume=resume, nice=0, ranking_sha=self.rsha, days=days)

    def test_recipe_env_is_exact_and_closed(self):
        by = {p.name: p for p in M.PRODUCTS}
        self.assertEqual(by["kt1_offblock2"].env, {"MOVING_KT": "1", "ADSB_OUT_DIR": "data/adsblol_kt1"})
        self.assertEqual(by["kt1_offblock5"].script, "src/adsb_offblock_v5.py")
        self.assertEqual(by["sw_kt1_offblock2"].env, {"MOVING_KT": "1", "STAND_WALK": "1", "ADSB_OUT_DIR": "data/adsblol_sw_kt1"})
        for n in ("visit", "visit_m", "head"):
            self.assertEqual(by[n].env, {})
        self.assertEqual([p.name for p in M.PRODUCTS], ["kt1_offblock2", "kt1_offblock3", "kt1_offblock5", "sw_kt1_offblock2", "visit", "visit_m", "head"])
        with patch.dict(os.environ, {"MOVING_KT": "7", "STAND_WALK": "2", "VISIT_PREV_DAY": "1", "ADSB_OUT_DIR": "x"}):
            e = M.producer_env(by["visit"], 3)
            self.assertNotIn("MOVING_KT", e); self.assertNotIn("STAND_WALK", e); self.assertNotIn("VISIT_PREV_DAY", e); self.assertNotIn("ADSB_OUT_DIR", e)
            self.assertEqual(e["POLARS_MAX_THREADS"], "3"); self.assertEqual(e["OMP_NUM_THREADS"], "3")
            self.assertEqual(M.producer_env(by["sw_kt1_offblock2"], 1)["STAND_WALK"], "1")

    def test_final_departures_rejects_wrong_sha_months_and_duplicates(self):
        with self.assertRaises(ValueError):
            M.final_departures(self.root / M.FINAL_RANKING, "0" * 64)
        dep = M.final_departures(self.root / M.FINAL_RANKING, self.rsha)
        self.assertEqual(dep.height, sum(NDEP.values())); self.assertEqual(M.days_of(dep), DAYS)
        bad = self.root / "data/prc-2026-datasets/bad.parquet"
        ranking().filter(pl.col("MVT_TIME_UTC_mvt").dt.month() != 2).write_parquet(bad)
        with self.assertRaises(ValueError):
            M.final_departures(bad, M.sha(bad))
        pl.concat([ranking(), ranking()]).write_parquet(bad)
        with self.assertRaises(ValueError):
            M.final_departures(bad, M.sha(bad))

    def test_product_contract_rows_ids_and_match_counts(self):
        dep = M.final_departures(self.root / M.FINAL_RANKING, self.rsha).filter(pl.col("day") == dt.date(2026, 1, 10))
        by = {p.name: p for p in M.PRODUCTS}
        f = self.root / "x.parquet"
        pl.DataFrame({"MVT_ID_mvt": [1000.0, 1001.0, 1002.0], "n_cand": [1, 0, 2], "coverage": [3, 0, 1]}).write_parquet(f)
        rec = M.check_product(by["kt1_offblock2"], f, dep)
        self.assertEqual((rec["rows"], rec["matched"], rec["coverage_ge2"], rec["coverage3"]), (3, 1, 1, 1))
        pl.DataFrame({"MVT_ID_mvt": [1000.0, 1001.0], "n_cand": [1, 1]}).write_parquet(f)   # a row short
        with self.assertRaises(ValueError):
            M.check_product(by["kt1_offblock2"], f, dep)
        pl.DataFrame({"MVT_ID_mvt": [1000.0, 1000.0, 1002.0], "sf_n_seg": [1.0, 1.0, None]}).write_parquet(f)   # duplicate id
        with self.assertRaises(ValueError):
            M.check_product(by["kt1_offblock3"], f, dep)
        pl.DataFrame({"MVT_ID_mvt": [1002.0], "hd_gs0": [4.0]}).write_parquet(f)   # head: matched rows only
        self.assertEqual(M.check_product(by["head"], f, dep)["matched"], 1)
        pl.DataFrame({"MVT_ID_mvt": [1002.0, 5555.0], "hd_gs0": [4.0, 4.0]}).write_parquet(f)   # head: a foreign id
        with self.assertRaises(ValueError):
            M.check_product(by["head"], f, dep)

    def test_run_receipts_resume_and_seal(self):
        self.surface("2026-01-10", matched=2); self.surface("2026-02-10", matched=0)   # 2026-06-10: no surface file
        s = self.run_days(DAYS[:3])
        self.assertEqual(s["status_counts"], {"processed": 1, "processed_no_matches": 1, "input_missing": 1})
        self.assertEqual(s["input_missing_days"], ["2026-06-10"]); self.assertEqual(s["departures_without_products"], 2)
        self.assertTrue((self.out / "incomplete.json").exists()); self.assertFalse((self.out / "complete.json").exists())
        r = json.loads((self.out / "receipts/2026-01-10.json").read_text())
        self.assertEqual(r["status"], "processed"); self.assertEqual(r["departures"], 3); self.assertEqual(r["matched_total"], 2 * 7)
        self.assertEqual(set(r["products"]), {p.name for p in M.PRODUCTS})
        self.assertEqual(r["products"]["head"]["rows"], 2); self.assertEqual(r["products"]["kt1_offblock2"]["coverage_ge2"], 2)
        self.assertEqual(r["products"]["kt1_offblock2"]["path"], "data/adsblol_kt1/offblock2_2026-01-10.parquet")
        self.assertEqual(r["surface"]["sha256"], M.sha(self.surf / "surface_2026-01-10.parquet"))
        r0 = json.loads((self.out / "receipts/2026-02-10.json").read_text())
        self.assertEqual(r0["status"], "processed_no_matches"); self.assertEqual(r0["products"]["visit"]["rows"], 2); self.assertEqual(r0["products"]["head"]["rows"], 0)
        rm = json.loads((self.out / "receipts/2026-06-10.json").read_text())
        self.assertEqual(rm["status"], "input_missing"); self.assertEqual(rm["products"], {})
        # the stubs saw exactly the recipe env
        env = lambda rel: json.loads((self.out / rel).read_text())
        self.assertEqual(env("data/adsblol_kt1/offblock2_2026-01-10.env.json"), {"MOVING_KT": "1", "STAND_WALK": None, "VISIT_PREV_DAY": None, "ADSB_OUT_DIR": "data/adsblol_kt1"})
        self.assertEqual(env("data/adsblol_sw_kt1/offblock2_2026-01-10.env.json"), {"MOVING_KT": "1", "STAND_WALK": "1", "VISIT_PREV_DAY": None, "ADSB_OUT_DIR": "data/adsblol_sw_kt1"})
        self.assertEqual(env("data/adsblol/visit_2026-01-10.env.json"), {"MOVING_KT": None, "STAND_WALK": None, "VISIT_PREV_DAY": None, "ADSB_OUT_DIR": None})
        self.assertEqual(env("data/adsblol/head_2026-01-10.env.json")["MOVING_KT"], None)
        # WORK layout: the producers read the final ranking under the old name, through a link
        self.assertEqual((self.out / "data/prc-2026-datasets/ranking.parquet").resolve(), (self.root / M.FINAL_RANKING).resolve())
        self.assertTrue((self.out / "src").is_symlink() and (self.out / "analysis/sources").is_symlink())
        man = (self.out / "MANIFEST.sha256").read_text().splitlines()
        self.assertEqual(len(man), 2 * 7 + 2 * 7 + 3)   # 7 parquet + 7 env.json per processed day, 3 receipts
        self.assertTrue(all(len(l.split("  ")[0]) == 64 for l in man))
        # a second run without --resume refuses; with --resume valid days are skipped and the missing one is completed
        with self.assertRaises(FileExistsError):
            self.run_days(DAYS[:3])
        self.surface("2026-06-10", matched=1)
        s2 = self.run_days(DAYS[:3], resume=True)
        self.assertEqual(s2["status_counts"], {"processed": 2, "processed_no_matches": 1, "input_missing": 0})
        self.assertTrue((self.out / "complete.json").exists())
        self.assertEqual(json.loads((self.out / "receipts/2026-01-10.json").read_text())["finished_utc"], r["finished_utc"])
        log = (self.out / "run.log").read_text()
        self.assertIn("2026-01-10: receipt valid, skipped", log); self.assertIn("2026-06-10: processed", log)
        with self.assertRaises(FileExistsError):   # sealed
            self.run_days(DAYS[:3], resume=True)

    def test_producer_failure_stops_without_receipt_or_summary(self):
        self.surface("2026-01-10", matched=1, fail="adsb_visit.py")
        with self.assertRaises(RuntimeError):
            self.run_days(DAYS[:1])
        self.assertFalse((self.out / "receipts/2026-01-10.json").exists())
        self.assertFalse((self.out / "incomplete.json").exists() or (self.out / "complete.json").exists())
        self.assertTrue((self.out / "data/adsblol_kt1/offblock2_2026-01-10.parquet").exists())   # earlier products of the day stay for inspection
        # rerun with resume: the partial outputs are this run's own leftovers and are removed first, then the day completes
        self.surface("2026-01-10", matched=1)
        s = self.run_days(DAYS[:1], resume=True)
        self.assertEqual(s["status_counts"]["processed"], 1)
        self.assertIn("removing partial data/adsblol_kt1/offblock2_2026-01-10.parquet", (self.out / "run.log").read_text())

    def test_row_mismatch_against_final_departures_is_rejected(self):
        self.surface("2026-01-10", matched=1, drop_rows=1)
        with self.assertRaises(ValueError):
            self.run_days(DAYS[:1])
        self.assertFalse((self.out / "receipts/2026-01-10.json").exists())

    def test_days_outside_final_ranking_and_wrong_ranking_are_refused(self):
        with self.assertRaises(ValueError):
            self.run_days(["2026-03-10"])
        with self.assertRaises(ValueError):
            M.run(self.root, self.out, self.surf, 1, 1, False, nice=0, ranking_sha="1" * 64, days=DAYS[:1])

    def test_full_run_requires_the_registered_day_and_departure_counts(self):
        self.surface("2026-01-10", matched=1)
        with self.assertRaises(ValueError):   # the synthetic ranking has 4 days, not 120
            M.run(self.root, self.out, self.surf, 1, 1, False, nice=0, ranking_sha=self.rsha, days=None)


if __name__ == "__main__":
    unittest.main()

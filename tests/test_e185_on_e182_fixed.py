"""Synthetic inputs only: fixed E185 addition, lineage and original gate."""
import importlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    E = importlib.import_module("e185_on_e182_fixed")
except ModuleNotFoundError:
    E = None

ACCEPTED = "cv_preds_e172_rawf_lobtf_schedf_eobtf_aobtf_mlpa_f1b"
MEMBERS = ("rawf", "lobtf", "schedf", "eobtf", "aobtf", "mlpa")
WEIGHTS = ((.05, .15, .15, .10, .15, .30), (.10, .20, .10, .15, .15, .20),
           *((0., .15, .15, .15, .15, .30),) * 4)


def fixture(root):
    rows, log = [], []
    for a, weights in enumerate(WEIGHTS, 1):
        name = f"fold_{a}_{a+6}.npz"
        ids = np.arange(a*100, a*100+8)
        months = np.array([a]*4 + [a+6]*4)
        values = np.tile([-99., np.nan, 0., 0., 0., -99.], (8, 1))
        z = dict(mvt_id=ids, month=months, y=np.full(8, 1000.),
                 reg=np.full(8, 100.), pred=np.full(8, 107.), values=values,
                 proba=np.tile([1., 0., 0., 0., 0., 0.], (8, 1)))
        for directory, reg, pred in (("cv_preds_fin1b_hz", 100., 107.), (ACCEPTED, 910., 917.)):
            folder = root / "data" / directory
            folder.mkdir(parents=True, exist_ok=True)
            np.savez(folder / name, **{**z, "reg": np.full(8, reg), "pred": np.full(8, pred)})
        for member in (*MEMBERS, "motion"):
            folder = root / "data" / f"cv_e130{member}"
            folder.mkdir(exist_ok=True)
            np.savez(folder / name, mvt_id=ids[::-1], reg2=np.full(8, 1100. if member == "motion" else 1000.))
        log.append(f"{ACCEPTED} {name}: weights {weights}")
        for j, ident in enumerate(ids):
            rows.append(dict(MVT_ID_mvt=int(ident), month=int(months[j]), split="train",
                             TAXITIME_SEC_mvt=1000., ADEP_mvt="LIRF" if j == 2 else "EDDF",
                             FLIGHT_ID_mvt=1., LOBT_flt=1. if j == 3 else None,
                             d_mvt_sched=0., d_mvt_lobt=0. if j == 3 else None))
    frame = pl.DataFrame(rows, schema_overrides={"LOBT_flt": pl.Float64, "d_mvt_lobt": pl.Float64})
    for name in ("features_fin1b.parquet", "features_e106.parquet"):
        frame.write_parquet(root / "data" / name)
    (root / "data/e182_grid_5f.log").write_text("\n".join(log) + "\n")
    (root / "src").mkdir()
    for name in ("e172_members_cv.py", "e185_subset_grid_cv.py"):
        (root / "src" / name).write_text("# synthetic provenance fixture\n")


class FixedAdditivityTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(E, "fixed-weight additivity runner is not implemented")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fixture(self.root)
        self.out = self.root / "run"

    def run_fixture(self):
        return E.run(self.root, self.out, registration="a" * 40)

    def replace(self, directory, fold="fold_1_7.npz", **changes):
        path = self.root / "data" / directory / fold
        with np.load(path, allow_pickle=False) as archive:
            z = dict(archive)
        z.update(changes)
        np.savez(path, **z)

    def test_fixed_delta_preserved_mask_and_stale_values_rebuilt(self):
        result = self.run_fixture()
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["rows"], 48)
        with np.load(self.out / "candidate/fold_1_7.npz") as z:
            np.testing.assert_allclose(z["reg"], [976.5, 976.5, 910., 910., 976.5, 976.5, 976.5, 976.5], rtol=0, atol=1e-10)
            np.testing.assert_allclose(z["pred"], [983.5, 983.5, 917., 917., 983.5, 983.5, 983.5, 983.5], rtol=0, atol=1e-10)
            self.assertEqual(z["values"].dtype, np.float64)
            np.testing.assert_array_equal(z["values"][:, 0], z["reg"])
            np.testing.assert_array_equal(z["values"][:, 5], z["reg"] + 86400.)
            self.assertTrue(np.isnan(z["values"][:, 1]).all())
        with np.load(self.out / "control/fold_1_7.npz") as z:
            np.testing.assert_array_equal(z["pred"], np.full(8, 917.))
            np.testing.assert_array_equal(z["reg"], np.full(8, 910.))
        with np.load(self.root / "data" / ACCEPTED / "fold_1_7.npz") as original:
            np.testing.assert_array_equal(original["values"][:, 0], np.full(8, -99.))
        manifest = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(manifest["weight"], .35)
        self.assertEqual(len(manifest["output_sha256"]), 13)
        self.assertIn("data/features_e106.parquet", manifest["input_sha256"])

    def test_day_probability_and_lobt_clipping_preserve_existing_delta(self):
        z = dict(reg=np.array([100., 10000., 100.]), pred=np.array([1000., 3613., 107.]),
                 values=np.zeros((3, 6)),
                 proba=np.array([[.5, 0., 0., 0., 0., .5], [1., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0.]]))
        out = E.build_candidate(z, np.array([300., 1000., 300.]), np.array([True, True, False]), np.array([np.nan, 0., np.nan]))
        np.testing.assert_allclose(out["pred"], [1070., 3613., 107.], rtol=0, atol=1e-10)
        np.testing.assert_array_equal(out["reg"], [170., 6850., 100.])

    def test_original_gate_does_not_add_aggregate_decided_requirement(self):
        report = {"metrics": {"lower_decided": 4}, "sse_delta": {"decided": 99., "trimmed": -1., "body": -2.}}
        self.assertTrue(E.original_gate(report)["passed"])
        for field in ("trimmed", "body"):
            changed = {**report, "sse_delta": {**report["sse_delta"], field: 0.}}
            self.assertFalse(E.original_gate(changed)["passed"])
        self.assertFalse(E.original_gate({**report, "metrics": {"lower_decided": 3}})["passed"])

    def test_missing_sixth_member_fold_fails_without_published_output(self):
        (self.root / "data/cv_e130motion/fold_6_12.npz").unlink()
        with self.assertRaisesRegex(E.InvalidInput, "six folds"):
            self.run_fixture()
        self.assertFalse(self.out.exists())

    def test_duplicate_member_id_is_not_silently_overwritten(self):
        self.replace("cv_e130motion", mvt_id=np.array([100, 100, 102, 103, 104, 105, 106, 107]))
        with self.assertRaisesRegex(E.InvalidInput, "duplicate"):
            self.run_fixture()
        self.assertFalse(self.out.exists())

    def test_wrong_member_id_set_and_nonfinite_prediction_reject(self):
        self.replace("cv_e130motion", mvt_id=np.arange(101, 109))
        with self.assertRaisesRegex(E.InvalidInput, "ID set"):
            self.run_fixture()
        self.replace("cv_e130motion", mvt_id=np.arange(100, 108), reg2=np.full(8, np.inf))
        with self.assertRaisesRegex(E.InvalidInput, "finite"):
            self.run_fixture()

    def test_last_fold_target_mismatch_rejects_before_publication(self):
        for directory in (ACCEPTED, "cv_preds_fin1b_hz"):
            self.replace(directory, "fold_6_12.npz", y=np.full(8, 999.))
        with mock.patch.object(E.G, "evaluate", side_effect=AssertionError("metrics called before validation")):
            with self.assertRaisesRegex(E.InvalidInput, "target"):
                self.run_fixture()
        self.assertFalse(self.out.exists())

    def test_wrong_month_rejects(self):
        self.replace(ACCEPTED, month=np.full(8, 2))
        with self.assertRaisesRegex(E.InvalidInput, "month"):
            self.run_fixture()

    def test_saved_e182_reg_or_pred_must_rebuild_from_frozen_members(self):
        for field in ("reg", "pred"):
            with self.subTest(field=field):
                self.replace(ACCEPTED, reg=np.full(8, 910.), pred=np.full(8, 917.))
                self.replace(ACCEPTED, **{field: np.full(8, 999.)})
                with self.assertRaisesRegex(E.InvalidInput, "faithful"):
                    self.run_fixture()
                self.assertFalse(self.out.exists())

    def test_probabilities_and_nonregular_values_must_match_base(self):
        self.replace(ACCEPTED, proba=np.tile([0., 1., 0., 0., 0., 0.], (8, 1)))
        with self.assertRaisesRegex(E.InvalidInput, "proba"):
            self.run_fixture()

    def test_changed_original_feature_mask_rejects(self):
        path = self.root / "data/features_e106.parquet"
        pl.read_parquet(path).with_columns(pl.lit("LIRF").alias("ADEP_mvt")).write_parquet(path)
        with self.assertRaisesRegex(E.InvalidInput, "feature.*parity"):
            self.run_fixture()

    def test_changed_recorded_weights_rejects(self):
        path = self.root / "data/e182_grid_5f.log"
        path.write_text(path.read_text().replace("(0.05,", "(0.0,", 1))
        with self.assertRaisesRegex(E.InvalidInput, "weights"):
            self.run_fixture()

    def test_existing_output_refused_and_preserved(self):
        self.out.mkdir()
        marker = self.out / "keep"
        marker.write_text("old")
        with self.assertRaisesRegex(E.InvalidInput, "exists"):
            self.run_fixture()
        self.assertEqual(marker.read_text(), "old")

    def test_publish_failure_leaves_no_partial_final_output(self):
        with mock.patch.object(Path, "rename", side_effect=OSError("synthetic disk publication failure")):
            with self.assertRaisesRegex(OSError, "publication failure"):
                self.run_fixture()
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.root.glob(".run.partial-*")), [])

    def test_cli_rejected_run_is_distinct_from_invalid_input(self):
        # Exact motion==accepted reg means no improvement; fixed gate must reject.
        for a in range(1, 7):
            self.replace("cv_e130motion", f"fold_{a}_{a+6}.npz", reg2=np.full(8, 910.))
        args = [sys.executable, str(Path(E.__file__)), "--repo-root", str(self.root),
                "--output", str(self.out), "--registration", "a"*40]
        child = subprocess.run(args, capture_output=True, text=True, timeout=30)
        self.assertEqual(child.returncode, 1, child.stderr)
        self.assertEqual(json.loads(child.stdout)["status"], "rejected")
        again = subprocess.run(args, capture_output=True, text=True, timeout=30)
        self.assertEqual(again.returncode, 2, again.stderr)
        self.assertIn("already exists", again.stderr)
        self.assertNotIn("Traceback", again.stderr)

    def test_import_and_help_do_not_read_data_or_compute_metrics(self):
        child = subprocess.run([sys.executable, str(Path(E.__file__)), "--help"], cwd=self.root,
                               capture_output=True, text=True, timeout=30)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertIn("--registration", child.stdout)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()

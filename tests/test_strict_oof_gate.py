"""Synthetic, read-only input checks for the standalone OOF acceptance gate."""
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
import zipfile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[name] = "2"
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    gate = importlib.import_module("strict_oof_gate")
except ModuleNotFoundError:
    gate = None


class StrictOOFGateTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(gate, "standalone strict OOF gate is not implemented")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.control, self.candidate = self.root / "control", self.root / "candidate"
        self.control.mkdir()
        self.candidate.mkdir()
        self.features = self.root / "features.parquet"
        rows = []
        for a in range(1, 7):
            ids = np.arange(a * 100, a * 100 + 8)
            months = np.array([a] * 4 + [a + 6] * 4)
            # 0: untouched; 1: strict-only exclusion; 2,3: tail exclusion.
            # 3: large body target. Others are untouched, including null D.
            y = np.array([100, 100, 100, 8000, 100, 100, 100, 100.], dtype=float)
            pred = y + 10
            z = dict(mvt_id=ids, month=months, y=y, pred=pred)
            name = f"fold_{a}_{a + 6}.npz"
            np.savez(self.control / name, **z)
            np.savez(self.candidate / name, **{**z, "pred": y + 9})
            for j, ident in enumerate(ids):
                rows.append(dict(MVT_ID_mvt=int(ident), split="train", month=int(months[j]),
                                 TAXITIME_SEC_mvt=float(y[j]), ADEP_mvt="LIRF" if j in (1, 2, 3, 7) else "EGLL",
                                 FLIGHT_ID_mvt=None, d_mvt_sched=[0., 1., 14400., 20000., 0., 0., 0., None][j]))
        pl.DataFrame(rows).write_parquet(self.features)

    def replace(self, directory, name="fold_1_7.npz", **updates):
        path = directory / name
        with np.load(path) as z:
            data = dict(z)
        data.update(updates)
        np.savez(path, **data)

    def evaluate(self, **kwargs):
        return gate.evaluate(self.control, self.candidate, self.features, **kwargs)

    def invalid(self, pattern):
        with self.assertRaisesRegex(gate.InvalidOOF, pattern):
            self.evaluate()

    def test_valid_improvement_and_hand_calculated_legacy_metrics(self):
        evidence = self.evaluate(require_body=True)
        self.assertEqual(evidence["status"], "accepted")
        self.assertEqual(evidence["rows"], 48)
        self.assertEqual(evidence["metrics"]["lower_decided"], 6)
        # Per row delta = 9^2 - 10^2 = -19. Per fold: 6 decided,
        # 5 strict, 8 all, 7 y<=7200; remove three extrema at each end.
        denominator = 6 * 2 * 252.43 * 344841
        for key, sse in dict(decided=-684., strict=-570., all=-912., body=-798., trimmed=-228.).items():
            self.assertAlmostEqual(evidence["metrics"][key], sse / denominator, places=15)
            self.assertEqual(evidence["sse_delta"][key], sse)

    def test_shuffled_candidate_rows_are_aligned_by_id(self):
        for path in self.candidate.glob("*.npz"):
            with np.load(path) as z:
                data = {k: v[::-1] for k, v in dict(z).items()}
            np.savez(path, **data)
        self.assertEqual(self.evaluate()["sse_delta"]["all"], -912.)

    def test_identical_inputs_pass_only_audit_mode(self):
        rejected = gate.evaluate(self.control, self.control, self.features)
        self.assertEqual(rejected["status"], "rejected")
        audit = gate.evaluate(self.control, self.control, self.features, audit_only=True)
        self.assertEqual(audit["status"], "audit_valid")
        self.assertFalse(audit["gate_passed"])

    def test_missing_sixth_fold_rejects(self):
        (self.candidate / "fold_6_12.npz").unlink()
        self.invalid("fold")

    def test_extra_fold_rejects(self):
        np.savez(self.candidate / "fold_0_6.npz", mvt_id=[999])
        self.invalid("fold")

    def test_duplicate_id_within_fold_rejects(self):
        self.replace(self.candidate, mvt_id=np.array([100, 100, 102, 103, 104, 105, 106, 107]))
        self.invalid("duplicate")

    def test_duplicate_id_across_folds_rejects(self):
        for directory in (self.control, self.candidate):
            self.replace(directory, "fold_2_8.npz", mvt_id=np.array([100, 201, 202, 203, 204, 205, 206, 207]))
        self.invalid("across folds")

    def test_legacy_float_ids_are_accepted_when_exactly_integral(self):
        self.replace(self.candidate, mvt_id=np.arange(100, 108, dtype=float))
        self.assertEqual(self.evaluate()["status"], "accepted")

    def test_fractional_and_unsafe_float_ids_reject(self):
        for value in (100.5, float(2**53 + 2), np.nan):
            self.replace(self.candidate, mvt_id=np.array([value, 101, 102, 103, 104, 105, 106, 107]))
            self.invalid("integer")

    def test_candidate_id_set_must_be_exact(self):
        self.replace(self.candidate, mvt_id=np.array([999, 101, 102, 103, 104, 105, 106, 107]))
        self.invalid("ID set")

    def test_incorrect_candidate_target_rejects(self):
        self.replace(self.candidate, y=np.array([101, 100, 100, 8000, 100, 100, 100, 100.]))
        self.invalid("target")

    def test_both_saved_targets_wrong_against_features_reject(self):
        for directory in (self.control, self.candidate):
            self.replace(directory, y=np.array([101, 100, 100, 8000, 100, 100, 100, 100.]))
        self.invalid("target")

    def test_missing_authoritative_feature_target_rejects(self):
        pl.read_parquet(self.features).drop("TAXITIME_SEC_mvt").write_parquet(self.features)
        self.invalid("TAXITIME_SEC_mvt")

    def test_nonfinite_target_and_prediction_reject(self):
        for field in ("y", "pred"):
            for bad in (np.nan, np.inf):
                with self.subTest(field=field, bad=bad):
                    self.replace(self.candidate, **{field: np.array([bad] + [100.] * 7)})
                    self.invalid("finite")

    def test_shapes_and_missing_fields_reject(self):
        self.replace(self.candidate, pred=np.zeros((8, 1)))
        self.invalid("shape")
        np.savez(self.candidate / "fold_1_7.npz", mvt_id=np.arange(100, 108), y=np.ones(8))
        self.invalid("missing")

    def test_malformed_npz_rejects(self):
        (self.candidate / "fold_1_7.npz").write_bytes(b"not an npz")
        self.invalid("read")

    def test_object_array_rejects_without_pickle(self):
        self.replace(self.candidate, pred=np.array(["unsafe"] * 8, dtype=object))
        self.invalid("read")

    def test_duplicate_npz_array_names_reject(self):
        path = self.candidate / "fold_1_7.npz"
        with zipfile.ZipFile(path, "a") as archive:
            old = archive.read("pred.npy")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr("pred.npy", old)
        self.invalid("duplicate.*array")

    def test_wrong_stored_month_rejects(self):
        self.replace(self.candidate, month=np.array([2] * 8))
        self.invalid("month")

    def test_wrong_feature_month_rejects_even_without_saved_month(self):
        frame = pl.read_parquet(self.features).with_columns(
            pl.when(pl.col("MVT_ID_mvt") == 100).then(2).otherwise(pl.col("month")).alias("month"))
        frame.write_parquet(self.features)
        for directory in (self.control, self.candidate):
            with np.load(directory / "fold_1_7.npz") as z:
                data = {k: v for k, v in dict(z).items() if k != "month"}
            np.savez(directory / "fold_1_7.npz", **data)
        self.invalid("month")

    def test_incomplete_training_coverage_rejects(self):
        frame = pl.read_parquet(self.features)
        extra = frame.head(1).with_columns(pl.lit(9999, dtype=pl.Int64).alias("MVT_ID_mvt"))
        pl.concat([frame, extra]).write_parquet(self.features)
        self.invalid("coverage")

    def test_feature_training_ids_are_unique(self):
        frame = pl.read_parquet(self.features)
        pl.concat([frame, frame.head(1)]).write_parquet(self.features)
        self.invalid("duplicate")

    def test_rank_features_do_not_count_toward_training_coverage(self):
        frame = pl.read_parquet(self.features)
        rank = frame.head(1).with_columns(pl.lit(9999, dtype=pl.Int64).alias("MVT_ID_mvt"), pl.lit("rank").alias("split"))
        pl.concat([frame, rank]).write_parquet(self.features)
        self.assertEqual(self.evaluate()["status"], "accepted")

    def test_valid_worse_candidate_is_rejected(self):
        for path in self.candidate.glob("*.npz"):
            with np.load(path) as z:
                data = dict(z)
            data["pred"] = data["y"] + 11
            np.savez(path, **data)
        evidence = self.evaluate()
        self.assertEqual(evidence["status"], "rejected")
        self.assertFalse(evidence["gate_passed"])

    def test_four_of_six_requirement_blocks_three_improving_folds(self):
        for a in range(4, 7):
            self.replace(self.candidate, f"fold_{a}_{a + 6}.npz",
                         pred=np.array([100, 100, 100, 8000, 100, 100, 100, 100.]) + 10.1)
        evidence = self.evaluate()
        self.assertLess(evidence["metrics"]["decided"], 0)
        self.assertEqual(evidence["metrics"]["lower_decided"], 3)
        self.assertEqual(evidence["status"], "rejected")

    def test_require_body_can_reject_a_tail_driven_gain(self):
        for path in self.candidate.glob("*.npz"):
            with np.load(path) as z:
                data = dict(z)
            data["y"] = np.array([8000.] * 7 + [100.])
            data["pred"] = data["y"] + np.array([9.] * 7 + [11.])
            np.savez(path, **data)
            np.savez(self.control / path.name, **{**data, "pred": data["y"] + 10})
        pl.read_parquet(self.features).with_columns(
            pl.when(pl.col("MVT_ID_mvt") % 100 == 7).then(100.).otherwise(8000.).alias("TAXITIME_SEC_mvt")
        ).write_parquet(self.features)
        self.assertEqual(self.evaluate()["status"], "accepted")
        self.assertEqual(self.evaluate(require_body=True)["status"], "rejected")

    def test_stale_values_are_reported_not_rejected_and_nan_plans_allowed(self):
        values = np.zeros((8, 6))
        values[:, 1] = np.nan
        self.replace(self.candidate, reg=np.ones(8), values=values, proba=np.ones((8, 6)) / 6)
        evidence = self.evaluate(check_values=True)
        self.assertEqual(evidence["status"], "accepted")
        self.assertFalse(evidence["values_consistency"]["candidate"]["fold_1_7.npz"]["consistent"])

    def test_invalid_optional_array_shapes_and_nonfinite_reject(self):
        self.replace(self.candidate, reg=np.zeros((8, 1)))
        self.invalid("shape")
        self.replace(self.candidate, reg=np.zeros(8), proba=np.full((8, 6), np.inf))
        self.invalid("finite")

    def test_cli_has_distinct_statuses_and_machine_readable_evidence(self):
        base = [str(self.control), str(self.candidate), "--features", str(self.features), "--json", "-"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = gate.main(base)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())["status"], "accepted")
        identical = [str(self.control), str(self.control), "--features", str(self.features), "--json", "-"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = gate.main(identical)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out.getvalue())["status"], "rejected")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = gate.main(identical + ["--audit-only"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())["status"], "audit_valid")
        (self.candidate / "fold_6_12.npz").unlink()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = gate.main(base)
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out.getvalue())["status"], "invalid")

    def test_json_destination_cannot_overwrite_an_input_or_existing_report(self):
        destination = self.candidate / "fold_1_7.npz"
        before = destination.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            code = gate.main([str(self.control), str(self.candidate), "--features", str(self.features),
                              "--json", str(destination)])
        self.assertEqual(code, 2)
        self.assertEqual(destination.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

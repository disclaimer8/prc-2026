"""Synthetic contract tests: no real features or model training."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import unittest
import tempfile
from unittest import mock


def load_runner():
    path = Path(__file__).parents[1] / "scripts/cat_fin1b_oof.py"
    assert path.exists(), "isolated Cat OOF wrapper is missing"
    spec = importlib.util.spec_from_file_location("cat_fin1b_oof", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample():
    return pl.DataFrame({"MVT_ID_mvt": [90, 11, 27, 12, 29],
                         "split": ["rank", "train", "train", "train", "train"],
                         "month": [1, 1, 7, 2, 9],
                         "TAXITIME_SEC_mvt": [None, 100., 200., 300., 400.],
                         "feature": [9., 1., 2., 3., 4.]})


def test_fold_has_no_2026_rows_no_heldout_labels_and_preserves_order(runner):
    view, meta = runner.fold_view(sample(), 1)
    assert view["MVT_ID_mvt"].to_list() == [11, 27, 12, 29]
    assert view["split"].to_list() == ["rank", "rank", "train", "train"]
    assert view["TAXITIME_SEC_mvt"].to_list() == [None, None, 300., 400.]
    assert view["feature"].to_list() == [1., 2., 3., 4.]
    assert meta["mvt_id"].tolist() == [11, 27]
    assert meta["y"].tolist() == [100., 200.]
    assert meta["month"].tolist() == [1, 7]
    assert sample()["TAXITIME_SEC_mvt"].to_list()[1:3] == [100., 200.]


def test_official_float64_integral_ids_are_preserved(runner, tmp_path):
    view, meta = runner.fold_view(sample().with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64)), 1)
    assert view.schema["MVT_ID_mvt"] == pl.Float64
    np.savez(tmp_path / "raw.npz", mvt_id=np.array([11., 27.]), reg=[100., 200.])
    runner.publish_seed(tmp_path / "raw.npz", tmp_path / "seed", meta["mvt_id"], {"seed": 0})
    assert runner.verify_seed(tmp_path / "seed", meta["mvt_id"], {"seed": 0}).tolist() == [100., 200.]


def test_nonintegral_ids_are_rejected(runner):
    frame = sample().with_columns((pl.col("MVT_ID_mvt") + .5).alias("MVT_ID_mvt"))
    with unittest.TestCase().assertRaises(runner.Invalid):
        runner.fold_view(frame, 1)


def test_fold_rejects_broken_training_population(runner, kind):
    df = sample()
    if kind == "duplicate":
        df = df.with_columns(pl.when(pl.col("MVT_ID_mvt") == 27).then(11).otherwise(pl.col("MVT_ID_mvt")).alias("MVT_ID_mvt"))
    elif kind == "null_y":
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("TAXITIME_SEC_mvt"))
    elif kind == "bad_month":
        df = df.with_columns(pl.lit(13).alias("month"))
    else:
        df = df.filter(~pl.col("month").is_in([1, 7]))
    with unittest.TestCase().assertRaises(runner.Invalid):
        runner.fold_view(df, 1)


def test_schedule_covers_every_fold_before_next_seed(runner):
    schedule = runner.schedule()
    assert schedule[:7] == [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 1)]
    assert schedule[-1] == (14, 6)
    assert len(schedule) == len(set(schedule)) == 90


def test_seed_publication_rejects_wrong_ids_or_nonfinite(runner, tmp_path, ids, reg):
    raw = tmp_path / "raw.npz"
    np.savez(raw, mvt_id=ids, reg=reg)
    with unittest.TestCase().assertRaises(runner.Invalid):
        runner.publish_seed(raw, tmp_path / "seed", np.array([11, 27]), {"seed": 0, "lineage": "abc"})
    assert not (tmp_path / "seed").exists()
    assert raw.exists()


def test_seed_publish_resume_and_tampering(runner, tmp_path):
    raw, final = tmp_path / "raw.npz", tmp_path / "seed"
    np.savez(raw, mvt_id=[11, 27], reg=[101., 201.])
    evidence = {"seed": 0, "lineage": "abc"}
    runner.publish_seed(raw, final, np.array([11, 27]), evidence)
    assert runner.verify_seed(final, np.array([11, 27]), evidence).tolist() == [101., 201.]
    with unittest.TestCase().assertRaises(runner.Invalid):
        runner.publish_seed(raw, final, np.array([11, 27]), evidence)
    with unittest.TestCase().assertRaises(runner.Invalid):
        runner.verify_seed(final, np.array([11, 27]), {"seed": 1, "lineage": "abc"})
    with (final / "prediction.npz").open("ab") as f:
        f.write(b"changed")
    with unittest.TestCase().assertRaises(runner.Invalid):
        runner.verify_seed(final, np.array([11, 27]), evidence)


def test_single_writer_lock_is_exclusive(runner, tmp_path):
    with runner.writer_lock(tmp_path):
        with unittest.TestCase().assertRaises(runner.Invalid):
            with runner.writer_lock(tmp_path):
                pass


def test_incomplete_attempt_preserved_in_quarantine(runner, tmp_path):
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    (attempts / "unfinished").mkdir()
    (attempts / "unfinished" / "log").write_text("diagnosis")
    runner.quarantine_attempts(tmp_path)
    assert not list(attempts.iterdir())
    files = list((tmp_path / "quarantine").glob("*/log"))
    assert len(files) == 1 and files[0].read_text() == "diagnosis"


def test_complete_mean_requires_all_fifteen_and_keeps_labels(runner, tmp_path):
    ids = np.array([11, 27])
    metadata = {"mvt_id": ids, "y": np.array([100., 200.]), "month": np.array([1, 7])}
    for seed in range(14):
        raw = tmp_path / f"raw{seed}.npz"
        np.savez(raw, mvt_id=ids, reg=[seed, seed + 100.])
        runner.publish_seed(raw, tmp_path / f"seed_{seed:02}", ids, {"seed": seed, "lineage": "abc", "fold": 1})
    with unittest.TestCase().assertRaises(runner.Invalid):
        runner.merge_fold(tmp_path, metadata, "abc", 1)
    np.savez(tmp_path / "raw14.npz", mvt_id=ids, reg=[14., 114.])
    runner.publish_seed(tmp_path / "raw14.npz", tmp_path / "seed_14", ids, {"seed": 14, "lineage": "abc", "fold": 1})
    runner.merge_fold(tmp_path, metadata, "abc", 1)
    with np.load(tmp_path / "complete/prediction.npz") as z:
        assert z["reg2"].tolist() == [7., 107.]
        assert z["per_seed"].shape == (15, 2)
        assert z["y"].tolist() == [100., 200.]
        assert z["month"].tolist() == [1, 7]
    before = (tmp_path / "complete/prediction.npz").read_bytes()
    runner.merge_fold(tmp_path, metadata, "abc", 1)
    assert before == (tmp_path / "complete/prediction.npz").read_bytes()


def test_child_environment_does_not_inherit_model_overrides(runner):
    with mock.patch.dict("os.environ", {"REG_OVERRIDE": '{"num_leaves":1}', "PROXY_AFFINE": "1", "PYTHONPATH": "/foreign/module"}):
        env = runner.child_environment(Path("/fold"), Path("/attempt"))
    assert not {"REG_OVERRIDE", "PROXY_AFFINE", "PYTHONPATH"}.intersection(env)
    assert env["CATBOOST_THREADS"] == "24"
    assert env["CAT_SEED_OUT"] == "/attempt/raw_seed{}.npz"


def test_atomic_json_refuses_overwrite(runner, tmp_path):
    target = tmp_path / "manifest.json"
    runner.write_json(target, {"a": 1})
    with unittest.TestCase().assertRaises(FileExistsError):
        runner.write_json(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 1}


class ContractTests(unittest.TestCase):
    pass


def bind(function, **parameters):
    def method(self):
        import inspect
        with tempfile.TemporaryDirectory() as directory:
            args = {"runner": load_runner(), "tmp_path": Path(directory), **parameters}
            function(**{name: args[name] for name in inspect.signature(function).parameters})
    return method


for _name, _function in list(globals().items()):
    if _name.startswith("test_"):
        if _name == "test_fold_rejects_broken_training_population":
            for _kind in ("duplicate", "null_y", "bad_month", "empty_holdout"):
                setattr(ContractTests, _name + "_" + _kind, bind(_function, kind=_kind))
        elif _name == "test_seed_publication_rejects_wrong_ids_or_nonfinite":
            for _i, (_ids, _reg) in enumerate((([27, 11], [1., 2.]), ([11, 11], [1., 2.]), ([11, 27], [1., np.nan]), ([11], [1.]))):
                setattr(ContractTests, _name + str(_i), bind(_function, ids=_ids, reg=_reg))
        else:
            setattr(ContractTests, _name, bind(_function))

if __name__ == "__main__":
    unittest.main()

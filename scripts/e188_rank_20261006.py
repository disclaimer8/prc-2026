"""Fixed E188 full-2025 fit and natural-ranking E64 replacement, 2026-10-06.

No CV, uploads, historical stage-module imports, or ranking augmentation.
Apply replaces only natural LIRF/NM rows in an otherwise complete E64 output.
The caller subsequently runs the existing E152/E81/LOBT/protect/LSZH tail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

for key in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "4"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["GOMP_SPINCOUNT"] = "0"
os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"] = "load_as_storage"

import numpy as np
import polars as pl

ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
ROWS = 344841
PROTOCOL = "E188-ranking-natural-LIRF-NM-nomonth-D3-keepall-five-seeds-20261006"
SCREEN_COMMIT = "da89a73846aa675d591ba5121bd2391241652a80"
SCREEN_MANIFEST_SHA = "57c1848c6d7634e87e041cd75bc67e3251e8274d3fa827eef7e72ac7203f40da"
SEEDS = [42, 1, 2, 3, 4]
ACTIVE = ["vi_first_seen", "vi_turnaround", "vi_gap", "vi_gs_first", "vi_has_prev"]
FEATS = ["ap", "coverage", "D", "p_nonreg", "base", "has_nm"] + ACTIVE
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
PARAMS = dict(objective="regression", learning_rate=.03, num_leaves=31,
              min_data_in_leaf=200, feature_fraction=.9, bagging_fraction=.8,
              bagging_freq=1, lambda_l2=10., verbose=-1, deterministic=True,
              force_row_wise=True, num_threads=4)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def validate_registration(root, commit):
    if not re.fullmatch("[a-f0-9]{40}", commit):
        raise RuntimeError("full registration commit required")
    source = Path(__file__).resolve()
    rel = str(source.relative_to(root))
    frozen = subprocess.check_output(["git", "show", f"{commit}:{rel}"], cwd=root)
    if hashlib.sha256(frozen).hexdigest() != sha(source):
        raise RuntimeError("running source differs from registration commit")
    return {rel: sha(source)}


def load_lgb():
    # macOS libomp ordering is required; never import an executable E64 module.
    import torch
    torch.set_num_threads(4)
    import lightgbm as lgb
    return lgb


def synthetic_mask(d):
    changed = np.zeros(d.height, dtype=bool)
    for name in ACTIVE:
        old, new = [d[p + name].cast(pl.Float64).to_numpy() for p in ("base_", "air_")]
        changed |= ~((old == new) | (np.isnan(old) & np.isnan(new)))
    observable = d.select(((pl.col("ADEP_mvt") == "LIRF") & pl.col("has_nm") &
        (pl.col("coverage") >= 2) & (pl.col("air_vi_has_prev") == 1) &
        pl.col("base_icao").is_not_null() & (pl.col("base_icao") == pl.col("air_icao"))).fill_null(False)).to_series().to_numpy()
    return observable & changed


def fit(root, bundle, extracted, output, registration):
    sources = validate_registration(root, registration)
    if output.exists():
        raise RuntimeError("fresh fit output required; no automatic resume")
    manifest_path, extraction_path = bundle / "manifest.json", extracted / "extraction.json"
    manifest = json.loads(manifest_path.read_text())
    extraction = json.loads(extraction_path.read_text())
    if (sha(manifest_path) != SCREEN_MANIFEST_SHA or manifest["registration_commit"] != SCREEN_COMMIT
        or extraction["registration_commit"] != SCREEN_COMMIT or extraction["manifest_sha256"] != SCREEN_MANIFEST_SHA
        or manifest["features"] != FEATS or manifest["seeds"] != SEEDS or manifest["synthetic_weight"] != .25
        or len(extraction["days"]) != 365 or len(extraction["output_sha256"]) != 365):
        raise RuntimeError("registered screen provenance differs")
    training_path = bundle / "training.parquet"
    expected_train = manifest["bundle_sha256"][str(training_path.relative_to(root))]
    if sha(training_path) != expected_train:
        raise RuntimeError("training projection SHA differs")
    hashes = {str(manifest_path.relative_to(root)): sha(manifest_path),
              str(extraction_path.relative_to(root)): sha(extraction_path),
              str(training_path.relative_to(root)): expected_train}
    frames = []
    for entry in extraction["days"]:
        name = f"days/{entry['day']}.parquet"
        p = extracted / name
        if sha(p) != extraction["output_sha256"][name]:
            raise RuntimeError(f"extracted data changed: {name}")
        hashes[str(p.relative_to(root))] = extraction["output_sha256"][name]
        frames.append(pl.read_parquet(p))
    audit = pl.concat([f.select(frames[0].columns) for f in frames])
    raw = pl.read_parquet(training_path)
    if raw.height != 2085047 or raw[ID].n_unique() != raw.height or audit[ID].n_unique() != audit.height:
        raise RuntimeError("training/extraction population differs")
    d = raw.join(audit.drop("month", "ADEP_mvt", "has_nm"), on=ID, validate="1:1", maintain_order="left")
    if d.height != raw.height or not np.array_equal(d[ID].to_numpy(), raw[ID].to_numpy()):
        raise RuntimeError("training join lost/reordered IDs")
    d = d.with_columns(*[pl.col("base_" + c).alias(c) for c in ACTIVE], pl.col("coverage").fill_null(-1))
    natural = d.filter(((pl.col("coverage") < 2) & (pl.col("vi_has_prev") == 1)).fill_null(False))
    extra = d.filter(pl.Series(synthetic_mask(d))).with_columns(
        pl.lit(1).cast(pl.Int64).alias("coverage"), *[pl.col("air_" + c).alias(c) for c in ACTIVE])
    if extra.height != extraction["synthetic_total"] or extra.height != 60327:
        raise RuntimeError("synthetic eligibility differs from passed screen")
    x = np.concatenate([natural.select(FEATS).to_numpy(), extra.select(FEATS).to_numpy()])
    y = np.concatenate([np.clip((f[Y] - f["base"]).to_numpy(), -1800, 1800) for f in (natural, extra)])
    weight = np.r_[np.ones(natural.height), np.full(extra.height, .25)]
    if not np.isfinite(y).all() or np.isinf(x).any():
        raise RuntimeError("invalid training values")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "fit_started.json", dict(protocol=PROTOCOL, registration_commit=registration,
         source_sha256=sources, input_sha256=hashes, natural_rows=natural.height,
         synthetic_rows=extra.height, synthetic_weight=.25, features=FEATS, seeds=SEEDS,
         rounds=400, params=PARAMS, training_parent="cv_preds_nomonth", full_chain_cv_waived=True))
    print(json.dumps(dict(event="fit_start", natural_rows=natural.height, synthetic_rows=extra.height,
                          features=FEATS, seeds=SEEDS)), flush=True)
    lgb = load_lgb()
    model_hashes = {}
    for seed in SEEDS:
        start = time.monotonic()
        model = lgb.train({**PARAMS, "seed": seed}, lgb.Dataset(x, y, weight=weight, categorical_feature=[0]), num_boost_round=400)
        p = output / f"seed{seed}.txt"
        model.save_model(str(p))
        model_hashes[p.name] = sha(p)
        print(json.dumps(dict(event="fit_complete", seed=seed, seconds=time.monotonic() - start,
                              model_sha256=model_hashes[p.name])), flush=True)
    for name, expected in hashes.items():
        if sha(root / name) != expected:
            raise RuntimeError(f"input changed during fit: {name}")
    record = json.loads((output / "fit_started.json").read_text())
    record.update(model_sha256=model_hashes, fit_complete=True, lightgbm_version=lgb.__version__)
    save(output / "fit_manifest.json", record)
    print(json.dumps(dict(event="fit_ready", output=str(output), manifest_sha256=sha(output / "fit_manifest.json"))), flush=True)


def checked_submission(path, ids=None):
    d = pl.read_parquet(path)
    if d.columns != [ID, Y] or d.height != ROWS or d[ID].n_unique() != ROWS or d[Y].null_count():
        raise RuntimeError(f"incomplete submission: {path}")
    if not np.isfinite(d[Y].to_numpy()).all():
        raise RuntimeError(f"nonfinite submission: {path}")
    if ids is not None and not np.array_equal(np.sort(d[ID].to_numpy()), np.sort(ids)):
        raise RuntimeError(f"ID mismatch: {path}")
    return d


def apply(root, models, pre_path, post_path, parts_path, output, registration):
    sources = validate_registration(root, registration)
    if output.exists() or output.with_suffix(".json").exists():
        raise RuntimeError("fresh apply output required")
    m = json.loads((models / "fit_manifest.json").read_text())
    if (m["protocol"] != PROTOCOL or not m["fit_complete"] or m["registration_commit"] != registration
        or m["features"] != FEATS or m["seeds"] != SEEDS or m["rounds"] != 400
        or set(m["model_sha256"]) != {f"seed{s}.txt" for s in SEEDS}):
        raise RuntimeError("incomplete/different fit")
    paths = [models / "fit_manifest.json", pre_path, post_path, parts_path,
             root / "data/prc-2026-datasets/ranking.parquet"]
    for name, expected in m["model_sha256"].items():
        p = models / name
        if sha(p) != expected:
            raise RuntimeError(f"model changed: {name}")
        paths.append(p)
    post = checked_submission(post_path)
    ids = post[ID].to_numpy()
    pre = checked_submission(pre_path, ids)
    if pre.schema != post.schema:
        raise RuntimeError("pre/post schema differs")
    cols = [ID, "PHASE_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
    raw = pl.read_parquet(paths[4], columns=cols).filter(pl.col("PHASE_mvt") == "DEP")
    if raw.height != ROWS or raw[ID].n_unique() != ROWS or not np.array_equal(np.sort(raw[ID]), np.sort(ids)):
        raise RuntimeError("ranking raw population differs")
    with np.load(parts_path, allow_pickle=False) as parts:
        if (len(parts["mvt_id"]) != ROWS or not np.array_equal(np.sort(parts["mvt_id"]), np.sort(ids))
            or parts["proba"].shape != (ROWS, 6) or not np.isfinite(parts["proba"]).all()):
            raise RuntimeError("incomplete/mismatched regime parts")
        prob = pl.DataFrame({ID: parts["mvt_id"], "p_nonreg": 1 - parts["proba"][:, 0]})
    frames = []
    for directory, pattern, fields in [("adsblol_kt1", "offblock2_2026-*.parquet", [ID, "coverage"]),
                                      ("adsblol", "visit_2026-*.parquet", [ID] + ACTIVE)]:
        found = sorted((root / "data" / directory).glob(pattern))
        if not found:
            raise RuntimeError("ranking sensor files missing")
        paths.extend(found)
        f = pl.concat([pl.read_parquet(p, columns=fields) for p in found])
        if f[ID].n_unique() != f.height:
            raise RuntimeError("duplicate sensor IDs")
        frames.append(f)
    hashes = {str(p): sha(p) for p in paths}
    d = (post.select(ID).join(raw, on=ID, validate="1:1", maintain_order="left")
         .join(pre.rename({Y: "base"}), on=ID, validate="1:1", maintain_order="left")
         .join(prob, on=ID, validate="1:1", maintain_order="left")
         .join(frames[0], on=ID, how="left", validate="1:1", maintain_order="left")
         .join(frames[1], on=ID, how="left", validate="1:1", maintain_order="left")
         .with_columns(pl.col("coverage").fill_null(-1), pl.col("base").cast(pl.Float64),
             pl.col("ADEP_mvt").replace_strict(dict(zip(AIRPORTS, range(10))), return_dtype=pl.Int32).alias("ap"),
             pl.col("FLIGHT_ID_mvt").is_not_null().cast(pl.Int8).alias("has_nm"),
             (pl.col("MVT_TIME_UTC_mvt").dt.epoch("s") - pl.col("SCHED_TIME_UTC_mvt").dt.epoch("s")).cast(pl.Float64).alias("D")))
    if d.height != ROWS or not np.array_equal(d[ID].to_numpy(), ids) or d["base"].null_count() or d["p_nonreg"].null_count():
        raise RuntimeError("ranking join is incomplete or reordered")
    use = d.select(((pl.col("ADEP_mvt") == "LIRF") & (pl.col("has_nm") == 1) &
                    (pl.col("coverage") < 2) & (pl.col("vi_has_prev") == 1)).fill_null(False)).to_series().to_numpy()
    if not use.any():
        raise RuntimeError("empty natural ranking target")
    x = d.filter(pl.Series(use)).select(FEATS).to_numpy()
    lgb = load_lgb()
    pred = np.mean([lgb.Booster(model_file=str(models / f"seed{s}.txt")).predict(x, num_threads=4) for s in SEEDS], axis=0)
    if not np.isfinite(pred).all():
        raise RuntimeError("nonfinite model prediction")
    replacement = np.maximum(d["base"].to_numpy()[use] + pred, 0)
    # e64_apply uses Polars round(0), followed by the existing target dtype.
    rounded = pl.Series(replacement).round(0).cast(post.schema[Y]).to_numpy()
    final = post[Y].to_numpy().copy()
    final[use] = rounded
    result = post.with_columns(pl.Series(Y, final).cast(post.schema[Y]))
    if not np.isfinite(final).all() or np.any(final < 0) or not np.array_equal(final[~use], post[Y].to_numpy()[~use]):
        raise RuntimeError("invalid candidate or off-mask mutation")
    for name, expected in hashes.items():
        if sha(name) != expected:
            raise RuntimeError(f"apply input changed: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output)
    checked_submission(output, ids)
    delta = final.astype(float) - post[Y].to_numpy().astype(float)
    save(output.with_suffix(".json"), dict(protocol=PROTOCOL, registration_commit=registration,
         sources=sources, input_sha256=hashes, output_sha256=sha(output), rows=ROWS,
         target_rows=int(use.sum()), changed_rows=int(np.count_nonzero(delta)),
         rms_change=float(np.sqrt(np.mean(delta**2))), max_abs_change=float(np.max(np.abs(delta))),
         offmask_unchanged=True, ranking_augmentation=False,
         required_remaining_tail=["E152 .95", "E81 v44 .1214 + v60 .027", "LOBT lift/clip", "protect42", "LSZH half-blend"],
         full_chain_cv_waived=True, final_submission=False))
    print(json.dumps(dict(event="stage_ready", output=str(output), rows=ROWS, target_rows=int(use.sum()),
                          changed_rows=int(np.count_nonzero(delta)), sha256=sha(output))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["fit", "apply"])
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--registration-commit", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bundle", type=Path, default=Path("data/visit_mask_inputs_20261005_v2"))
    p.add_argument("--extracted", type=Path, default=Path("data/visit_mask_20261005_v2"))
    p.add_argument("--models", type=Path)
    p.add_argument("--pre-e64", type=Path)
    p.add_argument("--post-e64", type=Path)
    p.add_argument("--parts", type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    resolve = lambda path: (root / path).resolve()
    if a.command == "fit":
        fit(root, resolve(a.bundle), resolve(a.extracted), resolve(a.output), a.registration_commit)
    else:
        if any(x is None for x in [a.models, a.pre_e64, a.post_e64, a.parts]):
            p.error("apply requires --models --pre-e64 --post-e64 --parts")
        apply(root, resolve(a.models), resolve(a.pre_e64), resolve(a.post_e64), resolve(a.parts), resolve(a.output), a.registration_commit)


if __name__ == "__main__":
    main()

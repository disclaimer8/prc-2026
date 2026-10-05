"""Bounded pre-ADS-B E182+E185 member-OOF diagnostics; no fitting or selection.

This is not v12048/Cat15/full-stage OOF. Label-dependent probability oracles
are descriptive diagnostics, never a forecast gain or an implementable rule.
Only the all-regular comparison is an outcome-independent intervention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "POLARS_MAX_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for _key in THREAD_KEYS:
    os.environ.setdefault(_key, "8")
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import e185_on_e182_fixed as E
import strict_oof_gate as G
import numpy as np
import polars as pl

InvalidInput = G.InvalidOOF
ID = "MVT_ID_mvt"
TARGET = G.TARGET
SOURCE = "data/e185_on_e182_w035_20261005"
TOLERANCE = E.TOLERANCE


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_parent(parent):
    expected = {"registration_commit": "6fb22b43c69465729d0ee3463c43c205771274c5",
                "weight": .35, "ranking_weight": .2333333333333333,
                "subset": "LOBT_flt IS NULL AND ADEP_mvt != LIRF"}
    if not isinstance(parent, dict) or any(parent.get(key) != value for key, value in expected.items()):
        raise InvalidInput("parent lineage differs from registered E182+E185 protocol")
    for field, name in (("control_source", "cv_preds_e172_rawf_lobtf_schedf_eobtf_aobtf_mlpa_f1b"),
                        ("member_source", "cv_e130motion")):
        if Path(str(parent.get(field, ""))).name != name:
            raise InvalidInput(f"parent lineage {field} mismatch")
    gate = parent.get("gate", {})
    if gate.get("passed") is not True or any(gate.get("flags", {}).get(key) is not True
                                             for key in ("four_of_six", "trimmed_lower", "body_lower")):
        raise InvalidInput("parent lineage did not pass the registered additivity gate")
    recorded_code = parent.get("code_sha256", {})
    for module in (E, G):
        path = Path(module.__file__).resolve()
        hashes = [digest for source, digest in recorded_code.items() if Path(source).name == path.name]
        if hashes != [sha256(path)]:
            raise InvalidInput(f"parent lineage pure-helper code mismatch: {path.name}")


def clip(values, lobt):
    """The same monotone clipping map as E.mix, before existing correction."""
    return np.maximum(np.where(np.isnan(lobt), values,
                               np.clip(values, lobt-E.WINDOW, lobt+E.WINDOW)), 0.)


def decompose(z, lobt):
    y = z["y"].astype(np.float64)
    mixture = E.mix(z, z["reg"], lobt)
    correction = z["pred"].astype(np.float64) - mixture
    rebuilt = mixture + correction
    if not np.allclose(rebuilt, z["pred"], rtol=0., atol=TOLERANCE):
        raise InvalidInput("mixture plus recorded correction failed reconstruction")
    values = z["values"].astype(np.float64)
    finite_lo, finite_hi = clip(np.nanmin(values, axis=1), lobt), clip(np.nanmax(values, axis=1), lobt)
    # Missing production candidates are literally zero after nan_to_num. Keep
    # that implementation support separate from the finite physical candidates.
    actual_values = np.nan_to_num(values)
    actual_lo, actual_hi = clip(actual_values.min(axis=1), lobt), clip(actual_values.max(axis=1), lobt)
    production_projection = np.clip(y-correction, actual_lo, actual_hi) + correction
    # Stored probabilities are validated to 1e-6, not symbolically normalized.
    # Retain current as an explicit feasible option even at roundoff boundaries.
    fallback = (z["pred"]-y)**2 < (production_projection-y)**2
    out = {"current": z["pred"].astype(np.float64), "mixture": mixture,
           "existing_correction": correction,
           "all_regular": clip(z["reg"].astype(np.float64), lobt) + correction,
           "finite_oracle": np.clip(y-correction, finite_lo, finite_hi) + correction,
           "production_oracle": np.where(fallback, z["pred"], production_projection),
           "production_current_fallback": fallback,
           "missing_probability_mass": np.sum(z["proba"] * ~np.isfinite(values), axis=1),
           "current_outside_finite_interval": (mixture < finite_lo-TOLERANCE) | (mixture > finite_hi+TOLERANCE),
           "rebuild_abs_error": np.abs(rebuilt-z["pred"])}
    if not all(np.isfinite(v).all() for v in out.values()):
        raise InvalidInput("nonfinite decomposition arithmetic")
    if np.any((out["production_oracle"]-y)**2 > (out["current"]-y)**2 + 1e-4):
        raise InvalidInput("production oracle unexpectedly worse than feasible current prediction")
    return out


def validate_fold(z, path, meta, number):
    ids = E.validate_arrays(z, path)
    consistency = G._values_consistency(z)
    if not consistency.get("consistent"):
        raise InvalidInput(f"{path}: stale or invalid regular/day endpoints")
    idx = np.searchsorted(meta["ids"], ids)
    if (idx >= len(meta["ids"])).any() or not np.array_equal(meta["ids"][idx], ids):
        raise InvalidInput(f"{path}: unknown training IDs")
    month = meta["month"][idx]
    if not np.isin(month, [number, number+6]).all():
        raise InvalidInput(f"{path}: wrong held-out month pair")
    if "month" in z and not np.array_equal(z["month"], month):
        raise InvalidInput(f"{path}: stored month mismatch")
    if not np.array_equal(z["y"], meta["y"][idx]):
        raise InvalidInput(f"{path}: target mismatch")
    return idx


def sensor_paths(root):
    groups = {}
    for year in (2025, 2026):
        for kind, directory, prefix in (("own", "adsblol_kt1", "offblock2"), ("visit", "adsblol", "visit")):
            paths = sorted((root / "data" / directory).glob(f"{prefix}_{year}-*.parquet"))
            if not paths:
                raise InvalidInput(f"no {year} {kind} sensor files")
            groups[f"{year}_{kind}"] = paths
    return groups


def load_features(root, sensors):
    path = root / "data/features_fin1b.parquet"
    schema = pl.read_parquet_schema(path)
    columns = [ID, "split", "ADEP_mvt", "FLIGHT_ID_mvt", "d_mvt_sched", "d_mvt_lobt"]
    month_col = "month" if "month" in schema else "MVT_TIME_UTC_mvt"
    frame = pl.read_parquet(path, columns=[*columns, month_col])
    if month_col != "month":
        frame = frame.with_columns(pl.col(month_col).dt.month().alias("month")).drop(month_col)
    ids = G._integer_ids(frame[ID].to_numpy(), "all feature IDs")
    if len(np.unique(ids)) != len(ids) or frame["ADEP_mvt"].null_count():
        raise InvalidInput("duplicate feature IDs or absent airports")
    unknown_splits = set(frame["split"].unique().to_list()) - {"train", "rank"}
    if unknown_splits:
        raise InvalidInput(f"unknown feature splits: {unknown_splits}")
    parts = {}
    evidence = {}
    for split, year in (("train", 2025), ("ranking", 2026)):
        d = frame.filter(pl.col("split") == ("rank" if split == "ranking" else split))
        if not d.height:
            raise InvalidInput(f"empty {split} features")
        evidence[split] = {}
        for kind in ("own", "visit"):
            cols = [ID, "coverage"] if kind == "own" else [ID, "vi_has_prev", "vi_first_seen"]
            paths = sensors[f"{year}_{kind}"]
            sensor = pl.concat([pl.read_parquet(p, columns=cols) for p in paths], how="vertical_relaxed")
            sensor_ids = G._integer_ids(sensor[ID].to_numpy(), f"{year} {kind} IDs")
            if len(np.unique(sensor_ids)) != len(sensor_ids):
                raise InvalidInput(f"duplicate sensor IDs in {year} {kind}")
            sensor = sensor.with_columns(pl.lit(True).alias(f"{kind}_record_present"))
            evidence[split][kind] = {"files": len(paths), "sensor_rows": sensor.height,
                                    "outside_feature_split_rows": sensor.join(d.select(ID), on=ID, how="anti").height}
            d = d.join(sensor, on=ID, how="left", validate="1:1", maintain_order="left")
        d = d.with_columns(
            has_nm=pl.col("FLIGHT_ID_mvt").is_not_null(),
            own_track=(pl.col("coverage") >= 2).fill_null(False),
            previous_visit=(pl.col("vi_has_prev") == 1).fill_null(False),
            airborne_seen=pl.col("vi_first_seen").cast(pl.Float64).is_finite().fill_null(False),
            own_record_present=pl.col("own_record_present").fill_null(False),
            visit_record_present=pl.col("visit_record_present").fill_null(False),
        ).sort(ID)
        parts[split] = d
    return parts, evidence


def observed_masks(frame):
    airport = frame["ADEP_mvt"].to_numpy()
    nm, own, prev = [frame[key].to_numpy() for key in ("has_nm", "own_track", "previous_visit")]
    july = frame["month"].to_numpy() == 7
    base = {"all": np.ones(frame.height, bool), "NM_LTFM": nm & (airport == "LTFM"),
            "NM_LIRF_July": nm & (airport == "LIRF") & july}
    paths = {"own_and_previous": own & prev, "own_only": own & ~prev,
             "previous_only": ~own & prev, "no_own_no_previous": ~own & ~prev,
             "missing_own_record": ~frame["own_record_present"].to_numpy(),
             "missing_visit_record": ~frame["visit_record_present"].to_numpy(),
             "airborne_seen": frame["airborne_seen"].to_numpy()}
    return {**base, **{f"{group}/{path}": mask & selected
                      for group, mask in base.items() for path, selected in paths.items()}}


def stats(y, z, selected):
    n = int(selected.sum())
    errors = {key: (z[key][selected]-y[selected])**2
              for key in ("current", "all_regular", "finite_oracle", "production_oracle")}
    allocation = errors["current"] - errors["all_regular"]
    result = {"n": n, **{f"{key}_sse": float(value.sum()) for key, value in errors.items()},
              "allocation_helps_n": int((allocation < 0).sum()),
              "allocation_hurts_n": int((allocation > 0).sum()),
              "allocation_unchanged_n": int((allocation == 0).sum()),
              "allocation_help_sse": float(-allocation[allocation < 0].sum()),
              "allocation_harm_sse": float(allocation[allocation > 0].sum()),
              "allocation_delta_sse": float(allocation.sum()),
              "current_error_sum": float((z["current"][selected]-y[selected]).sum()),
              "existing_correction_sum": float(z["existing_correction"][selected].sum()),
              "existing_correction_abs_sum": float(np.abs(z["existing_correction"][selected]).sum()),
              "existing_correction_abs_max": float(np.abs(z["existing_correction"][selected]).max()) if n else 0.,
              "missing_probability_mass_sum": float(z["missing_probability_mass"][selected].sum()),
              "missing_probability_rows": int((z["missing_probability_mass"][selected] > 0).sum()),
              "current_outside_finite_interval_n": int(z["current_outside_finite_interval"][selected].sum()),
              "production_current_fallback_n": int(z["production_current_fallback"][selected].sum()),
              "rebuild_abs_error_max": float(z["rebuild_abs_error"][selected].max()) if n else 0.}
    for key, values in errors.items():
        result[f"_{key}_top20"] = np.sort(values)[-20:].tolist()
    return result


def combine(a, b):
    if a is None:
        return b
    for key, value in b.items():
        if key.startswith("_"):
            a[key] = sorted(a[key] + value)[-20:]
        elif key.endswith("_max"):
            a[key] = max(a[key], value)
        else:
            a[key] += value
    return a


def finish_stats(row):
    n = row["n"]
    for key in ("current", "all_regular", "finite_oracle", "production_oracle"):
        sse = row[f"{key}_sse"]
        row[f"{key}_rmse"] = float(np.sqrt(sse/n)) if n else None
        top = row.pop(f"_{key}_top20")
        for count in (4, 20):
            subtotal = sum(top[-count:])
            row[f"{key}_top{count}_sse_share"] = subtotal/sse if sse else 0.
            row[f"{key}_sse_excluding_local_top{count}"] = max(0., sse-subtotal)
    row["finite_oracle_signed_sse_reduction"] = row["current_sse"]-row["finite_oracle_sse"]
    row["production_oracle_sse_reduction"] = row["current_sse"]-row["production_oracle_sse"]
    row["current_bias"] = row["current_error_sum"]/n if n else None
    return row


def update_table(table, masks, y, decomposition):
    for label, mask in masks.items():
        group = table.setdefault(label, {})
        for leg, submask in (("all", mask), ("body", mask & (y <= 7200))):
            group[leg] = combine(group.get(leg), stats(y, decomposition, submask))


def run(root, output, *, registration):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise InvalidInput(f"output already exists: {output}")
    if not re.fullmatch(r"[0-9a-f]{40}", registration):
        raise InvalidInput("registration must be the full code+journal commit SHA")
    folder = root / SOURCE
    folds = G._fold_paths(folder / "candidate")
    sensors = sensor_paths(root)
    feature = root / "data/features_fin1b.parquet"
    manifest_path = folder / "manifest.json"
    inputs = [*folds, feature, manifest_path, *(p for paths in sensors.values() for p in paths)]
    before = {str(p.relative_to(root)): sha256(p) for p in inputs}
    source_code = [Path(__file__).resolve(), Path(E.__file__).resolve(), Path(G.__file__).resolve()]
    code_sha = {str(p): sha256(p) for p in source_code}
    parent = json.loads(manifest_path.read_text())
    validate_parent(parent)
    if parent.get("input_sha256", {}).get("data/features_fin1b.parquet") != before["data/features_fin1b.parquet"]:
        raise InvalidInput("feature SHA differs from parent lineage manifest")
    for p in folds:
        if parent.get("output_sha256", {}).get(f"candidate/{p.name}") != before[str(p.relative_to(root))]:
            raise InvalidInput(f"candidate SHA differs from parent lineage manifest: {p.name}")
    meta = G._load_features(feature)
    frames, sensor_evidence = load_features(root, sensors)
    train, ranking = frames["train"], frames["ranking"]
    if not np.array_equal(train[ID].to_numpy(), meta["ids"]):
        raise InvalidInput("sorted feature ID order differs from authoritative training metadata")
    # First pass validates all six folds and complete coverage before outcomes.
    seen = np.zeros(len(meta["ids"]), bool)
    validation = []
    for number, path in enumerate(folds, 1):
        z = G._load_fold(path)
        idx = validate_fold(z, path, meta, number)
        if seen[idx].any():
            raise InvalidInput("duplicate IDs across folds")
        seen[idx] = True
        validation.append({"name": path.name, "rows": len(idx), "values": G._values_consistency(z)})
    del z
    if not seen.all():
        raise InvalidInput("six folds do not cover the full training ID set")
    tables = {"aggregate": {}, "month": {}, "fold": {}}
    composition = {}
    for number, path in enumerate(folds, 1):
        z = G._load_fold(path)
        idx = validate_fold(z, path, meta, number)
        frame = train[idx]
        lobt = frame["d_mvt_lobt"].cast(pl.Float64).to_numpy()
        G._numeric(lobt, "LOBT", (len(idx),), allow_nan=True)
        d = decompose(z, lobt)
        y, months = z["y"], meta["month"][idx]
        masks = observed_masks(frame)
        update_table(tables["aggregate"], masks, y, d)
        update_table(tables["fold"].setdefault(path.stem, {}), masks, y, d)
        for month in (number, number+6):
            update_table(tables["month"].setdefault(str(month), {}),
                         {name: mask & (months == month) for name, mask in masks.items()}, y, d)
        # Labels here depend on y and cannot define a ranking intervention.
        planned = np.any(np.isfinite(z["values"][:, 1:5]) &
                         (np.abs(z["values"][:, 1:5]-y[:, None]) <= 6.), axis=1)
        day = np.abs(z["values"][:, 5]-y) <= 6.
        labels = {"matches_plan_within_6s": planned,
                  "matches_reg_plus_day_only_within_6s": ~planned & day,
                  "neither_plan_nor_reg_plus_day": ~planned & ~day}
        for name in ("all", "NM_LTFM", "NM_LIRF_July"):
            update_table(composition.setdefault(name, {}),
                         {label: mask & masks[name] for label, mask in labels.items()}, y, d)
        print(f"validated/measured {path.name}: {len(idx)} rows", file=sys.stderr, flush=True)
    for table in [tables["aggregate"], *tables["month"].values(), *tables["fold"].values(), *composition.values()]:
        for legs in table.values():
            for row in legs.values():
                finish_stats(row)
    rank_masks = observed_masks(ranking)
    rank_month = ranking["month"].to_numpy()
    report = {
        "status": "diagnostic_only", "registration_commit": registration,
        "root": str(root),
        "runtime": {"python": sys.version, "numpy": np.__version__, "polars": pl.__version__,
                    "polars_threads": pl.thread_pool_size()},
        "scope": "Current E182+E185 member OOF before ADS-B; not v12048, not Cat15, not full-stage final OOF",
        "rows": len(meta["ids"]), "ranking_rows": ranking.height,
        "validation": {"fold_count": 6, "full_training_ids": True, "exact_targets_months": True, "folds": validation},
        "definitions": {
            "all_regular": "Outcome-independent p[0]=1; exact production clipping; existing pred-minus-mixture correction preserved",
            "allocation_delta_sse": "current SSE minus all-regular SSE; positive means allocation harms",
            "existing_correction": "Observed pred minus rebuilt clipped mixture; reconstructed algebraically, parent hashes provide provenance, not independent correction validation",
            "finite_oracle": "Label-dependent projection onto clipped convex hull of finite candidate endpoints plus existing correction. Signed reduction; not an upper bound when current uses missing endpoints",
            "production_oracle": "Label-dependent best of current prediction and projection onto clipped convex hull of production nan_to_num endpoints plus existing correction; current explicitly retained for probability-sum roundoff",
            "oracle_warning": "Diagnostic only, no deployable selector, no forecast gain, no new acceptance gate",
            "body": "y <= 7200; outcome-defined and never used for ranking counts",
            "composition": "Mutually exclusive y-dependent exact-plan/reg-plus-day proximity, 6 seconds. Neither category is a proven regular regime",
            "sensor_paths": "Own: adsblol_kt1 offblock2 coverage>=2; previous: adsblol visit vi_has_prev==1; availability only, no ADS-B prediction stages applied",
            "ranking_counts": "Observed input masks only; overlapping populations, counts are availability not proven useful signal",
            "concentration": "Top 4/20 local to each population/leg/method, merged exactly across folds",
        },
        "ranking_counts": {name: int(mask.sum()) for name, mask in rank_masks.items()},
        "ranking_counts_by_month": {str(month): {name: int((mask & (rank_month == month)).sum()) for name, mask in rank_masks.items()}
                                    for month in sorted(np.unique(rank_month))},
        "sensor_inventory": sensor_evidence, "tables": tables,
        "y_dependent_composition_diagnostic": composition,
        "input_sha256": before, "code_sha256": code_sha,
        "parent_lineage_manifest": parent,
    }
    # Rehash every consumed source, including sensors, before atomic publication.
    if {str(p.relative_to(root)): sha256(p) for p in inputs} != before:
        raise InvalidInput("input changed during diagnostic")
    if sensor_paths(root) != sensors or G._fold_paths(folder / "candidate") != folds:
        raise InvalidInput("input file inventory changed during diagnostic")
    if any(sha256(path) != digest for path, digest in code_sha.items()):
        raise InvalidInput("code changed during diagnostic")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    try:
        with (stage / "report.json").open("w") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if output.exists():
            raise InvalidInput(f"output already exists: {output}")
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 8:
        parser.error("threads must be in 1..8")
    for key in THREAD_KEYS:
        os.environ[key] = str(args.threads)
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=args.threads):
            report = run(args.root, args.output, registration=args.registration)
    except (InvalidInput, OSError, ValueError, pl.exceptions.PolarsError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "rows": report["rows"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Read-only validator for the single final file ``gentle-igloo_final.parquet``.

Checks the candidate against the organiser template (``final_submitting.parquet``)
exactly as the ranking script would reject it (name, columns, dtypes, row count,
ID set, duplicates, extra/missing rows) and adds sanity diagnostics joined through
``final_ranking.parquet`` (per month x airport distribution, share of rows that
equal take-off minus a scheduled/estimated off-block time, i.e. the fallback
regime). With ``--reference`` it compares the overlapping Jan/Jul rows with a
scored submission and bounds the resulting Jan/Jul RMSE by the Gram identity.

Nothing is written except the optional ``--json`` report. Nothing is fixed
silently: a wrong dtype is reported, never cast.

Usage (from the repo root)::

    .venv/bin/python scripts/validate_final_submission.py \
        --file submissions/gentle-igloo_final.parquet \
        --reference submissions/gentle-igloo_v12075.parquet --json out.json

Exit code 0 only if every HARD check passes; 1 if any HARD check fails;
2 on an unreadable input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# The organiser templates are written by R (arrow extension type "arrow.r.vctrs");
# polars warns unless told to load the storage type. Must be set before import.
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("POLARS_MAX_THREADS", "4")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

TEAM = "gentle-igloo"
FINAL_NAME = f"{TEAM}_final.parquet"
NAME_PATTERN = re.compile(rf"^{re.escape(TEAM)}_final\.parquet$")
ID_COL = "MVT_ID_mvt"
TARGET_COL = "TAXITIME_SEC_mvt"
MAX_SECONDS = 86400 * 2
REGIME_TOLERANCE_SEC = 15
# Off-block proxies in final_ranking: prediction == MVT_TIME - proxy means the model
# fell back to "take-off minus planned off-block" instead of a learned taxi time.
REGIME_PROXIES = ("SCHED_TIME_UTC_mvt", "IOBT_flt", "LOBT_flt", "EOBT_1_flt", "AOBT_3_flt")
DEFAULT_TEMPLATE = "data/prc-2026-datasets/final_submitting.parquet"
DEFAULT_RANKING = "data/prc-2026-datasets/final_ranking.parquet"
DEFAULT_REFERENCE_RMSE = 230.9387  # public Jan/Jul RMSE of submissions/gentle-igloo_v12075.parquet


class Report:
    """Collects HARD (gate) checks and INFO diagnostics; prints a compact text report."""

    def __init__(self):
        self.checks = []  # dicts: name, level (HARD/INFO), ok (bool|None), detail (str), data (json-able)

    def hard(self, name, ok, detail="", **data):
        self.checks.append({"name": name, "level": "HARD", "ok": bool(ok), "detail": detail, **data})
        return ok

    def info(self, name, detail="", **data):
        self.checks.append({"name": name, "level": "INFO", "ok": None, "detail": detail, **data})

    @property
    def passed(self):
        return all(c["ok"] for c in self.checks if c["level"] == "HARD")

    def to_json(self):
        return {"passed": self.passed, "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "checks": self.checks}

    def render(self):
        lines = []
        for c in self.checks:
            if c["level"] == "HARD":
                mark = "PASS" if c["ok"] else "FAIL"
            else:
                mark = "INFO"
            lines.append(f"[{mark}] {c['name']}: {c['detail']}")
            for extra in c.get("lines", []):
                lines.append(f"       {extra}")
        lines.append("RESULT: " + ("ALL HARD CHECKS PASS" if self.passed else "HARD CHECK FAILED - do not upload"))
        return "\n".join(lines)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dtype_name(dt) -> str:
    return str(dt)


def check_name(report: Report, path: Path):
    ok = NAME_PATTERN.match(path.name) is not None
    report.hard("file name", ok, f"{path.name!r}" + ("" if ok else f" does not match {FINAL_NAME!r}"))


def check_schema(report: Report, sub: pl.DataFrame, template: pl.DataFrame) -> bool:
    sub_cols, tpl_cols = list(sub.columns), list(template.columns)
    ok_cols = sub_cols == tpl_cols
    report.hard("columns", ok_cols,
                f"{sub_cols}" + ("" if ok_cols else f" != template {tpl_cols} (order and names must match)"))
    if not ok_cols:
        return False
    mismatches = [f"{col}: file {dtype_name(sub.schema[col])} vs template {dtype_name(template.schema[col])}"
                  for col in tpl_cols if sub.schema[col] != template.schema[col]]
    ok = not mismatches
    report.hard("dtypes", ok,
                "; ".join(f"{col}={dtype_name(template.schema[col])}" for col in tpl_cols) if ok
                else "mismatch (not cast - rebuild the file): " + "; ".join(mismatches))
    return ok


def check_rows_and_ids(report: Report, sub: pl.DataFrame, template: pl.DataFrame) -> bool:
    n_sub, n_tpl = sub.height, template.height
    ok_rows = report.hard("row count", n_sub == n_tpl,
                          f"{n_sub:,}" + ("" if n_sub == n_tpl else f" != template {n_tpl:,}"))

    ids = sub[ID_COL].to_numpy()
    tpl_ids = template[ID_COL].to_numpy()
    nan_ids = int(np.isnan(ids).sum()) if ids.dtype.kind == "f" else 0
    null_ids = int(sub[ID_COL].null_count())
    report.hard("ID null/NaN", nan_ids == 0 and null_ids == 0, f"null={null_ids}, NaN={nan_ids}")
    if ids.dtype.kind == "f":
        finite = ids[np.isfinite(ids)]
        non_int = int((finite != np.floor(finite)).sum())
        report.info("ID integrality", f"{non_int} non-integer Float64 IDs (template has {int((tpl_ids != np.floor(tpl_ids)).sum())})")

    clean = ids[~np.isnan(ids)] if ids.dtype.kind == "f" else ids
    uniq, counts = np.unique(clean, return_counts=True)
    dups = int((counts > 1).sum())
    report.hard("ID duplicates", dups == 0,
                f"{dups} duplicated IDs" + (f", e.g. {uniq[counts > 1][:5].tolist()}" if dups else ""))

    tpl_sorted = np.sort(tpl_ids)
    missing = np.setdiff1d(tpl_sorted, uniq, assume_unique=True)
    extra = np.setdiff1d(uniq, tpl_sorted, assume_unique=True)
    ok_set = len(missing) == 0 and len(extra) == 0 and nan_ids == 0 and null_ids == 0
    detail = "identical to template" if ok_set else (
        f"missing={len(missing):,} extra={len(extra):,}"
        + (f"; missing e.g. {missing[:5].tolist()}" if len(missing) else "")
        + (f"; extra e.g. {extra[:5].tolist()}" if len(extra) else ""))
    report.hard("ID set equality", ok_set, detail)
    return ok_rows and ok_set and dups == 0


def check_values(report: Report, sub: pl.DataFrame):
    col = sub[TARGET_COL]
    nulls = int(col.null_count())
    vals = col.cast(pl.Float64).to_numpy()
    nan = int(np.isnan(vals).sum()) if col.dtype.is_float() else 0
    inf = int(np.isinf(vals).sum()) if col.dtype.is_float() else 0
    report.hard("no null/NaN/inf", nulls == 0 and nan == 0 and inf == 0, f"null={nulls}, NaN={nan}, inf={inf}")
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        report.hard("value range", False, "no finite values")
        return
    neg = int((finite < 0).sum())
    too_big = int((finite > MAX_SECONDS).sum())
    report.hard("value range", neg == 0 and too_big == 0,
                f"min={finite.min():.0f} max={finite.max():.0f} negative={neg} >{MAX_SECONDS}={too_big}")
    non_int = int((finite != np.floor(finite)).sum())
    report.hard("integer seconds", non_int == 0, f"{non_int} non-integer values" if non_int else "all integer")
    q = np.percentile(finite, [1, 50, 95, 99])
    report.info("value summary",
                f"n={finite.size:,} mean={finite.mean():.1f} p1={q[0]:.0f} median={q[1]:.0f} p95={q[2]:.0f} p99={q[3]:.0f}"
                f" zeros={int((finite == 0).sum())} >7200s={int((finite > 7200).sum())}")


def load_ranking(path: Path) -> pl.DataFrame:
    schema = pl.scan_parquet(path).collect_schema()
    want = [ID_COL, "ADEP_mvt", "MVT_TIME_UTC_mvt"] + [c for c in REGIME_PROXIES if c in schema]
    lf = pl.scan_parquet(path)
    if "PHASE_mvt" in schema:
        lf = lf.filter(pl.col("PHASE_mvt") == "DEP")
    return lf.select(want).collect()


def check_distribution(report: Report, sub: pl.DataFrame, ranking: pl.DataFrame):
    j = ranking.join(sub.select(ID_COL, pl.col(TARGET_COL).cast(pl.Float64).alias("pred")), on=ID_COL, how="inner")
    unmatched = sub.height - j.height
    report.info("join with final_ranking (DEP)", f"{j.height:,} of {sub.height:,} rows matched; unmatched={unmatched:,}"
                + ("" if unmatched == 0 else " (template IDs absent from the ranking file?)"))
    j = j.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("month"))
    dist = (j.group_by("month", "ADEP_mvt")
            .agg(pl.len().alias("n"), pl.col("pred").median().alias("median"),
                 pl.col("pred").quantile(0.95).alias("p95"), pl.col("pred").max().alias("max"),
                 pl.col("pred").mean().alias("mean"))
            .sort("month", "ADEP_mvt"))
    lines = [f"{'month':7} {'apt':5} {'n':>7} {'mean':>7} {'median':>7} {'p95':>7} {'max':>7}"]
    for r in dist.iter_rows(named=True):
        lines.append(f"{r['month']:7} {r['ADEP_mvt']:5} {r['n']:>7,} {r['mean']:>7.0f} {r['median']:>7.0f} {r['p95']:>7.0f} {r['max']:>7.0f}")
    by_month = j.group_by("month").agg(pl.len().alias("n"), pl.col("pred").mean().alias("mean"),
                                       pl.col("pred").median().alias("median")).sort("month")
    report.info("per month x airport distribution",
                "; ".join(f"{r['month']}: n={r['n']:,} mean={r['mean']:.0f} median={r['median']:.0f}"
                          for r in by_month.iter_rows(named=True)),
                lines=lines, table=dist.to_dicts())

    # Regime mixture: share of predictions equal to take-off minus an off-block proxy within +-tolerance.
    proxies = [c for c in REGIME_PROXIES if c in j.columns]
    exprs = []
    for c in proxies:
        gap = (pl.col("MVT_TIME_UTC_mvt") - pl.col(c)).dt.total_seconds().cast(pl.Float64)
        exprs.append(((pl.col("pred") - gap).abs() <= REGIME_TOLERANCE_SEC).fill_null(False).alias(f"eq_{c}"))
        exprs.append(pl.col(c).is_null().alias(f"null_{c}"))
    j = j.with_columns(exprs)
    agg = [pl.len().alias("n")] + [pl.col(f"eq_{c}").mean().alias(f"eq_{c}") for c in proxies] \
        + [pl.col(f"null_{c}").mean().alias(f"null_{c}") for c in proxies]
    reg = j.group_by("month", "ADEP_mvt").agg(agg).sort("month", "ADEP_mvt")
    short = {c: c.replace("_TIME_UTC_mvt", "").replace("_flt", "").replace("_1", "").replace("_3", "") for c in proxies}
    head = f"{'month':7} {'apt':5} {'n':>7} " + " ".join(f"{'='+short[c]:>7}" for c in proxies) + "   " \
        + " ".join(f"{'~'+short[c]:>7}" for c in proxies)
    lines = [head, "       (=X: share of rows with |pred - (take-off - X)| <= %d s; ~X: share with X null)" % REGIME_TOLERANCE_SEC]
    for r in reg.iter_rows(named=True):
        lines.append(f"{r['month']:7} {r['ADEP_mvt']:5} {r['n']:>7,} "
                     + " ".join(f"{100*r['eq_'+c]:>6.2f}%" for c in proxies) + "   "
                     + " ".join(f"{100*r['null_'+c]:>6.2f}%" for c in proxies))
    overall = j.group_by("month").agg(agg).sort("month")
    summary = "; ".join(f"{r['month']}: " + ", ".join(f"{short[c]} {100*r['eq_'+c]:.2f}%" for c in proxies)
                        for r in overall.iter_rows(named=True))
    report.info(f"fallback regime share (pred == take-off - X within +-{REGIME_TOLERANCE_SEC}s)", summary,
                lines=lines, table=reg.to_dicts())


def check_reference(report: Report, sub: pl.DataFrame, ref_path: Path, ranking: pl.DataFrame | None, ref_rmse: float):
    ref = pl.read_parquet(ref_path)
    if ref.columns != [ID_COL, TARGET_COL]:
        report.info("reference", f"{ref_path.name}: unexpected columns {ref.columns}; skipped")
        return
    j = ref.select(ID_COL, pl.col(TARGET_COL).cast(pl.Float64).alias("ref")).join(
        sub.select(ID_COL, pl.col(TARGET_COL).cast(pl.Float64).alias("new")), on=ID_COL, how="inner")
    n_ref, n_over = ref.height, j.height
    if n_over == 0:
        report.info("reference overlap", f"{ref_path.name}: no overlapping IDs")
        return
    d = (j["new"] - j["ref"]).to_numpy()
    d = d[np.isfinite(d)]
    rms = float(np.sqrt(np.mean(d ** 2))) if d.size else float("nan")
    mx = float(np.abs(d).max()) if d.size else float("nan")
    changed = int((d != 0).sum())
    months = ""
    if ranking is not None:
        m = (ranking.select(ID_COL, pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("month"))
             .join(j.select(ID_COL), on=ID_COL, how="inner").group_by("month").len().sort("month"))
        months = " (" + ", ".join(f"{r['month']}: {r['len']:,}" for r in m.iter_rows(named=True)) + ")"
    report.info("reference overlap",
                f"{ref_path.name}: {n_over:,} of {n_ref:,} reference IDs overlap{months}; "
                f"changed={changed:,} ({100*changed/max(n_over,1):.2f}%) RMS diff={rms:.3f} s max |diff|={mx:.0f} s",
                overlap=n_over, reference_rows=n_ref, changed=changed, rms_diff=rms, max_abs_diff=mx)
    # Gram identity on the overlap: ||new - y||^2 = ||ref - y||^2 + ||d||^2 + 2<ref - y, d>.
    # With ||ref - y|| = ref_rmse*sqrt(n) and Cauchy-Schwarz on the cross term the new RMSE lies in
    # [|ref_rmse - rms|, ref_rmse + rms]; zero correlation gives sqrt(ref_rmse^2 + rms^2).
    lo, hi = abs(ref_rmse - rms), ref_rmse + rms
    orth = float(np.sqrt(ref_rmse ** 2 + rms ** 2))
    full = n_over == n_ref
    report.info("Gram-identity RMSE range on the overlap",
                f"[{lo:.4f}, {hi:.4f}] s; uncorrelated-change estimate {orth:.4f} s "
                f"(assumes: reference scored {ref_rmse} s on exactly these {n_over:,} rows"
                + ("" if full else f" - WARNING overlap is partial, {n_ref - n_over:,} reference rows missing")
                + "; labels unchanged between ranking.parquet and final_ranking.parquet; "
                "bounds are the triangle inequality, the point estimate assumes the change is uncorrelated with the residual)",
                lower=lo, upper=hi, uncorrelated=orth, reference_rmse=ref_rmse, full_overlap=full)


def run(args) -> Report:
    report = Report()
    path = Path(args.file)
    report.info("file", f"{path} ({path.stat().st_size:,} bytes, mtime {datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec='seconds')})")
    report.info("sha256", sha256_file(path), sha256=sha256_file(path))
    check_name(report, path)
    sub = pl.read_parquet(path)
    template = pl.read_parquet(args.template)
    report.info("template", f"{args.template}: {template.height:,} rows, "
                + ", ".join(f"{c}={dtype_name(t)}" for c, t in template.schema.items()))
    schema_ok = check_schema(report, sub, template)
    if ID_COL not in sub.columns or TARGET_COL not in sub.columns:
        report.hard("required columns present", False, f"need {ID_COL} and {TARGET_COL}")
        return report
    check_rows_and_ids(report, sub, template)
    check_values(report, sub)
    ranking = None
    if args.ranking and Path(args.ranking).exists():
        ranking = load_ranking(Path(args.ranking))
        report.info("final_ranking", f"{args.ranking}: {ranking.height:,} DEP rows")
        try:
            check_distribution(report, sub, ranking)
        except Exception as exc:  # diagnostics never mask the hard verdict
            report.info("distribution", f"skipped: {type(exc).__name__}: {exc}")
    else:
        report.info("final_ranking", f"{args.ranking} not found - distribution/regime diagnostics skipped")
    if args.reference:
        if Path(args.reference).exists():
            check_reference(report, sub, Path(args.reference), ranking, args.reference_rmse)
        else:
            report.info("reference", f"{args.reference} not found - skipped")
    if not schema_ok:
        report.info("note", "dtype/column mismatch is reported, never auto-cast; rebuild the file with the template schema")
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", required=True, help="candidate parquet (must be named gentle-igloo_final.parquet)")
    p.add_argument("--template", default=DEFAULT_TEMPLATE)
    p.add_argument("--ranking", default=DEFAULT_RANKING)
    p.add_argument("--reference", default=None, help="scored submission to diff against (e.g. submissions/gentle-igloo_v12075.parquet)")
    p.add_argument("--reference-rmse", type=float, default=DEFAULT_REFERENCE_RMSE,
                   help="public RMSE of --reference on its rows (default: v12075 = %(default)s)")
    p.add_argument("--json", default=None, help="write the full report as JSON to this path")
    args = p.parse_args(argv)
    try:
        report = run(args)
    except (FileNotFoundError, OSError, pl.exceptions.PolarsError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(report.render())
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_json(), indent=1, default=str))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

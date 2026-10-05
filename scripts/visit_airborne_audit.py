"""Protocol289: label-free 124-day audit of canonical ground removal before E64.

pack projects departures without labels; run re-extracts all original vi fields.
No model, target, prediction, previous-day extension or existing artifact change.
"""
from __future__ import annotations
import argparse
import ast
import calendar
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

os.environ.setdefault("POLARS_MAX_THREADS", "6")
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import adsb_visit as V
from adsb_offblock_v2 import surface_mask

PROTOCOL = "visit-airborne-label-free-124days-v1"
SOURCES = {"adsb_visit.py": "3921f23cc8956faf36fe3caf80843fa68f78a885274f78d5a1b68c1035f7e1a3",
           "adsb_offblock_v2.py": "c843aae5e7643c78d5968ae530af8022e45a6164efe8e6c26f7ffb6f61b2500b"}
CONFIG_SOURCES = {"e185_submission.py": "b1031e156dcd5ab65f808def63664b71eadfa99574d92112cd18b80c9dfef46c",
                  "e182_e185_submission.py": "02800595cb2984953b0dae2dd7b8e9ace56967c227d10e757a4281657450a87d"}
MONTHS = ["2025-01", "2025-07", "2026-01", "2026-07"]
DAYS = [f"{m}-{d:02}" for m in MONTHS for d in range(1, calendar.monthrange(int(m[:4]), int(m[5:]))[1] + 1)]
DEP_COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "CALLSIGN_flt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "FLIGHT_ID_mvt"]
POINT_COLS = ["icao", "airport", "t", "alt", "gs", "callsign", "on_ground"]
ACTIVE = [n for n in V.NAMES if n not in {"vi_n_pts", "vi_prev_pts", "vi_alt_first"}]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def save(path, data):
    with Path(path).open("x") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def read(path):
    return json.loads(Path(path).read_text())


def airborne(points):
    # Exact canonical complement: unknown masks are dropped by Polars filter.
    # Missing altitude with speed >40 and on_ground=False remains airborne.
    return points.filter(~surface_mask(points))


def extract(points, deps):
    """Literal adsb_visit.main extraction, without file I/O or labels."""
    cmap = {}
    for cs, ap, ic in points.filter(pl.col("callsign").is_not_null()).select("callsign", "airport", "icao").unique().iter_rows():
        cmap.setdefault((cs, ap), []).append(ic)
    by = {(k[0], k[1]): g.select("t", "alt", "gs").sort("t") for k, g in points.group_by(["icao", "airport"])}
    out, identity = [], {}
    for mid, ap, cs, to in deps.select("MVT_ID_mvt", "ADEP_mvt", "cs", "to").iter_rows():
        ic = cmap.get((cs, ap), [])
        identity[mid] = ic[0] if len(ic) == 1 and (ic[0], ap) in by else None
        if identity[mid] is None:
            out.append((mid, *([None] * 7), 0)); continue
        g = by[(ic[0], ap)]
        t = g["t"].to_numpy()
        sel = (t >= to - 12 * 3600) & (t <= to + 600)
        if sel.sum() < 2:
            out.append((mid, *([None] * 7), 0)); continue
        tt, alt, gs = t[sel], g["alt"].to_numpy()[sel], g["gs"].to_numpy()[sel]
        big = np.where(np.diff(tt) > 1800)[0]
        k = int(big[-1]) + 1 if len(big) else 0
        first, n = float(to - tt[k]), int(len(tt) - k)
        a0 = float(alt[k]) if np.isfinite(alt[k]) else None
        g0 = float(gs[k]) if np.isfinite(gs[k]) else None
        if len(big):
            end = float(tt[int(big[-1])]); start = 0 if len(big) == 1 else int(big[-2]) + 1
            out.append((mid, first, float(to-end), float(tt[k]-end), n, int(int(big[-1])-start+1), a0, g0, 1))
        else:
            out.append((mid, first, None, None, n, None, a0, g0, 0))
    schema = {"MVT_ID_mvt": pl.Float64, **{n: (pl.Int64 if n in ("vi_n_pts", "vi_prev_pts", "vi_has_prev") else pl.Float64) for n in V.NAMES}}
    return pl.DataFrame(out, schema=schema, orient="row"), identity


def same(a, b):
    return (a == b) | (np.isnan(a) & np.isnan(b))


def compare_control(control, saved):
    a, b = control.sort("MVT_ID_mvt"), saved.select(control.columns).sort("MVT_ID_mvt")
    if a.height != b.height or a["MVT_ID_mvt"].n_unique() != a.height or not np.array_equal(a["MVT_ID_mvt"], b["MVT_ID_mvt"]):
        raise RuntimeError("saved visit population or ID order differs")
    return {n: int((~same(a[n].cast(pl.Float64).to_numpy(), b[n].cast(pl.Float64).to_numpy())).sum()) for n in V.NAMES}


def sources(root):
    for n, h in {**SOURCES, **CONFIG_SOURCES}.items():
        if sha(root / "src" / n) != h:
            raise RuntimeError(f"source drift: {n}")
    tree = ast.parse((root / "src/e185_submission.py").read_text())
    drop = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "D3" for t in n.targets))
    if ACTIVE != [n for n in V.NAMES if n not in drop.split(",")]:
        raise RuntimeError("active vi fields differ from production D3")


def pack(root, target, registration):
    root, target = Path(root).resolve(), Path(target).resolve()
    sources(root)
    if target.exists():
        raise RuntimeError("fresh bundle required")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=target.parent))
    for name in ["surface", "visit", "coverage", "departures", "src", "scripts"]:
        (stage / name).mkdir()
    originals = {}
    for day in DAYS:
        for directory, old, new in [("adsblol", "surface", "surface"), ("adsblol", "visit", "visit"), ("adsblol_kt1", "offblock2", "coverage")]:
            p = root / "data" / directory / f"{old}_{day}.parquet"
            originals[str(p)] = sha(p)
            copied = stage / new / f"{day}.parquet"
            shutil.copy2(p, copied)
            if sha(copied) != originals[str(p)]:
                raise RuntimeError("observation copy differs from original")
    # Explicit projection is mandatory; target columns are never loaded.
    rawfiles = {"2025-01": "training_2025-01-01_2025-02-01.parquet", "2025-07": "training_2025-07-01_2025-08-01.parquet",
                "2026-01": "ranking.parquet", "2026-07": "ranking.parquet"}
    for month, name in rawfiles.items():
        p = root / "data/prc-2026-datasets" / name
        originals[str(p)] = sha(p)
        df = pl.read_parquet(p, columns=DEP_COLS).filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m") == month))
        if not df.height or df["MVT_ID_mvt"].n_unique() != df.height:
            raise RuntimeError("empty or duplicate departure IDs")
        df.write_parquet(stage / "departures" / f"{month}.parquet")
    for n in {**SOURCES, **CONFIG_SOURCES}:
        shutil.copy2(root / "src" / n, stage / "src" / n)
    shutil.copy2(Path(__file__), stage / "scripts" / Path(__file__).name)
    if any(sha(p) != h for p, h in originals.items()):
        raise RuntimeError("original inputs changed during packing")
    sources(root)
    inventory = {str(p.relative_to(stage)): sha(p) for p in stage.rglob("*") if p.is_file()}
    save(stage / "manifest.json", {"protocol": PROTOCOL, "registration_commit": registration, "days": DAYS,
        "source_sha256": SOURCES, "configuration_source_sha256": CONFIG_SOURCES,
        "active_vi_fields": ACTIVE, "original_input_sha256": originals, "files": inventory,
        "departure_columns": DEP_COLS, "point_columns": POINT_COLS})
    verify(stage, registration)
    stage.rename(target)
    print(json.dumps({"bundle": str(target), "manifest_sha256": sha(target / "manifest.json"), "days": len(DAYS)}), flush=True)


def verify(bundle, registration):
    bundle = Path(bundle)
    m = read(bundle / "manifest.json")
    expected = {f"{d}/{day}.parquet" for d in ["surface", "visit", "coverage"] for day in DAYS}
    expected |= {f"departures/{month}.parquet" for month in MONTHS} | {f"src/{n}" for n in {**SOURCES, **CONFIG_SOURCES}} | {f"scripts/{Path(__file__).name}"}
    if (m["protocol"] != PROTOCOL or m["registration_commit"] != registration or m["days"] != DAYS
            or m["source_sha256"] != SOURCES or m["configuration_source_sha256"] != CONFIG_SOURCES or m["active_vi_fields"] != ACTIVE
            or m["departure_columns"] != DEP_COLS or m["point_columns"] != POINT_COLS
            or set(m["files"]) != expected):
        raise RuntimeError("bundle protocol or inventory differs")
    for n, h in m["files"].items():
        if sha(bundle / n) != h:
            raise RuntimeError(f"bundle changed: {n}")
    sources(bundle)
    sources(Path(__file__).resolve().parents[1])
    if sha(Path(__file__)) != m["files"][f"scripts/{Path(__file__).name}"]:
        raise RuntimeError("active audit source differs from bundle")
    return m


def run(bundle, output, registration):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    manifest = verify(bundle, registration)
    if output.exists():
        raise RuntimeError("fresh output required; partial artifacts preserved")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    (stage / "days").mkdir()
    groups, daily, mismatches = {}, [], 0
    for month in MONTHS:
        deps = pl.read_parquet(bundle / "departures" / f"{month}.parquet", columns=DEP_COLS).with_columns(
            cs=pl.col("CALLSIGN_flt").fill_null(pl.col("FLIGHT_mvt")).str.replace_all(" ", ""),
            to=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64))
        for day in [d for d in DAYS if d.startswith(month)]:
            dep = deps.filter(pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m-%d") == day)
            pts = pl.read_parquet(bundle / "surface" / f"{day}.parquet", columns=POINT_COLS)
            apoints = airborne(pts)
            base, bid = extract(pts, dep)
            air, aid = extract(apoints, dep)
            parity = compare_control(base, pl.read_parquet(bundle / "visit" / f"{day}.parquet", columns=["MVT_ID_mvt", *V.NAMES]))
            mismatches += sum(parity.values())
            if sum(parity.values()):
                save(stage / "control_parity_failure.json", {"day": day, "mismatches": parity})
                raise RuntimeError(f"saved control parity failed on {day}: {parity}")
            cov = pl.read_parquet(bundle / "coverage" / f"{day}.parquet", columns=["MVT_ID_mvt", "coverage"])
            if (cov["MVT_ID_mvt"].n_unique() != cov.height or cov.height != dep.height
                    or not np.array_equal(np.sort(cov["MVT_ID_mvt"].to_numpy()), np.sort(dep["MVT_ID_mvt"].to_numpy()))):
                raise RuntimeError("coverage IDs differ from complete daily departure population")
            frame = (dep.select("MVT_ID_mvt", "ADEP_mvt", pl.col("FLIGHT_ID_mvt").is_not_null().alias("has_nm"))
                .join(cov, on="MVT_ID_mvt", how="left")
                .join(base.rename({n: "base_"+n for n in V.NAMES}), on="MVT_ID_mvt")
                .join(air.rename({n: "air_"+n for n in V.NAMES}), on="MVT_ID_mvt"))
            frame = frame.with_columns(pl.Series("base_icao", [bid[x] for x in frame["MVT_ID_mvt"]], dtype=pl.String),
                                       pl.Series("air_icao", [aid[x] for x in frame["MVT_ID_mvt"]], dtype=pl.String))
            if frame.height != dep.height:
                raise RuntimeError("audit joins changed population")
            frame.write_parquet(stage / "days" / f"{day}.parquet")
            mask = pts.select(surface_mask(pts).alias("surface")).to_series()
            daily.append({"day": day, "departures": dep.height, "points": pts.height, "retained_airborne": apoints.height,
                          "retained_unknown_altitude": int((~apoints["alt"].is_finite()).fill_null(True).sum()),
                          "unknown_mask": mask.null_count(), "surface_points": int(mask.fill_null(False).sum()), "parity_mismatches": parity})
            for key, g in frame.group_by(["ADEP_mvt", "has_nm"]):
                label = f"{month}|{key[0]}|nm={int(key[1])}"
                groups.setdefault(label, []).append(g)
            print(json.dumps(daily[-1]), flush=True)
    summary = {}
    for key, parts in groups.items():
        g = pl.concat(parts)
        low = g["coverage"].fill_null(-1).to_numpy() < 2
        bp = g["base_vi_has_prev"].to_numpy() == 1; ap = g["air_vi_has_prev"].to_numpy() == 1
        bident, aident = g["base_icao"].to_numpy(), g["air_icao"].to_numpy()
        changed = np.zeros(g.height, dtype=bool); active_changed = changed.copy(); deltas = {}
        for name in V.NAMES:
            b = g["base_"+name].cast(pl.Float64).to_numpy(); a = g["air_"+name].cast(pl.Float64).to_numpy()
            diff = ~same(a,b); changed |= diff
            if name in ACTIVE: active_changed |= diff
            delta = (a-b)[np.isfinite(a) & np.isfinite(b)]
            deltas[name] = {"changed": int(diff.sum()), "both_finite": len(delta),
                           "delta_p10_p50_p90": np.quantile(delta, [.1,.5,.9]).tolist() if len(delta) else None}
        stable_identity = np.asarray([b is not None and b == a for b,a in zip(bident,aident)])
        summary[key] = {"rows": g.height, "coverage_lt2": int(low.sum()), "baseline_prev": int(bp.sum()), "airborne_prev": int(ap.sum()),
            "current_e64_reachable": int((low & bp).sum()), "airborne_e64_reachable": int((low & ap).sum()),
            "synthetic_training_rows_from_coverage_ge2": int((~low & ap & stable_identity & active_changed).sum()),
            "synthetic_same_identity": int((~low & ap & stable_identity).sum()),
            "coverage_ge2_with_airborne_prev": int((~low & ap).sum()),
            "same_airframe": int(sum(b is not None and b == a for b,a in zip(bident,aident))),
            "identity_lost": int(sum(b is not None and a is None for b,a in zip(bident,aident))),
            "identity_gained": int(sum(b is None and a is not None for b,a in zip(bident,aident))),
            "identity_switched": int(sum(b is not None and a is not None and b != a for b,a in zip(bident,aident))),
            "changed_any_vi": int(changed.sum()), "changed_active_vi": int(active_changed.sum()), "delta": deltas}
    # Feasibility only: no claim of improvement, no target or RMSE gate.
    lirf = [summary[f"{m}|LIRF|nm=1"] for m in MONTHS]
    gate = {"exact_control_parity": mismatches == 0,
            "no_identity_switches": all(v["identity_switched"] == 0 for v in summary.values()),
            "ranking_july_previous_visit_survives": lirf[3]["airborne_e64_reachable"] > 0,
            "training_jan_and_july_have_synthetic_rows": all(v["synthetic_training_rows_from_coverage_ge2"] > 0 for v in lirf[:2]),
            "training_july_changes_active_vi": lirf[1]["changed_active_vi"] > 0}
    verify(bundle, registration)
    report = {"protocol": PROTOCOL, "registration_commit": registration, "input_manifest_sha256": sha(bundle / "manifest.json"),
              "parity_mismatched_cells": mismatches, "gate": gate, "feasible": all(gate.values()), "summary": summary, "days": daily,
              "output_sha256": {p.name: sha(p) for p in sorted((stage / "days").glob("*.parquet"))}}
    save(stage / "report.json", report)
    stage.rename(output)
    print(json.dumps({"output": str(output), "gate": gate, "report_sha256": sha(output / "report.json")}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["pack", "verify", "run"])
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--registration-commit", required=True)
    a = p.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", a.registration_commit): p.error("full code+journal registration commit required")
    if a.command == "run" and not a.output: p.error("--output required")
    target = a.output if a.command == "run" else a.input
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent / ("."+target.name+".lock")).open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.command == "pack": pack(a.root, a.input, a.registration_commit)
        elif a.command == "run": run(a.input, a.output, a.registration_commit)
        else:
            verify(a.input, a.registration_commit)
            print(json.dumps({"verified": str(a.input), "manifest_sha256": sha(a.input / "manifest.json")}))


if __name__ == "__main__":
    main()

"""Seven daily ADS-B derivative products for every final day, against the final MVT (08.10.2026).

The producers are run UNCHANGED, in a fresh child process per (day, product), from a separate WORK root in
which the hard-coded paths they read resolve to the final inputs:
    data/prc-2026-datasets/ranking.parquet  -> data/prc-2026-datasets/final_ranking.parquet (SHA-checked)
    data/adsblol/surface_<day>.parquet      -> the extracted adsb.lol surface points (MVT-independent raw input)
    src/, analysis/sources/                  -> the repository's own (producer code, OSM stand tables)
Recipe of the seven products = src/run_final_any_period.sh:224-231 (the env-only variants of the same code):
    kt1/offblock2,3,5    src/adsb_offblock_v{2,3,5}.py  MOVING_KT=1 ADSB_OUT_DIR=data/adsblol_kt1
    sw_kt1/offblock2     src/adsb_offblock_v2.py        MOVING_KT=1 STAND_WALK=1 ADSB_OUT_DIR=data/adsblol_sw_kt1
    visit, visit_m       src/adsb_visit.py, src/e69_visit_merge.py   (no VISIT_PREV_DAY, as the accepted files)
    head                 src/e134_head_features.py      (MOVING_KT default 3 kt)
Same-day matching, thresholds and identity semantics are those of the producers; nothing here touches them.

Receipts: one JSON per day distinguishing `processed` (N matched airframes), `processed_no_matches` (every product
ran and matched nothing) and `input_missing` (no surface file -> nothing is produced for the day). A producer failure
or a row/ID mismatch against the final departures of the day stops the run; there is no fallback to old files.

    .venv/bin/python scripts/final_adsb_derivatives_20261008.py run    [--out DIR] [--workers 4] [--resume]
    .venv/bin/python scripts/final_adsb_derivatives_20261008.py report [--out DIR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

PROTOCOL = "final-adsb-daily-derivatives-7products-finalMVT-unchanged-producers-v1"
SELF = "scripts/final_adsb_derivatives_20261008.py"
FINAL_RANKING = "data/prc-2026-datasets/final_ranking.parquet"
FINAL_RANKING_SHA = "4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f"
FINAL_MONTHS = {(2026, 1), (2026, 2), (2026, 6), (2026, 7)}
EXPECTED_DAYS = 120
EXPECTED_DEP = 670790
DEFAULT_OUT = "data/final_adsb_derivatives_20261008"
DEFAULT_SURFACE_DIR = "data/adsblol"
DISK_FLOOR = 4 * 1024**3
RANKING_COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]
SOURCES = ["src/adsb_offblock_v2.py", "src/adsb_offblock_v3.py", "src/adsb_offblock_v5.py", "src/adsb_visit.py",
           "src/e69_visit_merge.py", "src/e134_head_features.py", "src/geometry.py"]


@dataclass(frozen=True)
class Product:
    name: str
    script: str
    env: dict
    out_dir: str          # relative to WORK, as the producer writes it
    prefix: str           # <prefix>_<day>.parquet
    match_col: str        # a row has an own-track product when this column is not null (or == 1 for n_cand)
    aligned: bool = True  # one row per final departure of the day (head writes matched rows only)
    extra: tuple = field(default_factory=tuple)  # extra counted columns: (name, col, kind)

    def path(self, day: str) -> str:
        return f"{self.out_dir}/{self.prefix}_{day}.parquet"

    def matched(self, df: pl.DataFrame) -> int:
        if self.match_col == "n_cand":
            return int((df["n_cand"] == 1).sum())
        return int(df[self.match_col].is_not_null().sum())


KT1 = {"MOVING_KT": "1", "ADSB_OUT_DIR": "data/adsblol_kt1"}
SW_KT1 = {"MOVING_KT": "1", "STAND_WALK": "1", "ADSB_OUT_DIR": "data/adsblol_sw_kt1"}
COV = (("coverage_ge2", "coverage", "ge2"), ("coverage3", "coverage", "eq3"))
PRODUCTS = (
    Product("kt1_offblock2", "src/adsb_offblock_v2.py", KT1, "data/adsblol_kt1", "offblock2", "n_cand", True, COV),
    Product("kt1_offblock3", "src/adsb_offblock_v3.py", KT1, "data/adsblol_kt1", "offblock3", "sf_n_seg"),
    Product("kt1_offblock5", "src/adsb_offblock_v5.py", KT1, "data/adsblol_kt1", "offblock5", "gs_stand_known"),
    Product("sw_kt1_offblock2", "src/adsb_offblock_v2.py", SW_KT1, "data/adsblol_sw_kt1", "offblock2", "n_cand", True, COV),
    Product("visit", "src/adsb_visit.py", {}, "data/adsblol", "visit", "vi_first_seen"),
    Product("visit_m", "src/e69_visit_merge.py", {}, "data/adsblol", "visit_m", "vm_n_pts"),
    Product("head", "src/e134_head_features.py", {}, "data/adsblol", "head", "hd_gs0", False),
)
# env names the producers read: none of them may leak in from the caller's shell
PRODUCER_KNOBS = ("MOVING_KT", "STAND_WALK", "STAND_M", "ADSB_OUT_DIR", "VISIT_GAP", "VISIT_PREV_DAY", "MERGE_D_END",
                  "MERGE_D_MIN", "SURFACE_OUT")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def producer_env(product: Product, threads: int) -> dict:
    """A closed environment: PATH/HOME, the thread caps, and exactly the product's recipe knobs."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/"),
           "POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR": "load_as_storage",
           "POLARS_MAX_THREADS": str(threads), "OMP_NUM_THREADS": str(threads), "OMP_THREAD_LIMIT": str(threads),
           "PYTHONUNBUFFERED": "1"}
    for k in PRODUCER_KNOBS:
        assert k not in env
    env.update(product.env)
    return env


def final_departures(ranking: Path, expected_sha: str | None = FINAL_RANKING_SHA) -> pl.DataFrame:
    """DEP rows of the final ranking with their UTC day; refuses a file that is not the registered final one."""
    got = sha(ranking)
    if expected_sha is not None and got != expected_sha:
        raise ValueError(f"final ranking SHA mismatch: {got} != {expected_sha}")
    r = pl.read_parquet(ranking, columns=RANKING_COLS).filter(pl.col("PHASE_mvt") == "DEP")
    r = r.with_columns(day=pl.col("MVT_TIME_UTC_mvt").dt.date())
    months = {(d.year, d.month) for d in r["day"].unique().to_list()}
    if months != FINAL_MONTHS:
        raise ValueError(f"final months {sorted(months)} != {sorted(FINAL_MONTHS)}")
    if r["MVT_ID_mvt"].n_unique() != r.height:
        raise ValueError("duplicate MVT_ID in the final departures")
    return r


def days_of(dep: pl.DataFrame) -> list[str]:
    return [d.isoformat() for d in sorted(dep["day"].unique().to_list())]


def materialize(root: Path, out: Path, surface_dir: Path, days: list[str], ranking_sha: str | None) -> dict:
    """WORK = out itself: producer code and OSM tables by symlink, final ranking under the name the producers read,
    one symlink per surface day that exists. Returns {day: surface path or None}."""
    for sub in ("data/prc-2026-datasets", "data/adsblol", "data/adsblol_kt1", "data/adsblol_sw_kt1", "receipts", "logs", "analysis"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    links = {out / "src": root / "src", out / "analysis" / "sources": root / "analysis" / "sources",
             out / "data/prc-2026-datasets/ranking.parquet": root / FINAL_RANKING}
    if ranking_sha is not None and sha(root / FINAL_RANKING) != ranking_sha:
        raise ValueError("final ranking SHA mismatch at materialize")
    for link, target in links.items():
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError(f"{link} points elsewhere: {link.resolve()} != {target.resolve()}")
            continue
        if link.exists():
            raise FileExistsError(link)
        link.symlink_to(target.resolve())
    surfaces = {}
    for day in days:
        src = surface_dir / f"surface_{day}.parquet"
        link = out / "data/adsblol" / f"surface_{day}.parquet"
        if not src.is_file() or src.stat().st_size == 0:
            surfaces[day] = None
            if link.is_symlink():
                link.unlink()  # a dangling link from an earlier materialize
            continue
        if link.is_symlink():
            if link.resolve() != src.resolve():
                raise ValueError(f"{link} points elsewhere")
        elif link.exists():
            raise FileExistsError(f"{link} is a real file inside WORK; surface files must stay in {surface_dir}")
        else:
            link.symlink_to(src.resolve())
        surfaces[day] = src
    return surfaces


def check_product(product: Product, path: Path, day_dep: pl.DataFrame) -> dict:
    """Row/ID contract of one product file against the final departures of the day, plus the counts."""
    df = pl.read_parquet(path)
    if "MVT_ID_mvt" not in df.columns or product.match_col not in df.columns:
        raise ValueError(f"{path}: columns {df.columns} lack MVT_ID_mvt/{product.match_col}")
    ids = set(df["MVT_ID_mvt"].to_list())
    want = set(day_dep["MVT_ID_mvt"].to_list())
    if len(ids) != df.height:
        raise ValueError(f"{path}: duplicate MVT_ID rows")
    if product.aligned and ids != want:
        raise ValueError(f"{path}: {df.height} rows vs {len(want)} final departures of the day (ids differ by {len(ids ^ want)})")
    if not product.aligned and not ids <= want:
        raise ValueError(f"{path}: {len(ids - want)} rows outside the final departures of the day")
    rec = {"rows": df.height, "matched": product.matched(df), "bytes": path.stat().st_size, "sha256": sha(path)}
    for name, col, kind in product.extra:
        rec[name] = int((df[col] >= 2).sum()) if kind == "ge2" else int((df[col] == 3).sum())
    return rec


def remove_partial(out: Path, day: str, log) -> None:
    """Outputs of a day that has no receipt are this run's own leftovers (crash mid-day); remove before re-running."""
    for p in PRODUCTS:
        f = out / p.path(day)
        if f.is_symlink() or not f.resolve().is_relative_to(out.resolve()):
            raise ValueError(f"refusing to remove {f}")
        if f.exists():
            log(f"{day}: removing partial {f.relative_to(out)}")
            f.unlink()


def run_day(root: Path, out: Path, day: str, surface: Path | None, day_dep: pl.DataFrame, threads: int, py: str, log) -> dict:
    receipt = {"day": day, "protocol": PROTOCOL, "departures": day_dep.height, "started_utc": utc(), "products": {}}
    if surface is None:
        receipt.update(status="input_missing", surface=None, finished_utc=utc())
        return receipt
    receipt["surface"] = {"path": str(surface), "bytes": surface.stat().st_size, "sha256": sha(surface)}
    remove_partial(out, day, log)
    for product in PRODUCTS:
        target = out / product.path(day)
        t0 = time.monotonic()
        with (out / "logs" / f"{day}_{product.name}.log").open("w") as lf:
            rc = subprocess.run([py, "-u", product.script, day], cwd=out, env=producer_env(product, threads),
                                stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise RuntimeError(f"{day} {product.name}: {product.script} exit {rc}, see logs/{day}_{product.name}.log")
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f"{day} {product.name}: producer wrote no {target.relative_to(out)}")
        rec = check_product(product, target, day_dep)
        rec["path"] = product.path(day)
        rec["seconds"] = round(time.monotonic() - t0, 2)
        receipt["products"][product.name] = rec
    receipt["matched_total"] = sum(r["matched"] for r in receipt["products"].values())
    receipt["status"] = "processed" if receipt["matched_total"] > 0 else "processed_no_matches"
    receipt["finished_utc"] = utc()
    return receipt


def receipt_valid(out: Path, receipt: dict) -> bool:
    if receipt.get("protocol") != PROTOCOL or receipt.get("status") not in ("processed", "processed_no_matches"):
        return False
    if set(receipt.get("products", {})) != {p.name for p in PRODUCTS}:
        return False
    for rec in receipt["products"].values():
        f = out / rec["path"]
        if not f.is_file() or f.stat().st_size != rec["bytes"] or sha(f) != rec["sha256"]:
            return False
    return True


def manifest(out: Path) -> Path:
    files = sorted(p for p in out.rglob("*") if p.is_file() and not p.is_symlink()
                   and p.suffix in (".parquet", ".json") and p.parent.name != "logs" and p.name != "MANIFEST.sha256")
    lines = [f"{sha(p)}  {p.relative_to(out)}" for p in files]
    m = out / "MANIFEST.sha256"
    m.write_text("\n".join(lines) + "\n")
    return m


def run(root: Path, out: Path, surface_dir: Path, workers: int, threads: int, resume: bool, nice: int = 10,
        ranking_sha: str | None = FINAL_RANKING_SHA, days: list[str] | None = None) -> dict:
    py = sys.executable
    venv_py = (root / ".venv/bin/python")
    if venv_py.exists() and Path(py).resolve() != venv_py.resolve():
        raise RuntimeError(f"run with {venv_py}, not {py}")
    if nice:
        try:
            os.nice(nice)
        except OSError:
            pass
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        raise FileExistsError(f"{out}/complete.json exists: this output root is sealed")
    existing = sorted((out / "receipts").glob("*.json")) if (out / "receipts").exists() else []
    if existing and not resume:
        raise FileExistsError(f"{out} has {len(existing)} receipts; pass --resume to continue it")
    log_f = (out / "run.log").open("a")

    def log(msg: str) -> None:
        line = f"[{utc()}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n"); log_f.flush()

    dep = final_departures(root / FINAL_RANKING, ranking_sha)
    all_days = days_of(dep)
    if days is None:
        days = all_days
        if len(days) != EXPECTED_DAYS or dep.height != EXPECTED_DEP:
            raise ValueError(f"final departures: {len(days)} days / {dep.height} rows, expected {EXPECTED_DAYS}/{EXPECTED_DEP}")
    elif not set(days) <= set(all_days):
        raise ValueError(f"days outside the final ranking: {sorted(set(days) - set(all_days))}")
    surfaces = materialize(root, out, surface_dir, days, ranking_sha)
    log(f"{PROTOCOL}: {len(days)} days, {sum(s is not None for s in surfaces.values())} with a surface file, "
        f"workers={workers} threads/producer={threads} nice={nice} python={py}")
    by_day = {d: g for (d,), g in dep.partition_by("day", as_dict=True).items()}
    t_start = time.monotonic()

    def one(day: str) -> dict:
        rp = out / "receipts" / f"{day}.json"
        if rp.exists():
            old = json.loads(rp.read_text())
            if receipt_valid(out, old):
                log(f"{day}: receipt valid, skipped")
                return old
            if old.get("status") == "input_missing" and surfaces[day] is None:
                log(f"{day}: still no surface file")
                return old
            log(f"{day}: stale receipt ({old.get('status')}), redoing")
            rp.unlink()
        if shutil.disk_usage(out).free < DISK_FLOOR:
            raise RuntimeError("disk floor 4 GiB reached")
        day_dep = by_day[dt.date.fromisoformat(day)]
        r = run_day(root, out, day, surfaces[day], day_dep, threads, py, log)
        save(rp, r)
        if r["status"] == "input_missing":
            log(f"{day}: INPUT MISSING ({r['departures']} departures without any product)")
        else:
            ob = r["products"]["kt1_offblock2"]
            secs = sum(p["seconds"] for p in r["products"].values())
            log(f"{day}: {r['status']} dep={r['departures']} identified={ob['matched']} cov>=2={ob.get('coverage_ge2')} "
                f"visit={r['products']['visit']['matched']} head={r['products']['head']['rows']} in {secs:.0f}s")
        return r

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            receipts = list(ex.map(one, days))  # any exception propagates and stops the run
    except BaseException as e:
        log(f"STOPPED: {type(e).__name__}: {e}")
        log_f.close()
        raise
    counts = {s: sum(r["status"] == s for r in receipts) for s in ("processed", "processed_no_matches", "input_missing")}
    m = manifest(out)
    summary = {"protocol": PROTOCOL, "utc": utc(), "seconds": round(time.monotonic() - t_start, 1), "days": len(days),
               "status_counts": counts, "input_missing_days": [r["day"] for r in receipts if r["status"] == "input_missing"],
               "departures": int(dep.height), "departures_without_products": int(sum(r["departures"] for r in receipts if r["status"] == "input_missing")),
               "final_ranking": {"path": FINAL_RANKING, "sha256": ranking_sha},
               "sources_sha256": {s: sha(root / s) for s in SOURCES if (root / s).exists()},
               "self_sha256": sha(root / SELF) if (root / SELF).exists() else None,
               "products": [{"name": p.name, "script": p.script, "env": p.env, "out": p.out_dir, "prefix": p.prefix} for p in PRODUCTS],
               "manifest": {"path": m.name, "sha256": sha(m), "lines": len(m.read_text().splitlines())},
               "runtime": {"python": sys.version, "polars": pl.__version__, "platform": platform.platform(), "workers": workers, "threads": threads},
               "git_head": git_head(root)}
    name = "complete.json" if counts["input_missing"] == 0 else "incomplete.json"
    save(out / name, summary)
    log(f"{name}: {counts} in {summary['seconds']}s; manifest {summary['manifest']['lines']} lines")
    log_f.close()
    return summary


def git_head(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


# ---------------------------------------------------------------- coverage report
def coverage(dep: pl.DataFrame, kt1_dir: Path, visit_dir: Path, head_dir: Path, year: int) -> pl.DataFrame:
    """airport x month: share of departures with an identified airframe, a surface segment (coverage>=2), a stationary
    stand dwell (coverage 3), a visit record and a HEAD record."""
    def cat(pattern, cols=None):
        files = sorted(Path(pattern).parent.glob(Path(pattern).name))
        if not files:
            raise FileNotFoundError(pattern)
        return pl.concat([pl.read_parquet(f, columns=cols) for f in files]).unique("MVT_ID_mvt")
    ob = cat(f"{kt1_dir}/offblock2_{year}-*.parquet", ["MVT_ID_mvt", "coverage", "n_cand"])
    vi = cat(f"{visit_dir}/visit_{year}-*.parquet", ["MVT_ID_mvt", "vi_first_seen"])
    hd = cat(f"{head_dir}/head_{year}-*.parquet", ["MVT_ID_mvt", "hd_gs0"])
    d = dep.join(ob, on="MVT_ID_mvt", how="left").join(vi, on="MVT_ID_mvt", how="left").join(hd, on="MVT_ID_mvt", how="left")
    return d.group_by(ap=pl.col("ADEP_mvt"), month=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m")).agg(
        n=pl.len(), file_rows=pl.col("coverage").is_not_null().sum(),
        ident=(pl.col("n_cand") == 1).mean(), seg=(pl.col("coverage") >= 2).mean(), stand=(pl.col("coverage") == 3).mean(),
        visit=pl.col("vi_first_seen").is_not_null().mean(), head=pl.col("hd_gs0").is_not_null().mean()).sort("ap", "month")


def report(root: Path, out: Path, analysis_dir: Path) -> dict:
    status = out / "complete.json"
    if not status.exists():
        status = out / "incomplete.json"
    summary = json.loads(status.read_text())
    receipts = [json.loads(p.read_text()) for p in sorted((out / "receipts").glob("*.json"))]
    fin_dep = final_departures(root / FINAL_RANKING, summary["final_ranking"]["sha256"])
    fin = coverage(fin_dep, out / "data/adsblol_kt1", out / "data/adsblol", out / "data/adsblol", 2026)
    old_rank = pl.read_parquet(root / "data/prc-2026-datasets/ranking.parquet", columns=RANKING_COLS).filter(pl.col("PHASE_mvt") == "DEP")
    old = coverage(old_rank, root / "data/adsblol_kt1", root / "data/adsblol", root / "data/adsblol", 2026)
    tr = pl.concat([pl.read_parquet(f, columns=RANKING_COLS) for f in sorted((root / "data/prc-2026-datasets").glob("training_*.parquet"))]).filter(pl.col("PHASE_mvt") == "DEP")
    tr = tr.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in([1, 2, 6, 7]))
    y25 = coverage(tr, root / "data/adsblol_kt1", root / "data/adsblol", root / "data/adsblol", 2025)
    y25 = y25.with_columns(month=pl.col("month").str.replace("2025-", "2026-"))  # same calendar month, one year earlier

    j = fin.join(y25.select("ap", "month", seg25=pl.col("seg"), ident25=pl.col("ident"), visit25=pl.col("visit"), n25=pl.col("n")), on=["ap", "month"], how="left")
    j = j.join(old.select("ap", "month", seg_old=pl.col("seg"), ident_old=pl.col("ident"), visit_old=pl.col("visit"), n_old=pl.col("n")), on=["ap", "month"], how="left")
    j = j.with_columns(
        collapse=((pl.col("seg") < pl.col("seg25") - 0.20) | ((pl.col("seg25") >= 0.30) & (pl.col("seg") < 0.5 * pl.col("seg25")))),
        drift_vs_old=((pl.col("seg") - pl.col("seg_old")).abs() > 0.02) & pl.col("seg_old").is_not_null()).sort("ap", "month")

    flagged = j.filter(pl.col("collapse") | pl.col("drift_vs_old"))
    analysis_dir.mkdir(parents=True, exist_ok=True)
    j.write_csv(analysis_dir / "coverage_airport_month.csv")
    pct = lambda v: "—" if v is None else f"{100*v:.1f}"
    lines = [f"# Final daily ADS-B derivatives — coverage ({utc()})", "",
             f"Output root `{out}`; status file `{status.name}`: {summary['status_counts']}, {summary['days']} days, "
             f"{summary['departures']} final departures, {summary['departures_without_products']} without any product "
             f"(input_missing days: {summary['input_missing_days'] or 'none'}). Manifest `{summary['manifest']['path']}` "
             f"sha256 `{summary['manifest']['sha256']}` ({summary['manifest']['lines']} lines). Final ranking sha256 `{summary['final_ranking']['sha256']}`.", "",
             "Columns: `seg` = share of final departures with a surface segment in kt1/offblock2 (coverage>=2), `ident` = airframe "
             "identified (n_cand==1), `stand` = stationary dwell seen (coverage 3), `visit` = visit record, `head` = HEAD record. "
             "`*25` = same calendar month of 2025 from the historical products (the stage models' training coverage); `*old` = the "
             "old Jan/Jul 2026 products built on the old ranking MVT. `collapse` = seg < seg25 − 20 pp, or seg < ½·seg25 when seg25 ≥ 30 %. "
             "`drift_vs_old` = |seg − seg_old| > 2 pp on Jan/Jul.", "",
             "| airport | month | n | seg % | seg25 % | seg old % | ident % | ident25 % | stand % | visit % | visit25 % | head % | flags |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in j.iter_rows(named=True):
        flags = " ".join(f for f, on in (("COLLAPSE", r["collapse"]), ("drift-vs-old", r["drift_vs_old"])) if on)
        lines.append(f"| {r['ap']} | {r['month']} | {r['n']} | {pct(r['seg'])} | {pct(r['seg25'])} | {pct(r['seg_old'])} | {pct(r['ident'])} | "
                     f"{pct(r['ident25'])} | {pct(r['stand'])} | {pct(r['visit'])} | {pct(r['visit25'])} | {pct(r['head'])} | {flags} |")
    lines += ["", f"## Flags: {flagged.height} airport-months", ""]
    if flagged.height == 0:
        lines.append("No airport-month collapses against 2025 and no Jan/Jul drift beyond 2 pp against the old products.")
    for r in flagged.iter_rows(named=True):
        lines.append(f"- {r['ap']} {r['month']}: seg {pct(r['seg'])} % vs 2025 {pct(r['seg25'])} % vs old {pct(r['seg_old'])} %"
                     + (" — COLLAPSE" if r["collapse"] else "") + (" — drift vs old" if r["drift_vs_old"] else ""))
    tot = j.group_by("month").agg(n=pl.col("n").sum(), seg=(pl.col("seg") * pl.col("n")).sum() / pl.col("n").sum(),
                                   seg25=(pl.col("seg25") * pl.col("n")).sum() / pl.col("n").sum(),
                                   visit=(pl.col("visit") * pl.col("n")).sum() / pl.col("n").sum()).sort("month")
    lines += ["", "## Per month (departure-weighted)", "", "| month | n | seg % | seg25 % | visit % |", "|---|---|---|---|---|"]
    for r in tot.iter_rows(named=True):
        lines.append(f"| {r['month']} | {r['n']} | {pct(r['seg'])} | {pct(r['seg25'])} | {pct(r['visit'])} |")
    lines += ["", "## Per-day receipts", "", "| day | status | departures | identified | cov>=2 | visit | head rows | seconds |", "|---|---|---|---|---|---|---|---|"]
    for r in receipts:
        if r["status"] == "input_missing":
            lines.append(f"| {r['day']} | input_missing | {r['departures']} | — | — | — | — | — |")
        else:
            p = r["products"]
            lines.append(f"| {r['day']} | {r['status']} | {r['departures']} | {p['kt1_offblock2']['matched']} | {p['kt1_offblock2'].get('coverage_ge2')} | "
                         f"{p['visit']['matched']} | {p['head']['rows']} | {sum(x['seconds'] for x in p.values()):.0f} |")
    (analysis_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    res = {"flagged": flagged.select("ap", "month", "seg", "seg25", "seg_old", "collapse", "drift_vs_old").to_dicts(),
           "months": tot.to_dicts(), "status": summary["status_counts"]}
    save(analysis_dir / "coverage_summary.json", res)
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", default=DEFAULT_OUT)
    r.add_argument("--surface-dir", default=DEFAULT_SURFACE_DIR)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--threads", type=int, default=2, help="POLARS/OMP threads per producer process")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--days", nargs="*", help="subset of final days (smoke test); default = all 120")
    p = sub.add_parser("report")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--analysis", default="analysis/final_adsb_derivatives_20261008")
    a = ap.parse_args(argv)
    root = Path.cwd()
    if a.cmd == "run":
        run(root, Path(a.out), Path(a.surface_dir), a.workers, a.threads, a.resume, days=a.days)
    else:
        res = report(root, Path(a.out), Path(a.analysis))
        print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()

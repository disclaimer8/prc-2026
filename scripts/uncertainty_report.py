"""Calibrated predictive distributions from the regime mixture (docs/uncertainty.md).

    .venv/bin/python scripts/uncertainty_report.py --data-root data

Run from the repository root. No model is trained; nothing is written outside --tables / --figures.
Every number used in the document is printed with a `[U..]` prefix or written to a CSV table.

The point prediction of src/train.py is a mixture: six regime probabilities `proba` (classes in
`train.CLASSES` order: regular, sched, iobt, lobt, eobt, dayshift) over candidate values `values`
(regular = base regressor `reg`; planned regimes = take-off minus the reported planning time; dayshift =
reg + 86,400 s). This script turns the same arrays into a full predictive distribution per flight:

    F(t) = sum_{k>=1} p_k 1[v_k <= t]  +  p_0 * G_airport(t - c_0)

where G_airport is the empirical distribution of residuals y - c_0 on flights whose true regime is
regular, stored as a 99-point quantile table per airport. For the out-of-fold evaluation the table of a
held-out fold is fitted on the other five folds only (leave-one-fold-out), so no row sees its own residual.

Two centres c_0 of the regular regime are evaluated:
  base   c_0 = reg, the base-regressor candidate inside the registered mixture (cv_preds_fin1b_hz)
  chain  c_0 chosen so that the mixture mean equals the production out-of-fold point forecast `joint`
         (e195_pre_e81_20261007_v2/predictions.parquet): c_0 = (joint - sum_{k>=1} p_k v_k) / p_0 when
         p_0 >= 0.2, otherwise reg. This makes the median of the distribution consistent with the chain.

Reported on the 2025 out-of-fold rows: coverage of central 50/80/90 % intervals, PIT histograms per
airport and regime, CRPS against the point forecast (whose CRPS is its absolute error) and against a
fold-honest per-airport climatology. The reusable function is `predictive_quantiles`.

Inputs (paths under --data-root):
  cv_preds_fin1b_hz/fold_*.npz                 mvt_id, pred, reg, y, proba (n,6), values (n,6)
  features_fin1b.parquet                       airport, month, planning-time differences (regime labels)
  e195_pre_e81_20261007_v2/predictions.parquet column `joint` for the chain-consistent centre
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
REGIME_ENV = {"REGIME_TOL": "6", "REGIME_EXCLUDE": "EDDF,EHAM,LSZH", "SCHED_TOL_LONG": "0"}
ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
QGRID = np.arange(1, 100) / 100.0  # 99 residual quantiles per airport
QS_REPORT = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
CHUNK = 100_000
P0_MIN_CHAIN = 0.2

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
C50, C80, C90 = "#0d366b", "#256abf", "#86b6ef"  # ordinal blue ramp: nested intervals, dark = narrowest
ACCENT, CONTEXT = "#2a78d6", "#c3c2b7"
DPI = 180


# ----------------------------------------------------------------------------- distribution

def fit_residual_table(resid: np.ndarray, airports: np.ndarray, qgrid: np.ndarray = QGRID) -> dict[str, np.ndarray]:
    """Per-airport residual quantiles on the given regular-regime rows: {airport: array(len(qgrid))}."""
    out = {}
    for ap in np.unique(airports):
        r = resid[airports == ap]
        out[ap] = np.quantile(r, qgrid) if len(r) else np.zeros(len(qgrid))
    return out


def mixture_atoms(proba: np.ndarray, values: np.ndarray, airports: np.ndarray, table: dict[str, np.ndarray],
                  centre: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Weighted atoms (sorted) of the predictive distribution for each row.

    Returns (x, w) with shape (n, 5 + m): the five non-regular candidates with their probabilities and m
    equally weighted residual quantiles around the regular centre. Rows are sorted by x; w sums to 1.
    """
    n, k = proba.shape
    m = len(next(iter(table.values())))
    c0 = values[:, 0].astype(float) if centre is None else centre.astype(float)
    res = np.empty((n, m))
    for ap, q in table.items():
        res[airports == ap] = q
    missing = ~np.isin(airports, list(table))
    if missing.any():  # unseen airport: no residual spread, point mass at the centre
        res[missing] = 0.0
    x = np.concatenate([np.nan_to_num(values[:, 1:].astype(float)), c0[:, None] + res], axis=1)
    w = np.concatenate([proba[:, 1:].astype(float), np.repeat(proba[:, :1].astype(float) / m, m, axis=1)], axis=1)
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    order = np.argsort(x, axis=1, kind="stable")
    return np.take_along_axis(x, order, 1), np.take_along_axis(w, order, 1)


def quantiles_from_atoms(x: np.ndarray, w: np.ndarray, qs) -> np.ndarray:
    """Lower quantile function of the discrete distribution: smallest atom with cumulative weight >= q."""
    W = np.cumsum(w, axis=1)
    out = np.empty((x.shape[0], len(qs)))
    for j, q in enumerate(qs):
        idx = np.minimum((W < q - 1e-12).sum(axis=1), x.shape[1] - 1)
        out[:, j] = x[np.arange(x.shape[0]), idx]
    return out


def crps_from_atoms(x: np.ndarray, w: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS = E|X - y| - 0.5 E|X - X'| for a discrete distribution with sorted atoms."""
    e1 = (w * np.abs(x - y[:, None])).sum(axis=1)
    W = np.cumsum(w, axis=1)
    e2 = 2.0 * (x * w * (2.0 * W - w - 1.0)).sum(axis=1)
    return e1 - 0.5 * e2


def pit_from_atoms(x: np.ndarray, w: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomised probability integral transform F(y-) + V (F(y) - F(y-)), V ~ U(0, 1)."""
    f_lo = (w * (x < y[:, None])).sum(axis=1)
    f_hi = (w * (x <= y[:, None])).sum(axis=1)
    return f_lo + rng.random(len(y)) * (f_hi - f_lo)


def predictive_quantiles(proba: np.ndarray, values: np.ndarray, airports: np.ndarray, table: dict[str, np.ndarray],
                         qs=(0.1, 0.5, 0.9), centre: np.ndarray | None = None, chunk: int = CHUNK) -> np.ndarray:
    """Quantiles (default 10/50/90 %) per flight from the mixture arrays and a residual table.

    proba, values: (n, 6) arrays as saved by src/train.py (classes regular, sched, iobt, lobt, eobt, dayshift;
    probability of a missing candidate is already zero). airports: (n,) ICAO codes. table: output of
    fit_residual_table, or the pooled table in docs/tables/uncertainty_residual_quantiles.csv
    (fold_excluded == 'none'). centre: optional replacement of the regular candidate values[:, 0].
    """
    out = np.empty((len(airports), len(qs)))
    for s in range(0, len(airports), chunk):
        sl = slice(s, s + chunk)
        x, w = mixture_atoms(proba[sl], values[sl], airports[sl], table, None if centre is None else centre[sl])
        out[sl] = quantiles_from_atoms(x, w, qs)
    return out


def chain_centre(proba: np.ndarray, values: np.ndarray, joint: np.ndarray) -> np.ndarray:
    """Regular candidate that makes the mixture mean equal `joint`; falls back to reg when p_0 < P0_MIN_CHAIN."""
    p0 = proba[:, 0].astype(float)
    rest = (proba[:, 1:].astype(float) * np.nan_to_num(values[:, 1:].astype(float))).sum(axis=1)
    c = np.where(p0 >= P0_MIN_CHAIN, (joint - rest) / np.where(p0 > 0, p0, 1.0), values[:, 0].astype(float))
    return c


# ----------------------------------------------------------------------------- load

def load(root: Path) -> pl.DataFrame:
    files = sorted(glob.glob(str(root / "cv_preds_fin1b_hz" / "fold_*.npz")))
    if len(files) != 6:
        sys.exit(f"expected six fold files in {root / 'cv_preds_fin1b_hz'}, found {len(files)}")
    parts = []
    for fn in files:
        d = np.load(fn)
        fold = Path(fn).stem.replace("fold_", "")
        parts.append(pl.DataFrame({ID: d["mvt_id"], "pred_base": d["pred"], "y": d["y"].astype(np.float64),
                                   "fold": np.repeat(fold, len(d["y"])),
                                   **{f"p{k}": d["proba"][:, k].astype(np.float64) for k in range(6)},
                                   **{f"v{k}": d["values"][:, k].astype(np.float64) for k in range(6)}}))
    b = pl.concat(parts)
    assert b.height == 2_085_047 and b[ID].n_unique() == b.height, "unexpected OOF inventory"
    f = pl.read_parquet(root / "features_fin1b.parquet",
                        columns=[ID, "split", Y, "ADEP_mvt", "month", "FLIGHT_ID_mvt",
                                 "d_mvt_sched", "d_mvt_iobt", "d_mvt_lobt", "d_mvt_eobt"]).filter(pl.col("split") == "train")
    x = b.join(f.drop("split"), on=ID, how="left", validate="1:1")
    assert x["ADEP_mvt"].null_count() == 0 and (x["y"] == x[Y].cast(pl.Float64)).all()
    p = pl.read_parquet(root / "e195_pre_e81_20261007_v2" / "predictions.parquet", columns=[ID, "joint", "evaluation_mask"])
    x = x.join(p, on=ID, how="left", validate="1:1")
    os.environ.update(REGIME_ENV)
    sys.path.insert(0, str(REPO / "src"))
    import train as T  # noqa: E402

    lab = T.regime_label(x.select(Y, "ADEP_mvt", "d_mvt_sched", "d_mvt_iobt", "d_mvt_lobt", "d_mvt_eobt"))
    x = x.with_columns(regime=pl.Series(np.array(list(T.CLASSES), dtype=object)[lab].astype(str)),
                       has_nm=pl.col("FLIGHT_ID_mvt").is_not_null(),
                       joint=pl.col("joint").cast(pl.Float64))
    return x


# ----------------------------------------------------------------------------- evaluation

def evaluate(x: pl.DataFrame, variant: str, rng: np.random.Generator):
    """Leave-one-fold-out residual tables, then per-row quantiles, coverage flags, PIT and CRPS."""
    n = x.height
    proba = x.select([f"p{k}" for k in range(6)]).to_numpy()
    values = x.select([f"v{k}" for k in range(6)]).to_numpy()
    airports = x["ADEP_mvt"].to_numpy().astype(str)
    folds = x["fold"].to_numpy().astype(str)
    y = x["y"].to_numpy()
    regular = x["regime"].to_numpy().astype(str) == "regular"
    centre = values[:, 0].copy() if variant == "base" else chain_centre(proba, values, x["joint"].to_numpy())
    resid_all = y - centre
    tables = {}
    for f in np.unique(folds):
        m = (folds != f) & regular
        tables[f] = fit_residual_table(resid_all[m], airports[m])
    tables["none"] = fit_residual_table(resid_all[regular], airports[regular])
    # climatology baseline: fold-honest per-airport quantiles of y itself (equal weights)
    clim = {f: fit_residual_table(y[folds != f], airports[folds != f]) for f in np.unique(folds)}

    Q = np.empty((n, len(QS_REPORT)))
    crps = np.empty(n)
    crps_clim = np.empty(n)
    pit = np.empty(n)
    for f in np.unique(folds):
        idx = np.where(folds == f)[0]
        for s in range(0, len(idx), CHUNK):
            ii = idx[s:s + CHUNK]
            xa, wa = mixture_atoms(proba[ii], values[ii], airports[ii], tables[f], centre[ii])
            Q[ii] = quantiles_from_atoms(xa, wa, QS_REPORT)
            crps[ii] = crps_from_atoms(xa, wa, y[ii])
            pit[ii] = pit_from_atoms(xa, wa, y[ii], rng)
            # climatology atoms: 99 quantiles of y at this airport on the other folds
            xc = np.stack([clim[f][a] for a in airports[ii]])
            wc = np.full_like(xc, 1.0 / xc.shape[1])
            crps_clim[ii] = crps_from_atoms(xc, wc, y[ii])
    point = x["pred_base"].to_numpy() if variant == "base" else x["joint"].to_numpy()
    cols = {
        "q05": Q[:, 0], "q10": Q[:, 1], "q25": Q[:, 2], "q50": Q[:, 3], "q75": Q[:, 4], "q90": Q[:, 5], "q95": Q[:, 6],
        "in50": (y >= Q[:, 2]) & (y <= Q[:, 4]), "in80": (y >= Q[:, 1]) & (y <= Q[:, 5]), "in90": (y >= Q[:, 0]) & (y <= Q[:, 6]),
        "below90": y < Q[:, 0], "above90": y > Q[:, 6],
        "w50": Q[:, 4] - Q[:, 2], "w80": Q[:, 5] - Q[:, 1], "w90": Q[:, 6] - Q[:, 0],
        "crps": crps, "crps_clim": crps_clim, "pit": pit,
        "ae_point": np.abs(point - y), "se_point": (point - y) ** 2, "se_median": (Q[:, 3] - y) ** 2,
    }
    return x.with_columns(**{k: pl.Series(v) for k, v in cols.items()}), tables


def summarise(e: pl.DataFrame, by: list[str] | None) -> pl.DataFrame:
    agg = [pl.len().alias("rows"),
           pl.col("in50").mean().alias("cov50"), pl.col("in80").mean().alias("cov80"), pl.col("in90").mean().alias("cov90"),
           pl.col("below90").mean().alias("below_q05"), pl.col("above90").mean().alias("above_q95"),
           pl.col("w50").median().alias("width50_median"), pl.col("w80").median().alias("width80_median"),
           pl.col("w90").median().alias("width90_median"),
           pl.col("crps").mean().alias("crps"), pl.col("ae_point").mean().alias("crps_point_mae"),
           pl.col("crps_clim").mean().alias("crps_climatology"),
           pl.col("se_point").mean().sqrt().alias("rmse_point"), pl.col("se_median").mean().sqrt().alias("rmse_median")]
    g = e.group_by(by).agg(agg).sort(by) if by else e.select(agg)
    return g.with_columns(crps_skill_vs_point=1 - pl.col("crps") / pl.col("crps_point_mae"),
                          crps_skill_vs_climatology=1 - pl.col("crps") / pl.col("crps_climatology"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data", type=Path)
    ap.add_argument("--tables", default=REPO / "docs" / "tables", type=Path)
    ap.add_argument("--figures", default=REPO / "docs" / "figures", type=Path)
    ap.add_argument("--seed", default=20261008, type=int)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    a.tables.mkdir(parents=True, exist_ok=True)
    a.figures.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    x = load(a.data_root)
    x = x.filter(pl.col("evaluation_mask"))  # same 2,084,921 rows as docs/validation.md
    print(f"[U1] {x.height:,} evaluated OOF rows; regime shares: "
          + ", ".join(f"{r['regime']} {100 * r['len'] / x.height:.2f}%" for r in x.group_by("regime").len().sort("len", descending=True).to_dicts()))
    print(f"[U1] mixture point RMSE: base {np.sqrt((x['pred_base'] - x['y']).pow(2).mean()):.2f} s, "
          f"chain joint {np.sqrt((x['joint'] - x['y']).pow(2).mean()):.2f} s")
    p0 = x["p0"].to_numpy()
    print(f"[U1] chain centre falls back to reg on {100 * (p0 < P0_MIN_CHAIN).mean():.2f}% of rows (p_regular < {P0_MIN_CHAIN})")

    summaries, pits, resid_rows = [], [], []
    for variant in ("base", "chain"):
        e, tables = evaluate(x, variant, rng)
        e = e.with_columns(group=pl.when(pl.col("month").is_in([1, 7])).then(pl.lit("Jan+Jul"))
                           .when(pl.col("month").is_in([2, 6])).then(pl.lit("Feb+Jun")).otherwise(pl.lit("other")))
        tot = summarise(e, None).with_columns(level=pl.lit("all"), key=pl.lit("all"))
        parts = [tot]
        for level, col in (("airport", "ADEP_mvt"), ("regime", "regime"), ("month", "month"), ("group", "group"),
                           ("nm", "has_nm"), ("body", None)):
            if level == "body":
                s = summarise(e.filter(pl.col("y") <= 7200), None).with_columns(level=pl.lit("body"), key=pl.lit("y<=7200"))
            else:
                s = summarise(e, [col]).with_columns(level=pl.lit(level), key=pl.col(col).cast(pl.String)).drop(col)
            parts.append(s)
        summ = pl.concat(parts, how="diagonal").with_columns(variant=pl.lit(variant))
        summaries.append(summ)
        t = tot.to_dicts()[0]
        print(f"[U2] {variant:<5} all rows: coverage 50/80/90 = {100 * t['cov50']:.1f} / {100 * t['cov80']:.1f} / {100 * t['cov90']:.1f} %; "
              f"below q05 {100 * t['below_q05']:.1f} %, above q95 {100 * t['above_q95']:.1f} %; median widths {t['width50_median']:.0f}/{t['width80_median']:.0f}/{t['width90_median']:.0f} s; "
              f"CRPS {t['crps']:.1f} s vs point MAE {t['crps_point_mae']:.1f} s (skill {100 * t['crps_skill_vs_point']:.1f} %) "
              f"vs climatology {t['crps_climatology']:.1f} s (skill {100 * t['crps_skill_vs_climatology']:.1f} %); RMSE point {t['rmse_point']:.2f}, median {t['rmse_median']:.2f}")
        for lvl in ("airport", "regime", "group", "nm", "body"):
            for r in summ.filter(pl.col("level") == lvl).to_dicts():
                print(f"[U3] {variant:<5} {lvl:<7} {str(r['key']):<8} rows {r['rows']:>9,}  cov50/80/90 {100 * r['cov50']:5.1f} {100 * r['cov80']:5.1f} {100 * r['cov90']:5.1f}  "
                      f"w80 {r['width80_median']:6.0f}  CRPS {r['crps']:6.1f}  MAE {r['crps_point_mae']:6.1f}  clim {r['crps_climatology']:6.1f}  "
                      f"skill {100 * r['crps_skill_vs_point']:5.1f}%")
        mon = summ.filter(pl.col("level") == "month").sort(pl.col("key").cast(pl.Int32))
        print(f"[U3] {variant:<5} month   cov80: " + " ".join(f"{MONTH_ABBR[int(r['key']) - 1]} {100 * r['cov80']:.1f}" for r in mon.to_dicts()))
        print(f"[U3] {variant:<5} month   cov90: " + " ".join(f"{MONTH_ABBR[int(r['key']) - 1]} {100 * r['cov90']:.1f}" for r in mon.to_dicts()))
        print(f"[U3] {variant:<5} month   CRPS : " + " ".join(f"{MONTH_ABBR[int(r['key']) - 1]} {r['crps']:.1f}" for r in mon.to_dicts()))
        # PIT histograms (10 bins) per airport and per regime
        e = e.with_columns(pit_bin=(pl.col("pit") * 10).floor().clip(0, 9).cast(pl.Int32))
        for level, col in (("airport", "ADEP_mvt"), ("regime", "regime"), ("all", None)):
            if col is None:
                h = e.group_by("pit_bin").len().with_columns(key=pl.lit("all"))
            else:
                h = e.group_by(col, "pit_bin").len().with_columns(key=pl.col(col).cast(pl.String)).drop(col)
            h = h.with_columns(share=pl.col("len") / pl.col("len").sum().over("key"), level=pl.lit(level), variant=pl.lit(variant))
            pits.append(h.select("variant", "level", "key", "pit_bin", pl.col("len").alias("rows"), "share"))
        hall = pits[-1].sort("pit_bin")
        print(f"[U4] {variant:<5} PIT deciles, all rows (uniform = 10.0 each): " + " ".join(f"{100 * s:.1f}" for s in hall["share"].to_list()))
        # residual tables (CSV): leave-one-fold-out tables used above and the pooled table for deployment
        for fexc, tab in tables.items():
            for ap_, q in tab.items():
                for qq, v in zip(QGRID, q):
                    resid_rows.append({"variant": variant, "airport": ap_, "fold_excluded": fexc, "q": round(float(qq), 2), "residual_s": float(v)})
        pooled = tables["none"]
        print(f"[U5] {variant:<5} pooled regular residual quantiles q10/q50/q90 by airport: "
              + ", ".join(f"{ap_} {pooled[ap_][9]:.0f}/{pooled[ap_][49]:.0f}/{pooled[ap_][89]:.0f}" for ap_ in AIRPORTS))
        if variant == "chain":
            e_chain = e

    summary = pl.concat(summaries, how="diagonal").select(
        "variant", "level", "key", "rows", "cov50", "cov80", "cov90", "below_q05", "above_q95",
        "width50_median", "width80_median", "width90_median", "crps", "crps_point_mae", "crps_climatology",
        "crps_skill_vs_point", "crps_skill_vs_climatology", "rmse_point", "rmse_median")
    summary.write_csv(a.tables / "uncertainty_coverage.csv", float_precision=4)
    pl.concat(pits).sort("variant", "level", "key", "pit_bin").write_csv(a.tables / "uncertainty_pit.csv", float_precision=5)
    pl.DataFrame(resid_rows).write_csv(a.tables / "uncertainty_residual_quantiles.csv", float_precision=2)

    # sanity: the exported function reproduces the evaluation quantiles on a sample (pooled table, chain centre)
    smp = e_chain.head(5000)
    pr = smp.select([f"p{k}" for k in range(6)]).to_numpy()
    va = smp.select([f"v{k}" for k in range(6)]).to_numpy()
    tab = {ap_: np.array([r["residual_s"] for r in resid_rows if r["variant"] == "chain" and r["fold_excluded"] == "none" and r["airport"] == ap_]) for ap_ in AIRPORTS}
    q = predictive_quantiles(pr, va, smp["ADEP_mvt"].to_numpy().astype(str), tab, centre=chain_centre(pr, va, smp["joint"].to_numpy()))
    print(f"[U6] predictive_quantiles check on 5,000 rows (pooled table): median of q50 {np.median(q[:, 1]):.0f} s, "
          f"mean 80% width {np.mean(q[:, 2] - q[:, 0]):.0f} s; RMSE of q50 vs y {np.sqrt(np.mean((q[:, 1] - smp['y'].to_numpy()) ** 2)):.1f} s")

    if not a.no_figures:
        fig_calibration(summary, pl.concat(pits), a.figures / "fig_uncertainty_calibration.png")
    print("[done] tables in", a.tables, "figures in", a.figures)


# ----------------------------------------------------------------------------- figure

def fig_calibration(summary: pl.DataFrame, pits: pl.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.labelsize": 9.5,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK,
        "axes.titlepad": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False, "legend.fontsize": 9,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "figure.dpi": DPI,
    })
    s = summary.filter((pl.col("variant") == "chain") & (pl.col("level") == "airport")).sort("key")
    fig = plt.figure(figsize=(11.5, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=0.75)
    ax = fig.add_subplot(gs[0])
    ax.set_title("Coverage of central intervals by airport, chain-centred mixture (2025 out-of-fold)", loc="left")
    xs = np.arange(len(AIRPORTS))
    for nominal, col, colour, lab in ((0.5, "cov50", C50, "50 % interval"), (0.8, "cov80", C80, "80 % interval"), (0.9, "cov90", C90, "90 % interval")):
        ax.axhline(nominal, color=GRID, lw=0.8, zorder=0)
        vals = s[col].to_numpy()
        ax.plot(xs, vals, "o", color=colour, ms=7, label=lab, zorder=3)
        ax.text(len(AIRPORTS) - 0.55, nominal, f"nominal {int(100 * nominal)} %", va="bottom", ha="right", fontsize=8, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels(s["key"].to_list())
    ax.set_ylim(0.3, 1.0)
    ax.set_yticks([0.3, 0.5, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["30 %", "50 %", "80 %", "90 %", "100 %"])
    ax.set_ylabel("share of labels inside the interval")
    ax.legend(loc="lower left", ncol=3)
    # PIT per airport (small multiples)
    inner = gs[1].subgridspec(2, 5, hspace=0.6, wspace=0.25)
    ph = pits.filter((pl.col("variant") == "chain") & (pl.col("level") == "airport"))
    for i, ap_ in enumerate(AIRPORTS):
        axp = fig.add_subplot(inner[i // 5, i % 5])
        h = ph.filter(pl.col("key") == ap_).sort("pit_bin")
        axp.bar(h["pit_bin"].to_numpy() + 0.5, h["share"].to_numpy(), width=0.86, color=ACCENT, linewidth=0)
        axp.axhline(0.1, color=INK2, lw=0.8)
        axp.set_xlim(0, 10)
        axp.set_ylim(0, 0.25)
        axp.set_xticks([0, 5, 10])
        axp.set_xticklabels(["0", "0.5", "1"])
        axp.set_yticks([0, 0.1, 0.2])
        axp.set_yticklabels(["0", "10 %", "20 %"] if i % 5 == 0 else ["", "", ""])
        axp.set_title(ap_, loc="left", fontsize=9.5, pad=3)
        axp.grid(axis="y", color=GRID, lw=0.5)
        axp.set_axisbelow(True)
    fig.text(0.125, 0.515, "PIT histograms by airport (10 bins; line = uniform 10 %). Bars above the line in the outer bins mean the "
             "distribution is too narrow there; a hump in the middle means too wide.", ha="left", fontsize=8.5, color=INK2)
    fig.text(0.01, 0.01, "Residual quantiles per airport fitted on the other five folds; planned-regime candidates are point masses. "
             "2025 out-of-fold rows, not the 2026 final.", fontsize=8, color=INK2)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()

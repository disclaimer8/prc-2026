"""Figures for docs/data-insights.md: the data properties that shaped the model.

    .venv/bin/python scripts/make_doc_figures.py --data-root data

Run from the repository root. No model is trained and nothing is written outside --out.
Every number drawn in a figure is also printed, so the document can quote it.

Inputs (paths under --data-root unless noted):
  F1  docs/tables/leaderboard_gentle-igloo.csv (repository file, --leaderboard)
  F2  prc-2026-datasets/training_*.parquet
  F3  prc-2026-datasets/training_*.parquet
  F4  prc-2026-datasets/training_*.parquet (January and July 2025), prc-2026-datasets/ranking.parquet,
      adsblol_kt1/offblock2_<day>.parquet (surface-track extraction read by the production ADS-B stages)
  F5  e195_pre_e81_20261007_v2/predictions.parquet (out-of-fold predictions, column `joint`),
      the training parquets (regime labels) and adsblol_kt1/offblock2_2025-*.parquet
A figure whose inputs are missing is skipped with a message; the others are still produced.

Recording-regime labels come from src/train.py itself (`regime_label`), with the registered base
settings pinned below, so this figure cannot drift from the labels the model was trained on.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("POLARS_MAX_THREADS", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path as MPath

REPO = Path(__file__).resolve().parents[1]
# Registered base settings of src/train.py (docs/model-design.md, "Recording-regime mixture").
REGIME_ENV = {"REGIME_TOL": "6", "REGIME_EXCLUDE": "EDDF,EHAM,LSZH", "SCHED_TOL_LONG": "0"}

Y = "TAXITIME_SEC_mvt"
ID = "MVT_ID_mvt"
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
DPI = 180

# Colours: the validated reference palette of the dataviz method, checked against a white surface.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
ACCENT = "#2a78d6"  # single-series / "share of squared error" colour
CONTEXT = "#c3c2b7"  # de-emphasised "share of rows" colour
RAMP = ["#86b6ef", "#5598e7", "#256abf", "#184f95", "#0d366b"]  # ordinal blue ramp, light -> dark
YEAR_COLOURS = {2025: "#86b6ef", 2026: "#184f95"}
# planned-time regimes in src/train.py CLASSES order; categorical slots 2-6 (adjacent pairs validated)
REGIME_COLOURS = {"sched": "#eb6834", "iobt": "#1baf7a", "lobt": "#eda100", "eobt": "#e87ba4", "dayshift": "#008300"}
REGIME_NAMES = {"regular": "regular", "sched": "schedule (SCHED)", "iobt": "initial OBT (IOBT)",
                "lobt": "last OBT (LOBT)", "eobt": "estimated OBT (EOBT_1)", "dayshift": "one-day shift"}
REGIME_SHORT = {"regular": "regular", "sched": "SCHED", "iobt": "IOBT", "lobt": "LOBT", "eobt": "EOBT_1",
                "dayshift": "day shift"}
MINUS = "−"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.titlepad": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9.5,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "figure.dpi": DPI,
    "lines.solid_capstyle": "round",
})


# ----------------------------------------------------------------------------- helpers

def fmt_share(s: float, n: int | None = None) -> str:
    """Percent with precision that fits the size; very small shares are given as a row count."""
    if n is not None and s < 0.001:
        return f"{n:,} row" + ("" if n == 1 else "s")
    if s >= 0.1:
        return f"{100 * s:.0f}%" if s >= 0.995 else f"{100 * s:.1f}%"
    if s >= 0.01:
        return f"{100 * s:.1f}%"
    if s >= 0.001:
        return f"{100 * s:.2f}%"
    return "<0.1%" if s > 0 else "0%"


def px(ax) -> tuple[float, float]:
    """Data units per output pixel along x and y (linear axes, limits already fixed)."""
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(0.0, 0.0), (1.0, 1.0)])
    return abs(x1 - x0), abs(y1 - y0)


def hbar(ax, yc, x0, x1, h, colour, round_end=True, radius_pt=3.0, zorder=3):
    """Horizontal bar, square at the baseline, rounded data end (radius in points)."""
    if x1 - x0 <= 0:
        return
    ux, uy = px(ax)
    r = radius_pt * ax.figure.dpi / 72.0
    rx, ry = min(r * ux, x1 - x0), min(r * uy, h / 2)
    yb, yt = yc - h / 2, yc + h / 2
    if round_end:
        verts = [(x0, yb), (x1 - rx, yb), (x1, yb), (x1, yb + ry), (x1, yt - ry), (x1, yt), (x1 - rx, yt), (x0, yt), (x0, yb)]
        codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
                 MPath.LINETO, MPath.CLOSEPOLY]
    else:
        verts = [(x0, yb), (x1, yb), (x1, yt), (x0, yt), (x0, yb)]
        codes = [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=colour, edgecolor="none", zorder=zorder))


def stacked_hbar(ax, yc, widths, colours, h, gap_pt=1.5):
    """Left-anchored stacked bar; a surface gap separates segments, only the last segment is rounded."""
    ux, _ = px(ax)
    gap = gap_pt * ax.figure.dpi / 72.0 * ux
    segs, x = [], 0.0
    for w, c in zip(widths, colours):
        if w > 0:
            segs.append((x, x + w, c))
        x += w
    for i, (a, b, c) in enumerate(segs):
        last = i == len(segs) - 1
        end = b if last else b - gap
        if end - a > 0.6 * ux:
            hbar(ax, yc, a, end, h, c, round_end=last)
    return x


def paired_bars(ax, items, thick_pt=7.0, gap_pt=2.0, fontsize=7.8):
    """Per category a grey 'share of rows' bar above a blue 'share of error' bar, both direct-labelled.

    items: list of (row_share, error_share, n_rows) in category order. Thickness is fixed in points so
    panels with different category counts draw identical bars.
    """
    _, uy = px(ax)
    t = thick_pt * ax.figure.dpi / 72.0 * uy
    g = gap_pt * ax.figure.dpi / 72.0 * uy
    xmax = ax.get_xlim()[1]
    for i, (row, err, n) in enumerate(items):
        for val, yc, col, ink, label in [(row, i - (t + g) / 2, CONTEXT, MUTED, fmt_share(row, n)),
                                         (err, i + (t + g) / 2, ACCENT, INK, fmt_share(err))]:
            hbar(ax, yc, 0, val, t, col)
            if val > 0.86 * xmax:  # no room past the end: label inside, ink chosen by the fill
                ax.text(val - 0.015 * xmax, yc, label, ha="right", va="center", fontsize=fontsize,
                        color=INK if col == CONTEXT else SURFACE, zorder=4)
            else:
                ax.text(val + 0.012 * xmax, yc, label, ha="left", va="center", fontsize=fontsize, color=ink, zorder=4)


def note(ax, x, y, text, **kw):
    """Muted annotation on a surface-coloured backing so gridlines do not cut through it."""
    kw = {"va": "center", "fontsize": 8.5, "color": MUTED, **kw}
    ax.text(x, y, text, bbox=dict(fc=SURFACE, ec="none", pad=1.2), zorder=4, **kw)


def ring_dot(ax, x, y, colour, size=7.5, zorder=5):
    ax.plot([x], [y], "o", ms=size, mfc=colour, mec=SURFACE, mew=1.6, zorder=zorder, clip_on=False)


def style_value_axis(ax, xlim, ticks, ticklabels=None, xlabel=None, grid=True):
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    if ticklabels is not None:
        ax.set_xticklabels(ticklabels)
    if grid:
        for t in ticks:
            ax.axvline(t, color=GRID, lw=0.8, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=3, width=0.8)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=6)


def category_axis(ax, labels, pad=0.6):
    n = len(labels)
    ax.set_ylim(n - 1 + pad, -pad)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)


def save(fig, out: Path, name: str) -> None:
    path = out / name
    fig.savefig(path, dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    print(f"wrote {path}")


def need(paths: list[Path], figure: str) -> bool:
    missing = [str(p) for p in paths if not (p.exists() if "*" not in p.name else glob.glob(str(p)))]
    if missing:
        print(f"SKIP {figure}: missing input(s): {', '.join(missing)}")
        return False
    return True


# ----------------------------------------------------------------------------- data

def regime_module():
    os.environ.update(REGIME_ENV)
    sys.path.insert(0, str(REPO / "src"))
    import features as F  # noqa: E402  (src/features.py: _secs, the interval definition used in training)
    import train as T  # noqa: E402  (src/train.py: regime_label, CLASSES)
    assert T.TOL == 6 and T.PLANNED_REGIME_EXCLUDED == {"EDDF", "EHAM", "LSZH"}, "regime settings drift"
    return F, T


def load_departures(root: Path) -> pl.DataFrame:
    """All 2025 departures with the planning-time intervals of src/features.py and the src/train.py regime label."""
    F, T = regime_module()
    cols = [ID, "FLIGHT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
            "SCHED_TIME_UTC_mvt", "IOBT_flt", "LOBT_flt", "EOBT_1_flt", "AOBT_3_flt", Y]
    files = sorted(glob.glob(str(root / "prc-2026-datasets" / "training_*.parquet")))
    raw = pl.concat([pl.read_parquet(f, columns=cols) for f in files]).filter(pl.col("PHASE_mvt") == "DEP")
    df = raw.with_columns(
        d_mvt_aobt=F._secs("MVT_TIME_UTC_mvt", "AOBT_3_flt"),
        d_mvt_eobt=F._secs("MVT_TIME_UTC_mvt", "EOBT_1_flt"),
        d_mvt_lobt=F._secs("MVT_TIME_UTC_mvt", "LOBT_flt"),
        d_mvt_iobt=F._secs("MVT_TIME_UTC_mvt", "IOBT_flt"),
        d_mvt_sched=F._secs("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
        month=pl.col("MVT_TIME_UTC_mvt").dt.month(),
        y=pl.col(Y).cast(pl.Float64),
    )
    lab = T.regime_label(df)
    df = df.with_columns(regime=pl.Series(np.array(T.CLASSES)[lab]))
    assert df[ID].n_unique() == df.height
    print(f"[data] {len(files)} training files, {df.height:,} departures of 2025; regime settings {REGIME_ENV}; "
          f"classes {T.CLASSES}")
    return df


def surface_coverage(root: Path, pattern: str) -> pl.DataFrame:
    files = sorted(glob.glob(str(root / "adsblol_kt1" / pattern)))
    c = pl.concat([pl.read_parquet(f, columns=[ID, "coverage"]) for f in files])
    assert c[ID].n_unique() == c.height
    return c


# ----------------------------------------------------------------------------- F1

MILESTONES = {  # version -> short label (sources: HYPOTHESES_LOG.md, docs/model-design.md)
    2: "regime mixture",
    8: "NM-proxy residual + LIRF tail table",
    13: "TabPFN regime probabilities, LIRF tail",
    27: "own-aircraft ADS-B surface stage (E25)",
    316: "airline prefix in TabPFN tail",
    12075: "ordered-trajectory member (E203)",
}
# text position (days after 13 Sep 00:00 UTC, RMSE) and panel for each label
LABEL_POS = {2: (2.2, 420, "top", "left"), 8: (2.6, 285.6, "bot", "left"), 13: (4.3, 275.6, "bot", "left"),
             27: (8.6, 259.5, "bot", "left"), 316: (12.6, 247.0, "bot", "left"),
             12075: (24.7, 240.5, "bot", "right")}


def fig1(lb_path: Path, out: Path) -> None:
    if not need([lb_path], "F1"):
        return
    lb = (pl.read_csv(lb_path)
          .with_columns(t=pl.col("processed_at_utc").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_zone="UTC"),
                        v=pl.col("file").str.extract(r"_v(\d+)\.parquet", 1).cast(pl.Int64))
          .sort("t", maintain_order=True))
    lb = lb.with_columns(best=pl.col("rmse_s").cum_min())
    assert np.allclose(lb["best"].to_numpy(), lb["best_so_far_s"].to_numpy()), "best_so_far_s column disagrees"
    prev = lb["best"].shift(1)
    lb = lb.with_columns(step=(pl.col("best") - prev).fill_null(0.0))
    improving = lb.filter(pl.col("step") < 0)
    print(f"[F1] {lb.height} scored submissions {lb['t'].min()} .. {lb['t'].max()}; first {lb['rmse_s'][0]:.4f} s, "
          f"best {lb['best'][-1]:.4f} s (v{lb.filter(pl.col('rmse_s') == pl.col('rmse_s').min())['v'][0]})")
    print("[F1] largest best-so-far steps:",
          [(f"v{r['v']}", round(r["step"], 2), round(r["best"], 2)) for r in improving.sort("step").head(10).to_dicts()])
    for v in MILESTONES:
        r = lb.filter(pl.col("v") == v).to_dicts()[0]
        print(f"[F1] milestone v{v}: {r['rmse_s']:.4f} s, step {r['step']:+.2f} s  ({MILESTONES[v]})")
    v7 = lb.filter(pl.col("v") == 7)
    if v7.height:
        print(f"[F1] v7 (control without the LIRF tail table, scored after v8): {v7['rmse_s'][0]:.4f} s")
    after = lb.filter(pl.col("v") == 316)["best"][0] - lb["best"][-1]
    print(f"[F1] improvement after v316: {after:.2f} s")

    t0 = np.datetime64("2026-09-13T00:00")
    days = lambda t: (np.asarray(t, dtype="datetime64[s]") - t0) / np.timedelta64(1, "D")  # noqa: E731
    x_all, y_all = days(lb["t"].dt.replace_time_zone(None).to_numpy()), lb["rmse_s"].to_numpy()
    y_best = lb["best"].to_numpy()
    x_end = days(np.datetime64("2026-10-08T06:00"))

    fig = plt.figure(figsize=(9, 5.6), dpi=DPI)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.6], hspace=0.07, left=0.085, right=0.975, top=0.835, bottom=0.1)
    top, bot = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    lims = {"top": (296, 512), "bot": (226, 289)}
    for ax, key, ticks in [(top, "top", [300, 400, 500]), (bot, "bot", [230, 240, 250, 260, 270, 280])]:
        ax.set_ylim(*lims[key])
        ax.set_xlim(-0.4, x_end)
        ax.set_yticks(ticks)
        for t in ticks:
            ax.axhline(t, color=GRID, lw=0.8, zorder=0)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, labelcolor=MUTED)
        ax.scatter(x_all, y_all, s=11, color=MUTED, alpha=0.55, lw=0, zorder=2)
        ax.step(np.r_[x_all, x_end], np.r_[y_best, y_best[-1]], where="post", color=ACCENT, lw=2, zorder=3)
    top.spines["bottom"].set_visible(False)
    top.tick_params(axis="x", bottom=False, labelbottom=False)
    bot.spines["bottom"].set_color(AXIS)
    # axis-break marks on the left edge
    for ax, yfrac in [(top, 0.0), (bot, 1.0)]:
        ax.plot([0], [yfrac], transform=ax.transAxes, marker=[(-1, -0.6), (1, 0.6)], ms=9, ls="none", mec=INK2,
                mew=1.0, clip_on=False, zorder=6)
    ticks = [np.datetime64(d) for d in ["2026-09-13", "2026-09-17", "2026-09-21", "2026-09-25", "2026-09-29",
                                         "2026-10-03", "2026-10-07"]]
    bot.set_xticks(days(ticks))
    bot.set_xticklabels([str(d.astype(object).day) + " " + d.astype(object).strftime("%b") for d in ticks])
    bot.tick_params(axis="x", length=3, width=0.8)
    bot.set_xlabel("Date the submission was scored (UTC, 2026)", labelpad=6)
    fig.text(0.012, 0.47, "Leaderboard RMSE (s)", rotation=90, va="center", ha="left", color=INK2, fontsize=9.5)

    for v, label in MILESTONES.items():
        r = lb.filter(pl.col("v") == v).to_dicts()[0]
        xv, yv = float(days(np.datetime64(r["t"].replace(tzinfo=None)))), r["rmse_s"]
        tx, ty, panel, ha = LABEL_POS[v]
        ax = top if panel == "top" else bot
        ring_dot(ax if lims[panel][0] <= yv <= lims[panel][1] else bot, xv, yv, ACCENT, size=7)
        txt = f"v{v}  {label}\n{MINUS}{abs(r['step']):.1f} s, now {r['rmse_s']:.1f} s"
        ax.annotate(txt, xy=(xv, yv), xytext=(tx, ty), fontsize=8.8, color=INK, va="center", ha=ha,
                    linespacing=1.3,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=2, shrinkB=5,
                                    connectionstyle="arc3,rad=0"))
    handles = [Line2D([], [], color=ACCENT, lw=2, label="best score so far"),
               Line2D([], [], marker="o", ls="", ms=4.5, mfc=MUTED, mec="none", alpha=0.7,
                      label=f"every scored submission (n = {lb.height})")]
    top.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.02), ncols=2, handlelength=1.6,
               columnspacing=1.4)
    fig.text(0.02, 0.975, "Leaderboard progression of gentle-igloo", fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 0.925, "RMSE on the ranking set (January and July 2026, 344,841 departures). The y-axis is broken "
             "between 289 s and 296 s.\nLabelled points are design milestones; the step is the change of the best "
             "score at that submission.", fontsize=9.3, color=INK2, va="top", linespacing=1.35)
    save(fig, out, "fig1_leaderboard_progression.png")


# ----------------------------------------------------------------------------- F2

def fig2(df: pl.DataFrame, out: Path) -> None:
    planned = ["sched", "iobt", "lobt", "eobt", "dayshift"]
    n_ap = df.group_by("ADEP_mvt").len().rename({"len": "n"})
    cnt = df.group_by("ADEP_mvt", "regime").len()
    share = {(r["ADEP_mvt"], r["regime"]): r["len"] for r in cnt.to_dicts()}
    tot = {r["ADEP_mvt"]: r["n"] for r in n_ap.to_dicts()}
    longd = df.filter(pl.col("y") > 3600)
    lcnt = {(r["ADEP_mvt"], r["regime"]): r["len"] for r in longd.group_by("ADEP_mvt", "regime").len().to_dicts()}
    ltot = {r["ADEP_mvt"]: r["len"] for r in longd.group_by("ADEP_mvt").len().to_dicts()}
    mu = df["y"].mean()
    tss = float(((df["y"] - mu) ** 2).sum())
    by_cls = (df.group_by("regime").agg(n=pl.len(), ss=((pl.col("y") - mu) ** 2).sum(), long=(pl.col("y") > 3600).sum())
              .with_columns(row=pl.col("n") / df.height, ss_share=pl.col("ss") / tss))
    cls_order = ["regular", *planned]
    by_cls = {r["regime"]: r for r in by_cls.to_dicts()}

    print(f"[F2] mean label {mu:.1f} s; RMSE of a constant prediction {np.sqrt(tss / df.height):.1f} s")
    for c in cls_order:
        r = by_cls[c]
        print(f"[F2] class {c:9s} rows {r['n']:>9,} ({100 * r['row']:.4f}%), share of sum of squared deviations "
              f"{100 * r['ss_share']:.2f}%, labels > 1 h {r['long']:,}")
    for ap in AIRPORTS:
        k = {c: share.get((ap, c), 0) for c in planned}
        print(f"[F2] {ap}: non-regular {100 * sum(k.values()) / tot[ap]:.2f}% of {tot[ap]:,} "
              f"({', '.join(f'{c} {v:,}' for c, v in k.items())}); labels > 1 h {ltot.get(ap, 0):,}, "
              f"of which non-regular {sum(lcnt.get((ap, c), 0) for c in planned):,}")
    # same-minute evidence: block time in the scheduled minute vs the neighbouring minutes
    d = (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()
    sm = (df.group_by("ADEP_mvt").agg(same=(d.abs() <= 6).mean(), before=d.is_between(-66, -54).mean(),
                                      after=d.is_between(54, 66).mean()).sort("ADEP_mvt"))
    for r in sm.to_dicts():
        print(f"[F2] {r['ADEP_mvt']}: |BLOCK - SCHED| <= 6 s {100 * r['same']:.2f}%, previous minute "
              f"{100 * r['before']:.2f}%, next minute {100 * r['after']:.2f}%")
    band = lambda c: ((pl.col(c).dt.second() <= 6) | (pl.col(c).dt.second() >= 54)).mean()  # noqa: E731
    edge = df.group_by("ADEP_mvt").agg(block=band("BLOCK_TIME_UTC_mvt"), takeoff=band("MVT_TIME_UTC_mvt")).sort("ADEP_mvt")
    print("[F2] share of block / take-off times with second-of-minute in [54, 59] or [0, 6]:",
          {r["ADEP_mvt"]: (round(r["block"], 4), round(r["takeoff"], 4)) for r in edge.to_dicts()})
    print(f"[F2] all non-regular labels: {(df['regime'] != 'regular').sum():,} "
          f"({100 * (df['regime'] != 'regular').mean():.2f}% of departures)")
    jj = df.filter(pl.col("month").is_in([1, 7]))
    yj = jj["y"].to_numpy()
    dev = (yj - yj.mean()) ** 2
    print(f"[F2] Jan+Jul 2025: {len(yj):,} departures, {int((yj > 14400).sum())} labels > 4 h holding "
          f"{100 * dev[yj > 14400].sum() / dev.sum():.1f}% of the sum of squared deviations from the mean label")
    ll = df.filter(pl.col("y") > 3600)
    print(f"[F2] labels > 1 h: {ll.height:,}; at LIRF {ll.filter(pl.col('ADEP_mvt') == 'LIRF').height:,}; "
          f"non-regular overall {100 * (ll['regime'] != 'regular').mean():.1f}%")

    lab_max = {r["ADEP_mvt"]: r for r in df.group_by("ADEP_mvt").agg(
        mx=pl.col("y").max(), neg=(pl.col("y") < 0).sum()).to_dicts()}
    print("[F2] largest label by airport (s):", {ap: int(lab_max[ap]["mx"]) for ap in AIRPORTS},
          f"; negative labels: {sum(int(r['neg']) for r in lab_max.values()):,}")

    fig = plt.figure(figsize=(9, 7.8), dpi=DPI)
    gs = fig.add_gridspec(2, 2, height_ratios=[10, 6], width_ratios=[1, 1], hspace=0.45, wspace=0.40,
                          left=0.075, right=0.965, top=0.79, bottom=0.07)
    a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    h = 0.55
    excluded = {"EDDF", "EHAM", "LSZH"}
    # (a) share of all departures
    category_axis(a, AIRPORTS)
    style_value_axis(a, (0, 0.30), [0, 0.1, 0.2, 0.3], ["0%", "10%", "20%", "30%"],
                     xlabel="share of all 2025 departures")
    a.set_title("a  Label = take-off minus a planning time", fontsize=10)
    for i, ap in enumerate(AIRPORTS):
        if ap in excluded:
            note(a, 0.003, i, "not labelled: block time to the second")
            continue
        w = [share.get((ap, cl), 0) / tot[ap] for cl in planned]
        end = stacked_hbar(a, i, w, [REGIME_COLOURS[k] for k in planned], h)
        a.text(end + 0.006, i, fmt_share(end), va="center", fontsize=8.8, color=INK2)
    # (b) share of labels > 1 h
    category_axis(b, AIRPORTS)
    style_value_axis(b, (0, 1.0), [0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"],
                     xlabel="share of departures whose label exceeds 1 h")
    b.set_title("b  The same, among labels longer than 1 hour", fontsize=10)
    for i, ap in enumerate(AIRPORTS):
        n = ltot.get(ap, 0)
        if ap in excluded:
            note(b, 0.01, i, f"not labelled (n = {n:,})")
            continue
        if n < 20:
            note(b, 0.01, i, f"n = {n:,}: labels stop at {int(lab_max[ap]['mx']):,} s")
            continue
        w = [lcnt.get((ap, cl), 0) / n for cl in planned]
        end = stacked_hbar(b, i, w, [REGIME_COLOURS[k] for k in planned], h)
        if end > 0.7:
            b.text(end - 0.025, i, f"{fmt_share(end)} of {n:,}", va="center", ha="right", fontsize=8.8,
                   color=SURFACE, zorder=4)
        else:
            b.text(end + 0.02, i, f"{fmt_share(end)} of {n:,}", va="center", fontsize=8.8, color=INK2)
    # (c) classes: rows vs sum of squared deviations
    category_axis(c, [REGIME_SHORT[k] for k in cls_order], pad=0.6)
    style_value_axis(c, (0, 1.0), [0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    c.set_title("c  Share of departures and of the label's squared deviation from its mean, by regime",
                fontsize=10)
    paired_bars(c, [(by_cls[k]["row"], by_cls[k]["ss_share"], by_cls[k]["n"]) for k in cls_order], thick_pt=6.0,
                gap_pt=1.5)
    c.legend(handles=[Patch(color=CONTEXT, label="share of departures"),
                      Patch(color=ACCENT, label="share of \u03a3 (label \u2212 mean label)\u00b2")],
             loc="lower right", ncols=1, handlelength=1.2)
    leg = [Patch(color=REGIME_COLOURS[k], label=REGIME_NAMES[k]) for k in planned]
    fig.legend(handles=leg, loc="upper left", bbox_to_anchor=(0.02, 0.89), ncols=5, handlelength=1.0,
               columnspacing=1.0, fontsize=8.6)
    fig.text(0.02, 0.975, "Recording regimes in the 2025 labels", fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 0.935, f"{df.height:,} departures, labelled by regime_label in src/train.py: \u00b16 s tolerance; "
             "one-day shift checked first, then SCHED, IOBT, LOBT, EOBT_1.", fontsize=9.3, color=INK2, va="top")
    save(fig, out, "fig2_recording_regimes.png")


# ----------------------------------------------------------------------------- F3

def fig3(df: pl.DataFrame, out: Path) -> None:
    e = df.with_columns(err=pl.col("d_mvt_aobt") - pl.col("y"))  # = BLOCK - AOBT_3
    have = e.drop_nulls("err")
    q = (have.group_by("ADEP_mvt").agg(
        n=pl.len(), p10=pl.col("err").quantile(0.1, "linear"), p25=pl.col("err").quantile(0.25, "linear"),
        p50=pl.col("err").median(), p75=pl.col("err").quantile(0.75, "linear"), p90=pl.col("err").quantile(0.9, "linear"),
        mad=pl.col("err").abs().median(), within60=(pl.col("err").abs() <= 60).mean(),
        within120=(pl.col("err").abs() <= 120).mean()).sort("ADEP_mvt"))
    edges = [0, 60, 120, 300, 900, np.inf]
    bin_names = ["≤ 1 min", "1–2 min", "2–5 min", "5–15 min", "> 15 min"]
    absd = pl.col("err").abs()
    bins = e.group_by("ADEP_mvt").agg(
        n=pl.len(), missing=pl.col("err").is_null().mean(),
        **{f"b{i}": (absd.is_not_null() & (absd > lo if i else absd >= lo) & (absd <= hi)).mean()
           for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))}).sort("ADEP_mvt")
    qd = {r["ADEP_mvt"]: r for r in q.to_dicts()}
    bd = {r["ADEP_mvt"]: r for r in bins.to_dicts()}
    allerr = have["err"].to_numpy()
    body = have.filter(pl.col("y") <= 7200)["err"].to_numpy()
    print(f"[F3] AOBT_3 present on {have.height:,} of {e.height:,} departures ({100 * have.height / e.height:.2f}%); "
          f"missing {100 * (1 - have.height / e.height):.2f}%, no NM flight record {100 * df['FLIGHT_ID_mvt'].is_null().mean():.2f}%")
    print(f"[F3] AOBT_3 second-of-minute = 0 on {100 * have.select((pl.col('AOBT_3_flt').dt.second() == 0).mean()).item():.1f}% "
          f"of values")
    print(f"[F3] proxy (MVT - AOBT_3) as a prediction: RMSE {np.sqrt(np.mean(allerr ** 2)):.1f} s on all rows with AOBT_3, "
          f"{np.sqrt(np.mean(body ** 2)):.1f} s on labels <= 7200 s; |error| <= 60 s {100 * np.mean(np.abs(allerr) <= 60):.1f}%, "
          f"<= 120 s {100 * np.mean(np.abs(allerr) <= 120):.1f}%, <= 300 s {100 * np.mean(np.abs(allerr) <= 300):.1f}%")
    for ap in AIRPORTS:
        r, s = qd[ap], bd[ap]
        print(f"[F3] {ap}: median {r['p50']:+.0f} s, IQR [{r['p25']:+.0f}, {r['p75']:+.0f}], P10-P90 [{r['p10']:+.0f}, "
              f"{r['p90']:+.0f}], median |err| {r['mad']:.0f} s; bins " +
              ", ".join(f"{bn} {100 * s[f'b{i}']:.1f}%" for i, bn in enumerate(bin_names)) +
              f", no AOBT_3 {100 * s['missing']:.2f}%")

    fig = plt.figure(figsize=(9, 5.4), dpi=DPI)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.08], wspace=0.13, left=0.075, right=0.89, top=0.74, bottom=0.12)
    a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    category_axis(a, AIRPORTS)
    style_value_axis(a, (-1050, 1050), [-900, -600, -300, 0, 300, 600, 900],
                     [f"{MINUS}900", f"{MINUS}600", f"{MINUS}300", "0", "300", "600", "900"],
                     xlabel="(take-off − AOBT_3) − label, s   (positive: proxy too long)")
    a.axvline(0, color=INK2, lw=0.9, zorder=1)
    a.set_title("a  Signed proxy error", fontsize=10)
    for i, ap in enumerate(AIRPORTS):
        r = qd[ap]
        a.plot([r["p10"], r["p90"]], [i, i], color=RAMP[0], lw=2, zorder=2, solid_capstyle="butt")
        a.plot([r["p25"], r["p75"]], [i, i], color=ACCENT, lw=7, zorder=3, solid_capstyle="butt")
        ring_dot(a, r["p50"], i, RAMP[4], size=7)
    a.legend(handles=[Line2D([], [], color=RAMP[0], lw=2, label="10th–90th percentile"),
                      Line2D([], [], color=ACCENT, lw=7, label="interquartile range"),
                      Line2D([], [], marker="o", ls="", ms=6, mfc=RAMP[4], mec=SURFACE, label="median")],
             loc="upper left", bbox_to_anchor=(-0.02, 1.25), ncols=3, handlelength=1.4, columnspacing=1.0,
             fontsize=8.5)
    category_axis(b, AIRPORTS)
    b.set_yticklabels([])
    style_value_axis(b, (0, 1.0), [0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"],
                     xlabel="share of 2025 departures")
    b.set_title("b  Size of the error |proxy − label|", fontsize=10)
    cols = RAMP + [CONTEXT]
    for i, ap in enumerate(AIRPORTS):
        s = bd[ap]
        w = [s[f"b{k}"] for k in range(5)] + [s["missing"]]
        stacked_hbar(b, i, w, cols, 0.5)
        b.text(1.03, i, fmt_share(s["b0"] + s["b1"]), va="center", fontsize=8.8, color=INK2, transform=b.get_yaxis_transform())
    b.text(1.03, -0.85, "≤ 2 min", fontsize=8.6, color=INK2, transform=b.get_yaxis_transform(), va="bottom")
    b.legend(handles=[Patch(color=cl, label=nm) for cl, nm in zip(cols, bin_names + ["no AOBT_3"])],
             loc="upper left", bbox_to_anchor=(-0.02, 1.25), ncols=3, handlelength=1.0, columnspacing=0.9,
             fontsize=8.5)
    fig.text(0.02, 0.975, "NM actual off-block time (AOBT_3) as a taxi-out proxy", fontsize=13, fontweight="bold",
             va="top")
    fig.text(0.02, 0.925, f"{e.height:,} departures of 2025, {have.height:,} with an AOBT_3 value. The proxy is "
             "take-off minus AOBT_3, so its error equals BLOCK − AOBT_3.", fontsize=9.3, color=INK2, va="top")
    save(fig, out, "fig3_nm_proxy_error.png")


# ----------------------------------------------------------------------------- F4

def fig4(root: Path, out: Path) -> None:
    ds = root / "prc-2026-datasets"
    if not need([ds / "ranking.parquet", root / "adsblol_kt1" / "offblock2_2026-01-01.parquet",
                 root / "adsblol_kt1" / "offblock2_2025-01-01.parquet"], "F4"):
        return
    cols = [ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "FLIGHT_ID_mvt"]
    tr = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(ds / "training_2025-0[17]-*.parquet")))])
    rk = pl.read_parquet(ds / "ranking.parquet", columns=cols)
    frames = []
    for year, d, pat in [(2025, tr, "offblock2_2025-0[17]-*.parquet"), (2026, rk, "offblock2_2026-0[17]-*.parquet")]:
        dep = d.filter(pl.col("PHASE_mvt") == "DEP").with_columns(month=pl.col("MVT_TIME_UTC_mvt").dt.month(),
                                                                   year=pl.lit(year))
        cov = surface_coverage(root, pat)
        j = dep.join(cov, on=ID, how="left")
        alt = sorted(glob.glob(str(root / "adsblol" / pat)))
        if alt:  # the default 3-kt extraction read by src/e146_stage_drift.py; own-track sets should coincide
            o = pl.concat([pl.read_parquet(f, columns=[ID, "coverage"]) for f in alt]).join(cov, on=ID, suffix="_kt1")
            print(f"[F4] {year}: own-track disagreement adsblol vs adsblol_kt1: "
                  f"{int(((o['coverage'] >= 2) != (o['coverage_kt1'] >= 2)).sum())} of {o.height:,} departures")
        print(f"[F4] {year} Jan+Jul: {dep.height:,} departures, {len(glob.glob(str(root / 'adsblol_kt1' / pat)))} daily "
              f"files, {j['coverage'].null_count():,} departures absent from the files")
        frames.append(j)
    x = pl.concat(frames)
    g = (x.group_by("ADEP_mvt", "year", "month")
         .agg(n=pl.len(), own=(pl.col("coverage") >= 2).mean(), stat=(pl.col("coverage") == 3).mean())
         .sort("ADEP_mvt", "month", "year"))
    gd = {(r["ADEP_mvt"], r["year"], r["month"]): r for r in g.to_dicts()}
    nm = (x.group_by("year", "ADEP_mvt", "month").agg(miss=pl.col("FLIGHT_ID_mvt").is_null().mean())
          .sort("miss", descending=True))
    for yr in (2025, 2026):
        s = x.filter(pl.col("year") == yr)
        top = nm.filter(pl.col("year") == yr).row(0, named=True)
        print(f"[F4] {yr} Jan+Jul: no NM flight record on {100 * s['FLIGHT_ID_mvt'].is_null().mean():.2f}% of departures; "
              f"highest airport-month {top['ADEP_mvt']} month {top['month']} {100 * top['miss']:.2f}%")
    for yr in (2025, 2026):
        s = x.filter(pl.col("year") == yr)
        print(f"[F4] {yr}: own surface track (coverage >= 2) {100 * (s['coverage'] >= 2).mean():.1f}%, "
              f"stationary-before-movement (coverage = 3) {100 * (s['coverage'] == 3).mean():.1f}%")
    for ap in AIRPORTS:
        print(f"[F4] {ap}: " + "; ".join(
            f"{ {1: 'Jan', 7: 'Jul'}[m]} {100 * gd[(ap, 2025, m)]['own']:.1f}% -> {100 * gd[(ap, 2026, m)]['own']:.1f}% "
            f"(coverage 3: {100 * gd[(ap, 2025, m)]['stat']:.1f}% -> {100 * gd[(ap, 2026, m)]['stat']:.1f}%)"
            for m in (1, 7)))

    fig = plt.figure(figsize=(9, 5.2), dpi=DPI)
    gs = fig.add_gridspec(1, 2, wspace=0.32, left=0.075, right=0.93, top=0.735, bottom=0.12)
    for k, m in enumerate((1, 7)):
        ax = fig.add_subplot(gs[k])
        category_axis(ax, AIRPORTS)
        if k:
            ax.set_yticklabels([])
        style_value_axis(ax, (0, 1.0), [0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"],
                         xlabel="share of departures with an own surface track")
        ax.set_title(["a  January", "b  July"][k], fontsize=10)
        for i, ap in enumerate(AIRPORTS):
            v0, v1 = gd[(ap, 2025, m)]["own"], gd[(ap, 2026, m)]["own"]
            ax.plot([v0, v1], [i, i], color=AXIS, lw=2, zorder=2)
            ring_dot(ax, v0, i, YEAR_COLOURS[2025])
            ring_dot(ax, v1, i, YEAR_COLOURS[2026])
            dpp = 100 * (v1 - v0)
            ax.text(1.04, i, f"{'+' if dpp >= 0.5 else (MINUS if dpp <= -0.5 else '')}{abs(dpp):.0f}", va="center",
                    ha="left", fontsize=8.8, color=INK2 if abs(dpp) >= 10 else MUTED, transform=ax.get_yaxis_transform())
        ax.text(1.04, -0.85, "Δ pp", fontsize=8.6, color=INK2, transform=ax.get_yaxis_transform(), va="bottom")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=7.5, mfc=YEAR_COLOURS[2025], mec=SURFACE,
                               label="2025 (training)"),
                        Line2D([], [], marker="o", ls="", ms=7.5, mfc=YEAR_COLOURS[2026], mec=SURFACE,
                               label="2026 (ranking)")],
               loc="upper left", bbox_to_anchor=(0.02, 0.865), ncols=2, handlelength=1.0, columnspacing=1.4)
    fig.text(0.02, 0.975, "ADS-B surface coverage changed between training and ranking", fontsize=13,
             fontweight="bold", va="top")
    fig.text(0.02, 0.925, "Own surface track = the departing aircraft has an adsb.lol surface segment before take-off "
             "(coverage class ≥ 2 in adsblol_kt1/offblock2;\nthe production surface stages are applied only to "
             "these rows). Same calendar months, all departures.", fontsize=9.3, color=INK2, va="top", linespacing=1.35)
    save(fig, out, "fig4_adsb_coverage_drift.png")


# ----------------------------------------------------------------------------- F5

def fig5(root: Path, df: pl.DataFrame, out: Path) -> None:
    pred_path = root / "e195_pre_e81_20261007_v2" / "predictions.parquet"
    if not need([pred_path], "F5"):
        return
    p = pl.read_parquet(pred_path, columns=[ID, Y, "month", "ADEP_mvt", "has_nm", "evaluation_mask", "joint"])
    x = p.join(df.select(ID, "regime", "d_mvt_sched", "FLIGHT_ID_mvt", pl.col(Y).alias("y_raw")), on=ID, how="left",
               validate="1:1")
    assert x.height == p.height and (x[Y] == x["y_raw"]).all(), "prediction rows do not match the training labels"
    cov_files = glob.glob(str(root / "adsblol_kt1" / "offblock2_2025-*.parquet"))
    if cov_files:
        x = x.join(surface_coverage(root, "offblock2_2025-*.parquet"), on=ID, how="left")
    # the 126 excluded rows: LIRF, no NM record, 4 h < take-off - SCHED <= 30 h (separate TabPFN tail model)
    tail = ((pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null() & (pl.col("d_mvt_sched") > 14400)
            & (pl.col("d_mvt_sched") <= 108000)).fill_null(False)
    n_tail = x.select(tail.sum()).item()
    assert n_tail == (~x["evaluation_mask"]).sum(), "evaluation_mask is not the TabPFN tail segment"
    yf = pl.col(Y).cast(pl.Float64)
    print(f"[F5] {p.height:,} OOF rows; evaluation_mask excludes {n_tail} rows (LIRF, no NM record, 4-30 h after SCHED)")
    print(f"[F5] labels > 4 h in 2025: {x.filter(yf > 14400).height}, of which in the excluded tail segment "
          f"{x.filter((yf > 14400) & tail).height}")
    ev = x.filter(pl.col("evaluation_mask")).with_columns(se=(pl.col("joint").cast(pl.Float64) - yf) ** 2)
    N, TOT = ev.height, float(ev["se"].sum())
    allrmse = float(np.sqrt(((x["joint"] - x[Y]).cast(pl.Float64) ** 2).mean()))
    print(f"[F5] RMSE on evaluation_mask rows {np.sqrt(TOT / N):.2f} s (n = {N:,}); on all rows {allrmse:.2f} s; "
          f"labels <= 7200 s: {np.sqrt(ev.filter(yf <= 7200)['se'].mean()):.2f} s")
    jj = ev.filter(pl.col("month").is_in([1, 7]))
    big = jj.filter(yf > 14400)
    print(f"[F5] Jan+Jul 2025 evaluated rows {jj.height:,}, RMSE {np.sqrt(jj['se'].mean()):.2f} s; labels > 4 h: "
          f"{big.height}, share of Jan+Jul SSE {100 * big['se'].sum() / jj['se'].sum():.1f}%")
    se_sorted = np.sort(ev["se"].to_numpy())[::-1]
    cum = np.cumsum(se_sorted) / TOT
    print(f"[F5] concentration: top 10 rows {100 * cum[9]:.1f}%, top 100 {100 * cum[99]:.1f}%, top 1% of rows "
          f"{100 * cum[int(N * 0.01) - 1]:.1f}% of SSE")

    def table(key_expr, name):
        body = yf <= 7200
        t = (ev.with_columns(k=key_expr).group_by("k")
             .agg(n=pl.len(), sse=pl.col("se").sum(), mse_body=pl.col("se").filter(body).mean())
             .with_columns(row=pl.col("n") / N, err=pl.col("sse") / TOT, rmse=(pl.col("sse") / pl.col("n")).sqrt()))
        d = {r["k"]: r for r in t.to_dicts()}
        for k, r in sorted(d.items(), key=lambda kv: str(kv[0])):
            rb = f"{np.sqrt(r['mse_body']):.1f} s" if r["mse_body"] is not None else "n/a"
            print(f"[F5] {name} {k}: rows {r['n']:,} ({100 * r['row']:.3f}%), SSE share {100 * r['err']:.1f}%, "
                  f"RMSE {r['rmse']:.1f} s, RMSE on labels <= 7200 s {rb}")
        return d

    by_ap = table(pl.col("ADEP_mvt"), "airport")
    cls_order = ["regular", "sched", "iobt", "lobt", "eobt", "dayshift"]
    by_cls = table(pl.col("regime"), "regime")
    ybins = [("≤ 30 min", 0, 1800), ("30–60 min", 1800, 3600), ("1–2 h", 3600, 7200),
             ("2–4 h", 7200, 14400), ("> 4 h", 14400, np.inf)]
    yexpr = pl.when(yf <= 1800).then(pl.lit(ybins[0][0]))
    for nm, lo, hi in ybins[1:]:
        yexpr = yexpr.when(yf <= hi).then(pl.lit(nm))
    by_y = table(yexpr, "label")
    by_nm = table(pl.when(pl.col("has_nm")).then(pl.lit("NM flight record present")).otherwise(
        pl.lit("NM flight record missing")), "nm")
    groups = [("NM flight record present", by_nm), ("NM flight record missing", by_nm)]
    if "coverage" in ev.columns:
        by_tr = table(pl.when(pl.col("coverage") >= 2).then(pl.lit("own surface track")).otherwise(
            pl.lit("no own surface track")), "track")
        groups += [("own surface track", by_tr), ("no own surface track", by_tr)]

    fig = plt.figure(figsize=(9, 8.4), dpi=DPI)
    gs = fig.add_gridspec(2, 2, height_ratios=[10, 6], hspace=0.30, wspace=0.55,
                          left=0.115, right=0.975, top=0.80, bottom=0.06)
    pct_ticks = lambda t: [f"{100 * v:.0f}%" for v in t]  # noqa: E731

    def panel(ax, keys, d, labels, title, xmax=1.0, ticks=(0, 0.25, 0.5, 0.75, 1.0)):
        category_axis(ax, labels, pad=0.6)
        style_value_axis(ax, (0, xmax), list(ticks), pct_ticks(ticks))
        ax.set_title(title, fontsize=10)
        empty = {"row": 0.0, "err": 0.0, "n": 0}
        paired_bars(ax, [(d.get(k, empty)["row"], d.get(k, empty)["err"], d.get(k, empty)["n"]) for k in keys])

    panel(fig.add_subplot(gs[0, 0]), AIRPORTS, by_ap, AIRPORTS, "a  By airport", xmax=0.40,
          ticks=(0, 0.1, 0.2, 0.3, 0.4))
    panel(fig.add_subplot(gs[0, 1]), cls_order, by_cls, [REGIME_SHORT[k] for k in cls_order],
          "b  By recording regime (src/train.py label)")
    panel(fig.add_subplot(gs[1, 0]), [b[0] for b in ybins], by_y, [b[0] for b in ybins], "c  By recorded label")
    merged = {k: d[k] for k, d in groups if k in d}
    panel(fig.add_subplot(gs[1, 1]), [k for k, _ in groups], merged,
          [k.replace(" flight record", " record") for k, _ in groups], "d  By NM match and ADS-B track")
    fig.legend(handles=[Patch(color=CONTEXT, label="share of evaluated rows"),
                        Patch(color=ACCENT, label="share of squared error")],
               loc="upper left", bbox_to_anchor=(0.02, 0.885), ncols=2, handlelength=1.2, columnspacing=1.6)
    fig.text(0.02, 0.975, "Where the squared error sits (2025 out-of-fold, not the leaderboard)", fontsize=13,
             fontweight="bold", va="top")
    fig.text(0.02, 0.935, f"E195 retrospective application of the current chain before E81 (column joint), six "
             f"month-pair folds. {N:,} rows in evaluation_mask; RMSE {np.sqrt(TOT / N):.1f} s.\nExcluded: {n_tail} LIRF "
             "rows without an NM record and 4–30 h after SCHED, which the separate TabPFN tail model predicts.",
             fontsize=9.3, color=INK2, va="top", linespacing=1.35)
    save(fig, out, "fig5_error_budget.png")


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", default="data", help="directory holding prc-2026-datasets/ and derived tables")
    ap.add_argument("--leaderboard", default=str(REPO / "docs" / "tables" / "leaderboard_gentle-igloo.csv"))
    ap.add_argument("--out", default=str(REPO / "docs" / "figures"))
    ap.add_argument("--only", default="1,2,3,4,5", help="comma-separated figure numbers")
    args = ap.parse_args()
    root, out = Path(args.data_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = {int(s) for s in args.only.split(",") if s}
    if 1 in only:
        fig1(Path(args.leaderboard), out)
    df = None
    if only & {2, 3, 5}:
        if need([root / "prc-2026-datasets" / "training_*.parquet"], "F2/F3/F5"):
            df = load_departures(root)
    if 2 in only and df is not None:
        fig2(df, out)
    if 3 in only and df is not None:
        fig3(df, out)
    if 4 in only:
        fig4(root, out)
    if 5 in only and df is not None:
        fig5(root, df, out)


if __name__ == "__main__":
    main()

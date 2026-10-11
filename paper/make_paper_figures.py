#!/usr/bin/env python3
"""Build paper/figures/ for the JOAS manuscript.

1. Copies the repository figures that the manuscript reuses from docs/figures/ (unchanged bytes).
2. Draws two figures that exist only in the paper:
   - nn_weight_by_month.png: change of the 2025 out-of-fold chain RMSE against the neural-member
     weight, per month, three- and ten-seed member (E212), from
     analysis/e212_nn_weight_10seed_20261008/per_month.json;
   - final_recipe_price.png: the price of the uniform final recipe on January/July measured by
     single uploads, and each upload against the forecast corridor written before it, from
     analysis/v12080_control_explanation_20261009/results/{gram.json,sigma_piecewise.json,sigma_cluster.json}
     and the forecasts recorded in docs/final-submission.md (sections 4 and 7) and
     analysis/v12090_upload_20261008/RESULTS.md; the three uploads of 11 October that measured the
     E218 PRU level stage (v12081, v12082, v12083) from analysis/v1208{1,2,3}_*_20261011/
     (score_response.json, forecast.json; the v12081 corridor as recorded in its RESULTS.md).

Run from anywhere (no data outside the repository is read):

    /path/to/prc-2026/.venv/bin/python paper/make_paper_figures.py

Requires matplotlib (3.11 used). Never run through `uv run`.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "figures"

# Figures reused from the repository documentation, in the order the paper cites them.
REUSED = [
    "fig2_recording_regimes.png",            # paper Figure 1
    "fig3_nm_proxy_error.png",               # Figure 2
    "fig4_adsb_coverage_drift.png",          # Figure 3
    "fig5_error_budget.png",                 # Figure 4
    "fig1_leaderboard_progression.png",      # Figure 6 (Figure 5 is the stage diagram in the text)
    "fig_generalisation_months.png",         # Figure 7
    "fig_generalisation_airport_month.png",  # Figure 8
    "fig_uncertainty_calibration.png",       # Figure 10 (Figure 9 is drawn below)
    "fig6_regime_detector_calibration.png",  # Figure 11
    "fig7_regime_effect_on_kpi.png",         # Figure 12 (Figure 13 is drawn below)
]

E212 = ROOT / "analysis/e212_nn_weight_10seed_20261008/per_month.json"
CTRL = ROOT / "analysis/v12080_control_explanation_20261009/results"
E218_RUNS = {  # the three single-change uploads of 11 October that measured the E218 stage
    "v12081": ROOT / "analysis/v12081_upload_20261011",
    "v12082": ROOT / "analysis/v12082_control_20261011",
    "v12083": ROOT / "analysis/v12083_upload_20261011",
}
# v12081: point and corridor as written before the upload (RESULTS.md / journal 455); its forecast.json
# stores the board-model deltas rather than the rounded corridor.
V12081_FORECAST = (230.408, (230.394, 230.422))

# Colour roles (validated light-mode palette slots 1-2 plus neutral text/grid tokens).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BAND = "#ecebe6"
SURFACE = "#ffffff"

FINAL_MONTHS = {1, 2, 6, 7}
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 200,
})


def copy_reused() -> None:
    for name in REUSED:
        src = ROOT / "docs/figures" / name
        if not src.is_file():
            sys.exit(f"missing source figure: {src}")
        shutil.copyfile(src, OUT / name)
        print("copied", name)


def nn_weight_by_month() -> None:
    data = json.loads(E212.read_text())
    weights = sorted(float(w) for w in data["m10"]["1"]["d_rmse"])
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.6), sharex=True)
    for m in range(1, 13):
        ax = axes[(m - 1) // 4][(m - 1) % 4]
        final = m in FINAL_MONTHS
        colour = BLUE if final else MUTED
        m10 = data["m10"][str(m)]
        m3 = data["m3"][str(m)]
        y10 = [m10["d_rmse"][f"{w:.2f}"] for w in weights]
        y3 = [m3["d_rmse"][f"{w:.2f}"] for w in weights]
        ax.axhline(0, color=INK2, linewidth=0.8, zorder=1)
        ax.axvline(0.35, color=ORANGE, linewidth=1.0, zorder=1)
        ax.plot(weights, y3, color=colour, linewidth=1.0, linestyle=(0, (3, 2)), zorder=2)
        ax.plot(weights, y10, color=colour, linewidth=2.0, zorder=3)
        w_star = m10["w_star"]
        ax.plot([w_star], [m10["d_rmse"][f"{w_star:.2f}"]], marker="o", markersize=5,
                color=colour, markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=4)
        v35 = m10["d_rmse"]["0.35"]
        ax.set_title(f"{MONTH_NAMES[m - 1]}{' (final)' if final else ''}", fontsize=8.5,
                     fontweight="bold" if final else "normal", loc="left",
                     color=INK if final else INK2)
        ax.set_title(f"{v35:+.3f} s".replace("-", "\u2212"), fontsize=7.3, loc="right", color=INK2)
        lo = min(min(y10), min(y3))
        hi = max(max(y10), max(y3))
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks([0.1, 0.25, 0.4, 0.55])
        ax.tick_params(labelsize=7.5, length=2)
        if (m - 1) % 4 == 0:
            ax.set_ylabel("ΔRMSE vs 0.25 (s)", fontsize=7.5)
        if m > 8:
            ax.set_xlabel("neural-member weight", fontsize=7.5)
    handles = [
        Line2D([], [], color=INK2, linewidth=2.0, label="ten-seed member"),
        Line2D([], [], color=INK2, linewidth=1.0, linestyle=(0, (3, 2)), label="three-seed member"),
        Line2D([], [], color=ORANGE, linewidth=1.0, label="chosen weight 0.35"),
        Line2D([], [], color=INK2, marker="o", linestyle="none", markersize=5, label="best grid weight (ten seeds)"),
        Patch(facecolor=BLUE, label="months of the blind final"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7.2,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = OUT / "nn_weight_by_month.png"
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path.name)


def final_recipe_price() -> None:
    gram = json.loads((CTRL / "gram.json").read_text())["scores"]
    sigma_rows = json.loads((CTRL / "sigma_piecewise.json").read_text())["sigma_FH"]  # binned calibration
    sigma_lin = json.loads((CTRL / "sigma_cluster.json").read_text())["F-H|ADEP_mvt_day"]["sigma_rows"]  # linear calibration
    s = {k: float(v) for k, v in gram.items()}
    for name, run in E218_RUNS.items():
        s[name] = float(json.loads((run / "score_response.json").read_text())["score"])
    e218_forecasts = {"v12081": V12081_FORECAST}
    for name in ("v12082", "v12083"):
        f = json.loads((E218_RUNS[name] / "forecast.json").read_text())["forecast"]
        e218_forecasts[name] = (float(f["point"]), tuple(float(x) for x in f["corridor"]))
    final_e218 = s["v12082"] - s["v12077"]
    e81 = s["v12078"] - s["v12077"]
    tail = s["v12079"] - s["v12077"]
    canary = s["v12090"] - s["v12075"]
    control = s["v12080"] - s["v12077"]
    expected = s["H"]
    # sigma of the control in RMSE units: sigma_MSE / (2 * RMSE)
    sig_lo, sig_hi = sorted((sigma_rows / (2 * expected), sigma_lin / (2 * expected)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 4.2), gridspec_kw={"width_ratios": [1.0, 1.0]})

    def expectation(ax, x, y):
        """Measured expectation with the two label-free 1-sigma estimates (thick: binned, thin: linear)."""
        ax.plot([x - sig_hi, x + sig_hi], [y, y], color=ORANGE, linewidth=1.0, zorder=4)
        ax.plot([x - sig_lo, x + sig_lo], [y, y], color=ORANGE, linewidth=3.0, solid_capstyle="butt", zorder=4)
        ax.plot([x], [y], marker="o", markersize=6, markerfacecolor=SURFACE, markeredgecolor=ORANGE,
                markeredgewidth=1.5, linestyle="none", zorder=5)

    # (a) price against the parent file, measured by single uploads
    rows = [
        ("E81 blend removed\n(v12078 − v12077)", [(e81, BLUE)]),
        ("fresh tail rows\n(v12079 − v12077)", [(tail, ORANGE)]),
        ("sum of the two parts", [(e81, BLUE), (tail, ORANGE)]),
        ("both at once, older members\n(canary v12090 − v12075)", [(canary, MUTED)]),
        ("final file, Jan/Jul rows\n(control v12080 − v12077)", [(control, INK2)]),
        ("final file with E218, Jan/Jul\n(control v12082 − v12077)", [(final_e218, INK)]),
    ]
    ys = [6.0, 5.0, 4.0, 3.0, 1.8, 0.8]
    for y, (_, parts) in zip(ys, rows):
        left = 0.0
        for width, colour in parts:
            ax1.barh(y, width, left=left, height=0.5, color=colour, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            left += width
        ax1.text(left + 0.03, y, f"+{left:.3f}", va="center", ha="left", fontsize=7.5, color=INK)
    exp_rel = expected - s["v12077"]
    expectation(ax1, exp_rel, 2.4)
    ax1.text(exp_rel + sig_hi + 0.04, 2.4, f"+{exp_rel:.3f}", fontsize=7, color=INK2, va="center")
    ax1.set_yticks(ys + [2.4], [r[0] for r in rows] + ["expected v12080 (H)"], fontsize=7.3)
    ax1.set_ylim(0.3, 6.5)
    ax1.set_xlim(0, 2.45)
    ax1.set_xlabel("RMSE change on January/July 2026 (s)", fontsize=7.8)
    ax1.grid(axis="y", visible=False)
    ax1.set_title("(a) Price of the uniform recipe", loc="left", fontsize=8.5, fontweight="bold")

    # (b) each upload against the corridor written before it
    uploads = [
        ("v12078", 230.60, (230.40, 231.20), s["v12078"]),
        ("v12079", 231.17, (230.83, 231.48), s["v12079"]),
        ("canary v12090", 231.6, (231.1, 232.3), s["v12090"]),
        ("control v12080", 231.7, (231.2, 232.3), s["v12080"]),
        ("v12081 (E218 on v12077)", *e218_forecasts["v12081"], s["v12081"]),
        ("control v12082 (E218 final)", *e218_forecasts["v12082"], s["v12082"]),
        ("v12083 (E218 without LIRF)", *e218_forecasts["v12083"], s["v12083"]),
    ]
    yb = [7.4, 6.4, 5.4, 4.4, 2.6, 1.6, 0.6]
    for y, (name, central, (lo, hi), observed) in zip(yb, uploads):
        ax2.barh(y, hi - lo, left=lo, height=0.42, color=BAND, edgecolor="none", zorder=2)
        ax2.plot([central, central], [y - 0.21, y + 0.21], color=INK2, linewidth=1.5, zorder=3)
        ax2.plot([observed], [y], marker="o", markersize=6, color=BLUE, markeredgecolor=SURFACE,
                 markeredgewidth=1.0, zorder=6)
        if hi - lo < 0.1:  # narrow corridors (the E218 uploads): label to the right of the corridor and point
            ax2.text(max(observed, hi) + 0.05, y, f"{observed:.4f}", fontsize=7, color=INK, ha="left", va="center")
        else:
            ax2.text(observed, y + 0.26, f"{observed:.4f}", fontsize=7, color=INK, ha="center", va="bottom")
    expectation(ax2, expected, 3.6)
    ax2.text(expected + sig_hi + 0.04, 3.6, f"{expected:.3f}", fontsize=7, color=INK2, va="center")
    ax2.axvline(s["v12077"], color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
    ax2.text(s["v12077"] + 0.03, 8.0, "v12077 230.4812", fontsize=6.8, color=INK2, ha="left", va="bottom")
    ax2.set_yticks(yb + [3.6], [u[0] for u in uploads] + ["expected v12080 (H)"], fontsize=7.3)
    ax2.set_ylim(0.2, 8.4)
    ax2.set_xlim(230.25, 232.75)
    ax2.set_xlabel("leaderboard RMSE, January/July 2026 (s)", fontsize=7.8)
    ax2.grid(axis="y", visible=False)
    ax2.set_title("(b) Scores against forecasts", loc="left", fontsize=8.5, fontweight="bold")
    handles = [
        Patch(facecolor=BAND, label="corridor written before the upload"),
        Line2D([], [], color=INK2, linewidth=1.5, label="central forecast"),
        Line2D([], [], color=BLUE, marker="o", linestyle="none", markersize=6, label="score"),
        Line2D([], [], color=ORANGE, marker="o", markerfacecolor=SURFACE, linewidth=3.0, markersize=6,
               label="measured expectation, ±1σ (thick/thin: two noise estimates)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    path = OUT / "final_recipe_price.png"
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path.name, f"(sigma of control {sig_lo:.3f}-{sig_hi:.3f} s; expectation {expected:.4f} s)")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    copy_reused()
    nn_weight_by_month()
    final_recipe_price()
    return 0


if __name__ == "__main__":
    sys.exit(main())

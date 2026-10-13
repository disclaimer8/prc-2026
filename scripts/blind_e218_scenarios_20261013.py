"""Journal 467: E218 scenario bounds for the blind February/June months.

Label-free, 0 fits. Reads the sealed E218 cell table and the board verdict
of v12083, solves the January/July transfer shares in the final-file
parametrisation, then prices February/June under registered beta scenarios.

Cell model (registered in journal 467): before the shift, the mean residual
of a cell is -2*beta*c, where c is the rounded stage shift, so
    dSSE(beta) = sum_cells n * c^2 * (1 - 4*beta).
beta = 0.25 is break-even; beta = 0 means the published gap is pure noise.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELLS = ROOT / "analysis/final_tail_20261011/rows_changed_by_cell.csv"
VERDICT = ROOT / "analysis/v12083_upload_20261011/verdict.json"
OUT_DIR = ROOT / "analysis/blind_e218_scenarios_20261013"
REF_RMSE = 231.0
N_JANJUL = 344_841
N_FEB = 144_158
N_JUN = 181_791
N_FEBJUN = N_FEB + N_JUN
N_ALL = 670_790


def load_cells() -> list[dict]:
    rows = []
    with CELLS.open() as fh:
        for r in csv.DictReader(fh):
            n = int(r["changed"])
            if n == 0:
                continue
            shift = float(r["shift_mode"])
            assert shift == int(shift)
            rows.append(
                {
                    "airport": r["ADEP_mvt"],
                    "month": int(r["month"]),
                    "n": n,
                    "c": int(shift),
                    "nc2": n * int(shift) ** 2,
                }
            )
    return rows


def sse_factor(cells: list[dict], beta: float) -> float:
    return sum(c["nc2"] for c in cells) * (1.0 - 4.0 * beta)


def solve_beta(cells: list[dict], measured_dsse: float) -> float:
    total = sum(c["nc2"] for c in cells)
    return (1.0 - measured_dsse / total) / 4.0


def drmse(dsse: float, n: int) -> float:
    dmse = dsse / n
    return (REF_RMSE**2 + dmse) ** 0.5 - REF_RMSE


def main() -> None:
    cells = load_cells()
    verdict = json.loads(VERDICT.read_text())
    split = verdict["split_v12082_minus_v12080"]

    janjul = [c for c in cells if c["month"] in (1, 7)]
    febjun = [c for c in cells if c["month"] in (2, 6)]
    lirf_jj = [c for c in janjul if c["airport"] == "LIRF"]
    nonlirf_jj = [c for c in janjul if c["airport"] != "LIRF"]
    lirf_fj = [c for c in febjun if c["airport"] == "LIRF"]
    nonlirf_fj = [c for c in febjun if c["airport"] != "LIRF"]

    # Registered cross-checks against verdict.json (journal 467).
    lirf_fj_nc2 = sum(c["nc2"] for c in lirf_fj)
    assert lirf_fj_nc2 == 6_977_098, lirf_fj_nc2
    beta_lirf = solve_beta(lirf_jj, split["lirf"]["dsse"])
    beta_nonlirf = solve_beta(nonlirf_jj, split["non_lirf"]["dsse"])
    # Solved betas must reproduce the measured January/July split.
    for part, part_cells, beta in (
        ("lirf", lirf_jj, beta_lirf),
        ("non_lirf", nonlirf_jj, beta_nonlirf),
    ):
        repro = drmse(sse_factor(part_cells, beta), N_JANJUL)
        assert abs(repro - split[part]["delta_s"]) < 0.002, (part, repro)

    measured_jj_dsse = split["total"]["dsse"]
    subsets = {
        "february": [c for c in febjun if c["month"] == 2],
        "june": [c for c in febjun if c["month"] == 6],
        "feb_jun": febjun,
    }
    subset_n = {"february": N_FEB, "june": N_JUN, "feb_jun": N_FEBJUN}

    scenarios: dict[str, float | None] = {
        "beta_0_gap_is_noise": 0.0,
        "beta_0.25_break_even": 0.25,
        "measured_split_janjul": None,
        "beta_0.516_board_v12082": 0.516,
        "beta_0.74_lirf_janjul": 0.74,
        "beta_0.93_year2025": 0.93,
        "beta_1.0": 1.0,
    }
    results: dict = {
        "protocol": "blind-e218-scenarios-20261013-v1",
        "reference_rmse_for_drmse": REF_RMSE,
        "solved_janjul_betas_final_parametrisation": {
            "beta_LIRF": beta_lirf,
            "beta_nonLIRF": beta_nonlirf,
        },
        "sum_nc2": {
            "jan_jul": sum(c["nc2"] for c in janjul),
            "feb_jun": sum(c["nc2"] for c in febjun),
            "feb_jun_LIRF": lirf_fj_nc2,
            "feb_jun_nonLIRF": sum(c["nc2"] for c in nonlirf_fj),
        },
        "subsets": {},
        "four_months": {},
        "tornado_feb_jun": sorted(
            febjun, key=lambda c: -c["nc2"]
        ),
    }
    for name, sub in subsets.items():
        entry = {}
        for scen, beta in scenarios.items():
            if beta is None:
                dsse = sse_factor(
                    [c for c in sub if c["airport"] == "LIRF"], beta_lirf
                ) + sse_factor(
                    [c for c in sub if c["airport"] != "LIRF"], beta_nonlirf
                )
            else:
                dsse = sse_factor(sub, beta)
            entry[scen] = {
                "dsse": dsse,
                "dmse_per_row": dsse / subset_n[name],
                "drmse_at_ref": drmse(dsse, subset_n[name]),
            }
        results["subsets"][name] = entry
    for scen, beta in scenarios.items():
        if beta is None:
            dsse_fj = sse_factor(lirf_fj, beta_lirf) + sse_factor(
                nonlirf_fj, beta_nonlirf
            )
        else:
            dsse_fj = sse_factor(febjun, beta)
        total = measured_jj_dsse + dsse_fj
        results["four_months"][scen] = {
            "dsse": total,
            "dmse_per_row": total / N_ALL,
            "drmse_at_ref": drmse(total, N_ALL),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False) + "\n"
    )
    top = results["tornado_feb_jun"][:5]
    print(f"beta_LIRF(janjul, final param) = {beta_lirf:.4f}")
    print(f"beta_nonLIRF(janjul, final param) = {beta_nonlirf:.4f}")
    print(f"sum n*c^2 feb+jun = {results['sum_nc2']['feb_jun']:,}")
    for name in ("february", "june", "feb_jun"):
        e = results["subsets"][name]
        print(
            f"{name}: beta0 {e['beta_0_gap_is_noise']['drmse_at_ref']:+.4f} | "
            f"split {e['measured_split_janjul']['drmse_at_ref']:+.4f} | "
            f"beta0.93 {e['beta_0.93_year2025']['drmse_at_ref']:+.4f}"
        )
    e4 = results["four_months"]
    print(
        f"four months: beta0 {e4['beta_0_gap_is_noise']['drmse_at_ref']:+.4f} | "
        f"split {e4['measured_split_janjul']['drmse_at_ref']:+.4f} | "
        f"beta0.93 {e4['beta_0.93_year2025']['drmse_at_ref']:+.4f}"
    )
    print("tornado top:", [(c["airport"], c["month"], c["nc2"]) for c in top])


if __name__ == "__main__":
    main()

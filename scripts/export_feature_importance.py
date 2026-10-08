"""Export LightGBM feature importance of the saved v870 classifier and regressor.

    .venv/bin/python scripts/export_feature_importance.py [--submissions-dir submissions] [--out-dir docs/tables]

Reads the two saved LightGBM text models that the best submission's chain loads through
`src/blend.py 870` (see docs/model-design.md):

    <submissions-dir>/clf_v870.txt   six-class recording-regime classifier
    <submissions-dir>/reg_v870.txt   single-seed LightGBM residual regressor (the chain itself uses
                                     the five-seed average in lgb_reg_seeds_v870.npz; those five
                                     boosters were not saved)

and writes, for every feature of each model, total gain and split counts:

    <out-dir>/importance_classifier_v870.csv
    <out-dir>/importance_regressor_v870.csv
    <out-dir>/feature_families.csv           columns per family and each family's share of total gain

Gain is LightGBM's `importance_type="gain"`: the summed loss reduction of all splits on the feature
(for the classifier, over the trees of all six classes). It describes how the fitted trees use a
feature, not its causal effect or its marginal value to RMSE; correlated features share gain.
No model is trained. If the neural member's prepared-input manifest is available
(`--neural-manifest`, default data/e190_rank_inputs_20261006/manifest.json), the family table also
counts its tabular input columns; otherwise that column is left empty.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]

# (family, source, plain description, predicate on the column name). The first match wins.
FAMILIES = [
    ("Target encodings", "2025 training labels, month-out",
     "Smoothed historical taxi-out by stand/runway/hour/operator/type/destination; tree models drop them (E99)",
     lambda c: c.startswith(("te_", "ter_"))),
    ("Planning-time intervals", "Challenge data: movement + NM flight fields",
     "Take-off minus AOBT_3/EOBT/LOBT/IOBT/schedule and related gaps; feasibility flags of AOBT_3",
     lambda c: c.startswith(("d_mvt_", "d_aobt_", "d_eobt_")) or c in {"flight_time_plan", "aobt_minute_exact", "aobt_infeasible"}),
    ("Planned-arrival residuals (E165)", "Challenge data: NM planned times",
     "Take-off minus planned arrival, and planned flight time, relative to route medians",
     lambda c: c.startswith("plan_")),
    ("Airport, runway, aircraft and flight identity", "Challenge data",
     "Categorical context: airport, runway, aircraft type, wake class, operator, airline prefix, market segment, destination region",
     lambda c: c in {"ADEP_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt",
                     "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt", "stand_prefix", "ap_runway", "ap_stand", "ades_region",
                     "airline_prefix", "rwy_config", "diverted"}),
    ("Local time", "Challenge data", "Local hour and day of week of the take-off", lambda c: c in {"local_hour", "local_dow"}),
    ("Airport traffic counts", "Challenge data: all movements at the airport",
     "Departures, arrivals and off-blocks in 10-60 min windows; aircraft taxiing at off-block; recent arrival taxi-in times",
     lambda c: c.startswith(("dep_before_", "dep_after_", "arr_before_", "arr_around_ob_", "offblocks_around_ob_", "taxi_in_"))
     or c in {"taxiing_at_ob", "takeoffs_between_ob_mvt"}),
    ("Runway traffic and configuration", "Challenge data: all movements on the runway",
     "Departures/arrivals on the same runway, time since/until the nearest arrival, active runway counts",
     lambda c: c.startswith(("rwy_dep_", "rwy_arr_", "rwy_since_", "rwy_to_")) or c in {"n_dep_rwys", "n_arr_rwys", "own_rwy_share_hour"}),
    ("Neighbour NM taxi proxies", "Challenge data: other departures' take-off minus AOBT_3",
     "Count/mean/std of other departures' NM taxi proxy around the take-off, at airport, runway and stand level",
     lambda c: c.startswith("nb_")),
    ("Stand turnaround", "Challenge data: arrivals on the same stand",
     "Age, taxi-in and aircraft type of the previous arrival on the stand; gap to the next one",
     lambda c: c.startswith(("stand_prev_arr_", "stand_next_arr_"))),
    ("Forward neighbours (E28)", "Challenge data: later departures",
     "The same NM proxy statistics over departures taking off in the next 15/60 min",
     lambda c: c.startswith("fw_")),
    ("Filed-time neighbours (E77)", "Challenge data: other departures' take-off minus LOBT / schedule",
     "Neighbour statistics built from LOBT and scheduled time instead of AOBT_3",
     lambda c: c.startswith(("nbf_", "nbs_"))),
    ("Weather (METAR)", "Iowa Environmental Mesonet METAR archive",
     "Temperature, dew-point spread, wind, gust, visibility, ceiling and present-weather flags at off-block",
     lambda c: c.startswith("wx_") or c == "metar_age_min"),
    ("Stand-to-runway geometry", "OpenStreetMap (ODbL)",
     "Straight distance and bearing offset from the stand to the runway start",
     lambda c: c.startswith("geo_")),
    ("Taxi-route length (E51)", "OpenStreetMap taxiway graph (ODbL)",
     "Shortest taxiway-graph distance stand to runway start and its detour over the straight line",
     lambda c: c.startswith("route_")),
    ("Wake separation (E157)", "Challenge data (runway movements, wake category)",
     "Gap to the previous take-off on the runway versus the median wake-pair gap; heavy aircraft ahead",
     lambda c: c.startswith("e157_")),
    ("Destination arrival ATFM delay (E163)", "EUROCONTROL Aviation Intelligence Portal (ansperformance.eu)",
     "Destination airport's daily arrival ATFM delay per arrival on the take-off UTC date",
     lambda c: c.startswith("dst_")),
]


def family(col: str) -> str:
    for name, _src, _desc, match in FAMILIES:
        if match(col):
            return name
    return "Other"


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def export(model_path: Path, out_path: Path) -> dict:
    booster = lgb.Booster(model_file=str(model_path))
    names = booster.feature_name()
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    total_gain, total_split = float(gain.sum()), float(split.sum())
    rows = sorted(zip(names, gain, split), key=lambda r: -r[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank_by_gain", "feature", "family", "gain", "gain_share", "split", "split_share"])
        for i, (n, g, s) in enumerate(rows, start=1):
            w.writerow([i, n, family(n), f"{g:.6g}", f"{g / total_gain:.6f}", int(s), f"{s / total_split:.6f}"])
    info = dict(model=model_path.name, features=len(names), trees=booster.num_trees(),
                iterations=booster.current_iteration(), out=str(out_path))
    print(json.dumps(info))
    return {"names": names, "gain": dict(zip(names, gain / total_gain))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--submissions-dir", default="submissions")
    ap.add_argument("--out-dir", default="docs/tables")
    ap.add_argument("--version", default="870")
    ap.add_argument("--neural-manifest", default="data/e190_rank_inputs_20261006/manifest.json")
    a = ap.parse_args()
    sub, out = resolve(a.submissions_dir), resolve(a.out_dir)
    clf = export(sub / f"clf_v{a.version}.txt", out / f"importance_classifier_v{a.version}.csv")
    reg = export(sub / f"reg_v{a.version}.txt", out / f"importance_regressor_v{a.version}.csv")

    neural_cols = None
    manifest = resolve(a.neural_manifest)
    if manifest.is_file():
        neural_cols = json.loads(manifest.read_text())["spec"]["columns"]
        print(f"neural tabular inputs: {len(neural_cols)} columns from {manifest.name}")
    all_cols = list(dict.fromkeys(clf["names"] + reg["names"] + (neural_cols or [])))
    with (out / "feature_families.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "source", "description", "example_columns", "n_classifier", "n_regressor",
                    "n_neural_tabular", "classifier_gain_share", "regressor_gain_share"])
        for name, src, desc, _m in FAMILIES + [("Other", "", "", None)]:
            cols = [c for c in all_cols if family(c) == name]
            if not cols:
                continue
            nc = sum(c in clf["gain"] for c in cols)
            nr = sum(c in reg["gain"] for c in cols)
            nn = "" if neural_cols is None else sum(c in neural_cols for c in cols)
            w.writerow([name, src, desc, " ".join(cols[:3]), nc, nr, nn,
                        f"{sum(clf['gain'].get(c, 0.0) for c in cols):.4f}",
                        f"{sum(reg['gain'].get(c, 0.0) for c in cols):.4f}"])
    print(f"wrote {out / 'feature_families.csv'}")


if __name__ == "__main__":
    main()

"""H24 (journal 203): E163 generalised from the destination airport to the en-route airspace. E163 (destination arrival ATFM
delay, CV -0.76 6/6) works because a regulated flight is held to its slot; en-route regulations (ert_dly_ansp_*, daily en-route
ATFM delay per ANSP, ansperformance.eu/data, downloaded by Denys) give flights crossing a regulated ANSP the same slot.
Route = great circle ADEP -> ADES sampled at 41 points; each point takes the ANSP of the nearest large/medium airport
(OurAirports, ICAO prefix -> ANSP; > 400 km from any airport = ocean, skipped). Per departure and UTC day of take-off:
    ert_route_sum / ert_route_max   en-route ATFM delay per flight over the ANSPs crossed (minutes per flight that day)
    ert_route_nreg                  ANSPs crossed with >= 0.5 min per flight
    ert_dep / ert_dst               departure-country / destination-country ANSP
    ert_net                         EUROCONTROL Area (MS), network level
Screen on the E163 OOF (cv_preds_e163_hz, the model already has dst_*): LightGBM stage on the residual, half -> half, controls
+ dst_* against controls + dst_* + ert_* (as H16). Writes data/h24_ert_features.parquet for a CV if it passes.
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, "src")
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.neighbors import BallTree

ANSP = {"LF": "DSNA", "ED": "DFS", "ET": "DFS", "LE": "ENAIRE", "GC": "ENAIRE", "LI": "ENAV", "LS": "Skyguide", "EG": "NATS (Continental)",
        "EI": "AirNav Ireland", "EB": "skeyes", "EH": "LVNL", "EL": "ANA LUX", "LO": "Austro Control", "LK": "ANS CR", "EP": "PANSA",
        "LH": "HungaroControl (EC)", "LZ": "LPS", "LJ": "Slovenia Control", "LD": "Croatia Control", "LY": "SMATSA", "LW": "M-NAV",
        "LR": "ROMATSA", "LB": "BULATSA", "LG": "HASP", "LT": "DHMI", "LC": "DCAC Cyprus", "LM": "MATS", "LA": "Albcontrol",
        "LP": "NAV Portugal (Continental)", "EK": "NAVIAIR", "EN": "Avinor", "ES": "LFV", "EF": "Fintraffic ANS", "EE": "EANS",
        "EV": "LGS", "EY": "Oro Navigacija", "BI": "Isavia", "UG": "Sakaeronavigatsia", "UD": "ARMATS", "LU": "MOLDATSA"}

ert = pl.concat([pl.read_csv(f"data/ext_pru/ert_dly_ansp_{y}.csv", infer_schema_length=0, encoding="utf8-lossy") for y in (2024, 2025, 2026)])
ert = ert.select(ansp=pl.col("ENTITY_NAME"), day=pl.col("FLT_DATE").str.slice(0, 10).str.to_date(),
                 per=(pl.col("DLY_ERT_1").cast(pl.Float64).fill_null(0.0) / pl.col("FLT_ERT_1").cast(pl.Float64)).fill_nan(0.0)).unique(["ansp", "day"])

ap = pl.read_csv("data/ext/ourairports_airports.csv", infer_schema_length=0).select(
    "ident", "type", "gps_code", lat=pl.col("latitude_deg").cast(pl.Float64), lon=pl.col("longitude_deg").cast(pl.Float64))
coord = {}
for r in ap.filter(pl.col("gps_code").is_not_null()).iter_rows(named=True):
    coord.setdefault(r["gps_code"], (r["lat"], r["lon"]))
for r in ap.iter_rows(named=True):
    coord[r["ident"]] = (r["lat"], r["lon"])
ref = ap.filter(pl.col("type").is_in(["large_airport", "medium_airport"]) & pl.col("ident").str.slice(0, 2).is_in(list(ANSP))
                & (pl.col("ident").str.len_chars() == 4))
tree = BallTree(np.radians(ref.select("lat", "lon").to_numpy()), metric="haversine")
ref_ansp = np.array([ANSP[i[:2]] for i in ref["ident"].to_list()])

cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt"]
files = sorted(glob.glob("data/prc-2026-datasets/training_*.parquet")) + ["data/prc-2026-datasets/ranking.parquet"]
dep = pl.concat([pl.read_parquet(p, columns=cols) for p in files]).filter(pl.col("PHASE_mvt") == "DEP").select(
    "MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", day=pl.col("MVT_TIME_UTC_mvt").dt.date(), month=pl.col("MVT_TIME_UTC_mvt").dt.month(),
    year=pl.col("MVT_TIME_UTC_mvt").dt.year())


def gc_points(a, b, n=41):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    p1 = np.array([np.cos(la1) * np.cos(lo1), np.cos(la1) * np.sin(lo1), np.sin(la1)])
    p2 = np.array([np.cos(la2) * np.cos(lo2), np.cos(la2) * np.sin(lo2), np.sin(la2)])
    om = np.arccos(np.clip(p1 @ p2, -1, 1))
    f = np.linspace(0, 1, n)[:, None]
    pts = p1 if om < 1e-9 else (np.sin((1 - f) * om) * p1 + np.sin(f * om) * p2) / np.sin(om)
    pts = np.atleast_2d(pts)
    return np.column_stack([np.arcsin(pts[:, 2]), np.arctan2(pts[:, 1], pts[:, 0])])


pairs = dep.select("ADEP_mvt", "ADES_mvt").unique()
rows, miss = [], 0
for a, b in pairs.iter_rows():
    if a not in coord or b not in coord:
        miss += 1
        continue
    dist, idx = tree.query(gc_points(coord[a], coord[b]), k=1)
    seen = list(dict.fromkeys(ref_ansp[idx[:, 0]][dist[:, 0] * 6371 < 400]))
    rows += [(a, b, s) for s in seen]
route = pl.DataFrame(rows, schema=["ADEP_mvt", "ADES_mvt", "ansp"], orient="row")
print(f"pairs {pairs.height}, without coordinates {miss}; ANSPs per route: median "
      f"{route.group_by('ADEP_mvt', 'ADES_mvt').len()['len'].median()}", flush=True)

x = dep.select("MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "day").join(route, on=["ADEP_mvt", "ADES_mvt"]).join(ert, on=["ansp", "day"], how="left")
agg = x.group_by("MVT_ID_mvt").agg(ert_route_sum=pl.col("per").sum(), ert_route_max=pl.col("per").max(), ert_route_nreg=(pl.col("per") >= 0.5).sum().cast(pl.Float64))
ctry = lambda c: pl.col(c).str.slice(0, 2).replace_strict(ANSP, default=None)
feat = (dep.with_columns(a_dep=ctry("ADEP_mvt"), a_dst=ctry("ADES_mvt"))
        .join(ert.rename({"ansp": "a_dep", "per": "ert_dep"}), on=["a_dep", "day"], how="left")
        .join(ert.rename({"ansp": "a_dst", "per": "ert_dst"}), on=["a_dst", "day"], how="left")
        .join(ert.filter(pl.col("ansp") == "EUROCONTROL Area (MS)").select("day", ert_net="per"), on="day", how="left")
        .join(agg, on="MVT_ID_mvt", how="left"))
NEW = ["ert_route_sum", "ert_route_max", "ert_route_nreg", "ert_dep", "ert_dst", "ert_net"]
feat.select("MVT_ID_mvt", *NEW).write_parquet("data/h24_ert_features.parquet")
for yr, g in feat.group_by("year"):
    print(f"year {yr[0]}: rows {g.height}, route known {g['ert_route_sum'].is_not_null().mean():.3f}, route delay > 0 "
          f"{(g['ert_route_sum'].fill_null(0) > 0).mean():.3f}, >= 0.5 min {(g['ert_route_sum'].fill_null(0) >= 0.5).mean():.3f}, "
          f"mean {g['ert_route_sum'].mean():.3f}")

# screen: residual of the E163 model
DST = ["dst_dly_per_arr", "dst_share_dly", "dst_w_per_arr", "dst_cap_per_arr", "dst_known"]
schema = pl.read_parquet_schema("data/features_e163.parquet")
CTRL = [c for c in ("d_mvt_aobt", "d_mvt_lobt", "taxiing_at_ob", "rwy_dep_before_10", "rwy_dep_before_30", "rwy_dep_after_10",
                    "dep_before_30", "arr_before_30", "offblocks_around_ob_30", "nbf_ap_30_0_mean", "nb_rwy_30_30_mean",
                    "nb_ap_30_30_mean", "taxi_in_mean_30") if c in schema] + DST
base = pl.read_parquet("data/features_e163.parquet", columns=["MVT_ID_mvt", "split", *CTRL]).filter(pl.col("split") == "train").drop("split")
oof = pl.concat([pl.DataFrame({k: np.load(p)[k] for k in ("mvt_id", "pred", "y")}) for p in sorted(glob.glob("data/cv_preds_e163_hz/fold_*.npz"))]).rename({"mvt_id": "MVT_ID_mvt"})
full = feat.select("MVT_ID_mvt", "month", *NEW).join(base, on="MVT_ID_mvt").join(oof, on="MVT_ID_mvt").with_columns(r=pl.col("y") - pl.col("pred"))
d = full.filter((pl.col("y") <= 7200) & (pl.col("r").abs() < 1800))
print(f"screen rows {d.height} (of {full.height})")


def rank(v):
    return pl.Series(v).fill_nan(None).rank().fill_null(0).to_numpy().astype(float)


def partial(yv, v, controls):
    Z = np.column_stack([np.ones(len(yv))] + [rank(c) for c in controls])
    res = lambda u: u - Z @ np.linalg.lstsq(Z, u, rcond=None)[0]
    return float(np.corrcoef(res(rank(yv)), res(rank(v)))[0, 1])


halves = {h: d.filter(pl.col("month").is_between(*rng)) for h, rng in (("H1", (1, 6)), ("H2", (7, 12)))}
for c in NEW:
    p = [partial(g["r"].to_numpy(), g[c].to_numpy(), [g[k].to_numpy() for k in CTRL]) for g in halves.values()]
    print(f"{c:16s} partial(r | controls + dst_*) H1 {p[0]:+.4f} / H2 {p[1]:+.4f}", flush=True)
prm = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.9, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=42, num_threads=4, deterministic=True)
print("stage on the residual, half -> half")
for tr_h, te_h in (("H1", "H2"), ("H2", "H1")):
    tr, te = halves[tr_h], halves[te_h]
    line = f"  {tr_h} -> {te_h}: no stage {np.sqrt((te['r'].to_numpy() ** 2).mean()):.3f}"
    for name, cs in (("controls + dst", CTRL), ("+ en-route ATFM", CTRL + NEW), ("+ route only", CTRL + NEW[:3])):
        m = lgb.train(prm, lgb.Dataset(tr.select(cs).to_numpy(), tr["r"].to_numpy()), 300)
        line += f" | {name} {np.sqrt(((te['r'].to_numpy() - m.predict(te.select(cs).to_numpy())) ** 2).mean()):.3f}"
    print(line, flush=True)

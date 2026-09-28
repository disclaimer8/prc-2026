"""E151 (journal 131): flows over the departure's OWN taxi interval, after Yuan, Liu & Chen 2024 (Appl. Sci. 14(21) 9968, table 2).

    FEATURES=data/features_e106.parquet FEATURES_OUT=data/features_e151.parquet .venv/bin/python src/e151_own_interval_flows.py

For departure i with interval [a_i, t_i] = [AOBT_3 (LOBT where missing), MVT] and other intervals [s_j, e_j]
(departures [AOBT_3, MVT]; arrivals [ALDT = MVT, AIBT = MVT + taxi-in], both known in the ranking file):
    D1/A1  s_j < a_i,       a_i < e_j < t_i     ahead of me, finished during my taxi
    D2/A2  s_j > a_i,       e_j < t_i           nested: started after me, finished before me (paper: corr 0.81 / 0.74)
    D3/A3  s_j < a_i,       e_j > t_i           spans my whole taxi
    D4/A4  a_i < s_j < t_i, e_j > t_i           started during my taxi, still going
Levels: ap (airport), rw (departure runway; arrivals landing on it), cor (runway x stand zone; zone = stand_prefix, else
a 600 m cell of the runway-relative stand position; arrivals into the same zone, letter prefixes only), plus dprop =
D / (D + A) per level. Rows whose own interval is missing or outside 0..7200 s get nulls. The probe on 2025
(src/e151_corridor_probe.py) reported partial Spearman with the base OOF residual up to |0.02|, sign stable by half year.
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

CELL = 600.0
KMAX = 400
DATA = "data/prc-2026-datasets/"


def counts(a, t, s, e):
    """D1..D4 for queries [a, t] against objects [s, e]; self (identical interval) is excluded by the strict inequalities."""
    o = np.argsort(s, kind="stable"); s, e = s[o], e[o]
    es = np.sort(e)
    n_s_lt_a = np.searchsorted(s, a, "left")
    n_e_lt_t = np.searchsorted(es, t, "left")
    start_in = np.searchsorted(s, t, "left") - np.searchsorted(s, a, "right")
    end_in = n_e_lt_t - np.searchsorted(es, a, "right")
    lo, hi = np.searchsorted(s, a, "right"), np.searchsorted(s, t, "left")
    nested = np.zeros(len(a))
    for k in range(min(KMAX, int((hi - lo).max(initial=0)))):
        j = lo + k
        ok = j < hi
        nested[ok] += e[j[ok]] < t[ok]
    return end_in - nested, nested, n_s_lt_a - n_e_lt_t + nested, start_in - nested


def build():
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt",
            "LOBT_flt", "TAXITIME_SEC_mvt"]
    files = sorted(glob.glob(DATA + "training_*.parquet")) + [DATA + "ranking.parquet"]
    raw = pl.concat([pl.read_parquet(p, columns=cols) for p in files])
    ep = lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)
    feat_in = os.environ.get("FEATURES", "data/features_e106.parquet")
    geo = pl.read_parquet(feat_in, columns=["MVT_ID_mvt", "stand_prefix", "geo_dist_along", "geo_dist_across"])
    cell = lambda c: (pl.col(c).fill_nan(None) / CELL).floor().cast(pl.Int32).cast(pl.Utf8)
    dep = (raw.filter(pl.col("PHASE_mvt") == "DEP")
           .select("MVT_ID_mvt", ap=pl.col("ADEP_mvt"), rwy=pl.col("RUNWAY_mvt"), t=ep("MVT_TIME_UTC_mvt"),
                   a=pl.coalesce(ep("AOBT_3_flt"), ep("LOBT_flt")))
           .filter(pl.col("a").is_not_null() & (pl.col("t") - pl.col("a")).is_between(0, 7200))
           .join(geo, on="MVT_ID_mvt", how="left")
           .with_columns(zone=pl.coalesce(pl.col("stand_prefix"), cell("geo_dist_along") + "_" + cell("geo_dist_across"))))
    arr = raw.filter(pl.col("PHASE_mvt") == "ARR").select(
        ap=pl.col("ADES_mvt"), rwy=pl.col("RUNWAY_mvt"), zone=pl.col("STAND_mvt").str.extract(r"^([A-Za-z]+)", 1),
        s=ep("MVT_TIME_UTC_mvt"), e=ep("MVT_TIME_UTC_mvt") + pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)).drop_nulls(["s", "e"])

    def level(name, dkeys, akey_of):
        akeys = {"ap": ["ap"], "rw": ["ap", "rwy"], "cor": ["ap", "zone"]}[name]
        agroups = {k: g for k, g in arr.drop_nulls(akeys).group_by(akeys)}
        parts = []
        for key, g in dep.drop_nulls(dkeys).group_by(dkeys):
            a, t = g["a"].to_numpy(), g["t"].to_numpy()
            out = {"MVT_ID_mvt": g["MVT_ID_mvt"]}
            for i, v in enumerate(counts(a, t, a.copy(), t.copy()), 1):
                out[f"e151_{name}_D{i}"] = v
            ag = agroups.get(akey_of(key))
            for i in range(1, 5):
                out[f"e151_{name}_A{i}"] = np.full(len(a), np.nan)
            if ag is not None and ag.height:
                for i, v in enumerate(counts(a, t, ag["s"].to_numpy(), ag["e"].to_numpy()), 1):
                    out[f"e151_{name}_A{i}"] = v
            parts.append(pl.DataFrame(out))
        res = pl.concat(parts)
        D = sum(pl.col(f"e151_{name}_D{i}") for i in range(1, 5)); A = sum(pl.col(f"e151_{name}_A{i}") for i in range(1, 5))
        return res.with_columns((D / (D + A)).alias(f"e151_{name}_dprop"))

    fl = level("ap", ["ap"], lambda k: (k[0],))
    fl = fl.join(level("rw", ["ap", "rwy"], lambda k: (k[0], k[1])), on="MVT_ID_mvt", how="left")
    fl = fl.join(level("cor", ["ap", "rwy", "zone"], lambda k: (k[0], k[2])), on="MVT_ID_mvt", how="left")
    return fl


if __name__ == "__main__":
    fl = build()
    feat = pl.read_parquet(os.environ.get("FEATURES", "data/features_e106.parquet"))
    out = feat.join(fl, on="MVT_ID_mvt", how="left")
    assert out.height == feat.height
    out.write_parquet(os.environ.get("FEATURES_OUT", "data/features_e151.parquet"))
    new = [c for c in fl.columns if c != "MVT_ID_mvt"]
    print(f"wrote {out.height} rows, {len(new)} E151 columns")
    print(out.group_by("split").agg([pl.col(c).is_not_null().mean().round(3).alias(c) for c in ("e151_ap_D2", "e151_rw_A2", "e151_cor_D1", "e151_cor_A2")]
                                    + [pl.col("e151_ap_D2").mean().round(2).alias("ap_D2_mean"), pl.col("e151_ap_A2").mean().round(2).alias("ap_A2_mean")]))

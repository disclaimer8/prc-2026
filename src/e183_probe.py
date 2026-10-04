"""E183 probe (04.10, no training): does the label-free ADS-B proxy of the neighbours' residual of OUR model carry
information about the residual of a row WITHOUT its own track? Rule and definitions written before the numbers in
analysis/e183_probe_2026-10-04.md.

    OMP_NUM_THREADS=2 POLARS_MAX_THREADS=2 .venv/bin/python src/e183_probe.py > data/e183_probe.log

s = production OOF of the FIN1b base (data/cv_preds_fin1b_hz), r = y - s.
Own track = coverage >= 2 in the production extraction data/adsblol_kt1 (the E141 stage population). "No ADS-B at all" =
coverage < 2 and not an E64 row (vi_has_prev != 1, data/adsblol/visit_*).
Neighbour quantities (label-free except the oracle):
  nbq  = clip((MVT - t_first_move) + lag_airport - s, +-1800) on neighbours with coverage == 3 (pushback seen);
  nbq2 = same on coverage >= 2 with the lag per (airport, coverage) (reported);
  nbp  = (MVT - AOBT_3) - s where MVT - AOBT_3 in [1, 5399] (NM proxy);
  orc  = r of neighbours with y <= 7200 (oracle, uses labels).
Windows +-15/30/60 min on MVT time, by airport and by (airport, runway), self excluded, >= 2 neighbours with a value.
Writes data/e183_probe_rows.parquet (per-row features of 2025) for reuse by the stage.
"""
from __future__ import annotations

import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

R = "/Users/denyskolomiiets/prc-2026/"
WINDOWS = (900, 1800, 3600)
KINDS = ("nbq", "nbq2", "nbp", "orc")


def load_oof(path: str) -> pl.DataFrame:
    parts = []
    for f in sorted(glob.glob(f"{path}/fold_*.npz")):
        z = np.load(f)
        parts.append(pl.DataFrame({"MVT_ID_mvt": z["mvt_id"].astype(np.int64), "s": z["pred"].astype(np.float64),
                                   "y": z["y"].astype(np.float64), "fold": os.path.basename(f)[5:-4]}))
    return pl.concat(parts)


def load_adsb(year: int) -> pl.DataFrame:
    ob = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage", "t_first", "t_first_move"])
                    for f in sorted(glob.glob(f"{R}data/adsblol_kt1/offblock2_{year}-*.parquet"))])
    ob = ob.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)).sort("coverage", descending=True).unique("MVT_ID_mvt", keep="first")
    vi = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "vi_has_prev"])
                    for f in sorted(glob.glob(f"{R}data/adsblol/visit_{year}-*.parquet"))])
    vi = vi.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)).sort("vi_has_prev", descending=True, nulls_last=True).unique(
        "MVT_ID_mvt", keep="first")
    return ob.join(vi, on="MVT_ID_mvt", how="full", coalesce=True)


def load_raw(paths: list[str]) -> pl.DataFrame:
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "RUNWAY_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt",
            "SCHED_TIME_UTC_mvt"]
    raw = pl.concat([pl.read_parquet(p, columns=cols) for p in paths]).filter(pl.col("PHASE_mvt") == "DEP")
    return raw.select(
        pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", "RUNWAY_mvt",
        has_nm=pl.col("FLIGHT_ID_mvt").is_not_null(),
        t=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64),
        aobt=pl.col("AOBT_3_flt").dt.epoch("s").cast(pl.Float64),
        sched=pl.col("SCHED_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64))


def with_track(d: pl.DataFrame) -> pl.DataFrame:
    """coverage (-1 = no file row), own-track flags, ADS-B taxi."""
    return d.with_columns(coverage=pl.col("coverage").fill_null(-1)).with_columns(
        own_track=pl.col("coverage") >= 2,
        e64_row=(pl.col("coverage") < 2) & (pl.col("vi_has_prev") == 1).fill_null(False),
        adsb_taxi=pl.when(pl.col("coverage") >= 2).then(pl.col("t") - pl.col("t_first_move")),
        own_p=pl.col("t") - pl.col("aobt"),
    ).with_columns(no_adsb=~pl.col("own_track") & ~pl.col("e64_row"))


def lags(d: pl.DataFrame) -> tuple[dict, dict]:
    """Per airport (coverage 3) and per (airport, coverage) median of y - ADS-B taxi; 2025 labels, a constant."""
    ok = d.filter(pl.col("adsb_taxi").is_between(30, 7200) & pl.col("y").is_between(1, 7200))
    l3 = dict(ok.filter(pl.col("coverage") == 3).group_by("ADEP_mvt").agg((pl.col("y") - pl.col("adsb_taxi")).median()).iter_rows())
    l2 = {(a, c): v for a, c, v in ok.group_by("ADEP_mvt", "coverage").agg((pl.col("y") - pl.col("adsb_taxi")).median()).iter_rows()}
    return l3, l2


def point_values(d: pl.DataFrame, l3: dict, l2: dict, with_oracle: bool) -> pl.DataFrame:
    """Per-row quantities whose neighbour means are the features (NaN where undefined)."""
    lag3 = pl.col("ADEP_mvt").replace_strict(l3, default=None, return_dtype=pl.Float64)
    l2df = pl.DataFrame([(a, c, v) for (a, c), v in l2.items()], schema={"ADEP_mvt": pl.String, "coverage": pl.Int64, "_lag2": pl.Float64},
                        orient="row")
    d = d.join(l2df, on=["ADEP_mvt", "coverage"], how="left", maintain_order="left")
    lag2 = pl.col("_lag2")
    ok_taxi = pl.col("adsb_taxi").is_between(30, 7200)
    out = d.with_columns(
        v_nbq=pl.when((pl.col("coverage") == 3) & ok_taxi).then((pl.col("adsb_taxi") + lag3 - pl.col("s")).clip(-1800, 1800)),
        v_nbq2=pl.when((pl.col("coverage") >= 2) & ok_taxi).then((pl.col("adsb_taxi") + lag2 - pl.col("s")).clip(-1800, 1800)),
        v_nbp=pl.when(pl.col("own_p").is_between(1, 5399)).then(pl.col("own_p") - pl.col("s")),
    )
    if with_oracle:
        out = out.with_columns(v_orc=pl.when(pl.col("y") <= 7200).then(pl.col("y") - pl.col("s")))
    return out


def win_mean(t: np.ndarray, v: np.ndarray, w: float) -> np.ndarray:
    """Mean of non-NaN v over |t_j - t_i| <= w, self excluded, NaN unless >= 2 values; t sorted."""
    ok = ~np.isnan(v)
    vv = np.where(ok, v, 0.0)
    cs, cn = np.concatenate([[0.0], np.cumsum(vv)]), np.concatenate([[0], np.cumsum(ok)])
    lo, hi = np.searchsorted(t, t - w, "left"), np.searchsorted(t, t + w, "right")
    s, n = cs[hi] - cs[lo] - vv, cn[hi] - cn[lo] - ok
    return np.where(n >= 2, s / np.maximum(n, 1), np.nan)


def neighbour_features(d: pl.DataFrame, kinds=KINDS) -> pl.DataFrame:
    """Columns <kind>_<scope>_<w> for every row of d (row order kept)."""
    d = d.with_columns(_i=pl.int_range(pl.len()))
    names = [f"{k}_{sc}_{w}" for k in kinds for sc in ("apt", "rwy") for w in WINDOWS]
    out = {c: np.full(d.height, np.nan) for c in names}
    for scope, keys in (("apt", ["ADEP_mvt"]), ("rwy", ["ADEP_mvt", "RUNWAY_mvt"])):
        for _, g in d.group_by(keys):
            g = g.sort("t")
            t, idx = g["t"].to_numpy(), g["_i"].to_numpy()
            for k in kinds:
                v = g[f"v_{k}"].fill_null(np.nan).to_numpy().astype(np.float64)
                for w in WINDOWS:
                    out[f"{k}_{scope}_{w}"][idx] = win_mean(t, v, w)
    return pl.DataFrame(out)


def corr(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 50:
        return float("nan"), int(m.sum())
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


def centred(df: pl.DataFrame, c: str, by=("ADEP_mvt", "day")) -> tuple[np.ndarray, np.ndarray]:
    g = df.filter(pl.col(c).is_not_nan() & pl.col(c).is_not_null())
    x = (g[c] - g.select(pl.col(c).mean().over(*by)).to_series()).to_numpy()
    r = (g["r"] - g.select(pl.col("r").mean().over(*by)).to_series()).to_numpy()
    return x, r


def main() -> None:
    oof = load_oof(R + "data/cv_preds_fin1b_hz")
    raw = load_raw(sorted(glob.glob(R + "data/prc-2026-datasets/training_*.parquet")))
    d = raw.join(oof, on="MVT_ID_mvt", how="inner").join(load_adsb(2025), on="MVT_ID_mvt", how="left")
    d = with_track(d)
    print(f"2025 departures with FIN1b OOF: {d.height:,}", flush=True)
    l3, l2 = lags(d)
    print("lag (coverage 3) by airport:", {k: round(v) for k, v in sorted(l3.items())}, flush=True)
    d = point_values(d, l3, l2, with_oracle=True)
    nb = neighbour_features(d)
    d = pl.concat([d, nb], how="horizontal").with_columns(
        r=pl.col("y") - pl.col("s"), day=(pl.col("t") // 86400).cast(pl.Int64),
        month=pl.from_epoch(pl.col("t").cast(pl.Int64), time_unit="s").dt.month())
    keep = ["MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "fold", "t", "day", "month", "s", "y", "coverage", "own_track", "e64_row",
            "no_adsb", "has_nm", "own_p", "sched"] + nb.columns
    d.select(keep).write_parquet(R + "data/e183_probe_rows.parquet")

    # shares of rows without own track, 2025 and ranking
    rk = load_raw([R + "data/prc-2026-datasets/ranking.parquet"]).join(load_adsb(2026), on="MVT_ID_mvt", how="left")
    rk = with_track(rk.with_columns(s=pl.lit(None, dtype=pl.Float64), y=pl.lit(None, dtype=pl.Float64)))
    sh = d.group_by("ADEP_mvt").agg(n25=pl.len(), no_track25=(~pl.col("own_track")).mean(), no_adsb25=pl.col("no_adsb").mean()).join(
        rk.group_by("ADEP_mvt").agg(n26=pl.len(), no_track26=(~pl.col("own_track")).mean(), no_adsb26=pl.col("no_adsb").mean()),
        on="ADEP_mvt").sort("ADEP_mvt")
    tot = pl.DataFrame({"ADEP_mvt": ["ALL"], "n25": [d.height], "no_track25": [float((~d["own_track"]).mean())],
                        "no_adsb25": [float(d["no_adsb"].mean())], "n26": [rk.height], "no_track26": [float((~rk["own_track"]).mean())],
                        "no_adsb26": [float(rk["no_adsb"].mean())]}).cast(sh.schema)
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=20, float_precision=3, tbl_hide_dataframe_shape=True,
                   tbl_hide_column_data_types=True):
        print("\nShare of departures without own track (coverage < 2) and without any ADS-B (not E64 either)")
        print(pl.concat([sh, tot]))

    for pop_name, pop in (("no own track (coverage < 2)", ~pl.col("own_track")), ("no ADS-B at all (not E64)", pl.col("no_adsb"))):
        T = d.filter(pop & pl.col("y").is_between(1, 7200))
        print(f"\n===== target rows: {pop_name}, 0 < y <= 7200: {T.height:,}", flush=True)
        rows = []
        for k in KINDS:
            for sc in ("apt", "rwy"):
                for w in WINDOWS:
                    c = f"{k}_{sc}_{w}"
                    a, n = corr(T[c].to_numpy(), T["r"].to_numpy())
                    b, _ = corr(*centred(T, c))
                    rows.append((c, n, a, b))
        with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=60, float_precision=4, tbl_hide_dataframe_shape=True,
                       tbl_hide_column_data_types=True):
            print(pl.DataFrame(rows, schema=["feature", "n", "corr_raw", "corr_centred"], orient="row"))
        # by airport, the decisive and companions
        rows = []
        for (ap,), g in sorted(T.group_by(["ADEP_mvt"]), key=lambda z: z[0]):
            row = {"airport": ap, "rows": g.height}
            for c in ("nbq_apt_1800", "nbq_rwy_1800", "nbq_apt_3600", "nbq2_apt_1800", "nbp_apt_1800", "orc_apt_1800"):
                row[f"{c}_n"] = int(g[c].is_not_nan().sum())
                row[c] = corr(*centred(g, c))[0]
            rows.append(row)
        with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=20, float_precision=3, tbl_hide_dataframe_shape=True,
                       tbl_hide_column_data_types=True, tbl_cols=30, tbl_width_chars=400):
            print("\nby airport, centred within (airport, UTC day):")
            print(pl.DataFrame(rows).select("airport", "rows", "nbq_apt_1800_n", "nbq_apt_1800", "nbq_rwy_1800", "nbq_apt_3600",
                                            "nbq2_apt_1800", "nbp_apt_1800", "orc_apt_1800_n", "orc_apt_1800"))
        rows = []
        for (mo,), g in sorted(T.group_by(["month"]), key=lambda z: z[0]):
            row = {"month": mo, "rows": g.height}
            for c in ("nbq_apt_1800", "nbq2_apt_1800", "nbp_apt_1800", "orc_apt_1800"):
                row[c] = corr(*centred(g, c))[0]
            rows.append(row)
        with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=20, float_precision=3, tbl_hide_dataframe_shape=True,
                       tbl_hide_column_data_types=True, tbl_cols=30, tbl_width_chars=400):
            print("\nby month (pooled over airports), centred within (airport, UTC day):")
            print(pl.DataFrame(rows))
        # airport x month sign table of the decisive number
        rows = []
        for (ap,), g in sorted(T.group_by(["ADEP_mvt"]), key=lambda z: z[0]):
            row = {"airport": ap}
            for (mo,), h in sorted(g.group_by(["month"]), key=lambda z: z[0]):
                row[str(mo)] = corr(*centred(h, "nbq_apt_1800"))[0]
            rows.append(row)
        with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=20, float_precision=2, tbl_hide_dataframe_shape=True,
                       tbl_hide_column_data_types=True, tbl_cols=30, tbl_width_chars=400):
            print("\nnbq_apt_1800 centred, airport x month:")
            print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()

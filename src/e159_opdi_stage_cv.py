"""E159 (journal 171): OPDI exit-parking stage for departures without our adsb.lol ground coverage.

Mechanism check where OPDI is good and labels exist: LSZH 2025, January <-> July (fit on one month, score the other). Base =
production OOF (data/cv_preds_e116prod). Target population = rows with an OPDI exit-parking event and adsb.lol coverage < 2.
Stage = small LightGBM on r = y - base with d_opdi = (MVT - exit) - base, t_opdi = MVT - exit, base, twy = entry-taxiway - exit.
Matching as src/h14_opdi_probe.py (call sign + ADEP, nearest OPDI anchor in [-900, +1800] s of MVT).
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import lightgbm as lgb
import numpy as np
import polars as pl

APS = [a for a in os.environ.get("E159_APS", "LSZH").split(",")]
TYPES = ["exit-parking_position", "entry-taxiway", "take-off", "first_seen", "first-xing-fl50"]


def opdi_wide() -> pl.DataFrame:
    ev = pl.concat([pl.read_parquet(p, columns=["flight_id", "flt_id", "adep", "type", "t"]) for p in sorted(glob.glob("data/opdi/ev_*.parquet"))])
    ev = ev.filter(pl.col("type").is_in(TYPES)).with_columns(t=pl.col("t").dt.epoch("s").cast(pl.Float64))
    w = ev.group_by("flight_id", "type").agg(pl.col("t").min(), pl.col("flt_id").first(), pl.col("adep").first()) \
        .pivot(on="type", index=["flight_id", "flt_id", "adep"], values="t")
    for c in TYPES:
        if c not in w.columns:
            w = w.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    return w.with_columns(anchor=pl.coalesce("take-off", "first-xing-fl50", "first_seen")).filter(pl.col("anchor").is_not_null())


def match(d: pl.DataFrame, w: pl.DataFrame) -> pl.DataFrame:
    """d: departures with MVT_ID_mvt, ADEP_mvt, cs, mvt (epoch s). Returns d + exit / twy for matched flights."""
    j = d.join(w, left_on=["cs", "ADEP_mvt"], right_on=["flt_id", "adep"], how="left").with_columns(dt=(pl.col("anchor") - pl.col("mvt")))
    j = j.with_columns(ok=pl.when(pl.col("dt").is_between(-900, 1800)).then(pl.col("dt").abs()).otherwise(None)) \
        .sort("MVT_ID_mvt", "ok", nulls_last=True).unique("MVT_ID_mvt", keep="first")
    return j.select(*d.columns, exit=pl.when(pl.col("ok").is_not_null()).then(pl.col("exit-parking_position")),
                    twy=pl.when(pl.col("ok").is_not_null()).then(pl.col("entry-taxiway")))


if __name__ == "__main__":
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt"]
    tr = pl.concat([pl.read_parquet(p, columns=cols) for p in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))])
    d = tr.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("ADEP_mvt").is_in(APS)).with_columns(
        mvt=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64), month=pl.col("MVT_TIME_UTC_mvt").dt.month(),
        cs=pl.coalesce("CALLSIGN_flt", "FLIGHT_mvt").str.replace_all(" ", "")).filter(pl.col("month").is_in([1, 7])).drop("CALLSIGN_flt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "PHASE_mvt")
    j = match(d, opdi_wide()).filter(pl.col("exit").is_not_null())
    cov = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage"]) for f in sorted(glob.glob("data/adsblol_kt1/offblock2_2025-0[17]-*.parquet"))]).unique("MVT_ID_mvt")
    oof = pl.concat([pl.DataFrame({k: np.load(p)[k] for k in ("mvt_id", "pred", "y")}) for p in sorted(glob.glob("data/cv_preds_e116prod/fold_*.npz"))]).rename({"mvt_id": "MVT_ID_mvt"})
    j = j.join(cov, on="MVT_ID_mvt", how="left").with_columns(pl.col("coverage").fill_null(0)).join(oof, on="MVT_ID_mvt", how="inner").with_columns(
        t_opdi=pl.col("mvt") - pl.col("exit"), twy_s=pl.col("twy") - pl.col("exit")).with_columns(d_opdi=pl.col("t_opdi") - pl.col("pred"), r=pl.col("y") - pl.col("pred"))
    F = ["d_opdi", "t_opdi", "pred", "twy_s"]
    print(f"airports {APS}: rows with OPDI exit and label {j.height}; target (coverage < 2) {int((j['coverage'] < 2).sum())}")
    P = dict(objective="regression", learning_rate=0.05, num_leaves=15, min_data_in_leaf=100, lambda_l2=10.0, feature_fraction=1.0,
             bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42, num_threads=4, deterministic=True)
    tot = {"base": 0.0, "stage": 0.0}
    for tr_m, te_m in ((1, 7), (7, 1)):
        a, b = j.filter(pl.col("month") == tr_m), j.filter(pl.col("month") == te_m)
        m = lgb.train(P, lgb.Dataset(a.select(F).to_numpy(), a["r"].to_numpy()), int(os.environ.get("E159_ROUNDS", "300")))
        b = b.with_columns(new=pl.col("pred") + pl.Series(m.predict(b.select(F).to_numpy())))
        for name, sub in (("target cov<2", b.filter(pl.col("coverage") < 2)), ("all with OPDI", b), ("y<=7200 target", b.filter((pl.col("coverage") < 2) & (pl.col("y") <= 7200)))):
            e0, e1 = (sub["y"] - sub["pred"]).to_numpy(), (sub["y"] - sub["new"]).to_numpy()
            dd = e1 ** 2 - e0 ** 2
            trim = np.sort(dd)[3:-3].sum() if len(dd) > 6 else float("nan")
            print(f"fit {tr_m} -> score {te_m} | {name:16s} n {sub.height:6d} RMSE base {np.sqrt(np.mean(e0 ** 2)):8.2f} stage {np.sqrt(np.mean(e1 ** 2)):8.2f} "
                  f"({(np.sqrt(np.mean(e1 ** 2)) / np.sqrt(np.mean(e0 ** 2)) - 1) * 100:+.1f} %) dSSE {dd.sum() / 1e6:+.2f}e6 trimmed {trim / 1e6:+.2f}e6")
            if name == "target cov<2":
                tot["base"] += float((e0 ** 2).sum()); tot["stage"] += float((e1 ** 2).sum())
    print(f"target rows both directions: dSSE {(tot['stage'] - tot['base']) / 1e6:+.2f}e6 -> per 344841 ranking rows {(tot['stage'] - tot['base']) / 344841:+.1f} s^2")


def airport_offsets(aps, ref="LSZH") -> dict:
    """Per-airport shift that puts an airport's OPDI exit-parking on LSZH's footing (journal 173): the offset exit - BLOCK is
    estimated through our adsb.lol detector, median(exit - t_first, 2026) + median(t_first - BLOCK, 2025, coverage >= 2), and the
    shift is that offset minus LSZH's. EDDF: exit lags BLOCK by ~291 s against ~152 s at LSZH (OPDI 2025 EDDF is unusable directly)."""
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt"]
    want = sorted(set(aps) | {ref})
    tr = pl.concat([pl.read_parquet(p, columns=cols) for p in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(
        (pl.col("PHASE_mvt") == "DEP") & pl.col("ADEP_mvt").is_in(want)).with_columns(blk=pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64))
    c25 = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage", "t_first"]) for f in sorted(glob.glob("data/adsblol_kt1/offblock2_2025-*.parquet"))]).unique("MVT_ID_mvt")
    a = tr.join(c25, on="MVT_ID_mvt", how="inner").filter(pl.col("coverage") >= 2).group_by("ADEP_mvt").agg(tb=(pl.col("t_first") - pl.col("blk")).median())
    rk = pl.read_parquet("data/prc-2026-datasets/ranking.parquet", columns=[c for c in cols if c != "BLOCK_TIME_UTC_mvt"]).filter(
        (pl.col("PHASE_mvt") == "DEP") & pl.col("ADEP_mvt").is_in(want)).with_columns(
        mvt=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64), cs=pl.coalesce("CALLSIGN_flt", "FLIGHT_mvt").str.replace_all(" ", "")).select("MVT_ID_mvt", "ADEP_mvt", "mvt", "cs")
    c26 = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage", "t_first"]) for f in sorted(glob.glob("data/adsblol_kt1/offblock2_2026-*.parquet"))]).unique("MVT_ID_mvt")
    b = match(rk, opdi_wide()).filter(pl.col("exit").is_not_null()).join(c26, on="MVT_ID_mvt", how="inner").filter(pl.col("coverage") >= 2) \
        .group_by("ADEP_mvt").agg(et=(pl.col("exit") - pl.col("t_first")).median())
    off = {r[0]: r[1] + r[2] for r in a.join(b, on="ADEP_mvt").select("ADEP_mvt", "tb", "et").iter_rows()}
    shift = {ap: off[ap] - off[ref] for ap in aps}
    print(f"OPDI exit - BLOCK via adsb.lol: {({k: round(v, 1) for k, v in off.items()})}; shift vs {ref}: {({k: round(v, 1) for k, v in shift.items()})}")
    return shift

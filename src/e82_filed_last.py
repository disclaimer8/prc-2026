"""E82: the last observed neighbour, by filed time — the form E77 left unbuilt (20.09).

    .venv/bin/python src/e82_filed_last.py      data/features_e77.parquet -> data/features_e82.parquet

E77 replicated the neighbour family from a filed off-block time instead of AOBT_3 and paid -3.26 s on the
board, with 46% of the gain at LTFM, where the AOBT proxy is a three-tooth comb. But it replicated only the
WINDOW half of that family. `add_neighbor_features` also carries, for every group, the most recently observed
neighbour and its age (`nb_ap_last`, `nb_ap_last_age`, `nb_rwy_last`, `nb_rwy_last_age`), and those have no
filed-time twin. The count is the argument: 23 columns built on the broken anchor, 14 on the sound one.

Mechanism, and why it is not just the window mean again: the window statistics average the last half hour,
while `last` is the single most recent observation — the fastest thing there is to a disruption that is
developing right now. At LTFM the AOBT version of it is as degenerate as the rest of that family (the same
comb generates it), so whatever the airport's current state contributes there, the model cannot see it.

The stand group (`nb_stand_4320_0_*`, a three-day window) is deliberately NOT twinned here: section 0 closes
functions of the stand as absorbed, and a three-day window over the ranking frame has its own trap, which the
AOBT version handles with a staleness cutoff. One question per experiment.

The same staleness rule as the original is applied: a "previous" neighbour more than three days back says
nothing, because the ranking frame puts January and July 2026 back to back.

REGISTERED BEFORE THE RUN (gate = the standing one: decided rows, 6 folds, >= 0.3 s, lower in >= 4 of 6, the
same on y <= 7200 and trimmed): prediction **-0.10 .. -0.50 s, centre -0.25**. The centre is well under
E77's -2.73 on purpose: the window means already carry most of the same information, so this is the residual
of a form, not a new observation. E77 itself is the caution in the other direction -- its centre was -0.30
and it returned -2.73, because there the absorption discount was wrong.
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

SRC = {"nbf": "d_mvt_lobt", "nbs": "d_mvt_sched"}
GROUPS = {"ap": ["ADEP_mvt"], "rwy": ["ADEP_mvt", "RUNWAY_mvt"]}
STALE = 3 * 86400


def add_filed_last(feat: pl.DataFrame) -> pl.DataFrame:
    out = [feat.select("MVT_ID_mvt")]
    for pref, col in SRC.items():
        dep = feat.select("MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "t_mvt",
                          prox=pl.when(pl.col(col).is_between(0, 7200)).then(pl.col(col)))
        for gname, keys in GROUPS.items():
            parts = []
            for _, g in dep.drop_nulls(keys).group_by(keys):
                obs = g.drop_nulls("prox").sort("t_mvt")
                times, vals = obs["t_mvt"].to_numpy(), obs["prox"].to_numpy()
                q = g["t_mvt"].to_numpy()
                idx = np.searchsorted(times, q, side="left") - 1
                has = idx >= 0
                last = np.where(has, vals[np.clip(idx, 0, None)] if len(vals) else np.nan, np.nan)
                age = np.where(has, q - (times[np.clip(idx, 0, None)] if len(times) else 0), np.nan)
                parts.append(pl.DataFrame({"MVT_ID_mvt": g["MVT_ID_mvt"].to_numpy(),
                                           f"{pref}_{gname}_last": last,
                                           f"{pref}_{gname}_last_age": age}))
            out.append(pl.concat(parts, how="diagonal_relaxed"))
    nb = out[0]
    for p_ in out[1:]:
        nb = nb.join(p_, on="MVT_ID_mvt", how="left")
    res = feat.join(nb, on="MVT_ID_mvt", how="left")
    # the ranking frame holds January and July 2026 back to back, as in add_neighbor_features
    return res.with_columns([
        pl.when(pl.col(f"{p}_{g}_last_age") > STALE).then(None).otherwise(pl.col(f"{p}_{g}_last")).alias(f"{p}_{g}_last")
        for p in SRC for g in GROUPS
    ])


if __name__ == "__main__":
    src = os.environ.get("FEATURES", "data/features_e77.parquet")
    dst = os.environ.get("FEATURES_OUT", "data/features_e82.parquet")
    df = pl.read_parquet(src)
    res = add_filed_last(df)
    new = [c for c in res.columns if c not in df.columns]
    if len(res) != len(df) or [c for c in df.columns if c not in res.columns]:
        sys.exit("refusing to write: the result is not a superset of the source")
    for c in df.columns:
        if not df[c].equals(res[c]):
            sys.exit(f"refusing to write: source column {c} changed")
    res.write_parquet(dst)
    print(f"{src} -> {dst}: {res.height} rows, {len(new)} new columns")
    print(", ".join(new))
    lt = res.filter(pl.col("ADEP_mvt") == "LTFM")
    print(f"LTFM spread of the last observed neighbour: AOBT {float(lt['nb_ap_last'].std()):.1f} s "
          f"vs LOBT {float(lt['nbf_ap_last'].std()):.1f} s vs SCHED {float(lt['nbs_ap_last'].std()):.1f} s")

"""E84: the forward-looking neighbour family, by filed time — found by the E83 scan (21.09).

    .venv/bin/python src/e84_filed_forward.py     data/features_e77.parquet -> data/features_e84.parquet

E83 scanned all 156 model features for the two shapes that produced E77 and turned up seven "comb" columns:
six are LTFM and all derive from `d_mvt_aobt`, which E77 addressed, and the seventh is `fw_ap_15_mean` at
LSZH (84.2 % of rows in three 60-second bins against 54.1 % elsewhere). `src/e28_forward_neighbours.py:22`
builds the whole fw_ family from `prox = d_mvt_aobt` — the same broken anchor — and E77 twinned only the
symmetric nb_ windows, never the forward ones. Half the neighbour apparatus still stands on the comb.

Twinned here: count, mean and std of the neighbours taking off in (t, t+15 min] and (t, t+60 min], for the
airport and for the runway, from LOBT and from SCHED. `fw_*_next_gap` is not twinned: it is a time to the
next take-off, not a function of the proxy.

Prefixes are `fwl_` and `fws_` so the control arm can drop exactly these and keep both the AOBT originals
and E77's accepted nbf_/nbs_ columns.

REGISTERED BEFORE THE RUN (standing gate: decided rows, 6 folds, >= 0.3 s, lower in >= 4 of 6, the same on
y <= 7200 and trimmed): prediction **-0.15 .. -0.80 s, centre -0.35**. Two reasons the centre sits far below
E77's -2.73 even though the mechanism is identical: the forward windows overlap the symmetric ones E77
already twinned, and E28's own forward family, on the AOBT anchor, won -1.39 in CV and then moved the board
by -0.01 over the ADS-B stage. E82 is the other caution -- the `last` form of the same idea returned -0.18.
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from features import _window_stats  # noqa: E402

SRC = {"fwl": "d_mvt_lobt", "fws": "d_mvt_sched"}
GROUPS = {"ap": ["ADEP_mvt"], "rwy": ["ADEP_mvt", "RUNWAY_mvt"]}


def add_filed_forward(feat: pl.DataFrame) -> pl.DataFrame:
    out = [feat.select("MVT_ID_mvt")]
    for pref, col in SRC.items():
        dep = feat.select("MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "t_mvt",
                          prox=pl.when(pl.col(col).is_between(0, 7200)).then(pl.col(col)))
        for gname, keys in GROUPS.items():
            parts = []
            for _, g in dep.drop_nulls(keys).group_by(keys):
                g = g.sort("t_mvt")
                obs = g.drop_nulls("prox")
                times, vals, q = obs["t_mvt"].to_numpy(), obs["prox"].to_numpy(), g["t_mvt"].to_numpy()
                cols = {"MVT_ID_mvt": g["MVT_ID_mvt"].to_numpy()}
                for m in (15, 60):
                    n, mean, std = _window_stats(times, vals, q, 1, m * 60, None)
                    cols[f"{pref}_{gname}_{m}_n"] = n
                    cols[f"{pref}_{gname}_{m}_mean"] = mean
                    cols[f"{pref}_{gname}_{m}_std"] = std
                parts.append(pl.DataFrame(cols))
            out.append(pl.concat(parts, how="diagonal_relaxed"))
    nb = out[0]
    for p_ in out[1:]:
        nb = nb.join(p_, on="MVT_ID_mvt", how="left")
    return feat.join(nb, on="MVT_ID_mvt", how="left")


if __name__ == "__main__":
    src = os.environ.get("FEATURES", "data/features_e77.parquet")
    dst = os.environ.get("FEATURES_OUT", "data/features_e84.parquet")
    df = pl.read_parquet(src)
    res = add_filed_forward(df)
    new = [c for c in res.columns if c not in df.columns]
    if len(res) != len(df) or [c for c in df.columns if c not in res.columns]:
        sys.exit("refusing to write: the result is not a superset of the source")
    for c in df.columns:
        if not df[c].equals(res[c]):
            sys.exit(f"refusing to write: source column {c} changed")
    res.write_parquet(dst)
    print(f"{src} -> {dst}: {res.height} rows, {len(new)} new columns")
    print(", ".join(new))
    z = res.filter(pl.col("ADEP_mvt") == "LSZH")
    print(f"LSZH spread of the 15-min forward mean: AOBT {float(z['fw_ap_15_mean'].std()):.1f} s "
          f"vs LOBT {float(z['fwl_ap_15_mean'].std()):.1f} s vs SCHED {float(z['fws_ap_15_mean'].std()):.1f} s")

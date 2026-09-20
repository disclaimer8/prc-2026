"""E77: the neighbour aggregates again, built from a FILED off-block time instead of AOBT_3 (20.09).

    .venv/bin/python src/e77_filed_twins.py        data/features_e28.parquet -> data/features_e77.parquet

`add_neighbor_features` in src/features.py measures a neighbour's taxi proxy as MVT - AOBT_3. At LTFM that
proxy is degenerate: 96.3 % of 2025 departures fall in three 60-second bins of MVT - AOBT_3, and the same
bins hold about 89 % of the 2026 ranking rows. The consequence is visible in the columns themselves --
`nb_ap_30_0_std` has a median of 31 s at LTFM against 105-306 s at every other airport, so there the
neighbour family is nearly constant and carries no state of the airport at all. The filed times do carry it:
at LTFM, corr(y, MVT - AOBT_3) = 0.157 against corr(y, MVT - LOBT) = 0.560, while corr(AOBT_3, LOBT) is only
0.078 -- a different observation, not a rescaling of the one we hold.

Both filed sources are built, because what was measured is "the neighbours' current delay level from any
filed time", not LOBT specifically: the SCHED-based probe scored -0.677 against -0.561 for the LOBT one, and
at LTFM EOBT_1 equals LOBT to sixteen digits.

Added beside the AOBT columns, never in place of them (REVIEW_BRIEF: change the input by adding, not by
substituting). Everything else is copied unchanged, so the control arm is this same file with
DROP_PREFIX=...,nbf_,nbs_ and the two arms differ in exactly these 28 columns.

Provenance: found by the Fable search wave of 19.09 (line `search:onset-invisible`), and none of its three
adversarial verifiers refuted it -- the registry check found no LOBT-derived aggregate over other rows
anywhere in E1-E76 or G1-G16, the mechanism re-derived exactly, and an independent pooled second-stage probe
put LTFM at -0.745 and LIRF at -0.488 with every other airport inside +-0.09.
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from features import _window_stats  # noqa: E402

SRC = {"nbf": "d_mvt_lobt", "nbs": "d_mvt_sched"}
GROUPS = {"ap": ["ADEP_mvt"], "rwy": ["ADEP_mvt", "RUNWAY_mvt"]}
WINDOWS = {"ap": [(-1800, 0), (-1800, 1800)], "rwy": [(-3600, 0), (-1800, 1800)]}


def add_filed_twins(feat: pl.DataFrame) -> pl.DataFrame:
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
                own = g["prox"].to_numpy().astype(float)
                cols = {"MVT_ID_mvt": g["MVT_ID_mvt"].to_numpy()}
                for lo, hi in WINDOWS[gname]:
                    self_in = np.where((lo <= 0) & (hi >= 0), own, np.nan)
                    n, mean, std = _window_stats(times, vals, q, lo, hi, self_in)
                    tag = f"{pref}_{gname}_{-lo // 60}_{hi // 60}"
                    cols[f"{tag}_n"], cols[f"{tag}_mean"], cols[f"{tag}_std"] = n, mean, std
                parts.append(pl.DataFrame(cols))
            out.append(pl.concat(parts, how="diagonal_relaxed"))
    nb = out[0]
    for p_ in out[1:]:
        nb = nb.join(p_, on="MVT_ID_mvt", how="left")
    return feat.join(nb, on="MVT_ID_mvt", how="left").with_columns(
        nbf_own_minus_ap=pl.col("d_mvt_lobt") - pl.col("nbf_ap_30_30_mean"),
        nbf_own_minus_rwy=pl.col("d_mvt_lobt") - pl.col("nbf_rwy_30_30_mean"),
        nbs_own_minus_ap=pl.col("d_mvt_sched") - pl.col("nbs_ap_30_30_mean"),
    )


if __name__ == "__main__":
    src = os.environ.get("FEATURES", "data/features_e28.parquet")
    dst = os.environ.get("FEATURES_OUT", "data/features_e77.parquet")
    df = pl.read_parquet(src)
    out = add_filed_twins(df)
    new = [c for c in out.columns if c not in df.columns]
    if len(out) != len(df) or [c for c in df.columns if c not in out.columns]:
        sys.exit("refusing to write: the result is not a superset of the source")
    for c in df.columns:
        if not df[c].equals(out[c]):
            sys.exit(f"refusing to write: source column {c} changed")
    out.write_parquet(dst)
    print(f"{src} -> {dst}: {out.height} rows, {len(new)} new columns")
    print(", ".join(new))
    lt = out.filter(pl.col("ADEP_mvt") == "LTFM")
    print(f"LTFM std of the airport window, AOBT {float(lt['nb_ap_30_0_std'].median()):.1f} s "
          f"vs LOBT {float(lt['nbf_ap_30_0_std'].median()):.1f} s vs SCHED {float(lt['nbs_ap_30_0_std'].median()):.1f} s")

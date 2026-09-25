"""Regime probabilities for LIRF departures without NM from TabPFN-2 instead of the smoothed table (E11 adopted).

    TABPFN_ALLOW_CPU_LARGE_DATASET=1 uv run --with tabpfn python src/lirf_tabpfn.py <base_version> <new_version>

TabPFN-2 weights: Prior Labs License (Apache-2.0 with an attribution requirement). Fit on all 2025 LIRF no-NM
departures; applied to the ranking rows in the table's bins (delay from schedule 4-30 h), with the same candidates:
schedule (D), one-day shift (86400 + median taxi) and the regular prediction of the base model.
"""
import os
import re
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, "src")

import numpy as np
import polars as pl
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

import lirf_table as L

Y = "TAXITIME_SEC_mvt"
FEATS = ["d_mvt_sched", "local_hour", "local_dow", "day_nonm_rate", "dep_before_30", "arr_before_30", "stand_num", "rwy_head", "fn_num", "fn_len", "fn_garbled"]


def prepare(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null()).with_columns(
        b=pl.col("d_mvt_sched").cut(L.BINS, labels=L.LABELS),
        stand_num=pl.col("STAND_mvt").str.extract(r"(\d+)", 1).cast(pl.Float64),
        rwy_head=pl.col("RUNWAY_mvt").str.extract(r"^(\d{2})", 1).cast(pl.Float64),
        fn_num=pl.col("FLIGHT_mvt").str.extract(r"(\d+)", 1).cast(pl.Float64),
        fn_len=pl.col("FLIGHT_mvt").str.len_chars().cast(pl.Float64),
        # garbled call signs like ITYTY680, RYRR90BN, EXSXS1GT: ICAO prefix followed by a repeat of its tail
        fn_garbled=pl.col("FLIGHT_mvt").map_elements(
            lambda s: float(bool(re.match(r"^([A-Z])([A-Z])([A-Z])(\3|\2\3)", s or ""))), return_dtype=pl.Float64
        ),
    )


EXTRA = [x for x in os.environ.get("TABPFN_EXTRA", "").split(",") if x]  # hypothesis 56: "pfx" (E118), "local_change" (E26a)


def add_extra(tr: pl.DataFrame, rank: pl.DataFrame):
    feats, cat = list(FEATS), []
    if "local_change" in EXTRA:  # E26a: local (Europe/Rome) date differs between schedule and take-off
        lc = lambda d: d.with_columns(local_change=(pl.col("MVT_TIME_UTC_mvt").dt.convert_time_zone("Europe/Rome").dt.date()
                                                    != pl.col("SCHED_TIME_UTC_mvt").dt.convert_time_zone("Europe/Rome").dt.date()).cast(pl.Float64))
        tr, rank = lc(tr), lc(rank); feats.append("local_change")
    if "pfx" in EXTRA:  # E118: airline prefix (E113 lookup) as a categorical code, codes for prefixes with >= 5 rows in 2025
        ap = pl.read_parquet(os.environ.get("PFX_FEATURES", "data/features_e113.parquet"), columns=["MVT_ID_mvt", "airline_prefix"])
        tr, rank = tr.join(ap, on="MVT_ID_mvt", how="left"), rank.join(ap, on="MVT_ID_mvt", how="left")
        vc = tr["airline_prefix"].value_counts().sort("count", descending=True)
        code = {p: i + 1 for i, p in enumerate(vc.filter(pl.col("count") >= 5)["airline_prefix"].drop_nulls().to_list())}
        enc = lambda d: d.with_columns(pfx=pl.col("airline_prefix").map_elements(lambda p: float(code.get(p, 0)), return_dtype=pl.Float64).fill_null(0.0))
        tr, rank = enc(tr), enc(rank); feats.append("pfx"); cat = [len(feats) - 1]
        print(f"prefix codes {len(code)}, coverage train {float((tr['pfx'] > 0).mean()):.3f} rank {float((rank['pfx'] > 0).mean()):.3f}")
    return tr, rank, feats, cat


def main() -> None:
    base, new = int(sys.argv[1]), int(sys.argv[2])
    cols = ["MVT_ID_mvt", "split", "ADEP_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "STAND_mvt", "RUNWAY_mvt", "d_mvt_sched",
            "local_hour", "local_dow", "day_nonm_rate", "dep_before_30", "arr_before_30", Y] + (["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"] if "local_change" in EXTRA else [])
    f = pl.read_parquet(os.environ.get("FEATURES", "data/features_v9.parquet"), columns=cols)
    tr = prepare(f.filter(pl.col("split") == "train")).with_columns(
        cls=pl.when((pl.col(Y) - pl.col("d_mvt_sched")).abs() <= 6).then(0).when(pl.col(Y).is_between(86400, 93600)).then(1).otherwise(2)
    )
    day_taxi = float(f.filter((pl.col("split") == "train") & pl.col(Y).is_between(86400, 93600)).select((pl.col(Y) - 86400).median()).item())
    rank = prepare(f.filter(pl.col("split") == "rank")).filter(pl.col("b").is_in(L.OVERRIDE))
    tr, rank, feats, cat = add_extra(tr, rank)
    print("TabPFN features:", feats, "categorical idx:", cat)
    # E126 (hypothesis 62): ensemble the tail classifier — several TabPFN seeds averaged (TABPFN_SEEDS="0,1,2,3,4") and/or
    # more internal estimators (TABPFN_N_EST). Capacity-free: only the variance of one draw is removed.
    seeds = [int(x) for x in os.environ.get("TABPFN_SEEDS", "0").split(",")]
    extra_kw = {"categorical_features_indices": cat} if cat else {}
    if os.environ.get("TABPFN_N_EST"):
        extra_kw["n_estimators"] = int(os.environ["TABPFN_N_EST"])
    p = np.zeros((rank.height, 3))
    for sd in seeds:
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=sd, **extra_kw)
        clf.fit(tr.select(feats).to_numpy(), tr["cls"].to_numpy())
        raw = clf.predict_proba(rank.select(feats).to_numpy())
        for j, c in enumerate(clf.classes_):
            p[:, int(c)] += raw[:, j] / len(seeds)
    print(f"TabPFN seeds {seeds}, extra {extra_kw}")

    parts = np.load(f"submissions/parts_v{base}.npz")
    reg = dict(zip(parts["mvt_id"], parts["reg"]))
    r = np.array([reg[m] for m in rank["MVT_ID_mvt"].to_numpy()])
    d = rank["d_mvt_sched"].to_numpy()
    new_pred = p[:, 0] * d + p[:, 1] * (86400 + day_taxi) + p[:, 2] * r
    adj = pl.DataFrame({"MVT_ID_mvt": rank["MVT_ID_mvt"], "new": new_pred, "p_sched": p[:, 0], "p_day": p[:, 1], "b": rank["b"].cast(pl.String)})
    print(adj.group_by("b").agg(n=pl.len(), p_sched=pl.col("p_sched").mean().round(3), p_day=pl.col("p_day").mean().round(3)).sort("b").to_dicts())

    sub = pl.read_parquet(f"submissions/gentle-igloo_v{base}.parquet")
    out = sub.join(adj.select("MVT_ID_mvt", "new"), on="MVT_ID_mvt", how="left").with_columns(
        pl.coalesce(pl.col("new").round(0).cast(sub.schema[Y]), pl.col(Y)).alias(Y)
    ).select(sub.columns)
    assert out.height == sub.height and out[Y].null_count() == 0
    out.write_parquet(f"submissions/gentle-igloo_v{new}.parquet")
    print(f"wrote v{new}: {adj.height} rows replaced")


if __name__ == "__main__":
    main()

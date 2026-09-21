"""E83: scan every model feature for the two defects that produced E77 (21.09).

E77 was found by noticing two things about one column family, and neither was looked for systematically:

  1. **near-constant where it should vary** -- nb_ap_30_0_std has a median of 31 s at LTFM against 105-306 s
     at every other airport, so the neighbour family carried no airport state there;
  2. **quantised into a few values** -- 96.3 % of LTFM departures fall in three 60-second bins of
     MVT - AOBT_3, i.e. the proxy is a comb rather than a measurement.

Fixing that one input paid -3.26 s on the board. This scans all model features for the same two shapes, per
airport, read-only. It proposes nothing: it produces a ranked list of inputs that are degenerate somewhere,
which is where a mechanism can then be looked for.

    .venv/bin/python src/e83_degeneracy_scan.py
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))

SRC = os.environ.get("FEATURES", "data/features_e82.parquet")
MIN_ROWS = 5000


def main() -> None:
    import train as T
    df = pl.read_parquet(SRC).filter(pl.col("split") == "train")
    cols = [c for c in T.feature_columns(df) if df.schema[c] in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8)]
    aps = sorted(df["ADEP_mvt"].unique().to_list())
    print(f"признаков {len(cols)}, аэропортов {len(aps)}, строк {df.height}\n")

    flat, comb = [], []
    by_ap = {a: df.filter(pl.col("ADEP_mvt") == a) for a in aps}
    for c in cols:
        spread, share3 = {}, {}
        for a in aps:
            v = by_ap[a][c].drop_nulls().to_numpy().astype(float)
            if len(v) < MIN_ROWS:
                continue
            q1, q3 = np.percentile(v, [25, 75])
            spread[a] = float(q3 - q1)
            # concentration: how much mass sits in the three most common values rounded to 60 s
            vals, cnt = np.unique(np.round(v / 60.0), return_counts=True)
            share3[a] = float(np.sort(cnt)[-3:].sum() / len(v))
        if len(spread) < 5:
            continue
        s = np.array(list(spread.values()))
        med = float(np.median(s))
        if med > 0:
            worst = min(spread, key=spread.get)
            ratio = spread[worst] / med
            if ratio < 0.35:
                flat.append((ratio, c, worst, spread[worst], med))
        worst_c = max(share3, key=share3.get)
        others = float(np.median([v for a, v in share3.items() if a != worst_c]))
        if share3[worst_c] > 0.80 and others < 0.60:
            comb.append((share3[worst_c] - others, c, worst_c, share3[worst_c], others))

    print("=== ПОЧТИ КОНСТАНТА в одном аэропорту (межквартильный размах против медианы по аэропортам) ===")
    for ratio, c, a, w, med in sorted(flat)[:15]:
        print(f"  {ratio:5.2f}x  {c:28s} {a}: {w:10.2f}  против медианы {med:10.2f}")
    print(f"  всего таких признаков: {len(flat)}")

    print("\n=== ГРЕБЁНКА: три значения держат >80 % строк аэропорта ===")
    for gap, c, a, w, others in sorted(comb, reverse=True)[:15]:
        print(f"  {w:5.1%} против {others:5.1%}  {c:28s} {a}")
    print(f"  всего таких признаков: {len(comb)}")


if __name__ == "__main__":
    main()

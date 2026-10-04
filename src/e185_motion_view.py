"""E185: the movement-only view of the feature table, shared by the CV (src/e185_motion_only_cv.py) and the ranking fit
(src/e185_motion_rank.py) so that both read exactly the same matrix.

  * HIDE: every column that comes from the row's own NM record (null on 100 % of the no-NM rows of fin1b, train and ranking,
    checked 04.10): the four NM categoricals, the NM time differences, NM flags, `*_own_minus_*` against NM proxies, plan_*
    (ARVT_1, EOBT_1) and e157_ladder_q (AOBT_3 / LOBT). d_mvt_sched stays: SCHED is a movement field, known on every row.
  * REPLACE: the columns of data/e185_motion_features.parquet (src/e185_motion_features.py): the 35 columns anchored at the row's
    own AOBT_3 recomputed at MVT - 900 s with the own AOBT out of the sets, and airline_prefix from FLIGHT_mvt, for every row;
    identical to fin1b on the rows without NM.
Encodings (te_/ter_/tq_) are not built: production DROP_PREFIX removes all of them, and the script refuses to run otherwise.
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import train as T  # noqa: E402

MOTION = os.environ.get("E185_MOTION", "data/e185_motion_features.parquet")
HIDE = ["MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
        "d_mvt_aobt", "d_mvt_eobt", "d_mvt_lobt", "d_mvt_iobt", "d_aobt_eobt", "d_eobt_iobt", "d_aobt_sched", "flight_time_plan",
        "diverted", "aobt_minute_exact", "aobt_infeasible", "nb_own_minus_", "nbf_own_minus_", "e157_ladder_q", "plan_taxi_res",
        "plan_nm_taxi"]
SUB = pl.col("LOBT_flt").is_null() & (pl.col("ADEP_mvt") != "LIRF")  # E185 subset: no NM record (all *_flt null), outside LIRF

prod_drop = [p for p in os.environ.get("DROP_PREFIX", "").split(",") if p]
if not {"te_", "ter_", "tq_"} <= set(prod_drop) or os.environ.get("REGIME_RATES") == "1":
    sys.exit("E185 skips add_encodings: DROP_PREFIX must hold te_, ter_ and tq_ (production does) and REGIME_RATES must be off")
os.environ["DROP_PREFIX"] = ",".join(dict.fromkeys(prod_drop + HIDE))  # T.feature_columns reads it at call time


def load(splits: list[str]) -> tuple[pl.DataFrame, list[str], list[str], list[str]]:
    """(frame, feature columns, categorical feature columns, replaced columns) for the given splits, with `_lab` (regime label)
    and `_y` (clip(y, 0, 7200); null on ranking) added."""
    schema = pl.read_parquet_schema(T.FEATURES_PATH)
    replaced = [c for c in pl.read_parquet_schema(MOTION) if c != "MVT_ID_mvt"]
    missing = [c for c in replaced if c not in schema]
    assert not missing, f"motion view columns absent from {T.FEATURES_PATH}: {missing}"
    cols = T.feature_columns(pl.DataFrame(schema=schema))
    need = ["MVT_ID_mvt", "split", "month", T.TARGET, "ADEP_mvt", *T.CANDIDATES.values(), "LOBT_flt"]
    read = [c for c in schema if c in set(need) | (set(cols) - set(replaced))]
    df = (pl.scan_parquet(T.FEATURES_PATH).select(read).filter(pl.col("split").is_in(splits))
          .join(pl.scan_parquet(MOTION), on="MVT_ID_mvt", how="left").select([c for c in schema if c in set(read) | set(replaced)]).collect())
    assert df["MVT_ID_mvt"].n_unique() == df.height
    cols = T.feature_columns(df)
    leak = [c for c in cols if c.endswith("_flt") or any(k in c for k in ("aobt", "lobt", "eobt", "iobt"))
            or c.startswith(("plan_", "nb_own_minus", "nbf_own_minus", "e157_ladder"))]
    assert not leak, f"NM columns in the movement-only matrix: {leak}"
    df = df.with_columns(_lab=pl.Series(T.regime_label(df)), _y=pl.col(T.TARGET).cast(pl.Float64).clip(*T.REG_CLIP))
    return df, cols, [c for c in T.CATEGORICAL if c in cols], replaced


def categories(df: pl.DataFrame, expr: pl.Expr, cat_cols: list[str]) -> dict:
    """T.to_pandas categories: sorted non-null values of the fit rows (all regimes, as e130 builds them)."""
    return {c: sorted(df.lazy().filter(expr).select(c).drop_nulls().unique().collect()[c].to_list()) for c in cat_cols}


def matrix(df: pl.DataFrame, expr: pl.Expr, cols: list[str], cats: dict) -> np.ndarray:
    """Rows of df matching expr as a column-major float64 matrix, filled column by column (peak = df + one matrix). The values
    LightGBM gets from T.to_pandas: numbers as float64 with NaN for null, categoricals as the index into the sorted category list
    (pandas cat.codes), NaN when absent. LightGBM 4 reads an F-order array without a copy. Checked 04.10: the smoke fold gives the
    same RMSE to the hundredth through this path and through T.to_pandas."""
    idx = np.flatnonzero(df.select(expr).to_series().to_numpy())
    X = np.empty((len(idx), len(cols)), dtype=np.float64, order="F")
    for j, c in enumerate(cols):
        s = df[c]
        if c in cats:
            s = s.replace_strict(cats[c], list(range(len(cats[c]))), default=None, return_dtype=pl.Float64)
        X[:, j] = s.gather(idx).cast(pl.Float64).to_numpy()
    return X


def label(df: pl.DataFrame, expr: pl.Expr) -> np.ndarray:
    return df["_y"].to_numpy()[df.select(expr).to_series().to_numpy()]  # same rows, same order as matrix()

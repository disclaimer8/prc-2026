"""E100 on the ranking set (23.09): the MLP regular candidate inside the v115 chain, blended into v115 by the CV weight.

    .venv/bin/python src/e100_rank.py 160 161
v160 = the partner: v115's classifier probabilities and candidates (submissions/parts_v111.npz, the E89 chain), the
regular candidate replaced by the MLP fitted on all of 2025, mixed and LOBT-clipped exactly as train.py does. Like the
OOF arm it carries no later stage, and it is written only as a component.
v161 = v115 + W * (v160 - v115) on strict-decided rows; the LIRF tail table and the E14 population keep v115, since
the CV weight was fitted on decided rows and nothing measured the dilution of those later stages (rule of E78).
W = 0.158, the mean LOFO-NNLS weight of src/e100_blend_eval.py (0.131..0.170 over folds).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import e100_nn_partner as N  # noqa: E402  (sets FEATURES / DROP_PREFIX / PROXY_SCALE defaults of the E89 chain)
import train as T  # noqa: E402

W = float(os.environ.get("E100_W", "0.158"))
BASE = 115
TGT = "TAXITIME_SEC_mvt"


def main() -> None:
    partner, out = int(sys.argv[1]), int(sys.argv[2])
    assert os.environ["DROP_PREFIX"] == "day_,tq_,wxr_,month", os.environ["DROP_PREFIX"]
    df = pl.read_parquet(T.FEATURES_PATH)
    fit, rank = df.filter(pl.col("split") == "train"), df.filter(pl.col("split") == "rank")
    del df
    r = N.regular_candidate(fit, rank)
    parts = np.load("submissions/parts_v111.npz")
    pos = {m: i for i, m in enumerate(rank["MVT_ID_mvt"].to_numpy())}
    idx = np.array([pos[m] for m in parts["mvt_id"]])
    rk, r = rank[idx], r[idx]
    values = parts["values"].astype(np.float64).copy()
    print(f"regular candidate: RMS change vs v111 GBDT blend {np.sqrt(np.mean((r - parts['reg']) ** 2)):.1f} s", flush=True)
    values[:, 0] = r
    values[:, -1] = r + T.DAY
    pred = np.clip(T.lobt_clip(np.sum(parts["proba"] * np.nan_to_num(values), axis=1), rk), 0, None)
    ids = parts["mvt_id"]
    tmpl = pl.read_parquet("data/prc-2026-datasets/submitting.parquet")
    p_df = pl.DataFrame({"MVT_ID_mvt": ids, "_p": pred})
    sub = tmpl.drop(TGT).join(p_df, on="MVT_ID_mvt", how="left").with_columns(pl.col("_p").round(0).cast(tmpl.schema[TGT]).alias(TGT)).drop("_p")
    sub.write_parquet(f"submissions/gentle-igloo_v{partner}.parquet")
    print("wrote partner", partner, flush=True)

    base = pl.read_parquet(f"submissions/gentle-igloo_v{BASE}.parquet")
    strict = rk.select(~((pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null() & (pl.col("d_mvt_sched") > 0)).fill_null(False)).to_series().to_numpy()
    s_df = pl.DataFrame({"MVT_ID_mvt": ids, "_p": pred, "_s": strict})
    j = base.join(s_df, on="MVT_ID_mvt", how="left")
    b = j[TGT].cast(pl.Float64).to_numpy()
    new = np.where(j["_s"].to_numpy(), b + W * (j["_p"].to_numpy() - b), b)
    print(f"blend: rows changed {(np.round(new) != b).sum()}, RMS change vs v{BASE} {np.sqrt(np.mean((new - b) ** 2)):.2f} s, "
          f"kept v{BASE} on {(~j['_s'].to_numpy()).sum()} rows", flush=True)
    out_df = j.select("MVT_ID_mvt").with_columns(pl.Series(TGT, np.round(new)).cast(base.schema[TGT]))
    out_df = tmpl.drop(TGT).join(out_df, on="MVT_ID_mvt", how="left")
    assert out_df[TGT].null_count() == 0
    out_df.write_parquet(f"submissions/gentle-igloo_v{out}.parquet")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()

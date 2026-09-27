"""E145 (27.09, journal 114): shift the regular-regime predictions of the airports where PRU excludes almost no flight
(2025 VALID_FL / TF >= 0.98: EGLL, LEMD, LEBL, LIRF, EHAM) so that their monthly mean matches the PRU monthly taxi-out
(ansperformance CSV) plus the same-month 2025 offset between PRU and our labels. The organiser allowed the published
monthly figures as model input (Discord, 27.09 16:14: "yes").

    python src/e145_pru_calib.py cv                 # honest 2025 CV: offset learned on the other months of the airport
    python src/e145_pru_calib.py apply <base> <new>  # 2026: offset = same month of 2025
Shift applies to rows with prediction <= 7200 s (PRU drops taxi-outs over 120 min).
"""
import glob
import sys

import numpy as np
import polars as pl

APS = ["EGLL", "LEMD", "LEBL", "LIRF", "EHAM"]
CAP = 7200


def pru(year):
    return pl.read_csv(f"data/taxi_out_additional_time_{year}.csv", infer_schema_length=0).filter(pl.col("APT_ICAO").is_in(APS)).select(
        ADEP_mvt=pl.col("APT_ICAO"), m=pl.col("MONTH_NUM").cast(pl.Int8),
        pru=(pl.col("TOTAL_REF_TIME_MIN").cast(pl.Float64) + pl.col("TOTAL_ADD_TIME_MIN").cast(pl.Float64)) * 60 / pl.col("TOTAL_REF_NB_FL").cast(pl.Float64))


def train_frame():
    raw = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
                     for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(pl.col("PHASE_mvt") == "DEP")
    oof = pl.concat([pl.DataFrame({"MVT_ID_mvt": (z := np.load(f))["mvt_id"], "p": z["pred"], "y": z["y"]})
                     for f in sorted(glob.glob("data/cv_preds_e106_air/fold_*.npz"))])
    return raw.join(oof.with_columns(pl.col("MVT_ID_mvt").cast(raw["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt").with_columns(
        m=pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int8))


def offsets(d):
    return d.filter(pl.col("ADEP_mvt").is_in(APS)).group_by("ADEP_mvt", "m").agg(
        ymean_f=pl.col("y").filter(pl.col("y") <= CAP).mean()).join(pru(2025), on=["ADEP_mvt", "m"]).with_columns(
        off=pl.col("ymean_f") - pl.col("pru")).select("ADEP_mvt", "m", "off", "pru")


def shift_frame(pred_df, target):  # pred_df: ADEP_mvt, m, p ; target: ADEP_mvt, m, target
    g = pred_df.filter(pl.col("p") <= CAP).group_by("ADEP_mvt", "m").agg(pmean_f=pl.col("p").mean())
    return g.join(target, on=["ADEP_mvt", "m"]).select("ADEP_mvt", "m", shift=pl.col("target") - pl.col("pmean_f"))


def cv():
    d = train_frame()
    off = offsets(d)
    folds = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
    parts = []
    for a, b in folds:
        oh = off.filter(~pl.col("m").is_in([a, b])).group_by("ADEP_mvt").agg(off_hat=pl.col("off").mean())
        tgt = off.filter(pl.col("m").is_in([a, b])).join(oh, on="ADEP_mvt").select("ADEP_mvt", "m", target=pl.col("pru") + pl.col("off_hat"))
        parts.append(shift_frame(d.filter(pl.col("m").is_in([a, b])), tgt))
    s = pl.concat(parts)
    x = d.join(s, on=["ADEP_mvt", "m"], how="left").with_columns(pl.col("shift").fill_null(0.0)).with_columns(
        p2=pl.when(pl.col("p") <= CAP).then(pl.col("p") + pl.col("shift")).otherwise(pl.col("p")))
    x = x.with_columns(de=(pl.col("p2") - pl.col("y")) ** 2 - (pl.col("p") - pl.col("y")) ** 2,
                       fold=pl.col("m").replace_strict({mm: i for i, (a, b) in enumerate(folds) for mm in (a, b)}))
    N = d.height
    r0 = float(np.sqrt(((d["p"] - d["y"]) ** 2).mean()))
    per = x.group_by("fold").agg(pl.col("de").sum()).sort("fold")
    trim = x.filter(pl.col("de").abs() < pl.col("de").abs().quantile(0.99999))
    r1 = float(np.sqrt((((x["p2"] - x["y"]) ** 2).sum()) / N))
    print(f"OOF RMSE {r0:.3f} -> {r1:.3f} ({r1 - r0:+.3f} s); folds lower {int((per['de'] < 0).sum())}/6; "
          f"dRMSE without the 0.001 % largest row changes {float(trim['de'].sum()) / N / (2 * r0):+.3f} s; median |shift| {s['shift'].abs().median():.1f} s")
    print("per fold dRMSE:", [round(float(v) / N / (2 * r0), 4) for v in per["de"].to_list()])


def apply(base_v, new_v):
    d = train_frame()
    off = offsets(d).select("ADEP_mvt", "m", "off")
    tgt = pru(2026).join(off, on=["ADEP_mvt", "m"]).select("ADEP_mvt", "m", target=pl.col("pru") + pl.col("off"))
    rk = pl.read_parquet("data/prc-2026-datasets/ranking.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]).filter(
        pl.col("PHASE_mvt") == "DEP").with_columns(m=pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int8))
    sub = pl.read_parquet(f"submissions/gentle-igloo_v{base_v}.parquet")
    Y = "TAXITIME_SEC_mvt"
    x = rk.join(sub.with_columns(pl.col("MVT_ID_mvt").cast(rk["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt").rename({Y: "p"}).with_columns(pl.col("p").cast(pl.Float64))
    s = shift_frame(x, tgt)
    print(s.sort("ADEP_mvt", "m").with_columns(pl.col("shift").round(1)).to_dicts())
    x = x.join(s, on=["ADEP_mvt", "m"], how="left").with_columns(pl.col("shift").fill_null(0.0)).with_columns(
        new=pl.when(pl.col("p") <= CAP).then((pl.col("p") + pl.col("shift")).clip(0, None)).otherwise(pl.col("p")))
    out = sub.join(x.select(pl.col("MVT_ID_mvt").cast(sub["MVT_ID_mvt"].dtype), "new"), on="MVT_ID_mvt", how="left").with_columns(
        pl.coalesce(pl.col("new").round(0).cast(sub.schema[Y]), pl.col(Y)).alias(Y)).select(sub.columns)
    dd = (out.sort("MVT_ID_mvt")[Y].cast(float) - sub.sort("MVT_ID_mvt")[Y].cast(float)).to_numpy()
    out.write_parquet(f"submissions/gentle-igloo_v{new_v}.parquet")
    print(f"wrote v{new_v}: changed {(dd != 0).sum()}, RMS change {np.sqrt((dd ** 2).mean()):.2f} s, mean {dd.mean():+.2f} s")


if __name__ == "__main__":
    cv() if sys.argv[1] == "cv" else apply(int(sys.argv[2]), int(sys.argv[3]))

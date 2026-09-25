"""E133 (26.09, hypothesis 70, external review): schedule-copy habit of a flight number on MATCHED Rome rows.

Band: ADEP = LIRF, NM record present, 1800 <= D = MVT - SCHED < 14400 s. Neither the model (FLIGHT_mvt is in DROP)
nor E14 (no-NM rows only) sees the flight number there. Rule, no threshold search: if in the OTHER folds (cv mode) /
in all of 2025 (apply mode) the same flight number has >= MIN_N departures in the band and the share of
"block = schedule" (|y - D| <= 6 s) among them is >= MIN_SHARE, predict the schedule candidate D for the row.

    python src/e133_lirf_matched_sched_rule.py cv                 # on data/cv_preds_e106_air (6 folds)
    python src/e133_lirf_matched_sched_rule.py apply <base> <new> # ranking: history from all 2025 rows
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

MIN_N = int(os.environ.get("RULE_MIN_N", "5"))
MIN_SHARE = float(os.environ.get("RULE_MIN_SHARE", "0.8"))
LO, HI, TOL = 1800.0, 14400.0, 6.0
FEATURES = os.environ.get("FEATURES", "data/features_e106.parquet")
cols = ["MVT_ID_mvt", "split", "ADEP_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "d_mvt_sched", "TAXITIME_SEC_mvt", "month"]
f = pl.read_parquet(FEATURES, columns=cols)
band = (pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_not_null() & pl.col("FLIGHT_mvt").is_not_null() & (pl.col("d_mvt_sched") >= LO) & (pl.col("d_mvt_sched") < HI)
tr = f.filter((pl.col("split") == "train") & band).with_columns(sched=((pl.col("TAXITIME_SEC_mvt") - pl.col("d_mvt_sched")).abs() <= TOL).cast(pl.Int8))
print(f"2025 LIRF matched rows in band [{LO:.0f},{HI:.0f}): {tr.height}, block=schedule share {tr['sched'].mean():.3f}, distinct flight numbers {tr['FLIGHT_mvt'].n_unique()}")


def history(rows: pl.DataFrame) -> pl.DataFrame:
    return rows.group_by("FLIGHT_mvt").agg(n=pl.len(), share=pl.col("sched").mean())


mode = sys.argv[1]
if mode == "cv":
    lb = 239.9169
    tot_d = 0.0; fired_all = []; per_fold = []
    for p in sorted(glob.glob("data/cv_preds_e106_air/fold_*.npz")):
        d = np.load(p)
        a, b = [int(x) for x in os.path.basename(p)[5:-4].split("_")]
        o = pl.DataFrame({"MVT_ID_mvt": d["mvt_id"].astype(np.int64), "pred": d["pred"].astype(np.float64), "y": d["y"].astype(np.float64)}).with_columns(pl.col("MVT_ID_mvt").cast(f.schema["MVT_ID_mvt"]))
        j = o.join(tr.drop("TAXITIME_SEC_mvt"), on="MVT_ID_mvt", how="inner")  # only band rows of the held-out months
        h = history(tr.filter(~pl.col("month").is_in([a, b])))
        j = j.join(h, on="FLIGHT_mvt", how="left").with_columns(fire=(pl.col("n") >= MIN_N) & (pl.col("share") >= MIN_SHARE))
        fr = j.filter(pl.col("fire"))
        new = fr["d_mvt_sched"].to_numpy(); yy = fr["y"].to_numpy(); old = fr["pred"].to_numpy()
        sse_old = float(((old - yy) ** 2).sum()); sse_new = float(((new - yy) ** 2).sum())
        prec = float((np.abs(yy - new) <= TOL).mean()) if fr.height else float("nan")
        rmse_old = np.sqrt(sse_old / max(fr.height, 1)); rmse_new = np.sqrt(sse_new / max(fr.height, 1))
        fold_sse = float(((o["pred"] - o["y"]) ** 2).sum())
        print(f"fold {a}+{b}: band rows {j.height}, fired {fr.height}, precision {prec:.3f}, RMSE fired {rmse_old:.0f} -> {rmse_new:.0f}, dSSE {sse_new - sse_old:+.3e}, fold RMSE {np.sqrt(fold_sse / o.height):.3f} -> {np.sqrt((fold_sse + sse_new - sse_old) / o.height):.3f}")
        tot_d += sse_new - sse_old; per_fold.append(sse_new - sse_old); fired_all.append(fr.height)
    lb_eq = np.sqrt(max(lb ** 2 + tot_d / 6 / 344841, 0)) - lb
    print(f"TOTAL fired {sum(fired_all)}, dSSE {tot_d:+.3e}, lower in {sum(1 for x in per_fold if x < 0)}/6 folds, LB-equivalent {lb_eq:+.3f} s")
else:
    base, new = int(sys.argv[2]), int(sys.argv[3])
    h = history(tr)
    rk = f.filter((pl.col("split") == "rank") & band).join(h, on="FLIGHT_mvt", how="left").filter((pl.col("n") >= MIN_N) & (pl.col("share") >= MIN_SHARE))
    s = pl.read_parquet(f"submissions/gentle-igloo_v{base}.parquet")
    j = s.join(rk.select("MVT_ID_mvt", "d_mvt_sched"), on="MVT_ID_mvt", how="left", maintain_order="left")
    p_old = j["TAXITIME_SEC_mvt"].to_numpy().astype(float); dsc = j["d_mvt_sched"].to_numpy()
    fire = np.isfinite(dsc); out = np.where(fire, dsc, p_old)
    print(f"ranking: band rows {f.filter((pl.col('split') == 'rank') & band).height}, fired {int(fire.sum())}, RMS change {np.sqrt(((out - p_old) ** 2).mean()):.3f} s, mean |change| on fired {np.abs(out - p_old)[fire].mean():.0f} s")
    s.with_columns(pl.Series("TAXITIME_SEC_mvt", np.round(out)).cast(s.schema["TAXITIME_SEC_mvt"])).write_parquet(f"submissions/gentle-igloo_v{new}.parquet")
    print(f"wrote v{new} = v{base} + E133 rule")

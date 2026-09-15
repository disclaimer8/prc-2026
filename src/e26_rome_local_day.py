"""E26: Rome local calendar day for LIRF departures without NM at D = 4-30 h (competitor idea, probe src/rome_local_day_probe.py).

    FEATURES=data/features_final.parquet TABPFN_ALLOW_CPU_LARGE_DATASET=1 .venv/bin/python src/e26_rome_local_day.py
Reuses the E16 frame (same rows, candidates, expectation, TabPFN-2 control with the E11 features).
E26a: TabPFN with the E11 features plus `local_change` (take-off and schedule on different Europe/Rome dates).
E26b: control probabilities with the one-day-shift probability set to 0 where local_change is false (renormalised).
Gate (REVIEW_BRIEF.md, E26) as E11: LOMO-SSE >= 5% lower, lower without the top-3 rows, better in >= 8 of 12 months;
if both arms pass, the larger gain. Per-row predictions: data/e26_lomo.parquet.
"""
import numpy as np
import polars as pl

src = open("src/e16_tail_regimes.py").read().split("\nrows, months = [], []")[0]
g = {"__name__": "e26"}
exec(src, g)
t, FE11, tabpfn, expected, L, Y = g["t"], g["FE11"], g["tabpfn"], g["expected"], g["L"], g["Y"]
t = t.with_columns(sched_t=pl.from_epoch((pl.col("t_mvt") - pl.col("d_mvt_sched")).cast(pl.Int64), time_unit="s").dt.replace_time_zone("UTC"),
                   mvt_t=pl.from_epoch(pl.col("t_mvt").cast(pl.Int64), time_unit="s").dt.replace_time_zone("UTC")).with_columns(
    local_change=(pl.col("mvt_t").dt.convert_time_zone("Europe/Rome").dt.date() != pl.col("sched_t").dt.convert_time_zone("Europe/Rome").dt.date()).cast(pl.Float64))
FE26 = FE11 + ["local_change"]
rows, months = [], []
for m in range(1, 13):
    tr, te = t.filter(pl.col("month") != m), t.filter((pl.col("month") == m) & pl.col("b").is_in(L.OVERRIDE))
    if te.height == 0:
        continue
    d, y = te["d_mvt_sched"].to_numpy(), te[Y].to_numpy()
    p_c = tabpfn(tr, te, FE11)
    p_b = p_c.copy()
    no_change = te["local_change"].to_numpy() == 0
    p_b[no_change, 1] = 0.0
    p_b = p_b / p_b.sum(axis=1, keepdims=True)
    preds = {"control": expected(p_c, d), "e26a": expected(tabpfn(tr, te, FE26), d), "e26b": expected(p_b, d)}
    months.append((m, {k: float(((v - y) ** 2).sum()) for k, v in preds.items()}))
    rows.append(te.select("MVT_ID_mvt", "month", "b", "local_change", "d_mvt_sched", Y).with_columns(**{f"pred_{k}": pl.Series(v) for k, v in preds.items()}))
    print(f"month {m:2d}: n={te.height:3d} " + " ".join(f"{k} SSE={v:.3e}" for k, v in months[-1][1].items()), flush=True)
res = pl.concat(rows)
res.write_parquet("data/e26_lomo.parquet")
se_c = ((res["pred_control"] - res[Y]) ** 2).to_numpy()
mask = np.ones(len(se_c), bool)
mask[np.argsort(-se_c)[:3]] = False
decisions = {}
for arm in ("e26a", "e26b"):
    se_a = ((res[f"pred_{arm}"] - res[Y]) ** 2).to_numpy()
    gain, gain_wo = 100 * (se_a.sum() / se_c.sum() - 1), 100 * (se_a[mask].sum() / se_c[mask].sum() - 1)
    wins = sum(1 for _, s in months if s[arm] < s["control"])
    ok = gain <= -5 and gain_wo < 0 and wins >= 8
    decisions[arm] = (ok, gain)
    print(f"{arm}: LOMO-SSE ({gain:+.1f}%), without top-3 ({gain_wo:+.1f}%), months better {wins}/{len(months)}, "
          f"LB-equivalent RMSE change {(se_a.sum() - se_c.sum()) / 6 / (2 * 273.41 * 344841):+.2f} s -> {'PASS' if ok else 'FAIL'}")
passed = [a for a, (ok, _) in decisions.items() if ok]
print("DECISION E26:", "ACCEPT " + min(passed, key=lambda a: decisions[a][1]) if passed else "REJECT")

"""E28: neighbour windows strictly after the own take-off (competitor idea, zestful-fountain v12) added to the canonical model.

    .venv/bin/python src/e28_forward_neighbours.py build     writes data/features_e28.parquet
    .venv/bin/python src/e28_forward_neighbours.py summary   decision against data/cv_preds_nomonth
New columns: for the airport and for the runway, count / mean / std of the NM taxi proxy (take-off - AOBT_3, same validity
rule as features.add_neighbor_features) of other departures taking off in (t, t+15 min] and (t, t+60 min]; time to the next
departure take-off at the airport and on the runway. Existing symmetric +-30 min windows stay.
Gate (REVIEW_BRIEF.md, E28): total SSE of the 6 folds lower by >= 0.25 s LB-equivalent and lower in >= 4 of 6 folds.
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

if sys.argv[1] == "build":
    from features import _window_stats
    feat = pl.read_parquet("data/features_final.parquet")
    dep = feat.select("MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "t_mvt",
                      prox=pl.when(pl.col("d_mvt_aobt").is_between(0, 7200) & (pl.col("aobt_infeasible") != 1)).then(pl.col("d_mvt_aobt")))
    out = []
    for gname, keys in {"ap": ["ADEP_mvt"], "rwy": ["ADEP_mvt", "RUNWAY_mvt"]}.items():
        parts = []
        for _, g in dep.drop_nulls(keys).group_by(keys):
            g = g.sort("t_mvt")
            obs = g.drop_nulls("prox")
            times, vals, q = obs["t_mvt"].to_numpy(), obs["prox"].to_numpy(), g["t_mvt"].to_numpy()
            cols = {"MVT_ID_mvt": g["MVT_ID_mvt"].to_numpy()}
            for m in (15, 60):
                n, mean, std = _window_stats(times, vals, q, 1, m * 60, None)
                cols[f"fw_{gname}_{m}_n"], cols[f"fw_{gname}_{m}_mean"], cols[f"fw_{gname}_{m}_std"] = n, mean, std
            all_t = g["t_mvt"].to_numpy()
            idx = np.searchsorted(all_t, q, side="right")
            cols[f"fw_{gname}_next_gap"] = np.where(idx < len(all_t), all_t[np.clip(idx, 0, len(all_t) - 1)] - q, np.nan)
            parts.append(pl.DataFrame(cols))
        out.append(pl.concat(parts, how="diagonal_relaxed"))
    new = feat
    for p_ in out:
        new = new.join(p_, on="MVT_ID_mvt", how="left")
    assert new.height == feat.height
    print("columns added:", [c for c in new.columns if c.startswith("fw_")])
    new.write_parquet("data/features_e28.parquet")
else:
    LB_SCALE = 6 * 2 * 273.41 * 344841
    rows = []
    for p in sorted(glob.glob("data/cv_preds_nomonth/fold_*.npz")):
        q = p.replace("cv_preds_nomonth", "cv_preds_e28")
        if not os.path.exists(q):
            continue
        c, e = np.load(p), np.load(q)
        assert np.array_equal(c["mvt_id"], e["mvt_id"])
        y = c["y"]
        dl = (e["pred"] - y) ** 2 - (c["pred"] - y) ** 2
        m = y <= 7200
        rows.append((os.path.basename(p), dl.sum(), np.sort(dl)[:3].sum()))
        print(f"{rows[-1][0]}: SSE change {dl.sum():+.3e}, best 3 rows {rows[-1][2]:+.3e}; regressor RMSE y<=7200 "
              f"{np.sqrt(np.mean((c['reg'][m] - y[m]) ** 2)):.2f} -> {np.sqrt(np.mean((e['reg'][m] - y[m]) ** 2)):.2f}")
    tot = sum(r[1] for r in rows)
    lower = sum(r[1] < 0 for r in rows)
    ok = len(rows) == 6 and -tot / LB_SCALE >= 0.25 and lower >= 4
    print(f"DECISION E28: {'ACCEPT' if ok else 'REJECT'} (folds {len(rows)}, lower in {lower}, total {tot:+.3e} = {tot / LB_SCALE:+.2f} s LB-equivalent)")

"""E57: how much of a change to the proxy the regressor simply undoes (17.09).

    .venv/bin/python src/e57_absorption.py     per fold and per airport, on whatever E57 folds exist

The absorption is an identity, not a coincidence. src/train.py:258-259 fits the regressor on `clip(y) - base_f`
and src/train.py:280 rebuilds the prediction as `reg.predict(Xp) + base_p`: add a constant b to the proxy and the
target drops by b, the output rises by b, the sum does not move - provided the regressor can express b from features
it already has. A per-airport constant is one split on the categorical ADEP_mvt, which it has; so are the proxy's own
inputs d_mvt_aobt / d_mvt_lobt / d_mvt_eobt, which sit among its 129 features (verified 17.09).

So the question to ask before spending a CV run is not "is my transform smooth" but "can the model already express it".
Measured here: the per-airport level is absorbed 98%, the row-by-row spread only 65% - trees approximate a*raw in steps.
That is also why E46 (a multiplier) transferred and E57 (an intercept) did not: the surviving third of a multiplier
carried signal, the surviving third of an intercept did not.
"""
import os, sys, contextlib, io
os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"] = "load_as_storage"
os.environ["FEATURES"] = "data/features_e28.parquet"
sys.path.insert(0, "src")
import numpy as np, polars as pl
import train as T

cols = ["MVT_ID_mvt", "ADEP_mvt", "month", "split", "d_mvt_aobt", "d_mvt_lobt", "d_mvt_eobt", "d_mvt_sched", "d_mvt_iobt", T.TARGET]
tr = pl.read_parquet(T.FEATURES_PATH, columns=cols).filter(pl.col("split") == "train")
lab = T.regime_label(tr)
tr = tr.with_columns(reg_row=pl.Series(lab == 0))

for a, b in [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]:
    p46, p57 = f"data/cv_preds_e46/fold_{a}_{b}.npz", f"data/cv_preds_e57/fold_{a}_{b}.npz"
    if not (os.path.exists(p46) and os.path.exists(p57)):
        continue
    z46, z57 = np.load(p46), np.load(p57)
    o = {m: i for i, m in enumerate(z57["mvt_id"])}
    idx = np.array([o[m] for m in z46["mvt_id"]])
    d_reg = z57["reg"][idx] - z46["reg"]

    fit = tr.filter(~pl.col("month").is_in([a, b]) & pl.col("reg_row"))
    val = tr.filter(pl.col("month").is_in([a, b]))
    order = {m: i for i, m in enumerate(val["MVT_ID_mvt"].to_numpy())}
    vidx = np.array([order[m] for m in z46["mvt_id"]])
    proxies = {}
    for mode in ("SCALE", "AFFINE"):
        for k in ("PROXY_SCALE", "PROXY_AFFINE", "PROXY_ISO"):
            os.environ.pop(k, None)
        os.environ["PROXY_" + mode] = "1"
        with contextlib.redirect_stdout(io.StringIO()):
            _, p = T.proxy_base(fit, val)
        proxies[mode] = p[vidx]
    d_proxy = proxies["AFFINE"] - proxies["SCALE"]
    ap = val["ADEP_mvt"].to_numpy()[vidx]

    print(f"\nfold {a}+{b}: added to the proxy {d_proxy.mean():+.1f} s on average, the sum `reg` moved {d_reg.mean():+.1f} s "
          f"-> absorbed {100 * (1 - abs(d_reg.mean()) / max(abs(d_proxy.mean()), 1e-9)):.1f}%")
    print("  airport   added to proxy    reg moved   absorbed")
    for a_name in sorted(set(ap)):
        m = ap == a_name
        dp, dr = d_proxy[m].mean(), d_reg[m].mean()
        print(f"  {a_name:6s}  {dp:+11.1f} s  {dr:+11.1f} s   {100 * (1 - abs(dr) / max(abs(dp), 1e-9)):6.1f}%")

    # Level vs shape: the per-airport mean is the level; the spread around it is what the slope change does row by row.
    mp = {k: d_proxy[ap == k].mean() for k in set(ap)}
    mr = {k: d_reg[ap == k].mean() for k in set(ap)}
    cen_p = d_proxy - np.array([mp[k] for k in ap])
    cen_r = d_reg - np.array([mr[k] for k in ap])
    print(f"  level  (per-airport mean, |.|): proxy {np.mean([abs(v) for v in mp.values()]):6.1f} s -> "
          f"reg {np.mean([abs(v) for v in mr.values()]):5.2f} s "
          f"({100 * (1 - np.mean([abs(v) for v in mr.values()]) / np.mean([abs(v) for v in mp.values()])):.1f}% absorbed)")
    print(f"  shape  (spread within airport): proxy {cen_p.std():6.1f} s -> reg {cen_r.std():5.1f} s "
          f"({100 * (1 - cen_r.std() / max(cen_p.std(), 1e-9)):.1f}% absorbed)")

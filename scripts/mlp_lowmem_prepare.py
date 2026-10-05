"""E100 prepare with identical arithmetic and bounded intermediate storage.

The caller supplies the unchanged trainer's categorical list. Quantiles and
interpolation remain float64; casting to float32 happens column by column.
Categorical value_counts order is deliberately unchanged (Polars can vary it).
"""
from __future__ import annotations

import numpy as np
import polars as pl


def prepare(fit: pl.DataFrame, pred: pl.DataFrame, cols: list[str], *, categorical):
    cats = [c for c in categorical if c in cols]
    nums = [c for c in cols if c not in cats]
    qs = np.linspace(0, 1, 257)
    plans = []
    width = 0
    for c in nums:
        a = fit[c].cast(pl.Float64).to_numpy()
        ok = np.isfinite(a)
        if ok.sum() < 100:
            continue
        q = np.unique(np.quantile(a[ok], qs))
        if len(q) < 2:
            continue
        missing = bool((~ok).any())
        plans.append((c, q, missing))
        width += 1 + missing
    # Match the original column_stack failure when either family is empty.
    if not width or not cats:
        raise ValueError("need at least one array to concatenate")
    xnf = np.empty((len(fit), width), dtype=np.float32)
    xnp = np.empty((len(pred), width), dtype=np.float32)
    j = 0
    for c, q, missing in plans:
        # Do not downcast the source or quantiles before interpolation.
        a = fit[c].cast(pl.Float64).to_numpy()
        b = pred[c].cast(pl.Float64).to_numpy()
        def tf(v):
            r = np.interp(v, q, np.linspace(-2, 2, len(q)))
            return np.where(np.isfinite(v), r, 0.0)
        xnf[:, j] = tf(a)
        xnp[:, j] = tf(b)
        j += 1
        if missing:
            xnf[:, j] = ~np.isfinite(a)
            xnp[:, j] = ~np.isfinite(b)
            j += 1
    xcf = np.empty((len(fit), len(cats)), dtype=np.int64)
    xcp = np.empty((len(pred), len(cats)), dtype=np.int64)
    card = []
    for j, c in enumerate(cats):
        vc = fit[c].cast(pl.String).value_counts().filter(pl.col("count") >= 20)
        mp = {v: i + 1 for i, v in enumerate(vc[c].to_list())}
        xcf[:, j] = np.fromiter((mp.get(v, 0) for v in fit[c].cast(pl.String).to_list()),
                                dtype=np.int64, count=len(fit))
        xcp[:, j] = np.fromiter((mp.get(v, 0) for v in pred[c].cast(pl.String).to_list()),
                                dtype=np.int64, count=len(pred))
        card.append(len(mp) + 1)
    return xnf, xcf, xnp, xcp, card

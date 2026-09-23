"""E104 (23.09): a blend partner whose REGIME CLASSIFIER is an MLP; candidates are the E89 fold's own.

    FOLDS=1,2 NN_OUT=data/cv_preds_e104_nnclf .venv/bin/python src/e104_nn_clf.py
Most of the error budget sits in regime choice (the tail), yet every partner so far varied the regressor and kept the
same LightGBM classifier. Here the six regime probabilities come from a network (plain cross-entropy with the E89 cost
weight, fixed epochs), and are mixed with the E89 fold's candidates (regular = the GBDT regressor), masked and
LOBT-clipped as train.py does. Same inputs as E100 (src/e100_nn_partner.prepare).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import e100_nn_partner as N  # noqa: E402  (env defaults of the E89 chain)
import train as T  # noqa: E402
from target_encoding import add_encodings  # noqa: E402

EPOCHS = int(os.environ.get("NN_EPOCHS", "6"))


class Clf(nn.Module):
    def __init__(self, n_num, card, k, width=512):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c, min(16, (c + 1) // 2 + 1)) for c in card])
        d = n_num + sum(e.embedding_dim for e in self.emb)
        self.body = nn.Sequential(nn.Linear(d, width), nn.SiLU(), nn.Dropout(0.1), nn.Linear(width, width), nn.SiLU(),
                                  nn.Dropout(0.1), nn.Linear(width, width // 2), nn.SiLU(), nn.Linear(width // 2, k))

    def forward(self, xn, xc):
        return self.body(torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(self.emb)], dim=1))


def cost_weight(fit: pl.DataFrame, lab: np.ndarray) -> np.ndarray:
    """E89's classifier weight: 1 while the candidates lie within 3600 s, (spread/3600)^2 up to 400 beyond."""
    med = fit.filter(pl.Series(lab == 0)).group_by("ADEP_mvt").agg(pl.col(T.TARGET).median().alias("m"))
    mm = dict(zip(med["ADEP_mvt"].to_list(), med["m"].to_list()))
    reg = np.array([mm.get(a, 900.0) for a in fit["ADEP_mvt"].to_list()], dtype=float)
    cand = np.column_stack([fit[c].to_numpy().astype(float) for c in T.CANDIDATES.values()])
    d = fit["d_mvt_lobt"].to_numpy().astype(float)[:, None]
    cand = np.where(~np.isnan(d) & (np.abs(np.nan_to_num(cand) - np.nan_to_num(d)) > T.LOBT_WINDOW), np.nan, cand)
    cand = np.column_stack([cand, reg])
    with np.errstate(invalid="ignore"):
        spread = np.nan_to_num(np.nanmax(cand, axis=1) - np.nanmin(cand, axis=1), nan=0.0)
    return np.where(spread <= 3600, 1.0, np.clip((spread / 3600) ** 2, 1.0, 400.0))


def proba_nn(fit: pl.DataFrame, pred: pl.DataFrame) -> np.ndarray:
    lab = T.regime_label(fit)
    w = cost_weight(fit, lab) if os.environ.get("NN_CLF_W", "1") == "1" else np.ones(len(lab))
    fit, pred = add_encodings(fit, pred, lab == 0, fit_labels=lab)
    cols = T.feature_columns(fit)
    xnf, xcf, xnp, xcp, card = N.prepare(fit, pred, cols)
    k = len(T.CLASSES)
    torch.manual_seed(0)
    net = Clf(xnf.shape[1], card, k).to(N.DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    bs, n = 4096, len(lab)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=EPOCHS * (n // bs), pct_start=0.1)
    Xn, Xc = torch.from_numpy(xnf), torch.from_numpy(xcf)
    Y, W = torch.from_numpy(lab.astype(np.int64)), torch.from_numpy((w / w.mean()).astype(np.float32))
    rng = np.random.default_rng(0)
    for ep in range(EPOCHS):
        net.train(); perm = rng.permutation(n); tot = 0.0
        for i in range(n // bs):
            idx = torch.from_numpy(perm[i * bs:(i + 1) * bs])
            logit = net(Xn[idx].to(N.DEV), Xc[idx].to(N.DEV))
            loss = (nn.functional.cross_entropy(logit, Y[idx].to(N.DEV), reduction="none") * W[idx].to(N.DEV)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        print(f"  epoch {ep + 1}: weighted ce {tot / (n // bs):.4f}", flush=True)
    net.eval()
    with torch.no_grad():
        out = [torch.softmax(net(torch.from_numpy(xnp[i:i + 65536]).to(N.DEV), torch.from_numpy(xcp[i:i + 65536]).to(N.DEV)), 1).cpu().numpy()
               for i in range(0, len(xnp), 65536)]
    return np.concatenate(out).astype(np.float64)


def mix(proba: np.ndarray, values: np.ndarray, rows: pl.DataFrame) -> np.ndarray:
    proba = np.where(np.isnan(values), 0.0, proba)
    proba = T.mask_infeasible(proba, values, rows)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return np.clip(T.lobt_clip(np.sum(proba * np.nan_to_num(values), axis=1), rows), 0, None), proba


def main() -> None:
    out = os.environ.get("NN_OUT", "data/cv_preds_e104_nnclf")
    os.makedirs(out, exist_ok=True)
    folds = [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5,6").split(",")]
    df = pl.read_parquet(T.FEATURES_PATH)
    tr = df.filter(pl.col("split") == "train")
    del df
    for a in folds:
        b = a + 6
        path = f"{out}/fold_{a}_{b}.npz"
        if os.path.exists(path):
            continue
        fit, val = tr.filter(~pl.col("month").is_in([a, b])), tr.filter(pl.col("month").is_in([a, b]))
        print(f"fold {a}+{b}", flush=True)
        p = proba_nn(fit, val)
        c = np.load(f"{N.CTRL}/fold_{a}_{b}.npz")
        pos = {m: i for i, m in enumerate(val["MVT_ID_mvt"].to_numpy())}
        idx = np.array([pos[m] for m in c["mvt_id"]])
        pv, values = val[idx], c["values"].astype(np.float64)
        pred, proba = mix(p[idx], values, pv)
        y = c["y"]
        lab = T.regime_label(pv)
        ll = lambda q: -np.mean(np.log(np.clip(q[np.arange(len(lab)), lab], 1e-9, 1)))  # noqa: E731
        print(f"  logloss GBDT {ll(c['proba']):.4f} NN {ll(proba):.4f}; mixture RMSE ctrl {np.sqrt(np.mean((c['pred'] - y) ** 2)):.2f} "
              f"NN {np.sqrt(np.mean((pred - y) ** 2)):.2f}; corr of errors {np.corrcoef(c['pred'] - y, pred - y)[0, 1]:.3f}", flush=True)
        np.savez(path, mvt_id=c["mvt_id"], pred=pred, reg=c["reg"], y=y, proba=proba.astype(np.float32), values=values.astype(np.float32))
    print("done", out, flush=True)


if __name__ == "__main__":
    main()

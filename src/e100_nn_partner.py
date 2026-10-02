"""E100 (23.09): a blend partner from another model family -- an MLP regressor in place of the GBDT one.

    FOLDS=1,2 NN_OUT=data/cv_preds_e100_nn .venv/bin/python src/e100_nn_partner.py      # OOF arm
Everything but the regular-regime candidate is taken from the E89 fold files (data/cv_preds_e89_cand): the classifier
probabilities, the planned-time candidates and the LOBT clip are the same objects, so the partner differs from the
control by one link only. Every blend partner so far was the same GBDT family (residual correlation 0.94-0.96 with the
chain), which is why blends buy tenths; a network fitted to the same residual target errs differently.
The network learns what the GBDT regressor learns: the residual against the calibrated NM proxy (train.proxy_base),
on regular-regime rows, target clipped as in train.py. Fixed epochs, no early stopping on the scored months.
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("PROXY_SCALE", "1")
os.environ.setdefault("FEATURES", "data/features_e77.parquet")
os.environ.setdefault("DROP_PREFIX", "day_,tq_,wxr_,month")
import numpy as np
import polars as pl
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(__file__))
import train as T  # noqa: E402
from target_encoding import add_encodings  # noqa: E402

CTRL = os.environ.get("NN_CTRL", "data/cv_preds_e89_cand")
EPOCHS = int(os.environ.get("NN_EPOCHS", "8"))
DEV = torch.device(os.environ.get("NN_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu"))


def prepare(fit: pl.DataFrame, pred: pl.DataFrame, cols: list[str]):
    """Numeric: rank-gauss from fit quantiles, NaN -> 0 plus an indicator. Categorical: frequency codes, rare -> 0."""
    cats = [c for c in T.CATEGORICAL if c in cols]
    nums = [c for c in cols if c not in cats]
    qs = np.linspace(0, 1, 257)
    xf, xp = [], []
    for c in nums:
        a = fit[c].cast(pl.Float64).to_numpy()
        b = pred[c].cast(pl.Float64).to_numpy()
        ok = np.isfinite(a)
        if ok.sum() < 100:
            continue
        q = np.unique(np.quantile(a[ok], qs))
        if len(q) < 2:
            continue
        def tf(v):
            r = np.interp(v, q, np.linspace(-2, 2, len(q)))
            return np.where(np.isfinite(v), r, 0.0)
        xf.append(tf(a)); xp.append(tf(b))
        if (~ok).any():
            xf.append((~ok).astype(float)); xp.append((~np.isfinite(b)).astype(float))
    cf, cp, card = [], [], []
    for c in cats:
        vc = fit[c].cast(pl.String).value_counts().filter(pl.col("count") >= 20)
        mp = {v: i + 1 for i, v in enumerate(vc[c].to_list())}
        cf.append(np.array([mp.get(v, 0) for v in fit[c].cast(pl.String).to_list()]))
        cp.append(np.array([mp.get(v, 0) for v in pred[c].cast(pl.String).to_list()]))
        card.append(len(mp) + 1)
    return (np.column_stack(xf).astype(np.float32), np.column_stack(cf).astype(np.int64),
            np.column_stack(xp).astype(np.float32), np.column_stack(cp).astype(np.int64), card)


class Net(nn.Module):
    def __init__(self, n_num: int, card: list[int], width: int = 512):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(k, min(16, (k + 1) // 2 + 1)) for k in card])
        d = n_num + sum(e.embedding_dim for e in self.emb)
        self.body = nn.Sequential(nn.Linear(d, width), nn.SiLU(), nn.Dropout(0.1),
                                  nn.Linear(width, width), nn.SiLU(), nn.Dropout(0.1),
                                  nn.Linear(width, width // 2), nn.SiLU(), nn.Linear(width // 2, 1))

    def forward(self, xn, xc):
        return self.body(torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(self.emb)], dim=1)).squeeze(1)


def fit_nn(xn, xc, y, card, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(xn.shape[1], card, width=int(os.environ.get("NN_WIDTH", "512"))).to(DEV)   # E181: NN_WIDTH (default = E100)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    bs, n = 4096, len(y)
    steps = EPOCHS * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    Xn, Xc, Y = torch.from_numpy(xn), torch.from_numpy(xc), torch.from_numpy(y.astype(np.float32))
    rng = np.random.default_rng(seed)
    for ep in range(EPOCHS):
        net.train(); perm = rng.permutation(n); tot = 0.0
        for i in range(n // bs):
            idx = torch.from_numpy(perm[i * bs:(i + 1) * bs])
            loss = nn.functional.mse_loss(net(Xn[idx].to(DEV), Xc[idx].to(DEV)), Y[idx].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        print(f"  epoch {ep + 1}: train mse {tot / (n // bs):.4f}", flush=True)
    return net


@torch.no_grad()
def predict(net, xn, xc):
    net.eval()
    out = [net(torch.from_numpy(xn[i:i + 65536]).to(DEV), torch.from_numpy(xc[i:i + 65536]).to(DEV)).cpu().numpy()
           for i in range(0, len(xn), 65536)]
    return np.concatenate(out)


def regular_candidate(fit: pl.DataFrame, pred: pl.DataFrame) -> np.ndarray:
    lab = T.regime_label(fit)
    fit, pred = add_encodings(fit, pred, lab == 0, fit_labels=lab)
    cols = T.feature_columns(fit)
    reg_mask = lab == 0
    base_f, base_p = T.proxy_base(fit.filter(pl.Series(reg_mask)), pred)
    y = np.clip(np.clip(fit[T.TARGET].to_numpy()[reg_mask], *T.REG_CLIP) - base_f, -7200, 7200)
    sd = float(y.std())
    xnf, xcf, xnp, xcp, card = prepare(fit.filter(pl.Series(reg_mask)), pred, cols)
    print(f"  rows {len(y)}, numeric {xnf.shape[1]}, categorical {len(card)}, resid sd {sd:.0f}", flush=True)
    # E101: NN_SEEDS=k averages k networks (seed 0 alone is bit-for-bit the E100 arm)
    k = int(os.environ.get("NN_SEEDS", "1"))
    out = np.mean([predict(fit_nn(xnf, xcf, y / sd, card, seed=s), xnp, xcp) for s in range(k)], axis=0)
    return out * sd + base_p


def main() -> None:
    out = os.environ.get("NN_OUT", "data/cv_preds_e100_nn")
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
        r = regular_candidate(fit, val)
        c = np.load(f"{CTRL}/fold_{a}_{b}.npz")
        pos = {m: i for i, m in enumerate(val["MVT_ID_mvt"].to_numpy())}
        idx = np.array([pos[m] for m in c["mvt_id"]])
        r = r[idx]
        values = c["values"].astype(np.float64).copy()
        values[:, 0] = r
        values[:, -1] = r + T.DAY
        proba = c["proba"].astype(np.float64)
        pv = val[idx]
        pred = np.clip(T.lobt_clip(np.sum(proba * np.nan_to_num(values), axis=1), pv), 0, None)
        y = c["y"]
        reg = T.regime_label(pv) == 0
        small = reg & (y <= 7200)
        print(f"  regular rows y<=7200: RMSE GBDT {np.sqrt(np.mean((c['reg'][small] - y[small]) ** 2)):.2f} "
              f"NN {np.sqrt(np.mean((r[small] - y[small]) ** 2)):.2f}; residual corr "
              f"{np.corrcoef(c['reg'][small] - y[small], r[small] - y[small])[0, 1]:.3f}; "
              f"mixture RMSE ctrl {np.sqrt(np.mean((c['pred'] - y) ** 2)):.2f} NN {np.sqrt(np.mean((pred - y) ** 2)):.2f}", flush=True)
        np.savez(path, mvt_id=c["mvt_id"], pred=pred, reg=r, y=y, proba=c["proba"], values=values.astype(np.float32))
    print("done", out, flush=True)


if __name__ == "__main__":
    main()

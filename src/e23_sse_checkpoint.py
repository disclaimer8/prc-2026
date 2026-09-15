"""E23: choose the number of regime-classifier rounds by SSE of the final mixture instead of multi_logloss (direction C, 15.09).

    FEATURES=data/features_final.parquet DROP_PREFIX=day_,tq_,wxr_ .venv/bin/python src/e23_sse_checkpoint.py <a> <b>
    .venv/bin/python src/e23_sse_checkpoint.py summary
Nested, no leakage: outer fold = months a and b; inner validation = months a+1 and b+1 (inside the 10 fit months).
The classifier is trained on the 8 remaining months up to MAX_ROUNDS; control picks the round with the lowest multi_logloss on the
inner months, E23 the round with the lowest mixture SSE there (regressor trained on the same 8 months, LOBT mask and clip as in
train.fit_predict). Both are refit on the 10 fit months with rounds x 1.1 and scored on the outer months with the stored OOF
regressor and candidates of data/cv_preds/fold_<a>_<b>.npz. Writes data/e23/fold_<a>_<b>.npz.
Gate (REVIEW_BRIEF.md, E23): total outer SSE lower by >= 0.5 s LB-equivalent and lower in >= 4 of 6 folds.
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

LB_SCALE = 6 * 2 * 274.28 * 344841  # 6 folds cover 12 months, ranking has 2

if sys.argv[1] == "summary":
    res = []
    for p in sorted(glob.glob("data/e23/fold_*.npz")):
        z = np.load(p)
        res.append((os.path.basename(p), int(z["k_ctrl"]), int(z["k_sse"]), float(z["sse_ctrl"]), float(z["sse_e23"]), z["top3"]))
        print(f"{res[-1][0]}: rounds logloss {res[-1][1]} vs SSE {res[-1][2]}; outer SSE {res[-1][3]:.4e} -> {res[-1][4]:.4e} "
              f"({res[-1][4] - res[-1][3]:+.3e}); top-3 row share of the change {float(z['top3']):.2f}")
    d = sum(r[4] - r[3] for r in res)
    lower = sum(r[4] < r[3] for r in res)
    ok = len(res) == 6 and -d / LB_SCALE >= 0.5 and lower >= 4
    print(f"DECISION E23: {'ACCEPT' if ok else 'REJECT'} (folds {len(res)}, lower in {lower}, total SSE change {d:+.3e} = {d / LB_SCALE:+.2f} s LB-equivalent RMSE)")
    sys.exit()

import lightgbm as lgb
import polars as pl

import train as T
from target_encoding import add_encodings

a, b = int(sys.argv[1]), int(sys.argv[2])
MAX_ROUNDS, STEP = 400, 10
inner = [m % 12 + 1 for m in (a, b)]
df = pl.read_parquet(T.FEATURES_PATH)
tr = df.filter(pl.col("split") == "train")
del df
fit_all = tr.filter(~pl.col("month").is_in([a, b]))
outer = tr.filter(pl.col("month").is_in([a, b]))
fit_in, val_in = fit_all.filter(~pl.col("month").is_in(inner)), fit_all.filter(pl.col("month").is_in(inner))


def design(fit, pred):
    lab = T.regime_label(fit)
    fit, pred = add_encodings(fit, pred, lab == 0)
    cols = T.feature_columns(fit)
    Xf, cats = T.to_pandas(fit, cols, None)
    Xp, _ = T.to_pandas(pred, cols, cats)
    return fit, pred, lab, Xf, Xp, [c for c in T.CATEGORICAL if c in cols]


def mixture(proba, values, pred_df):
    proba = np.where(np.isnan(values), 0.0, proba)
    proba = T.mask_infeasible(proba, values, pred_df)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return np.clip(T.lobt_clip(np.sum(proba * np.nan_to_num(values), axis=1), pred_df), 0, None)


# inner: classifier checkpoints and a regressor on the 8 months
f8, v2, lab8, Xf, Xp, cat_cols = design(fit_in, val_in)
clf = lgb.train(T.CLF_PARAMS, lgb.Dataset(Xf, lab8, categorical_feature=cat_cols), MAX_ROUNDS)
mask = lab8 == 0
base_f, base_p = T.proxy_base(f8.filter(pl.Series(mask)), v2)
y8 = f8[T.TARGET].to_numpy()
reg = lgb.train(T.REG_PARAMS, lgb.Dataset(Xf[mask], np.clip(np.clip(y8[mask], *T.REG_CLIP) - base_f, -7200, 7200), categorical_feature=cat_cols), 600)
r = reg.predict(Xp) + base_p
values = np.column_stack([r, *[v2[c].to_numpy() for c in T.CANDIDATES.values()], r + T.DAY])
yv, labv = v2[T.TARGET].to_numpy(), T.regime_label(v2)
raw = np.zeros((len(yv), len(T.CLASSES)))
ks, loglosses, sses = [], [], []
for start in range(0, MAX_ROUNDS, STEP):
    raw += clf.predict(Xp, start_iteration=start, num_iteration=STEP, raw_score=True)
    e = np.exp(raw - raw.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    ks.append(start + STEP)
    loglosses.append(float(-np.mean(np.log(np.clip(p[np.arange(len(labv)), labv], 1e-15, None)))))
    sses.append(float(np.sum((mixture(p, values, v2) - yv) ** 2)))
k_ctrl, k_sse = ks[int(np.argmin(loglosses))], ks[int(np.argmin(sses))]
print(f"fold {a}+{b}: inner months {inner}; best rounds logloss {k_ctrl} (SSE there {sses[ks.index(k_ctrl)]:.4e}), "
      f"best rounds SSE {k_sse} (SSE {min(sses):.4e})", flush=True)
del f8, v2, Xf, Xp, clf, reg

# outer: refit on the 10 months, score with the stored OOF regressor and candidates
z = np.load(f"data/cv_preds/fold_{a}_{b}.npz")
F, O, labF, XF, XO, cat_cols = design(fit_all, outer)
assert np.array_equal(O["MVT_ID_mvt"].to_numpy(), z["mvt_id"])
yo = O[T.TARGET].to_numpy()
preds = {}
for name, k in (("ctrl", k_ctrl), ("e23", k_sse)):
    if name == "e23" and k_sse == k_ctrl:
        preds["e23"] = preds["ctrl"]
        continue
    c = lgb.train(T.CLF_PARAMS, lgb.Dataset(XF, labF, categorical_feature=cat_cols), int(k * 1.1))
    preds[name] = mixture(c.predict(XO), z["values"], O)
se_c, se_e = (preds["ctrl"] - yo) ** 2, (preds["e23"] - yo) ** 2
dlt = se_e - se_c
top3 = float(np.sort(np.abs(dlt))[-3:].sum() / max(np.abs(dlt).sum(), 1e-9))
os.makedirs("data/e23", exist_ok=True)
np.savez(f"data/e23/fold_{a}_{b}.npz", k_ctrl=k_ctrl, k_sse=k_sse, ks=ks, loglosses=loglosses, sses=sses,
         sse_ctrl=se_c.sum(), sse_e23=se_e.sum(), top3=top3, pred_ctrl=preds["ctrl"], pred_e23=preds["e23"], mvt_id=z["mvt_id"])
print(f"fold {a}+{b}: outer SSE ctrl {se_c.sum():.4e} (stored OOF {np.sum((z['pred'] - yo) ** 2):.4e}), e23 {se_e.sum():.4e}", flush=True)

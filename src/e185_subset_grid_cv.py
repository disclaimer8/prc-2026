"""E185 step 2: the movement-only member (src/e185_motion_only_cv.py) blended into the regular candidate ONLY on the rows without an
NM record outside LIRF; every other row keeps the control's prediction bit for bit.

    FEATURES=data/features_e106.parquet .venv/bin/python src/e185_subset_grid_cv.py
    FEATURES=data/features_e106.parquet .venv/bin/python src/gate.py data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlpa_f1b data/e185_cv_sub

On top of a finished candidate dir (CAND, default the six with the E181a MLP under the FIN1b control,
data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlpa_f1b): on subset rows reg' = (1 - w) reg + w motion, the mixture recomputed with
the stored regime probabilities and candidates exactly as src/e172_members_cv.py (difference form: pred + mix(reg') - mix(reg)).
w on a grid 0..E185_WMAX step 0.05; the weight for fold k is chosen on the other five folds.

Selection criterion E185_SELECT (pre-registered: body):
  body  SSE of subset rows with y <= 7200. The member is a body model (target clip(y, 0, 7200)); the subset's 42 rows with y > 7200
        (up to 84 240 s) hold 85 % of its SSE in the control OOF, and on them the SSE change is -2 (y - pred) * delta, i.e. the sign of
        (motion - reg) on two or three rows would pick the weight.
  all   SSE of all subset rows (the E172 convention), printed in any case as a diagnostic.
Writes OUT (default data/e185_cv_sub)/fold_*.npz in the candidate format (pred and reg changed on subset rows only).
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import train as T  # noqa: E402

CAND = os.environ.get("CAND", "data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlpa_f1b")
MEMBER = os.environ.get("MEMBER", "data/cv_e130motion")
OUT = os.environ.get("OUT", "data/e185_cv_sub")
SELECT = os.environ.get("E185_SELECT", "body")
WMAX = float(os.environ.get("E185_WMAX", "1.0"))
GRID = np.round(np.arange(0.0, WMAX + 1e-9, 0.05), 2)
LB = 6 * 2 * 252.43 * 344841  # fold_metrics.compare: LB-equivalent seconds of a fold's SSE change
os.makedirs(OUT, exist_ok=True)
feat = (pl.read_parquet(T.FEATURES_PATH, columns=["MVT_ID_mvt", "split", "ADEP_mvt", "LOBT_flt", "d_mvt_lobt"])
        .filter(pl.col("split") == "train").drop("split"))


def mix(proba, values, reg, dl):
    """src/e172_members_cv.py mix on a row subset: regular candidate = reg, dayshift = reg + 1 day, T.lobt_clip, clip at 0."""
    v = values.astype(float).copy()
    v[:, 0], v[:, -1] = reg, reg + T.DAY
    d = np.sum(proba.astype(float) * np.nan_to_num(v), axis=1)
    return np.clip(np.where(np.isnan(dl), d, np.clip(d, dl - T.LOBT_WINDOW, dl + T.LOBT_WINDOW)), 0, None)


folds = []
for p in sorted(glob.glob(f"{CAND}/fold_*.npz")):
    nm = os.path.basename(p)
    if not os.path.exists(f"{MEMBER}/{nm}"):
        print(f"{nm}: no member fold in {MEMBER}, skipped", flush=True)
        continue
    z = dict(np.load(p))
    s = np.load(f"{MEMBER}/{nm}")
    order = {k: j for j, k in enumerate(s["mvt_id"])}
    R = s["reg2"][[order[k] for k in z["mvt_id"]]]
    d = pl.DataFrame({"MVT_ID_mvt": z["mvt_id"]}).join(feat, on="MVT_ID_mvt", how="left", maintain_order="left")
    S = (d["LOBT_flt"].is_null() & (d["ADEP_mvt"] != "LIRF")).to_numpy()
    y = z["y"].astype(float)
    P, V, reg, dl = z["proba"][S], z["values"][S], z["reg"][S], d["d_mvt_lobt"].to_numpy()[S]
    base = mix(P, V, reg, dl)
    gap = float(np.abs(base - z["pred"][S]).max())
    sse_all, sse_body, newp = {}, {}, {}
    b = y[S] <= 7200
    for w in GRID:
        new = z["pred"][S] + mix(P, V, (1 - w) * reg + w * R[S], dl) - base
        e = (new - y[S]) ** 2
        sse_all[w], sse_body[w], newp[w] = float(e.sum()), float(e[b].sum()), new
    folds.append(dict(nm=nm, z=z, S=S, R=R, y=y, b=b, sse_all=sse_all, sse_body=sse_body, newp=newp, ap=d["ADEP_mvt"].to_numpy()[S]))
    ys = y[S]
    print(f"{nm}: subset {S.sum()} rows ({b.sum()} with y<=7200), |mix(reg) - pred| max {gap:.2e}; on subset y<=7200: corr(motion, reg) "
          f"{np.corrcoef(R[S][b], reg[b])[0, 1]:.4f}, RMSE reg {np.sqrt(np.mean((reg[b] - ys[b]) ** 2)):.1f}, motion "
          f"{np.sqrt(np.mean((R[S][b] - ys[b]) ** 2)):.1f}, mean(motion - reg) {np.mean(R[S][b] - reg[b]):+.1f}", flush=True)
if not folds:
    sys.exit("no folds")

crit = {"body": "sse_body", "all": "sse_all"}
n_sub = sum(f["S"].sum() for f in folds)
n_body = sum(f["b"].sum() for f in folds)
for key in ("sse_body", "sse_all"):
    tot = {w: sum(f[key][w] for f in folds) for w in GRID}
    n = n_body if key == "sse_body" else n_sub
    best = min(GRID, key=lambda w: tot[w])
    print(f"pooled curve, subset {'y<=7200' if key == 'sse_body' else 'all y'} RMSE (in-sample): "
          + " ".join(f"{w:.2f}:{np.sqrt(tot[w] / n):.2f}" for w in GRID[::2]) + f" | best {best:.2f} {np.sqrt(tot[best] / n):.3f}", flush=True)


def choose(key):
    tot = {w: sum(f[key][w] for f in folds) for w in GRID}
    return [min(GRID, key=lambda w: tot[w] - f[key][w]) for f in folds]  # ties -> the smaller weight (GRID ascending)


for name in ("body", "all"):
    ws = choose(crit[name])
    d_all = [f["sse_all"][w] - f["sse_all"][0.0] for f, w in zip(folds, ws)]
    d_body = [f["sse_body"][w] - f["sse_body"][0.0] for f, w in zip(folds, ws)]
    trim = []
    for f, w in zip(folds, ws):
        dd = np.sort((f["newp"][w] - f["y"][f["S"]]) ** 2 - (f["newp"][0.0] - f["y"][f["S"]]) ** 2)
        trim.append(dd.sum() - dd[:3].sum() - dd[-3:].sum())
    tag = "PRE-REGISTERED" if name == SELECT else "diagnostic"
    print(f"selection on {name} ({tag}): weights {[float(w) for w in ws]}; subset change, LB-equivalent s: all {sum(d_all) / LB:+.4f} "
          f"(lower in {sum(x < 0 for x in d_all)}/{len(folds)}) | y<=7200 {sum(d_body) / LB:+.4f} (lower in {sum(x < 0 for x in d_body)}/{len(folds)}) "
          f"| trimmed {sum(trim) / LB:+.4f}", flush=True)
    if name != SELECT:
        continue
    # RMSE of the subset and of all rows, before and after (pooled over the folds present)
    e0 = np.concatenate([(f["newp"][0.0] - f["y"][f["S"]]) ** 2 for f in folds])
    e1 = np.concatenate([(f["newp"][w] - f["y"][f["S"]]) ** 2 for f, w in zip(folds, ws)])
    bb = np.concatenate([f["b"] for f in folds])
    all0 = sum(float(((f["z"]["pred"] - f["y"]) ** 2).sum()) for f in folds)
    nall = sum(len(f["y"]) for f in folds)
    all1 = all0 + float(e1.sum() - e0.sum())
    print(f"  RMSE subset all y {np.sqrt(e0.mean()):.2f} -> {np.sqrt(e1.mean()):.2f} | subset y<=7200 {np.sqrt(e0[bb].mean()):.2f} -> "
          f"{np.sqrt(e1[bb].mean()):.2f} ({100 * (np.sqrt(e1[bb].mean()) / np.sqrt(e0[bb].mean()) - 1):+.2f} %) | all rows {np.sqrt(all0 / nall):.3f} -> "
          f"{np.sqrt(all1 / nall):.3f}", flush=True)
    for f, w, da, db in zip(folds, ws, d_all, d_body):
        print(f"  {f['nm']}: w {w:.2f}, subset all {da / LB:+.4f}, y<=7200 {db / LB:+.4f}", flush=True)
    aps = np.concatenate([f["ap"] for f in folds])
    dd = e1 - e0
    print("  by airport (LB-equivalent s, all y | y<=7200): " + ", ".join(
        f"{a} {dd[aps == a].sum() / LB:+.4f}|{dd[(aps == a) & bb].sum() / LB:+.4f}" for a in sorted(set(aps))), flush=True)
    for f, w in zip(folds, ws):
        z = dict(f["z"])
        S = f["S"]
        reg = z["reg"].astype(float).copy()
        reg[S] = (1 - w) * reg[S] + w * f["R"][S]
        pred = z["pred"].astype(float).copy()
        pred[S] = f["newp"][w]
        z["reg"], z["pred"] = reg, pred
        np.savez(f"{OUT}/{f['nm']}", **z)
    tot = {w: sum(f[crit[name]][w] for f in folds) for w in GRID}
    print(f"wrote {OUT} ({len(folds)} folds); board weight (pooled {name} selection over these folds) {min(GRID, key=lambda w: tot[w]):.2f}", flush=True)

"""E179 (journal 235, external review 01.10, item 4): soft shrinkage of the ADS-B stage correction by regime uncertainty.

E152 (src/e152_stage_gate.py) is a hard gate: where the base is sure of a filed-time regime (max P(sched|iobt|lobt|eobt) >= 0.95)
the stage is dropped. Here pred = base + g * (staged - base), with g in [0, 1] looked up in a table of cells
P(regular) x mixture spread of the candidates (sqrt(sum p_k (v_k - mean)^2) over finite candidates, s). g per cell = the least-
squares shrink clip(sum d*(y - base) / sum d^2, 0, 1), fitted on the other five folds and applied to the sixth (cells with < 300
changed rows fall back to the E152 rule). Control = E152 at 0.95 on the same OOF. The stage already sees p_nonreg, so g is
estimated, never set to P(regular).
OOF: base data/cv_preds_nomonth (the base the stage chain was trained on), staged data/cv_preds_e150c_e64_keep (E141 keep-all +
E64 keep-all, the chain of E152/E159b). Writes data/cv_preds_e179_ctrl and data/cv_preds_e179_soft for src/gate.py.
"""
import glob
import os

import numpy as np

BASE, STAGED = "data/cv_preds_nomonth", "data/cv_preds_e150c_e64_keep"
P_EDGES = np.array([0.0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0001])
S_EDGES = np.array([0.0, 120.0, 600.0, 1800.0, 3600.0, 1e9])
MIN_ROWS = 300
GATE = 0.95


def load():
    out = []
    for p in sorted(glob.glob(f"{BASE}/fold_*.npz")):
        n = os.path.basename(p)
        b, s = np.load(p), np.load(f"{STAGED}/{n}")
        pos = {m: i for i, m in enumerate(s["mvt_id"])}
        idx = np.array([pos[m] for m in b["mvt_id"]])
        proba, values = b["proba"].astype(float), b["values"].astype(float)
        fin = np.isfinite(values)
        pv = np.where(fin, proba, 0.0); pv = pv / np.clip(pv.sum(1, keepdims=True), 1e-12, None)
        v0 = np.where(fin, values, 0.0)
        mean = (pv * v0).sum(1)
        spread = np.sqrt((pv * (v0 - mean[:, None]) ** 2).sum(1))
        conf = proba[:, 1:5].max(axis=1)
        out.append(dict(name=n, z=dict(b), base=b["pred"].astype(float), staged=s["pred"][idx].astype(float), y=b["y"].astype(float),
                        preg=proba[:, 0], spread=spread, conf=conf))
    return out


folds = load()
for f in folds:
    f["pc"] = np.clip(np.digitize(f["preg"], P_EDGES) - 1, 0, len(P_EDGES) - 2)
    f["sc"] = np.clip(np.digitize(f["spread"], S_EDGES) - 1, 0, len(S_EDGES) - 2)
    f["d"] = f["staged"] - f["base"]
    f["ctrl"] = np.where(f["conf"] >= GATE, f["base"], f["staged"])
nP, nS = len(P_EDGES) - 1, len(S_EDGES) - 1
for k, f in enumerate(folds):
    num = np.zeros((nP, nS)); den = np.zeros((nP, nS)); cnt = np.zeros((nP, nS))
    for j, o in enumerate(folds):
        if j == k:
            continue
        ch = o["d"] != 0
        np.add.at(num, (o["pc"][ch], o["sc"][ch]), (o["d"] * (o["y"] - o["base"]))[ch])
        np.add.at(den, (o["pc"][ch], o["sc"][ch]), (o["d"] ** 2)[ch])
        np.add.at(cnt, (o["pc"][ch], o["sc"][ch]), 1)
    g = np.clip(num / np.where(den > 0, den, 1), 0, 1)
    use = cnt >= MIN_ROWS
    gr = g[f["pc"], f["sc"]]
    soft = np.where(use[f["pc"], f["sc"]], f["base"] + gr * f["d"], f["ctrl"])
    f["soft"] = soft
    if k == 0:
        print("g table (fit without fold 1; rows: P(regular) bins", list(P_EDGES[:-1]), "; cols: spread bins", list(S_EDGES[:-1]), ")")
        for i in range(nP):
            print("  " + " ".join(f"{g[i, s]:.2f}({int(cnt[i, s]):>6d})" for s in range(nS)))
    sse_c = float(((f["ctrl"] - f["y"]) ** 2).sum()); sse_s = float(((soft - f["y"]) ** 2).sum())
    print(f"{f['name']}: SSE ctrl {sse_c:.4e} soft {sse_s:.4e} diff {(sse_s - sse_c) / len(f['y']):+.2f} s^2/row", flush=True)
for arm in ("ctrl", "soft"):
    out = f"data/cv_preds_e179_{arm}"
    os.makedirs(out, exist_ok=True)
    for f in folds:
        z = dict(f["z"]); z["pred"] = f[arm]
        np.savez(f"{out}/{f['name']}", **z)
print("wrote data/cv_preds_e179_ctrl and data/cv_preds_e179_soft")

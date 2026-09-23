"""E100 gate: LOFO-NNLS weight of the NN partner against the E89 stage output (the object on the board), decided rows."""
import numpy as np
from scipy.optimize import nnls
import fold_metrics as FM

REF, ARM = "data/cv_preds_e89_stage", "data/cv_preds_e100_nn"
FOLDS = [f"fold_{a}_{a + 6}" for a in range(1, 7)]
dec, st = FM.decided_map(), FM.decided_map(strict=True)
lb = 6 * 2 * 242.6556 * 344841
D = {}
for f in FOLDS:
    r, e = np.load(f"{REF}/{f}.npz"), np.load(f"{ARM}/{f}.npz")
    o = {m: i for i, m in enumerate(e["mvt_id"])}
    p = e["pred"][np.array([o[m] for m in r["mvt_id"]])]
    D[f] = (r["pred"], p, r["y"], np.array([dec[k] for k in r["mvt_id"]]), np.array([st[k] for k in r["mvt_id"]]))
tot = {k: 0.0 for k in ("decided", "strict", "y<=7200", "trimmed")}; lower = 0; ws = []
for f in FOLDS:
    A = np.concatenate([(D[g][1] - D[g][0])[D[g][3]] for g in FOLDS if g != f])
    b = np.concatenate([(D[g][2] - D[g][0])[D[g][3]] for g in FOLDS if g != f])
    w = float(nnls(A[:, None], b)[0][0]); ws.append(w)
    base, p, y, m, s = D[f]
    d = (base + w * (p - base) - y) ** 2 - (base - y) ** 2
    q = np.sort(d)
    parts = {"decided": d[m].sum(), "strict": d[s].sum(), "y<=7200": d[y <= 7200].sum(), "trimmed": d.sum() - q[:3].sum() - q[-3:].sum()}
    for k in tot: tot[k] += parts[k]
    lower += parts["decided"] < 0
    print(f + ": w %.3f " % w + " ".join(f"{k} {v / lb:+.3f}" for k, v in parts.items()))
ok = -tot["decided"] / lb >= 0.3 and lower >= 4 and -tot["trimmed"] / lb >= 0.3
print(f"DECISION E100: {'ACCEPT' if ok else 'REJECT'} (w {np.mean(ws):.3f}±{np.std(ws):.3f}, lower {lower}/6; "
      + ", ".join(f"{k} {v / lb:+.3f}" for k, v in tot.items()) + ")")

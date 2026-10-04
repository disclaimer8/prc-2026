"""Idea 5 (journal/_UPDATE_2026-10-04): which CV slice predicts the board delta, and where the CV gain sits by y-bin.
Read-only. Run: OMP_NUM_THREADS=2 .venv/bin/python analysis/cv_bins_vs_board.py
Part 1: pairs (change, CV slices, board delta, type) -> Pearson/Spearman/sign/slope per slice, overall and by type.
Part 2: for pairs with OOF dirs on both sides, split the CV gain into y<=3600, 3600<y<=7200, y>7200 (control y) and
        correlate each bin with the board delta.
Slices are LB-equivalent seconds as in src/fold_metrics.compare (positive = candidate worse)."""
import os, sys, glob, functools
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("FEATURES", "data/features_e28.parquet")
sys.path.insert(0, "src")
import numpy as np
import fold_metrics as fm
fm.decided_map = functools.lru_cache(maxsize=4)(fm.decided_map)
D = "data/cv_preds_"
NAN = float("nan")
SL = ["decided", "strict", "all", "y7200", "trim"]

# (id, change, type, board delta, board control, cv source, note)
# cv source: ("dirs", [(ctrl, cand), ...]) summed over the dir pairs | ("log", dict(decided=, strict=, all=, y7200=, trim=, k=))
# types: F features/capacity of the base, M mixture member, S ADS-B stage, T tail/TabPFN, W weights/scale, D seeds, R rule
P = [
 ("E46", "trust coefficient of submitted time", "W", -1.00, "v55 vs v30", ("log", dict(decided=-0.40)), "old metric; v55 also carries E36 stage"),
 ("E25", "ADS-B ground-track stage", "S", -15.2, "", ("log", dict(decided=-15.2)), "old metric (all rows)"),
 ("E30", "ADS-B stage E30", "S", -2.0, "", ("log", dict(decided=-1.4)), "old metric"),
 ("E36", "ADS-B stage + stand geometry", "S", -3.7, "v30", ("log", dict(decided=-3.0)), "old metric"),
 ("E28", "E28 features (on top of E25)", "F", -0.01, "", ("log", dict(decided=-1.39)), "old metric, same info as E25"),
 ("E32", "ADS-B features into the base model", "F", +0.33, "v50 vs v30", ("log", dict(decided=-1.41)), "old metric (leaky early stop)"),
 ("E38", "model per airport", "F", +3.79, "v43 vs v30", ("log", dict(decided=-1.82, all=-10.98)), "decided -1.82 / all -10.98 (journal sec.12)"),
 ("E50", "regime shares by flight number", "F", +5.78, "v65 vs v30", ("log", dict(decided=-2.79)), "old metric"),
 ("E64", "visit stage on air points of own aircraft", "S", -0.171, "v76 vs v55", ("dirs", [("e64_ctrl", "e64")]), "ctrl = stage ctrl"),
 ("E65", "visit features into stage E36", "S", -2.475, "v78 vs v76", ("dirs", [("e65_ctrl", "e65")]), ""),
 ("E70+E74", "stage E70 (E63/66/69 family) + 1kt extraction", "S", -1.024, "v82 vs v78", ("dirs", [("e70_ctrl", "e70"), ("e74_ctrl", "e74")]), "two CV runs summed"),
 ("E73", "MB3 blind (regressor on served rows)", "F", +0.010, "v88 vs v82", ("dirs", [("e73_ctrl", "e73")]), ""),
 ("E77", "features of submitted time (LOBT twins)", "F", -3.26, "v97 vs v82", ("dirs", [("e77_ctrl", "e77_cand")]), ""),
 ("E89", "classifier weight by error price (>3600)", "W", -1.18, "v115 vs v97", ("log", dict(decided=-1.02)), "journal: CV decided -1.02"),
 ("E90", "same, threshold 1800", "W", +0.29, "v125 vs v115", ("dirs", [("e89_cand", "e90_t1800")]), "ctrl = E89"),
 ("E99", "drop target encodings te_*", "F", -0.167, "v155 vs v115", ("dirs", [("e89_cand", "e99_note")]), "capacity removal"),
 ("E100", "MLP partner in the mix (v161)", "M", -0.255, "v161 vs v115", ("log", dict(decided=-0.657, k="6/6")), ""),
 ("E106", "airline prefix for rows without NM", "F", -1.28, "v195 vs v155", ("dirs", [("e99_note", "e106_air")]), "log: after E14 -0.58 (journal 24)"),
 ("E36lift", "lift at the lower LOBT window edge", "R", -0.008, "v227 vs v226", ("log", dict(decided=-0.139, k="6/6")), ""),
 ("E118", "airline prefix in LIRF tail TabPFN", "T", -3.60, "v316 vs v247", ("log", dict(decided=-0.27)), "LOMO SSE -5.0% (6/12) converted by journal 56 to ~-0.27; not CV-comparable"),
 ("E26a", "local-date shift in LIRF tail", "T", -0.46, "v326 vs v247", ("log", dict(decided=-0.30)), "LOMO -6.2% (5/12) ~-0.30"),
 ("E133", "schedule-habit rule, Rome rows with NM", "R", +0.36, "v249 vs v247", ("log", dict(decided=-0.72, k="5/6")), "LB-equivalent"),
 ("E141", "kinematic estimate of invisible start", "S", -0.040, "v404 vs v366", ("log", dict(decided=-0.137, trim=-0.144, k="6/6")), "after E14"),
 ("E142", "stage E70 x5 seeds", "D", -0.051, "v414 vs v404", ("log", dict(decided=-0.052, k="6/6")), "after E14"),
 ("E145", "PRU monthly calibration", "R", +0.114, "v448 vs v447", ("log", dict(decided=-0.045, k="6/6")), "CV 2025 only"),
 ("E147-70", "stage E70 without reception-density features", "F", -0.027, "v487 vs v467", ("log", dict(decided=+0.014, trim=+0.041, k="2/6")), "capacity removal; trim = stage vs cand_f"),
 ("E147-64", "stage E64 without reception-density features", "F", -0.035, "v497 vs v467", ("log", dict(decided=+0.01, k="2/6")), "capacity removal"),
 ("E150", "stage E70 on all rows (keep contradicting rows)", "S", -0.175, "v507 vs v467", ("log", dict(decided=-0.321, all=-0.559, y7200=+0.175, trim=-0.252, k="5/6")), "pre-E14 numbers; post-E14 decided -0.281"),
 ("E14-off", "remove E14 logit from the chain", "R", -0.227, "v588 vs v538", ("log", dict(decided=-0.376, k="3/6")), "CV: E14 vs production base +0.376, sign flipped"),
 ("E151", "flows in own taxi interval", "F", +0.041, "v548 vs v538", ("dirs", [("e116prod", "e151prod")]), ""),
 ("E163", "ATFM delay at destination (apt_dly)", "F", -0.174, "v778 vs v655", ("dirs", [("e116prod", "e163_hz")]), "OOF from hetzner vs mac control"),
 ("E171-3", "three anchor members raw/LOBT/SCHED", "M", -0.983, "v808 vs v655", ("dirs", [("e116prod", "e172_raw_lobt_sched")]), "board weights x2/3, CV weights x1"),
 ("E172-eobt", "4th member EOBT_1", "M", +0.037, "v829 vs v819", ("dirs", [("e172_raw_lobt_sched_eobt_ctrl", "e172_raw_lobt_sched_eobt")]), "board x2/3"),
 ("E173-aobt", "5th member AOBT_3", "M", +0.158, "v859 vs v839", ("dirs", [("e172_raw_lobt_sched_eobt_aobt_ctrl", "e172_raw_lobt_sched_eobt_aobt")]), "board: x1 weights, sum larger"),
 ("E176", "MLP member (6th)", "M", -0.775, "v899 vs v859", ("dirs", [("e172_raw_lobt_sched_eobt_aobt_mlp_ctrl", "e172_raw_lobt_sched_eobt_aobt_mlp")]), ""),
 ("FIN1b", "final table E163+E51+E157+E165 in base", "F", -0.218, "v959 vs v939", ("dirs", [("e163_hz", "fin1b_hz")]), "CatBoost still on E106 in v959"),
 ("E169", "stage E70 trained on production base", "S", -0.023, "v969 vs v939", ("log", dict(decided=-0.110, strict=-0.091, all=-0.201, y7200=-0.245, trim=-0.151, k="5/6")), ""),
 ("E159b-LSZH", "OPDI exit-parking stage, LSZH", "S", -0.131, "v729 vs v588", ("log", dict(decided=-1.2)), "journal 177 LOMO forecast (-38.6% RMSE on rows); not CV"),
]
NO_CV = [("E152", "R", -0.221, "no clean CV number (mean -0.012 per fold, 6/6)"),
         ("E159b-EDDF", "S", +1.132, "no labels at EDDF; OPDI offset guessed (outlier on the board)"),
         ("E103", "W", +1.77, "no CV recorded"), ("E105", "F", +1.45, "CV deliberately not run"),
         ("E110", "F", +0.40, "CV not recorded"), ("E112", "F", -1.06, "CV not recorded"), ("E116", "F", -0.20, "CV not recorded"),
         ("E117", "F", +0.62, "CV not recorded"), ("E119", "F", +1.17, "no CV"), ("E120", "F", +0.43, "no CV"),
         ("E121b", "D", -0.51, "variance calculation, no CV"), ("E144", "D", -0.083, "variance calculation"),
         ("E81", "M", -0.656, "mix by Gram identity, no CV gate"), ("E175", "F", -0.076, "no CV"), ("E169-E143..", "D", -0.02, "mixed")]

def lbscale(): return 6 * 2 * 252.43 * 344841

def compare_dirs(pairs):
    out = np.zeros(5); lower = n = 0
    for c, e in pairs:
        r = fm.compare(D + c, D + e, quiet=True)
        out += np.array([r[0], r[6], r[1], r[2], r[3]]); lower += r[4]; n += r[5]
    return out, lower, n

def bins(pairs, edges=(3600, 7200)):
    lb = lbscale(); res = np.zeros((2, 3)); cnt = np.zeros(3)
    for c, e in pairs:
        dec = fm.decided_map("data/features_e28.parquet")
        for p in sorted(glob.glob(f"{D}{c}/fold_*.npz")):
            q = p.replace(D + c, D + e)
            if not os.path.exists(q): continue
            a, b = np.load(p), np.load(q)
            order = {m: i for i, m in enumerate(b["mvt_id"])}
            idx = np.array([order[m] for m in a["mvt_id"]])
            d = (b["pred"][idx] - a["y"]) ** 2 - (a["pred"] - a["y"]) ** 2
            m = np.array([dec[k] for k in a["mvt_id"]])
            y = a["y"]
            for j, sel in enumerate([y <= edges[0], (y > edges[0]) & (y <= edges[1]), y > edges[1]]):
                res[0, j] += d[sel].sum() / lb; res[1, j] += d[sel & m].sum() / lb
                if c == pairs[0][0] and e == pairs[0][1]: cnt[j] += sel.sum()
    return res, cnt

def rank(x):
    o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x))
    for v in np.unique(x):
        s = x == v; r[s] = r[s].mean()
    return r

def stats(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); k = ~(np.isnan(x) | np.isnan(y)); x, y = x[k], y[k]
    if len(x) < 4: return None
    pe = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else NAN
    sp = np.corrcoef(rank(x), rank(y))[0, 1]
    nz = (x != 0) & (y != 0)
    sg = np.mean(np.sign(x[nz]) == np.sign(y[nz])) if nz.any() else NAN
    sl = (x * y).sum() / (x * x).sum()
    return len(x), pe, sp, sg, sl

def fmt(s):
    return "n/a" if s is None else f"n={s[0]:2d} pearson {s[1]:+.2f} spearman {s[2]:+.2f} sign {s[3]*100:3.0f}% slope {s[4]:+.2f}"

if __name__ == "__main__":
    rows = []
    for pid, ch, ty, bd, bc, (kind, src), note in P:
        if kind == "dirs":
            v, lw, n = compare_dirs(src); cv = dict(zip(SL, v)); cv["k"] = f"{n - lw}/{n}"
            cv["k"] = f"{lw}/{n}"
        else:
            cv = {s: src.get(s, NAN) for s in SL}; cv["k"] = src.get("k", "")
        rows.append(dict(id=pid, ch=ch, ty=ty, board=bd, bc=bc, cv=cv, note=note, kind=kind, src=src))
    print("== PAIRS (CV in LB-equivalent s, positive = worse; 'k' = folds where decided is lower)")
    print(f"{'id':11s} {'ty':2s} {'board':>7s} | " + " ".join(f"{s:>8s}" for s in SL) + "  k   src  note")
    for r in rows:
        print(f"{r['id']:11s} {r['ty']:2s} {r['board']:+7.3f} | " + " ".join(f"{r['cv'][s]:+8.3f}" for s in SL) + f"  {r['cv']['k']:>4s} {r['kind']:4s} {r['note']}")
    print("NO CV pairs:", NO_CV)
    big = {"E25", "E38", "E50", "E118", "E77", "E32"}
    recent = {"E141", "E142", "E145", "E147-70", "E147-64", "E150", "E14-off", "E151", "E163", "E171-3", "E172-eobt", "E173-aobt", "E176", "FIN1b", "E169", "E159b-LSZH", "E133", "E36lift"}
    def sel(f): return [r for r in rows if f(r)]
    groups = {"ALL": sel(lambda r: True), "no |board|>=3 outliers": sel(lambda r: abs(r["board"]) < 3),
              "ALL without E25": sel(lambda r: r["id"] != "E25"),
              "recent (>=26.09 era)": sel(lambda r: r["id"] in recent),
              "dirs-only (clean OOF)": sel(lambda r: r["kind"] == "dirs"),
              "dirs-only no outliers": sel(lambda r: r["kind"] == "dirs" and abs(r["board"]) < 3)}
    for g, rs in groups.items():
        print(f"\n== SLICE vs BOARD, group '{g}' ({len(rs)} pairs)")
        for s in SL:
            print(f"  {s:8s} {fmt(stats([r['cv'][s] for r in rs], [r['board'] for r in rs]))}")
    print("\n== BY TYPE (decided, plus ratio board/CV = sum board / sum CV, rows with CV)")
    for t in "FMSTWDR":
        rs = sel(lambda r: r["ty"] == t and not np.isnan(r["cv"]["decided"]))
        if not rs: continue
        x = np.array([r["cv"]["decided"] for r in rs]); y = np.array([r["board"] for r in rs])
        print(f"  {t} n={len(rs)} ids={[r['id'] for r in rs]}")
        print(f"     {fmt(stats(x, y))}  sum_board/sum_cv {y.sum()/x.sum():+.2f}  median(board/cv) {np.median(y/x):+.2f}")
        for s in SL[1:]:
            xs = np.array([r["cv"][s] for r in rs])
            if not np.isnan(xs).any(): print(f"     {s:8s} {fmt(stats(xs, y))}")
    # part 2
    print("\n== BINS (control y): LB-eq s of CV gain per bin: [all rows | decided rows], and #rows in fold union")
    brs = []
    for r in rows:
        if r["kind"] != "dirs": continue
        tot = np.zeros((2, 3)); cnt = None
        for pr in r["src"]:
            b, c = bins([pr]); tot += b; cnt = c if cnt is None else cnt + c
        brs.append((r, tot))
        a, d = tot
        print(f"{r['id']:11s} board {r['board']:+7.3f} | all: <=3600 {a[0]:+7.3f} 3600-7200 {a[1]:+7.3f} >7200 {a[2]:+7.3f} | decided: {d[0]:+7.3f} {d[1]:+7.3f} {d[2]:+7.3f}")
    for nm, ex in [("all pairs", set()), ("without E73/E65 (stage, tiny/odd)", {"E73"}), ("recent only", {"E64","E65","E70+E74","E73","E77","E90","E99","E106"})]:
        rs = [(r, t) for r, t in brs if r["id"] not in ex]
        print(f"\n-- bin vs board correlation, {nm} (n={len(rs)})")
        yb = [r["board"] for r, _ in rs]
        for lab, ri in [("all rows", 0), ("decided rows", 1)]:
            for j, bn in enumerate(["y<=3600", "3600<y<=7200", "y>7200"]):
                print(f"  {lab:12s} {bn:13s} {fmt(stats([t[ri, j] for _, t in rs], yb))}")
            print(f"  {lab:12s} {'body+mid<=7200':13s} {fmt(stats([t[ri, 0] + t[ri, 1] for _, t in rs], yb))}")
            print(f"  {lab:12s} {'sum':13s} {fmt(stats([t[ri].sum() for _, t in rs], yb))}")

    # robustness: leave-one-out of the bin/decided correlations on the recent pairs
    print("\n-- leave-one-out Pearson, recent n=7 (decided rows unless noted)")
    rec = [(r, t) for r, t in brs if r["id"] not in {"E64","E65","E70+E74","E73","E77","E90","E99","E106"}]
    for i in range(len(rec) + 1):
        rs = rec[:i] + rec[i + 1:] if i < len(rec) else rec
        nm = rec[i][0]["id"] if i < len(rec) else "(none dropped)"
        yb = [r["board"] for r, _ in rs]
        f = lambda v: stats(v, yb)[1] if stats(v, yb) else NAN
        print(f"  drop {nm:11s} body {f([t[1,0]+t[1,1] for _,t in rs]):+.2f}  y<=3600 {f([t[1,0] for _,t in rs]):+.2f}  dec-sum {f([t[1].sum() for _,t in rs]):+.2f}  all-rows-sum {f([t[0].sum() for _,t in rs]):+.2f}")

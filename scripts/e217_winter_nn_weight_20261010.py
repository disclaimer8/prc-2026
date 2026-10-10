"""E217: a winter NN-member weight as a fixed calendar rule, decided by a forward test on December 2025.

Registered in HYPOTHESES_LOG №449 before any outcome of this script was computed; verdict №450.

Rule under test (primary, variant A): NN-member weight w = w_winter in December, January, February and March,
w = 0.35 (current final, №425) in every other month; the other premix members keep their proportions exactly as
in E212 (`regular(w) = ((1-w)/0.75)*[(1-sum W)(.65 cat + .35 lgb) + sum_k W_k anchor_k] + w*NN`), i.e. the
anchors are scaled by (1-w)/0.75 and the tree pair keeps 1 - sum, as WEIGHTS['final'] does for 0.35.
Variant B (reported): winter = December, January, February only.

Data: no new chain runs. The E212 saved chain deltas (data/e212_chain_20261008/fold_*/chain_{m3,m10}.npz, int32
deltas vs the E203 `ordered` column at every grid weight 0.05..0.60) already hold the exact pre-E81 chain output
of every 2025 row at every candidate weight; every chain step (premix, upstream mixture, E185, E167 rules, E141/E64/
E188 stage predictions, E152 gate, feasibility flag) is row-local, so a per-month weight is evaluated exactly by
taking each month's rows from the run at that month's weight. Population = E212 `decided` rows (E203 evaluation
mask, minus rows whose feasibility changes anywhere on the grid for either member (0 in E212), minus LIRF no-NM
D >= 4 h). Primary member m10 (10-seed mean, the final's member); m3 reported for comparison only.

Decision procedure (registered):
  1. Selection: w_winter = argmin over {0.35, 0.40, 0.45, 0.50, 0.55} of the pooled decided SSE of January,
     February and March 2025 (variant B: January and February), m10; ties -> the smaller weight.
  2. Forward test, all three required for PASS:
     (i)   December 2025 (not used for the choice): decided SSE at w_winter < decided SSE at 0.35 (strict);
     (ii)  January 2025: decided SSE at w_winter <= decided SSE at 0.35;
     (iii) pooled 12-month decided SSE of the rule < pooled 12-month decided SSE of constant 0.35 (strict).
     If the selection returns 0.35 the rule equals the constant and (i) fails: FAIL (null).
  3. Verdict on variant A. Variant B is reported with the same three conditions; if A fails and B passes, B is
     NOT proposed automatically (choosing the variant after seeing the outcome is post hoc) - coordinator decides.
Informational only (not in the gate): pooled body (y <= 7200) and trim20 (-20/-20 per pair) of the rule vs 0.35,
per-month tables, and the expected effect on the four final months (January, February, June, July 2026), weighting
each month's 2025 decided delta-MSE by its 2026 DEP-row share in final_ranking.parquet.

Controls: m3 at w = 0.25 reproduces `ordered` on every row (E212 control); per-month delta RMSE vs 0.25 equals
analysis/e212_nn_weight_10seed_20261008/per_month.json (|diff| < 1e-9 s); decided rows = 2 084 920.

Output: analysis/e217_winter_nn_weight_20261010/result.json. Mac .venv, numpy/polars only, <= 4 threads, no fits,
no uploads, no edits to scripts/final_tail_assembly_20261008.py.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
for k, v in {'POLARS_MAX_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OMP_NUM_THREADS': '4',
             'POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR': 'load_as_storage'}.items():
    os.environ.setdefault(k, v)
import numpy as np
import polars as pl

PROTOCOL = 'E217-winter-nn-weight-calendar-rule-dec-forward-v1'
ID, Y = 'MVT_ID_mvt', 'TAXITIME_SEC_mvt'
CHAIN = 'data/e203_pre_e81_20261007/predictions.parquet'
DELTAS = 'data/e212_chain_20261008'
PER_MONTH = 'analysis/e212_nn_weight_10seed_20261008/per_month.json'
FINAL = 'data/prc-2026-datasets/final_ranking.parquet'
OUT = 'analysis/e217_winter_nn_weight_20261010/result.json'
GRID = tuple(k / 20 for k in range(1, 13))  # identical to E212
W25, BASE = GRID.index(0.25), GRID.index(0.35)
CANDIDATES = (0.35, 0.40, 0.45, 0.50, 0.55)
CJ = [GRID.index(w) for w in CANDIDATES]
VARIANTS = {'A_dec_jan_feb_mar': dict(winter=(12, 1, 2, 3), select=(1, 2, 3)),
            'B_dec_jan_feb': dict(winter=(12, 1, 2), select=(1, 2))}
PRIMARY, MEMBER, MEMBERS = 'A_dec_jan_feb_mar', 'm10', ('m3', 'm10')
FINAL_MONTHS = (1, 2, 6, 7)
FOLDS = (1, 2, 3, 4, 5, 6)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def load(root):
    chain_df = pl.read_parquet(root / CHAIN).sort(ID)
    if chain_df.height != 2085047: raise ValueError('chain population')
    F = []
    for fold in FOLDS:
        name = f'fold_{fold}_{fold + 6}'
        old = chain_df.filter(pl.col('month').is_in([fold, fold + 6]))
        ids = old[ID].to_numpy(); y = old[Y].to_numpy().astype(np.float64); ordered = old['ordered'].to_numpy().astype(np.float64)
        ok = old['evaluation_mask'].to_numpy(); airport = old['ADEP_mvt'].to_numpy(); nm = old['has_nm'].to_numpy(); D = old['D'].to_numpy()
        month = old['month'].to_numpy().astype(int)
        se = {}; unstable_any = np.zeros(len(ids), bool)
        for m in MEMBERS:
            with np.load(root / DELTAS / name / f'chain_{m}.npz') as z:
                if not np.array_equal(z['mvt_id'], ids) or not np.array_equal(z['grid'], np.array(GRID)) or not np.array_equal(z['excluded_e203'], ~ok):
                    raise ValueError('chain file alignment ' + name)
                pred = ordered[None, :] + z['delta'].astype(np.float64); unstable_any |= z['unstable'].any(axis=0)
            if m == 'm3' and np.any(pred[W25] != ordered): raise ValueError('CONTROL FAILED: m3 at 0.25 != ordered ' + name)
            se[m] = (pred - y[None, :]) ** 2
            del pred
        common = ok & ~unstable_any
        decided = common & ~((airport == 'LIRF') & ~nm & (D >= 14400)); body = common & (y <= 7200)
        F.append(dict(fold=fold, name=name, month=month, se=se, decided=decided, body=body, extra_unstable=int((ok & unstable_any).sum())))
    return F


def month_sse(F):
    """sse[m][mask][month] -> array over GRID; n[mask][month]."""
    sse = {m: {k: {} for k in ('decided', 'body')} for m in MEMBERS}; n = {k: {} for k in ('decided', 'body')}
    for X in F:
        for mo in (X['fold'], X['fold'] + 6):
            for k in ('decided', 'body'):
                mask = X[k] & (X['month'] == mo); n[k][mo] = int(mask.sum())
                for m in MEMBERS: sse[m][k][mo] = X['se'][m][:, mask].sum(axis=1)
    return sse, n


def rmse(s, n): return float(np.sqrt(s / n))


def evaluate_variant(F, sse, n, m, winter, select):
    tr = np.array([sum(sse[m]['decided'][mo][j] for mo in select) for j in CJ])
    jw = int(np.argmin(tr)); w = CANDIDATES[jw]; j = GRID.index(w)
    wmonth = {mo: (j if mo in winter else BASE) for mo in range(1, 13)}
    dec = lambda mo, jj: float(sse[m]['decided'][mo][jj])
    rule = sum(dec(mo, wmonth[mo]) for mo in range(1, 13)); const = sum(dec(mo, BASE) for mo in range(1, 13)); N = sum(n['decided'].values())
    body_rule = sum(float(sse[m]['body'][mo][wmonth[mo]]) for mo in range(1, 13)); body_const = sum(float(sse[m]['body'][mo][BASE]) for mo in range(1, 13))
    trim20 = 0.0
    for X in F:
        jr = np.where(np.isin(X['month'], winter), j, BASE)
        d = X['se'][m][jr, np.arange(len(jr))] - X['se'][m][BASE]
        s = np.sort(d[X['decided']]); trim20 += float(s[20:-20].sum())
    cond = dict(dec_lower=dec(12, j) < dec(12, BASE), jan_not_worse=dec(1, j) <= dec(1, BASE), pooled12_lower=rule < const)
    return dict(
        selection=dict(months=list(select), candidates=list(CANDIDATES), pooled_sse=dict(zip([f'{c:.2f}' for c in CANDIDATES], tr.tolist())),
                       pooled_delta_sse_vs_035=dict(zip([f'{c:.2f}' for c in CANDIDATES], (tr - tr[0]).tolist())), w_winter=w),
        winter_months=list(winter),
        december=dict(n=n['decided'][12], sse_w=dec(12, j), sse_035=dec(12, BASE), delta_sse=dec(12, j) - dec(12, BASE),
                      delta_rmse=rmse(dec(12, j), n['decided'][12]) - rmse(dec(12, BASE), n['decided'][12])),
        january=dict(n=n['decided'][1], sse_w=dec(1, j), sse_035=dec(1, BASE), delta_sse=dec(1, j) - dec(1, BASE),
                     delta_rmse=rmse(dec(1, j), n['decided'][1]) - rmse(dec(1, BASE), n['decided'][1])),
        pooled12=dict(n=N, sse_rule=rule, sse_035=const, delta_sse=rule - const, delta_rmse=rmse(rule, N) - rmse(const, N)),
        conditions=cond, passed=bool(w != 0.35 and all(cond.values())),
        informational=dict(body_delta_sse=body_rule - body_const, trim20_delta_sse=trim20,
                           per_month_delta_sse={mo: dec(mo, wmonth[mo]) - dec(mo, BASE) for mo in range(1, 13)}))


def final_shares(root):
    f = pl.read_parquet(root / FINAL, columns=['MVT_ID_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt']).filter(pl.col('PHASE_mvt') == 'DEP')
    c = f.group_by(pl.col('MVT_TIME_UTC_mvt').dt.month().alias('mo')).len().sort('mo')
    counts = {int(a): int(b) for a, b in zip(c['mo'], c['len'])}
    if sorted(counts) != list(FINAL_MONTHS): raise ValueError(f'final months {sorted(counts)}')
    return counts


def expected_final(sse, n, m, counts, wmonth):
    N = sum(counts.values()); share = {mo: counts[mo] / N for mo in FINAL_MONTHS}
    base = sum(share[mo] * sse[m]['decided'][mo][BASE] / n['decided'][mo] for mo in FINAL_MONTHS)
    dmse = sum(share[mo] * (sse[m]['decided'][mo][wmonth[mo]] - sse[m]['decided'][mo][BASE]) / n['decided'][mo] for mo in FINAL_MONTHS)
    return dict(share=share, base_mse_2025=float(base), delta_mse=float(dmse), delta_rmse_at_2025_level=float(np.sqrt(base + dmse) - np.sqrt(base)),
                delta_rmse_at_lb_231=float(dmse / (2 * 231.18)),
                per_month_delta_mse={mo: float((sse[m]['decided'][mo][wmonth[mo]] - sse[m]['decided'][mo][BASE]) / n['decided'][mo]) for mo in FINAL_MONTHS})


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path.cwd()); a = p.parse_args(); root = a.root.resolve()
    F = load(root); sse, n = month_sse(F)
    # controls
    if sum(n['decided'].values()) != 2084920: raise ValueError('decided population differs from E212')
    if any(X['extra_unstable'] for X in F): raise ValueError('feasibility population differs from E212')
    ref = json.loads((root / PER_MONTH).read_text()); worst = 0.0
    for m in MEMBERS:
        for mo in range(1, 13):
            r = ref[m][str(mo)]
            if r['n'] != n['decided'][mo]: raise ValueError(f'per-month n {m} {mo}')
            for j, w in enumerate(GRID):
                d = rmse(sse[m]['decided'][mo][j], n['decided'][mo]) - rmse(sse[m]['decided'][mo][W25], n['decided'][mo])
                worst = max(worst, abs(d - r['d_rmse'][f'{w:.2f}']))
    if worst > 1e-9: raise ValueError(f'per-month control vs E212 per_month.json: {worst}')
    N = sum(n['decided'].values()); pooled035 = rmse(sum(float(sse[MEMBER]['decided'][mo][BASE]) for mo in range(1, 13)), N)
    print('CONTROL OK: m3@0.25 == ordered on all rows; decided', N, '; per-month dRMSE = E212 (max |diff| %.1e s); m10@0.35 pooled decided RMSE %.3f s' % (worst, pooled035), flush=True)
    counts = final_shares(root)
    res = dict(protocol=PROTOCOL, registered='HYPOTHESES_LOG №449', primary_variant=PRIMARY, primary_member=MEMBER, candidates=list(CANDIDATES),
               inputs=dict(chain_sha256=sha(root / CHAIN), per_month_sha256=sha(root / PER_MONTH),
                           deltas_sha256={f'{X["name"]}/chain_{m}.npz': sha(root / DELTAS / X['name'] / f'chain_{m}.npz') for X in F for m in MEMBERS},
                           self_sha256=sha(Path(__file__).resolve())),
               controls=dict(m3_025_equals_ordered=True, decided_rows=N, per_month_max_abs_diff_vs_e212=worst, m10_035_pooled_decided_rmse=pooled035),
               n_decided=n['decided'], final_dep_rows_2026=counts,
               per_month={m: {mo: dict(n=n['decided'][mo],
                                      delta_sse_vs_035={f'{w:.2f}': float(sse[m]['decided'][mo][GRID.index(w)] - sse[m]['decided'][mo][BASE]) for w in GRID},
                                      delta_rmse_vs_035={f'{w:.2f}': rmse(sse[m]['decided'][mo][GRID.index(w)], n['decided'][mo]) - rmse(sse[m]['decided'][mo][BASE], n['decided'][mo]) for w in GRID})
                              for mo in range(1, 13)} for m in MEMBERS})
    res['variants'] = {}
    for m in MEMBERS:
        for v, spec in VARIANTS.items():
            r = evaluate_variant(F, sse, n, m, spec['winter'], spec['select'])
            j = GRID.index(r['selection']['w_winter']); wmonth = {mo: (j if mo in spec['winter'] else BASE) for mo in range(1, 13)}
            r['expected_final_2026'] = expected_final(sse, n, m, counts, wmonth)
            res['variants'][f'{m}/{v}'] = r
            print(m, v, 'w_winter', r['selection']['w_winter'], 'Dec dRMSE %.4f' % r['december']['delta_rmse'], 'Jan dRMSE %.4f' % r['january']['delta_rmse'],
                  'pooled12 dRMSE %.4f' % r['pooled12']['delta_rmse'], r['conditions'], 'PASS' if r['passed'] else 'FAIL', flush=True)
    # informational: expected final effect of every candidate if applied to the winter months that the final contains (Jan, Feb)
    res['expected_final_by_candidate'] = {m: {f'{w:.2f}': expected_final(sse, n, m, counts, {mo: (GRID.index(w) if mo in (1, 2) else BASE) for mo in range(1, 13)})
                                              for w in CANDIDATES} for m in MEMBERS}
    prim = res['variants'][f'{MEMBER}/{PRIMARY}']
    res['verdict'] = dict(variant=PRIMARY, member=MEMBER, w_winter=prim['selection']['w_winter'], passed=prim['passed'], conditions=prim['conditions'],
                          variant_B_passed=res['variants'][f'{MEMBER}/B_dec_jan_feb']['passed'])
    out = root / OUT; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    print('VERDICT', res['verdict'], flush=True); print('WROTE', out, flush=True)


if __name__ == '__main__':
    main()

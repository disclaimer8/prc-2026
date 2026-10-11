"""Label-free forecast for v12083 = v12082 with the E218 shift removed on LIRF rows (journal 460).

e = v12083 - v12082 = -d2 on the LIRF Jan/Jul rows of the E218 stage (d2 = v12082 - v12080: +9 Jan, +11 Jul), 0 elsewhere.
dSSE(83 vs 82) = -sum_L d2^2 - 2 sum_L d2 (v80 - v77) - 2 T,  T = sum_L d2 (v77 - y)  (the only label-dependent term).
The first two terms are exact. The boards fixed two projections of the v12077 residual:
  C1 = sum d1 (v77 - y) = (N (s81^2 - s77^2) - sum d1^2) / 2              (v12081 = v12077 + d1)
  C2 = sum d2 (v77 - y) = (N (s82^2 - s80^2) - sum d2^2) / 2 - sum d2 (v80 - v77)   (v12082 = v12080 + d2)
T = C2 - (non-LIRF part of C2). Readings: (a) one beta for every cell (cell residual = beta * x77, x77 = indicator gap
minus emu_hat of v12077) fitted to v12082 [and to v12081]; (b) non-LIRF cells at the 2025 beta-hat 0.932, LIRF = what is left
of C2 ("LIRF about neutral"); (c) both boards with non-LIRF at beta-hat solve the two LIRF cells separately (T as in b).
Sweep: non-LIRF beta 0.40 ... 1.10 with C2 fixed. Mac .venv, read-only, no labels.
"""
import hashlib, json, os, sys
import numpy as np, polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ID, Y = 'MVT_ID_mvt', 'TAXITIME_SEC_mvt'
FILES = {'v12077': 'submissions/gentle-igloo_v12077.parquet', 'v12081': 'submissions/gentle-igloo_v12081.parquet',
         'v12080': 'submissions/gentle-igloo_v12080.parquet', 'v12082': 'submissions/gentle-igloo_v12082.parquet',
         'ranking': 'data/prc-2026-datasets/ranking.parquet', 'cells77': 'analysis/v12081_upload_20261011/cells_v12077_e218.csv',
         'cells_final': 'analysis/final_tail_20261011/pru_level_cells.csv', 'rows_final': 'analysis/final_tail_20261011/rows_changed_by_cell.csv'}
PINS = {'v12077': 'f06c66bf', 'v12081': '71962226', 'v12080': 'ae1680cb', 'v12082': 'e87b7978', 'ranking': '9db75e3a',
        'cells77': 'b7426e99', 'cells_final': 'f1d8cb44'}
S77, S81, S80, S82 = 230.4812, 230.4478, 231.1805, 231.1445
BETA_HAT, BETA_SE = 0.9323329891592096, 0.0672   # evaluate.json №452 (LOPO over airports)
TOL = 0.003                                       # decision rule tolerance (coordinator, №460)
N_FINAL = 670790


def sha(p):
    return hashlib.sha256(open(os.path.join(ROOT, p), 'rb').read()).hexdigest()


def score(dsse, n, s):
    return float(np.sqrt(s ** 2 + dsse / n))


def main():
    shas = {k: sha(v) for k, v in FILES.items()}
    for k, p in PINS.items():
        if not shas[k].startswith(p):
            sys.exit(f'pin mismatch {k}')
    rd = lambda k, name: pl.read_parquet(os.path.join(ROOT, FILES[k])).select(ID, pl.col(Y).cast(pl.Float64).alias(name))
    rk = pl.read_parquet(os.path.join(ROOT, FILES['ranking']), columns=[ID, 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt']).filter(
        pl.col('PHASE_mvt') == 'DEP').select(ID, 'ADEP_mvt', month=pl.col('MVT_TIME_UTC_mvt').dt.month().cast(pl.Int64))
    d = rd('v12080', 'v80').join(rd('v12077', 'v77'), on=ID).join(rd('v12081', 'v81'), on=ID).join(rd('v12082', 'v82'), on=ID).join(rk, on=ID, how='left')
    N = d.height
    assert N == 344841 and d['ADEP_mvt'].null_count() == 0
    d = d.with_columns(d1=pl.col('v81') - pl.col('v77'), d2=pl.col('v82') - pl.col('v80'), L=pl.col('ADEP_mvt') == 'LIRF')
    d = d.with_columns(v83=pl.when(pl.col('L')).then(pl.col('v80')).otherwise(pl.col('v82')))
    d = d.with_columns(e=pl.col('v83') - pl.col('v82'))
    cells = d.group_by('ADEP_mvt', 'month').agg(
        n1=(pl.col('d1') != 0).sum(), d1=pl.col('d1').filter(pl.col('d1') != 0).first(), k1=pl.col('d1').filter(pl.col('d1') != 0).n_unique(),
        n2=(pl.col('d2') != 0).sum(), d2=pl.col('d2').filter(pl.col('d2') != 0).first(), k2=pl.col('d2').filter(pl.col('d2') != 0).n_unique(),
        same_rows=((pl.col('d1') != 0) == (pl.col('d2') != 0)).all(),
        b_term=(pl.col('d2') * (pl.col('v80') - pl.col('v77'))).sum()).with_columns(pl.col('d1', 'd2').fill_null(0.0)).sort('ADEP_mvt', 'month')
    assert cells['k1'].max() <= 1 and cells['k2'].max() <= 1
    x77 = pl.read_csv(os.path.join(ROOT, FILES['cells77'])).select('ADEP_mvt', pl.col('month').cast(pl.Int64), 'x_bias')
    cells = cells.join(x77, on=['ADEP_mvt', 'month'], how='left')
    assert cells['x_bias'].null_count() == 0
    L = (cells['ADEP_mvt'] == 'LIRF').to_numpy()
    n1, n2, d1c, d2c, x = (cells[c].to_numpy() for c in ('n1', 'n2', 'd1', 'd2', 'x_bias'))
    W1, W2 = n1 * d1c * x, n2 * d2c * x
    sum_d1sq, sum_d2sq = float((d['d1'] ** 2).sum()), float((d['d2'] ** 2).sum())
    B = float(cells['b_term'].sum())
    C1 = (N * (S81 ** 2 - S77 ** 2) - sum_d1sq) / 2
    D2 = (N * (S82 ** 2 - S80 ** 2) - sum_d2sq) / 2          # sum d2 (v80 - y)
    C2 = D2 - B
    A_L = float((d['e'] ** 2).sum())
    B_L = float(cells.filter(pl.col('ADEP_mvt') == 'LIRF')['b_term'].sum())
    assert A_L == float((d.filter('L')['d2'] ** 2).sum()) and int((d['e'] != 0).sum()) == 25017
    W2L, W2N = float(W2[L].sum()), float(W2[~L].sum())

    def outcome(T, label):
        dsse = -A_L - 2 * B_L - 2 * T
        s = score(dsse, N, S82)
        return dict(reading=label, T=T, dsse=dsse, dmse=dsse / N, score=s, delta=s - S82,
                    rule='rebuild final without LIRF cells' if round(s, 4) <= round(S82 + TOL, 4) else 'final stays bbae2cc5',
                    lambda_LIRF_v77=T / W2L)
    beta_a2, beta_a1 = C2 / W2.sum(), C1 / W1.sum()
    lam1 = (C1 - BETA_HAT * W1[~L].sum()) / W1[L].sum()
    readings = {
        'a_uniform_beta_fit_v12082': outcome(beta_a2 * W2L, f'one beta {beta_a2:.3f} for all cells, fitted to v12082'),
        'a1_uniform_beta_fit_v12081': outcome(beta_a1 * W2L, f'one beta {beta_a1:.3f} for all cells, fitted to v12081'),
        'b_nonLIRF_beta_hat_LIRF_from_v12082': outcome(C2 - BETA_HAT * W2N, 'non-LIRF at 2025 beta-hat 0.932, LIRF = rest of C2 (LIRF ~ neutral)'),
        'b1_nonLIRF_beta_hat_LIRF_from_v12081': outcome(lam1 * W2L, f'non-LIRF at beta-hat, LIRF lambda {lam1:.3f} fitted to v12081'),
        'beta_hat_everywhere_2025': outcome(BETA_HAT * W2L, 'beta-hat 0.932 in LIRF too (2025 relation, contradicted by both boards)'),
        'lirf_cross_zero': outcome(0.0, 'LIRF residual of v12077 zero on the moved rows (T = 0)'),
    }
    # (c) both boards, non-LIRF at beta-hat: the two LIRF cells separately
    iL = np.where(L & ((n1 > 0) | (n2 > 0)))[0]
    M = np.array([[n1[i] * d1c[i] for i in iL], [n2[i] * d2c[i] for i in iL]])
    rhs = np.array([C1 - BETA_HAT * W1[~L].sum(), C2 - BETA_HAT * W2N])
    r_L = np.linalg.solve(M, rhs)
    joint = {f"LIRF-{cells['month'][int(i)]}": float(r) for i, r in zip(iL, r_L)}
    # break-even and rule threshold in T, and the sweep over the non-LIRF beta with C2 fixed
    T_zero = -(A_L + 2 * B_L) / 2
    T_rule = -(A_L + 2 * B_L + ((S82 + TOL) ** 2 - S82 ** 2) * N) / 2
    sweep = {f'{b:.2f}': outcome(C2 - b * W2N, f'non-LIRF beta {b:.2f}, LIRF = rest of C2') for b in np.round(np.arange(0.40, 1.101, 0.05), 2)}
    s_lo, s_hi = (outcome(C2 - b * W2N, '')['score'] for b in (1.10, 0.40))
    # informational: the same LIRF question in the blind months of the final (F-parametrisation: residual = beta_L * x_F, x_F = -2 c)
    cf = pl.read_csv(os.path.join(ROOT, FILES['cells_final'])).filter(pl.col('ADEP_mvt') == 'LIRF')
    rf = pl.read_csv(os.path.join(ROOT, FILES['rows_final'])).filter(pl.col('ADEP_mvt') == 'LIRF')
    lf = cf.join(rf.select('month', 'changed', 'shift_mode'), on='month')
    ndc = {int(m): float(n * s * c) for m, n, s, c in lf.select('month', 'changed', 'shift_mode', 'c').iter_rows()}
    nd2 = {int(m): float(n * s * s) for m, n, s in lf.select('month', 'changed', 'shift_mode').iter_rows()}
    beta_F = lambda T: -(B_L + T) / (2 * (ndc[1] + ndc[7]))      # Jan/Jul: sum_L d2 (v80 - y) = -2 beta_F sum n d c
    def blind(bF):   # dSSE of removing the LIRF cells on Feb/Jun of the final; seconds on a 231 s level over 670 790 rows
        dsse = -(nd2[2] + nd2[6]) + 4 * bF * (ndc[2] + ndc[6])
        return dict(beta_F=bF, dsse_febjun=dsse, delta_final_s=dsse / (2 * N_FINAL * 231.0))
    for k, v in list(readings.items()) + list(sweep.items()):
        v['beta_F_LIRF'] = beta_F(v['T'])
        v['blind_febjun_remove_LIRF'] = blind(v['beta_F_LIRF'])
    a, b = readings['a_uniform_beta_fit_v12082']['score'], readings['b_nonLIRF_beta_hat_LIRF_from_v12082']['score']
    point = (a + b) / 2
    res = dict(inputs=shas, N=N, scores=dict(v12077=S77, v12081=S81, v12080=S80, v12082=S82),
               label_free=dict(changed=int((d['e'] != 0).sum()), sum_e_sq=A_L, rms_e=float(np.sqrt(A_L / N)), B_L_sum_d2_v80_minus_v77=B_L,
                               B_all=B, W2_LIRF=W2L, W2_nonLIRF=W2N, sum_d1_sq=sum_d1sq, sum_d2_sq=sum_d2sq),
               measured=dict(C1_sum_d1_resid77=C1, C2_sum_d2_resid77=C2, D2_sum_d2_resid80=D2, beta_uniform_v12081=beta_a1,
                             beta_uniform_v12082=beta_a2, lambda_LIRF_v12081=lam1),
               joint_two_boards_nonLIRF_beta_hat=dict(mean_resid_v77_on_moved_rows=joint,
                                                      note='ill-conditioned (d1 and d2 almost collinear); T itself does not depend on the split'),
               thresholds=dict(T_break_even=T_zero, lambda_break_even=T_zero / W2L, T_rule=T_rule, lambda_rule=T_rule / W2L,
                               rule_score=round(S82 + TOL, 4)),
               readings=readings, sweep_nonLIRF_beta=sweep,
               forecast=dict(point=round(point, 4), corridor=[round(min(s_lo, s_hi) - 0.001, 3), round(max(s_lo, s_hi) + 0.001, 3)],
                             corridor_basis='non-LIRF beta 0.40 ... 1.10 with the v12082 projection fixed, +-0.001 s',
                             rule=f'v12083 <= {S82 + TOL:.4f} (v12082 + 0.003) => final rebuilt with E218 excluding LIRF cells; otherwise final stays bbae2cc5'),
               cells=cells.drop('k1', 'k2').to_dicts())
    with open(os.path.join(ROOT, 'analysis/v12083_upload_20261011/forecast.json'), 'w') as f:
        json.dump(res, f, indent=1)
    pl.Config.set_tbl_rows(30); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(200)
    print(cells.drop('k1', 'k2'))
    print(json.dumps({k: v for k, v in res.items() if k not in ('cells', 'inputs', 'sweep_nonLIRF_beta')}, indent=1))
    for k, v in sweep.items():
        print(k, round(v['score'], 4), round(v['delta'], 4), round(v['lambda_LIRF_v77'], 3), round(v['beta_F_LIRF'], 3), round(v['blind_febjun_remove_LIRF']['delta_final_s'], 4), v['rule'])


if __name__ == '__main__':
    main()

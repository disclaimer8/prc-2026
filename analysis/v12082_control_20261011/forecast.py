"""Label-free forecast for the control v12082 (Jan/Jul rows of the E218 final, journal 458).

v12082 - v12080 is exactly the E218 shift d2 of the new final on Jan/Jul rows (constant per airport x month cell).
dSSE = sum d2^2 + 2 sum d2 (F_old - v77) + 2 sum d2 (v77 - y): the first two terms are label-free; the third needs the
per-cell residual of v12077, of which the board measured one projection through v12081 = v12077 + d1:
sum d1 (v77 - y) = (N (s81^2 - s77^2) - sum d1^2) / 2. Cell residuals are modelled as b = beta * x77
(x77 = indicator gap minus emu_hat of v12077, analysis/v12081_upload_20261011/cells_v12077_e218.csv) and fitted to that
projection two ways: (a) one beta for all cells; (b) non-LIRF cells at the 2025 beta 0.932, LIRF free.
Mac .venv, read-only, no labels.
"""
import hashlib, json, os, sys
import numpy as np, polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ID, Y = 'MVT_ID_mvt', 'TAXITIME_SEC_mvt'
FILES = {'v12077': 'submissions/gentle-igloo_v12077.parquet', 'v12081': 'submissions/gentle-igloo_v12081.parquet',
         'v12080': 'submissions/gentle-igloo_v12080.parquet', 'final_e218': 'data/final_tail_20261011/work/gentle-igloo_final.parquet',
         'ranking': 'data/prc-2026-datasets/ranking.parquet', 'cells77': 'analysis/v12081_upload_20261011/cells_v12077_e218.csv'}
PINS = {'v12077': 'f06c66bf', 'v12081': '71962226', 'v12080': 'ae1680cb', 'final_e218': 'bbae2cc5', 'ranking': '9db75e3a'}
S77, S81, S80 = 230.4812, 230.4478, 231.1805
BETA_HAT = 0.9323329891592096   # evaluate.json №452
CORRIDOR = 0.15                 # coordinator minimum (lesson of №446)


def sha(p):
    return hashlib.sha256(open(os.path.join(ROOT, p), 'rb').read()).hexdigest()


def main():
    shas = {k: sha(v) for k, v in FILES.items()}
    for k, p in PINS.items():
        if not shas[k].startswith(p):
            sys.exit(f'pin mismatch {k}')
    rd = lambda k, name: pl.read_parquet(os.path.join(ROOT, FILES[k])).select(ID, pl.col(Y).cast(pl.Float64).alias(name))
    rk = pl.read_parquet(os.path.join(ROOT, FILES['ranking']), columns=[ID, 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt']).filter(
        pl.col('PHASE_mvt') == 'DEP').select(ID, 'ADEP_mvt', month=pl.col('MVT_TIME_UTC_mvt').dt.month().cast(pl.Int64))
    d = rd('v12080', 'f_old').join(rd('v12077', 'v77'), on=ID).join(rd('v12081', 'v81'), on=ID).join(rk, on=ID, how='left')
    d = d.join(rd('final_e218', 'f_new'), on=ID, how='left')
    N = d.height
    assert N == 344841 and d['ADEP_mvt'].null_count() == 0 and d['f_new'].null_count() == 0
    d = d.with_columns(d1=pl.col('v81') - pl.col('v77'), d2=pl.col('f_new') - pl.col('f_old'))
    # v12082 = Jan/Jul rows of the new final = v12080 + d2 exactly
    cells = d.group_by('ADEP_mvt', 'month').agg(
        n1=(pl.col('d1') != 0).sum(), d1=pl.col('d1').filter(pl.col('d1') != 0).first(), d1_distinct=pl.col('d1').filter(pl.col('d1') != 0).n_unique(),
        n2=(pl.col('d2') != 0).sum(), d2=pl.col('d2').filter(pl.col('d2') != 0).first(), d2_distinct=pl.col('d2').filter(pl.col('d2') != 0).n_unique(),
        b_term=(pl.col('d2') * (pl.col('f_old') - pl.col('v77'))).sum(),
        mean_fold_minus_v77_on_moved=(pl.col('f_old') - pl.col('v77')).filter(pl.col('d2') != 0).mean()).with_columns(
        pl.col('d1', 'd2').fill_null(0.0)).sort('ADEP_mvt', 'month')
    assert cells['d1_distinct'].max() <= 1 and cells['d2_distinct'].max() <= 1   # one integer shift per cell
    x77 = pl.read_csv(os.path.join(ROOT, FILES['cells77'])).select('ADEP_mvt', pl.col('month').cast(pl.Int64), 'x_bias')
    cells = cells.join(x77, on=['ADEP_mvt', 'month'], how='left')
    assert cells['x_bias'].null_count() == 0
    A = float((d['d2'] ** 2).sum())
    B = float(cells['b_term'].sum())
    C1 = (N * (S81 ** 2 - S77 ** 2) - float((d['d1'] ** 2).sum())) / 2      # sum d1 (v77 - y), measured
    w1 = (cells['d1'] * cells['n1'] * cells['x_bias']).to_numpy()
    w2 = (cells['d2'] * cells['n2'] * cells['x_bias']).to_numpy()
    lirf = (cells['ADEP_mvt'] == 'LIRF').to_numpy()
    beta_a = C1 / w1.sum()
    lam = (C1 - BETA_HAT * w1[~lirf].sum()) / w1[lirf].sum()
    T = {'a_uniform_beta': beta_a * w2.sum(),
         'b_nonLIRF_2025beta_LIRF_fitted': BETA_HAT * w2[~lirf].sum() + lam * w2[lirf].sum(),
         'c_beta_hat_all_cells_(2025, already contradicted by v12081)': BETA_HAT * w2.sum(),
         'd_beta_0_gap_is_noise': 0.0}
    out = {}
    for k, t in T.items():
        dsse = A + 2 * B + 2 * t
        s = float(np.sqrt(S80 ** 2 + dsse / N))
        out[k] = dict(dsse=dsse, dmse=dsse / N, score=s, delta=s - S80)
    naive = S80 - (S77 - S81)
    point = float(np.mean([out['a_uniform_beta']['score'], out['b_nonLIRF_2025beta_LIRF_fitted']['score']]))
    res = dict(inputs=shas, N=N, scores=dict(v12077=S77, v12081=S81, v12080=S80),
               label_free=dict(sum_d2_sq=A, rms_d2=float(np.sqrt(A / N)), changed=int((d['d2'] != 0).sum()), mean_d2=float(d['d2'].mean()),
                               cross_d2_fold_minus_v77=B, sum_d1_sq=float((d['d1'] ** 2).sum())),
               measured_projection_sum_d1_resid77=C1, beta_uniform=beta_a, lambda_LIRF=lam, beta_hat_2025=BETA_HAT,
               scenarios=out, naive_transfer=dict(score=naive, delta=-(S77 - S81)),
               forecast=dict(point=round(point, 3), corridor=[round(point - CORRIDOR, 3), round(point + CORRIDOR, 3)],
                             rule='v12082 > 231.1805 (worse than v12080) => final stays 76c32426; otherwise the E218 file bbae2cc5 becomes the final'),
               cells=cells.to_dicts())
    path = os.path.join(ROOT, 'analysis/v12082_control_20261011/forecast.json')
    with open(path, 'w') as f:
        json.dump(res, f, indent=1)
    pl.Config.set_tbl_rows(30); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(200)
    print(cells.select('ADEP_mvt', 'month', 'n1', 'd1', 'n2', 'd2', 'x_bias', 'mean_fold_minus_v77_on_moved').with_columns(pl.col(pl.Float64).round(3)))
    print(json.dumps({k: v for k, v in res.items() if k not in ('cells', 'inputs')}, indent=1))


if __name__ == '__main__':
    main()

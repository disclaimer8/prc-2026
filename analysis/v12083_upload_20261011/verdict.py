"""Verdict arithmetic for v12083 (journal 461): exact split of the E218 Jan/Jul gain into LIRF and non-LIRF cells.

v12080 -> v12083 adds only the non-LIRF E218 cells; v12083 -> v12082 adds only the LIRF cells (same labels, same rows), so
the two board differences are the exact contributions (to the board's 4-decimal rounding). The label term T of the forecast is
now measured: T = -(N (s83^2 - s82^2) + sum e^2 + 2 B_L) / 2. Mac .venv, read-only, no labels.
"""
import json, os
import numpy as np, polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
F = json.load(open(os.path.join(ROOT, 'analysis/v12083_upload_20261011/forecast.json')))
S83 = json.load(open(os.path.join(ROOT, 'analysis/v12083_upload_20261011/score_response.json')))['score']
N, sc, lf, me = F['N'], F['scores'], F['label_free'], F['measured']
S80, S81, S82, S77 = sc['v12080'], sc['v12081'], sc['v12082'], sc['v12077']
A_L, B_L, W2L, W2N = lf['sum_e_sq'], lf['B_L_sum_d2_v80_minus_v77'], lf['W2_LIRF'], lf['W2_nonLIRF']
C1, C2 = me['C1_sum_d1_resid77'], me['C2_sum_d2_resid77']
cells = pl.DataFrame(F['cells'])
L = (cells['ADEP_mvt'] == 'LIRF').to_numpy()
W1 = (cells['n1'] * cells['d1'] * cells['x_bias']).to_numpy()
dsse83 = N * (S83 ** 2 - S82 ** 2)
T = -(dsse83 + A_L + 2 * B_L) / 2
lam = T / W2L
beta_nL = (C2 - T) / W2N
# rounding band of the board (each score +-0.00005)
band = [(-(N * ((S83 + a) ** 2 - (S82 + b) ** 2) + A_L + 2 * B_L) / 2) / W2L for a in (-5e-5, 5e-5) for b in (-5e-5, 5e-5)]
# consistency with v12081 under the measured LIRF lambda
beta_nL_81 = (C1 - lam * W1[L].sum()) / W1[~L].sum()
# exact split of v12082 - v12080
dsse82 = N * (S82 ** 2 - S80 ** 2)
split = dict(total=dict(dsse=dsse82, delta_s=S82 - S80), lirf=dict(dsse=-dsse83, delta_s=S82 - S83),
             non_lirf=dict(dsse=dsse82 + dsse83, delta_s=S83 - S80))
# F-parametrisation for the blind months (residual of the final in a LIRF cell = beta_F * (-2 c))
cf = pl.read_csv(os.path.join(ROOT, 'analysis/final_tail_20261011/pru_level_cells.csv')).filter(pl.col('ADEP_mvt') == 'LIRF')
rf = pl.read_csv(os.path.join(ROOT, 'analysis/final_tail_20261011/rows_changed_by_cell.csv')).filter(pl.col('ADEP_mvt') == 'LIRF')
lf2 = cf.join(rf.select('month', 'changed', 'shift_mode'), on='month')
ndc = {int(m): float(n * s * c) for m, n, s, c in lf2.select('month', 'changed', 'shift_mode', 'c').iter_rows()}
nd2 = {int(m): float(n * s * s) for m, n, s in lf2.select('month', 'changed', 'shift_mode').iter_rows()}
beta_F = -(B_L + T) / (2 * (ndc[1] + ndc[7]))
blind = {}
for bF, name in ((beta_F, 'measured_janjul_beta_F'), (0.25, 'break_even'), (0.0, 'beta_F_0')):
    dsse = -(nd2[2] + nd2[6]) + 4 * bF * (ndc[2] + ndc[6])
    blind[name] = dict(beta_F=bF, dsse_febjun_remove_LIRF=dsse, delta_final_file_s=dsse / (2 * 670790 * 231.0))
janjul_final_units = -dsse83 / (2 * 670790 * 231.0)
out = dict(score_v12083=S83, delta_vs_v12082=S83 - S82, delta_vs_v12080=S83 - S80, dsse_83_vs_82=dsse83, T_sum_LIRF_d_resid77=T,
           lambda_LIRF_v77=lam, lambda_rounding_band=[min(band), max(band)], beta_nonLIRF_from_v12082=beta_nL,
           beta_nonLIRF_from_v12081_given_lambda=beta_nL_81, split_v12082_minus_v12080=split, beta_F_LIRF_janjul=beta_F,
           blind_febjun=blind, janjul_LIRF_gain_in_final_file_units_s=janjul_final_units,
           forecast=dict(point=F['forecast']['point'], corridor=F['forecast']['corridor'],
                         inside=F['forecast']['corridor'][0] <= S83 <= F['forecast']['corridor'][1]),
           rule=dict(threshold=F['thresholds']['rule_score'], outcome='final stays bbae2cc5' if S83 > F['thresholds']['rule_score'] else 'rebuild without LIRF'))
json.dump(out, open(os.path.join(ROOT, 'analysis/v12083_upload_20261011/verdict.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))

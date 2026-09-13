# Корреляция дневных показателей EUROCONTROL AIU с дневной медианой taxi-out (training 2025).
import glob, os, pandas as pd, numpy as np
R = os.path.expanduser('~/prc-2026'); S = f'{R}/analysis/sources/samples'
cols = ['ADEP_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt']
d = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f'{R}/data/prc-2026-datasets/training_*.parquet'))])
d = d[d.PHASE_mvt == 'DEP']
d['taxi'] = (pd.to_datetime(d.MVT_TIME_UTC_mvt) - pd.to_datetime(d.BLOCK_TIME_UTC_mvt)).dt.total_seconds() / 60
d = d[(d.taxi >= 1) & (d.taxi <= 240)]
d['FLT_DATE'] = pd.to_datetime(d.BLOCK_TIME_UTC_mvt).dt.tz_convert(None).dt.normalize()
g = d.groupby(['ADEP_mvt', 'FLT_DATE']).taxi.agg(med='median', mean='mean', n='size').reset_index().rename(columns={'ADEP_mvt': 'APT_ICAO'})
g['p90'] = d.groupby(['ADEP_mvt', 'FLT_DATE']).taxi.quantile(0.9).values

def rd(p):
    x = pd.read_csv(p); x['FLT_DATE'] = pd.to_datetime(x.FLT_DATE); return x
a = rd(f'{S}/eurocontrol_apt_dly/Airport_Arrival_ATFM_Delay_10apt.csv')
dc = [c for c in a.columns if c.startswith('DLY_APT_ARR')]; a[dc] = a[dc].fillna(0)  # NaN = нет задержки
a['atfm_arr_per_flt'] = a.DLY_APT_ARR_1 / a.FLT_ARR_1
a['atfm_wx_per_flt'] = (a.DLY_APT_ARR_W_1 + a.DLY_APT_ARR_D_1) / a.FLT_ARR_1
a['atfm_cap_per_flt'] = (a.DLY_APT_ARR_C_1 + a.DLY_APT_ARR_G_1 + a.DLY_APT_ARR_S_1) / a.FLT_ARR_1
a['atfm_deicing_min'] = a.DLY_APT_ARR_D_1
p = rd(f'{S}/eurocontrol_atc_predep_dly/ATC_Pre-Departure_Delay_10apt.csv'); p['atc_predep_per_dep'] = p.DLY_ATC_PRE_3 / p.FLT_DEP_3  # источник airlines/CODA: у источника airport половина дней пустая
q = rd(f'{S}/eurocontrol_all_predep_dly/All_Pre-Departure_Delay_10apt.csv'); q['all_predep_per_dep'] = q.DLY_ALL_PRE_2 / q.FLT_DEP_IFR_2
s = rd(f'{S}/eurocontrol_slot_adherence/ATFM_Slot_Adherence_10apt.csv'); s['reg_dep_share'] = s.FLT_DEP_REG_1 / s.FLT_DEP_1
t = rd(f'{S}/eurocontrol_airport_traffic/Airport_Traffic_10apt.csv')
feats = ['atfm_arr_per_flt', 'atfm_wx_per_flt', 'atfm_cap_per_flt', 'atfm_deicing_min', 'atc_predep_per_dep', 'all_predep_per_dep', 'reg_dep_share', 'FLT_DEP_1']
m = g.merge(a[['APT_ICAO', 'FLT_DATE'] + feats[:4]], how='left').merge(p[['APT_ICAO', 'FLT_DATE', 'atc_predep_per_dep']], how='left') \
     .merge(q[['APT_ICAO', 'FLT_DATE', 'all_predep_per_dep']], how='left').merge(s[['APT_ICAO', 'FLT_DATE', 'reg_dep_share']], how='left') \
     .merge(t[['APT_ICAO', 'FLT_DATE', 'FLT_DEP_1']], how='left')
m = m.sort_values(['APT_ICAO', 'FLT_DATE'])
print('merged airport-days:', len(m), 'missing per feat:', m[feats].isna().sum().to_dict())
# внутри аэропорт×месяц: вычитаем среднее, остаётся дневной сигнал
m['ym'] = m.FLT_DATE.dt.to_period('M')
def dm(c): return m[c] - m.groupby(['APT_ICAO', 'ym'])[c].transform('mean')
rows = []
for f in feats:
    lag = m.groupby('APT_ICAO')[f].shift(1)
    for tgt in ['med', 'p90']:
        rows.append(dict(feat=f, target=tgt,
            spearman_raw_sameday=m[[f, tgt]].corr('spearman').iloc[0, 1],
            pearson_within_aptmonth_sameday=np.corrcoef(*m[[f, tgt]].dropna().pipe(lambda z: [z[f] - z.groupby(m.loc[z.index, 'APT_ICAO'] + m.loc[z.index, 'ym'].astype(str))[f].transform('mean'), z[tgt] - z.groupby(m.loc[z.index, 'APT_ICAO'] + m.loc[z.index, 'ym'].astype(str))[tgt].transform('mean')]))[0, 1],
            pearson_within_aptmonth_lagD1=pd.concat([lag - lag.groupby([m.APT_ICAO, m.ym]).transform('mean'), dm(tgt)], axis=1).dropna().corr().iloc[0, 1]))
print(pd.DataFrame(rows).round(3).to_string())
print('\nper-airport within-month pearson (same day) med vs atfm_arr_per_flt / all_predep_per_dep / reg_dep_share:')
for apt, z in m.groupby('APT_ICAO'):
    zz = z.assign(**{c: z[c] - z.groupby('ym')[c].transform('mean') for c in ['med', 'atfm_arr_per_flt', 'all_predep_per_dep', 'reg_dep_share']})
    print(apt, [round(zz[['med', c]].corr().iloc[0, 1], 3) for c in ['atfm_arr_per_flt', 'all_predep_per_dep', 'reg_dep_share']])
# месячный additional taxi-out vs наш месячный средний taxi-out
x = pd.read_csv(f'{S}/eurocontrol_taxi_out_add/Taxi-Out_Additional_Time_10apt.csv')
x = x[x.YEAR == 2025].assign(add_avg=lambda z: z.TOTAL_ADD_TIME_MIN / z.TOTAL_REF_NB_FL, ref_avg=lambda z: z.TOTAL_REF_TIME_MIN / z.TOTAL_REF_NB_FL)
d['MONTH_NUM'] = d.FLT_DATE.dt.month
mm = d.groupby(['ADEP_mvt', 'MONTH_NUM']).taxi.mean().rename('our_mean').reset_index().rename(columns={'ADEP_mvt': 'APT_ICAO'})
xm = x.merge(mm); xm['eu_total'] = xm.add_avg + xm.ref_avg
print('\nmonthly 2025, 120 airport-months: pearson our_mean vs EU ref+add =', round(xm[['our_mean', 'eu_total']].corr().iloc[0, 1], 3),
      '; mean abs diff min =', round((xm.our_mean - xm.eu_total).abs().mean(), 2))
xm['our_dm'] = xm.our_mean - xm.groupby('APT_ICAO').our_mean.transform('mean'); xm['add_dm'] = xm.add_avg - xm.groupby('APT_ICAO').add_avg.transform('mean')
print('within-airport monthly pearson our_mean vs add_avg =', round(xm[['our_dm', 'add_dm']].corr().iloc[0, 1], 3))
print(xm.groupby('APT_ICAO').apply(lambda z: pd.Series({'our': z.our_mean.mean(), 'eu_ref+add': z.eu_total.mean()})).round(2).to_string())

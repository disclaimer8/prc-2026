"""Сверка SUP AIP AIRAC 120/25 (закрытие 09R/27L LFPG H24 с 07.08.2025 по 08.12.2025) с RUNWAY_mvt:
доля движений LFPG на 09R/27L по месяцам 2025 и в ranking (янв/июл 2026) + медиана taxi-out вылетов."""
import glob, pandas as pd
P='/Users/denyskolomiiets/prc-2026/data/prc-2026-datasets/'
c=['ADEP_mvt','ADES_mvt','PHASE_mvt','RUNWAY_mvt','MVT_TIME_UTC_mvt','TAXITIME_SEC_mvt']
fs=sorted(glob.glob(P+'training_2025-*.parquet'))+[P+'ranking.parquet']
out=[]
for f in fs:
    x=pd.read_parquet(f); x=x[[k for k in c if k in x.columns]]
    x=x[((x.PHASE_mvt=='DEP')&(x.ADEP_mvt=='LFPG'))|((x.PHASE_mvt=='ARR')&(x.ADES_mvt=='LFPG'))]
    out.append(x)
d=pd.concat(out); d['ym']=d.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m')
d['r27L']=d.RUNWAY_mvt.isin(['09R','27L'])
g=d.groupby('ym').agg(n=('r27L','size'),share_09R27L=('r27L','mean'))
dep=d[d.PHASE_mvt=='DEP']
if 'TAXITIME_SEC_mvt' in dep: g['dep_taxi_med_min']=dep.groupby('ym').TAXITIME_SEC_mvt.median()/60
print(g.round(3).to_string())
wk=d[(d.MVT_TIME_UTC_mvt>='2025-07-25')&(d.MVT_TIME_UTC_mvt<'2025-08-20')]
print(wk.groupby(wk.MVT_TIME_UTC_mvt.dt.date).r27L.mean().round(3).to_string())

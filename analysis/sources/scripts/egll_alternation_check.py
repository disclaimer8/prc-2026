"""Сверка опубликованного графика дневного чередования ВПП Хитроу (PDF 2025) с RUNWAY_mvt вылетов EGLL 2025.
График: неделя, начинающаяся с пн 6 янв 2025 = посадки 27L до 15:00, 27R после; далее чередование каждую неделю.
Вылеты идут с другой ВПП: посадка 27L -> вылет 27R. Easterly (09x) не чередуется днём."""
import glob, pandas as pd, numpy as np
fs=sorted(glob.glob('/Users/denyskolomiiets/prc-2026/data/prc-2026-datasets/training_2025-*.parquet'))
d=pd.concat([pd.read_parquet(f,columns=['ADEP_mvt','PHASE_mvt','RUNWAY_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','TAXITIME_SEC_mvt']) for f in fs])
d=d[(d.PHASE_mvt=='DEP')&(d.ADEP_mvt=='EGLL')].copy()
t=d.MVT_TIME_UTC_mvt.dt.tz_convert('Europe/London')
d['hour']=t.dt.hour; d['date']=t.dt.tz_localize(None).dt.normalize()
wk=((d.date-pd.Timestamp('2025-01-06')).dt.days//7)
ldg_am=np.where(wk%2==0,'27L','27R'); ldg_pm=np.where(wk%2==0,'27R','27L')
ldg=np.where(d.hour<15,ldg_am,ldg_pm)
d['sched_dep']=np.where(ldg=='27L','27R','27L')
w=d[d.RUNWAY_mvt.isin(['27L','27R'])&(d.hour>=7)&(d.hour<22)]
print('westerly day deps',len(w),'agree with schedule',(w.RUNWAY_mvt==w.sched_dep).mean().round(4))
print(w.groupby(w.date.dt.month).apply(lambda g:(g.RUNWAY_mvt==g.sched_dep).mean()).round(3).to_string())
d['west']=d.RUNWAY_mvt.str.startswith('27')
print('share westerly deps',d.west.mean().round(3))

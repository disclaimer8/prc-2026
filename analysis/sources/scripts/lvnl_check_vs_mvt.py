"""Сверка периодов LVNL /runway/get (сэмпл 4 дня) с RUNWAY_mvt вылетов/прилётов EHAM.
Проверяем сдвиг времени: для каждого движения ищем период, считая start/end как UTC со сдвигом s часов."""
import json, pandas as pd, numpy as np
S='/Users/denyskolomiiets/prc-2026/analysis/sources/samples/lvnl_runway_use/lvnl_runway_get_4days.json'
P='/Users/denyskolomiiets/prc-2026/data/prc-2026-datasets/'
recs=[r for v in json.load(open(S)).values() for r in v]
per=pd.DataFrame(recs).drop_duplicates('id')
per['start']=pd.to_datetime(per.start.str.replace('Z','')); per['end']=pd.to_datetime(per.end.str.replace('Z',''))
cols=['ADEP_mvt','ADES_mvt','PHASE_mvt','RUNWAY_mvt','MVT_TIME_UTC_mvt']
mv=[]
for f in ['training_2025-01-01_2025-02-01.parquet','training_2025-07-01_2025-08-01.parquet','ranking.parquet']:
    x=pd.read_parquet(P+f); x=x[[c for c in cols if c in x.columns]]
    x=x[((x.PHASE_mvt=='DEP')&(x.ADEP_mvt=='EHAM'))|((x.PHASE_mvt=='ARR')&(x.ADES_mvt=='EHAM'))]
    mv.append(x)
mv=pd.concat(mv); mv['t']=mv.MVT_TIME_UTC_mvt.dt.tz_localize(None)
mv=mv[mv.t.dt.strftime('%m-%d').isin(['01-15','07-15'])&mv.RUNWAY_mvt.notna()]
print('movements with runway:',len(mv), mv.groupby([mv.t.dt.year,mv.t.dt.month]).size().to_dict())
for s in [-2,-1,0,1,2]:
    ok=n=0
    rows=[]
    for _,m in mv.iterrows():
        tt=m.t+pd.Timedelta(hours=s)
        p=per[(per.start<=tt)&(per.end>tt)]
        if len(p)==0: continue
        p=p.iloc[0]
        rw=[p.takeoff1,p.takeoff2,p.takeoff3] if m.PHASE_mvt=='DEP' else [p.landing1,p.landing2,p.landing3]
        rw=[r.strip() for r in rw if r and r.strip()]
        n+=1; hit=m.RUNWAY_mvt in rw; ok+=hit
        rows.append((m.t.year,m.t.month,hit))
    r=pd.DataFrame(rows,columns=['y','m','hit'])
    print(f'shift {s:+d}h: matched {n}, runway in LVNL set {ok/n:.3f}', r.groupby(['y','m']).hit.mean().round(3).to_dict())

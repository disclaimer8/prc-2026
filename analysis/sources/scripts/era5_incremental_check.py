# Проверка: добавляет ли ERA5 (Open-Meteo) снежный покров к METAR на уровне аэропорт-день (зима 2025).
import json, glob, pandas as pd, numpy as np
APTS=['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']
fr=['training_2025-01','training_2025-02','training_2025-03','training_2025-11','training_2025-12']
d=pd.concat([pd.read_parquet(f,columns=['ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt']) for f in sorted(glob.glob('data/prc-2026-datasets/training_*.parquet')) if any(x in f for x in fr)])
d=d[d.PHASE_mvt=='DEP']; d['tx']=(d.MVT_TIME_UTC_mvt-d.BLOCK_TIME_UTC_mvt).dt.total_seconds()/60
d=d[(d.tx>1)&(d.tx<240)]; d['day']=d.BLOCK_TIME_UTC_mvt.dt.tz_localize(None).dt.floor('D')
y=d.groupby(['ADEP_mvt','day']).tx.mean().rename('tx').reset_index()
rows=[]
for a in APTS:
    m=pd.read_csv(f'data/metar/{a}.csv',na_values='null'); m['valid']=pd.to_datetime(m.valid); m['day']=m.valid.dt.floor('D')
    m['frozen']=m.wxcodes.fillna('').str.contains('SN|SG|PL|GS|FZ|IC')
    mg=m.groupby('day').agg(tmin=('tmpc','min'),frozen=('frozen','sum'),vis=('vsby','min')).reset_index(); mg['ADEP_mvt']=a
    h=pd.DataFrame(json.load(open(f'analysis/sources/samples/open_meteo_era5/{a}_2025-01-01_2025-12-31.json'))['hourly']); h['time']=pd.to_datetime(h.time); h['day']=h.time.dt.floor('D')
    hg=h.groupby('day').agg(sd0=('snow_depth','first'),sdmax=('snow_depth','max'),sf=('snowfall','sum')).reset_index()
    rows.append(mg.merge(hg,on='day'))
X=pd.concat(rows).merge(y,on=['ADEP_mvt','day'])
X['frozen_any']=(X.frozen>0).astype(float); X['sd0_pos']=(X.sd0>0).astype(float)
X['tx_dm']=X.tx-X.groupby('ADEP_mvt').tx.transform('mean')
def r2(cols):
    Z=X[cols].copy(); Z=Z-Z.groupby(X.ADEP_mvt).transform('mean'); A=np.c_[np.ones(len(Z)),Z.fillna(0).values]
    b=np.linalg.lstsq(A,X.tx_dm.values,rcond=None)[0]; e=X.tx_dm.values-A@b; return 1-e.var()/X.tx_dm.var()
base=['tmin','frozen','frozen_any','vis']
print('N airport-days',len(X))
print('R2 METAR only        %.3f'%r2(base))
print('R2 METAR+ERA5 sd0    %.3f'%r2(base+['sd0','sd0_pos']))
print('R2 METAR+ERA5 all    %.3f'%r2(base+['sd0','sd0_pos','sdmax','sf']))
print(X[X.sd0>0.02].groupby('ADEP_mvt').size())

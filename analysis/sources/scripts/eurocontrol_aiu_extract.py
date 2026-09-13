# Лист DATA каждого xlsx -> срез по 10 аэропортам (csv), печать покрытия; + текст листа copyright.
import glob, os, pandas as pd
APT = ['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']
S = os.path.expanduser('~/prc-2026/analysis/sources/samples')
for f in sorted(glob.glob(f'{S}/eurocontrol_*/*.xlsx')):
    d = os.path.dirname(f); name = os.path.basename(f)[:-5]
    print('=' * 80); print(name)
    cp = pd.read_excel(f, sheet_name=0, engine='calamine', header=None)
    txt = [str(x) for x in cp.values.ravel() if isinstance(x, str) and len(x.strip()) > 3]
    open(f'{d}/{name}_copyright_sheet.txt', 'w').write('\n'.join(txt))
    df = pd.read_excel(f, sheet_name='DATA', engine='calamine')
    print('cols:', list(df.columns)); print('rows all:', len(df))
    df = df[df.APT_ICAO.isin(APT)].copy()
    if 'FLT_DATE' in df:
        df['FLT_DATE'] = pd.to_datetime(df.FLT_DATE); key = df.FLT_DATE.dt.strftime('%Y-%m')
        print('date range (10 apt):', df.FLT_DATE.min(), df.FLT_DATE.max())
    else:
        key = df.YEAR.astype(str) + '-' + df.MONTH_NUM.astype(int).astype(str).str.zfill(2)
        print('month range (10 apt):', key.min(), key.max())
    print('rows per apt:', df.APT_ICAO.value_counts().to_dict())
    for m in ['2025-01', '2025-07', '2026-01', '2026-06', '2026-07']:
        sub = df[key == m]
        print(f'  {m}: rows={len(sub)} apts={sub.APT_ICAO.nunique()}')
    df.to_csv(f'{d}/{name}_10apt.csv', index=False)

# Читает xlsx EUROCONTROL AIU, печатает листы/колонки/покрытие, сохраняет срез по 10 аэропортам в parquet.
import glob, os, pandas as pd
APT = ['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']
S = os.path.expanduser('~/prc-2026/analysis/sources/samples')
for f in sorted(glob.glob(f'{S}/eurocontrol_*/*.xlsx')):
    print('=' * 100); print(f, os.path.getsize(f) // 1e6, 'MB')
    sheets = pd.read_excel(f, sheet_name=None, engine='calamine', header=None, nrows=8)
    for sh, head in sheets.items():
        print('--- sheet', sh, head.shape); print(head.head(8).to_string()[:1500])

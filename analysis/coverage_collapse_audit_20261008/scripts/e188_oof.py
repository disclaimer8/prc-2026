import polars as pl, numpy as np, glob
S='/private/tmp/claude-501/-Users-denyskolomiiets/237b9694-d169-4305-ab5c-7ff3144974ed/scratchpad'
e=pl.concat([pl.read_parquet(p) for p in sorted(glob.glob('data/e195_pre_e81_20261007_v2/fold_*.parquet'))]).select('MVT_ID_mvt','control','e190','e188','joint','evaluation_mask')
d=pl.read_parquet(f'{S}/cv_rows_tok.parquet',columns=['MVT_ID_mvt','TAXITIME_SEC_mvt','month','ADEP_mvt','has_nm','coverage','vi_has_prev','visit','no_tok','fold']).join(e,on='MVT_ID_mvt',how='left').filter(pl.col('evaluation_mask'))
y=pl.col('TAXITIME_SEC_mvt').cast(pl.Float64)
def rm(c): return ((pl.col(c)-y)**2).mean().sqrt()
m=((pl.col('coverage')<2)&(pl.col('vi_has_prev')==1)).fill_null(False)&(pl.col('ADEP_mvt')=='LIRF')&pl.col('has_nm')
with pl.Config(tbl_rows=30,tbl_width_chars=220,float_precision=2):
    print('E188 natural mask rows (LIRF, NM, coverage<2, prev visit): joint (E188+E190) vs e190 (no E188), by month')
    print(d.filter(m).group_by('month').agg(n=pl.len(),e190=rm('e190'),joint=rm('joint'),d=rm('joint')-rm('e190'),changed=(pl.col('joint')!=pl.col('e190')).sum()).sort('month'))
    print(d.filter(m).select(n=pl.len(),e190=rm('e190'),joint=rm('joint'),d=rm('joint')-rm('e190')))
    print('E188 off-mask rows changed (joint vs e190):', d.filter(~m).select((pl.col('joint')!=pl.col('e190')).sum()).item())
    print('E190 (PLE member) effect by class, chain level: e190 vs control')
    d=d.with_columns(cls=pl.when(pl.col('coverage')>=2).then(pl.lit('own_track')).when(pl.col('vi_has_prev')==1).then(pl.lit('prev_visit')).otherwise(pl.lit('no_adsb')))
    print(d.group_by('cls').agg(n=pl.len(),control=rm('control'),e190=rm('e190'),d=rm('e190')-rm('control')).sort('cls'))

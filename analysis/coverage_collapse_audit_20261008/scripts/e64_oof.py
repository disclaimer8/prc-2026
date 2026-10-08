import polars as pl, numpy as np, glob
S='/private/tmp/claude-501/-Users-denyskolomiiets/237b9694-d169-4305-ab5c-7ff3144974ed/scratchpad'
d=pl.read_parquet(f'{S}/cv_rows_tok.parquet',columns=['MVT_ID_mvt','TAXITIME_SEC_mvt','month','ADEP_mvt','has_nm','coverage','vi_has_prev','visit','no_tok','fold'])
fr=[]
for f in range(1,7):
    n=f'fold_{f}_{f+6}'; a=np.load(f'data/cv_preds_e150c_e64_keep/{n}.npz'); b=np.load(f'data/cv_preds_nomonth/{n}.npz'); c=np.load(f'data/cv_preds_e70/{n}.npz'); e=np.load(f'data/cv_preds_e70_ctrl/{n}.npz')
    assert np.array_equal(a['mvt_id'],b['mvt_id'])
    t=pl.DataFrame({'MVT_ID_mvt':a['mvt_id'],'e64':a['pred'],'nomonth':b['pred']})
    t2=pl.DataFrame({'MVT_ID_mvt':c['mvt_id'],'e70':c['pred']}).join(pl.DataFrame({'MVT_ID_mvt':e['mvt_id'],'e70ctrl':e['pred']}),on='MVT_ID_mvt')
    fr.append(t.join(t2,on='MVT_ID_mvt',how='left'))
d=d.join(pl.concat(fr),on='MVT_ID_mvt',how='left')
print('nulls',d.null_count())
d=d.with_columns(e64mask=((pl.col('coverage')<2)&(pl.col('vi_has_prev')==1)).fill_null(False), e141mask=pl.col('coverage')>=2)
y=pl.col('TAXITIME_SEC_mvt').cast(pl.Float64)
def rm(c): return ((pl.col(c)-y)**2).mean().sqrt()
with pl.Config(tbl_rows=60,tbl_width_chars=220,float_precision=2):
    print('E64 stage (e150c keep) vs its base nomonth, on E64 mask rows, by airport')
    print(d.filter(pl.col('e64mask')).group_by('ADEP_mvt').agg(n=pl.len(),base=rm('nomonth'),e64=rm('e64'),d=rm('e64')-rm('nomonth'),changed=(pl.col('e64')!=pl.col('nomonth')).sum()).sort('ADEP_mvt'))
    print(d.filter(pl.col('e64mask')&(pl.col('ADEP_mvt')=='LIRF')).group_by('has_nm').agg(n=pl.len(),base=rm('nomonth'),e64=rm('e64'),d=rm('e64')-rm('nomonth')))
    print('E64 off-mask rows changed:', d.filter(~pl.col('e64mask')).select((pl.col('e64')!=pl.col('nomonth')).sum()).item())
    print('E70 (E141 family) vs ctrl on coverage>=2 rows by airport')
    print(d.filter(pl.col('e141mask')).group_by('ADEP_mvt').agg(n=pl.len(),ctrl=rm('e70ctrl'),e70=rm('e70'),d=rm('e70')-rm('e70ctrl'),changed=(pl.col('e70')!=pl.col('e70ctrl')).sum()).sort('ADEP_mvt'))
    print('E70 off-mask rows changed:', d.filter(~pl.col('e141mask')).select((pl.col('e70')!=pl.col('e70ctrl')).sum()).item(), 'of', d.filter(~pl.col('e141mask')).height)

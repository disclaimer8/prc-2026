import polars as pl, numpy as np
S='/private/tmp/claude-501/-Users-denyskolomiiets/237b9694-d169-4305-ab5c-7ff3144974ed/scratchpad'
z=np.load(f'{S}/tokens.npz')
tk=pl.DataFrame({'MVT_ID_mvt':z['cv_ids'],'tok_own':z['cv_ntok'][:,0].astype(np.int16),'tok_prev':z['cv_ntok'][:,1].astype(np.int16),'sup_own':z['cv_support'][:,0].astype(np.int8),'sup_prev':z['cv_support'][:,1].astype(np.int8)})
d=pl.read_parquet(f'{S}/cv_rows.parquet').join(tk,on='MVT_ID_mvt',how='left')
print('cv missing tokens', d['tok_own'].null_count())
d=d.with_columns(no_tok=(pl.col('tok_own')+pl.col('tok_prev'))==0)
# crosstab tokens vs coverage/visit classes
d=d.with_columns(cls=pl.when(pl.col('coverage')>=2).then(pl.lit('own_track')).when(pl.col('vi_has_prev')==1).then(pl.lit('prev_visit')).otherwise(pl.lit('no_adsb')))
with pl.Config(tbl_rows=100,tbl_cols=20,tbl_width_chars=250,float_precision=2):
    print(d.group_by('cls','no_tok').agg(n=pl.len(), visit=pl.col('visit').sum(), own0=(pl.col('tok_own')==0).sum(), prev0=(pl.col('tok_prev')==0).sum()).sort('cls','no_tok'))
    print(d.group_by('visit','no_tok').agg(n=pl.len()).sort('visit','no_tok'))
d.write_parquet(f'{S}/cv_rows_tok.parquet')
# final
ft=pl.DataFrame({'MVT_ID_mvt':z['final_ids'],'tok_own':z['final_ntok'][:,0].astype(np.int16),'tok_prev':z['final_ntok'][:,1].astype(np.int16)})
f=pl.read_parquet(f'{S}/final_rows.parquet').join(ft,on='MVT_ID_mvt',how='left')
print('final missing tokens', f['tok_own'].null_count(), f.height)
f=f.with_columns(no_tok=(pl.col('tok_own')+pl.col('tok_prev'))==0, no_tok_no_visit=((pl.col('tok_own')+pl.col('tok_prev'))==0)&(~pl.col('visit')))
g=f.group_by('ADEP_mvt','month').agg(n=pl.len(), no_tok=pl.col('no_tok').sum(), no_tok_no_visit=pl.col('no_tok_no_visit').sum(), own_tok=(pl.col('tok_own')>0).sum(), prev_tok=(pl.col('tok_prev')>0).sum()).with_columns((100*pl.col('no_tok')/pl.col('n')).round(1).alias('no_tok_%'),(100*pl.col('own_tok')/pl.col('n')).round(1).alias('own_tok_%'),(100*pl.col('prev_tok')/pl.col('n')).round(1).alias('prev_tok_%')).sort('ADEP_mvt','month')
with pl.Config(tbl_rows=100,tbl_cols=20,tbl_width_chars=250): print(g)
print(f.select(n=pl.len(), no_tok=pl.col('no_tok').sum(), no_tok_no_visit=pl.col('no_tok_no_visit').sum()))
f.write_parquet(f'{S}/final_rows_tok.parquet')
# 2025 same table by airport x month for LIRF/EDDF
c=d.group_by('ADEP_mvt','month').agg(n=pl.len(), no_tok=pl.col('no_tok').sum(), own_tok=(pl.col('tok_own')>0).sum(), prev_tok=(pl.col('tok_prev')>0).sum()).with_columns((100*pl.col('no_tok')/pl.col('n')).round(1).alias('no_tok_%'),(100*pl.col('own_tok')/pl.col('n')).round(1).alias('own_tok_%')).sort('ADEP_mvt','month')
with pl.Config(tbl_rows=200,tbl_cols=20,tbl_width_chars=250): print(c.filter(pl.col('ADEP_mvt').is_in(['LIRF','EDDF','LTFM','LEMD'])))

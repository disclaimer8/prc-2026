import polars as pl, glob, numpy as np
D='data/final_adsb_derivatives_20261008/data'
rk = pl.read_parquet('data/prc-2026-datasets/final_ranking.parquet', columns=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','FLIGHT_ID_mvt','MVT_TIME_UTC_mvt']).filter(pl.col('PHASE_mvt')=='DEP')
cov = pl.concat([pl.read_parquet(f, columns=['MVT_ID_mvt','coverage']) for f in sorted(glob.glob(f'{D}/adsblol_kt1/offblock2_2026-*.parquet'))])
vis = pl.concat([pl.read_parquet(f, columns=['MVT_ID_mvt','vi_has_prev','vi_first_seen']) for f in sorted(glob.glob(f'{D}/adsblol/visit_2026-*.parquet'))])
print('cov rows', cov.height, cov['MVT_ID_mvt'].n_unique(), 'vis rows', vis.height, vis['MVT_ID_mvt'].n_unique())
d = (rk.join(cov.unique('MVT_ID_mvt'), on='MVT_ID_mvt', how='left').join(vis.unique('MVT_ID_mvt'), on='MVT_ID_mvt', how='left')
     .with_columns(coverage=pl.col('coverage').fill_null(-1), has_nm=pl.col('FLIGHT_ID_mvt').is_not_null(),
                   month=pl.col('MVT_TIME_UTC_mvt').dt.strftime('%Y-%m'), visit=pl.col('vi_first_seen').is_not_null()))
d = d.with_columns(e141=pl.col('coverage')>=2, e64=((pl.col('coverage')<2)&(pl.col('vi_has_prev')==1)).fill_null(False))
d = d.with_columns(e188=pl.col('e64')&(pl.col('ADEP_mvt')=='LIRF')&pl.col('has_nm'), none=(~pl.col('e141'))&(~pl.col('e64')))
g = d.group_by('ADEP_mvt','month').agg(n=pl.len(), e141=pl.col('e141').sum(), e64=pl.col('e64').sum(), e188=pl.col('e188').sum(), untouched=pl.col('none').sum(), visit=pl.col('visit').sum(), has_prev=(pl.col('vi_has_prev')==1).sum(), nm=pl.col('has_nm').sum()).sort('ADEP_mvt','month')
g = g.with_columns((100*pl.col('e141')/pl.col('n')).round(1).alias('e141_%'), (100*pl.col('e64')/pl.col('n')).round(1).alias('e64_%'), (100*pl.col('untouched')/pl.col('n')).round(1).alias('untouched_%'))
with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=250): print(g)
tot = d.select(n=pl.len(), e141=pl.col('e141').sum(), e64=pl.col('e64').sum(), e188=pl.col('e188').sum(), untouched=pl.col('none').sum()); print(tot)
d.select('MVT_ID_mvt','ADEP_mvt','month','coverage','vi_has_prev','has_nm','visit').write_parquet('/private/tmp/claude-501/-Users-denyskolomiiets/237b9694-d169-4305-ab5c-7ff3144974ed/scratchpad/final_rows.parquet')

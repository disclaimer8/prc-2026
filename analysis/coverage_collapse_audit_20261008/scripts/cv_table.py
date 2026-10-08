import polars as pl, glob, numpy as np
S='/private/tmp/claude-501/-Users-denyskolomiiets/237b9694-d169-4305-ab5c-7ff3144974ed/scratchpad'
e = pl.read_parquet('data/e203_pre_e81_20261007/predictions.parquet')
cov = pl.concat([pl.read_parquet(f, columns=['MVT_ID_mvt','coverage']) for f in sorted(glob.glob('data/adsblol_kt1/offblock2_2025-*.parquet'))]).unique('MVT_ID_mvt')
vis = pl.concat([pl.read_parquet(f, columns=['MVT_ID_mvt','vi_has_prev','vi_first_seen']) for f in sorted(glob.glob('data/adsblol/visit_2025-*.parquet'))]).unique('MVT_ID_mvt')
d = e.join(cov, on='MVT_ID_mvt', how='left').join(vis, on='MVT_ID_mvt', how='left').with_columns(coverage=pl.col('coverage').fill_null(-1), visit=pl.col('vi_first_seen').is_not_null())
frames=[]
for f in range(1,7):
    name=f'fold_{f}_{f+6}'
    p = f'data/e197_trajectory_20261007/{name}/ordered/prediction.npz' if f in (1,4) else f'data/e202_trajectory_20261007/{name}/ordered/prediction.npz'
    z=np.load(p); b=np.load(f'data/cv_preds_fin1b_hz/{name}.npz'); g=np.load(f'data/e190_numemb_20261005/{name}/prediction.npz')
    t=pl.DataFrame({'MVT_ID_mvt':z['mvt_id'],'nn_member':z['member'],'nn_pred':z['pred'],'nn_reg':z['reg'],'y_chk':z['y']})
    t=t.join(pl.DataFrame({'MVT_ID_mvt':b['mvt_id'],'base_pred':b['pred'],'base_reg':b['reg']}),on='MVT_ID_mvt',how='left')
    t=t.join(pl.DataFrame({'MVT_ID_mvt':g['mvt_id'],'ple_member':g['member'],'ple_pred':g['pred']}),on='MVT_ID_mvt',how='left').with_columns(fold=pl.lit(f,dtype=pl.Int8))
    frames.append(t); print(name, t.height, 'nulls', t.null_count().sum_horizontal().item(), 'ple keys', list(g.files))
t=pl.concat(frames)
d=d.join(t,on='MVT_ID_mvt',how='left')
print(d.height, 'missing fold', d['fold'].null_count(), 'y mismatch', (d['TAXITIME_SEC_mvt']!=d['y_chk']).sum())
d.drop('y_chk').write_parquet(f'{S}/cv_rows.parquet')
print(d.schema)

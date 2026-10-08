import polars as pl, numpy as np
S='/private/tmp/claude-501/-Users-denyskolomiiets/237b9694-d169-4305-ab5c-7ff3144974ed/scratchpad'
d=pl.read_parquet(f'{S}/cv_rows_tok.parquet').join(pl.read_parquet(f'{S}/premix_candidates.parquet'),on='MVT_ID_mvt',how='left').filter(pl.col('evaluation_mask'))
d=d.with_columns(rule=pl.col('no_tok')&(~pl.col('visit')))
d=d.with_columns(cand=pl.when(pl.col('rule')).then(pl.col('pred_r2')).otherwise(pl.col('pred_nn_c')))
y=d['TAXITIME_SEC_mvt'].to_numpy().astype(float)
ec=(d['pred_nn_c'].to_numpy()-y)**2; ep=(d['cand'].to_numpy()-y)**2
print('whole eval population: ctrl RMSE %.4f cand RMSE %.4f dRMSE %+.4f n=%d; rows changed %d'%(np.sqrt(ec.mean()),np.sqrt(ep.mean()),np.sqrt(ep.mean())-np.sqrt(ec.mean()),len(y),int(d['rule'].sum())))
for m in [[1,7],[2,8],[6,12]]:
    g=d.filter(pl.col('month').is_in(m)); yy=g['TAXITIME_SEC_mvt'].to_numpy().astype(float)
    a=np.sqrt(((g['pred_nn_c'].to_numpy()-yy)**2).mean()); b=np.sqrt(((g['cand'].to_numpy()-yy)**2).mean()); print('months',m,'dRMSE %+.4f'%(b-a))
# member level on rule rows
g=d.filter(pl.col('rule')); yy=g['TAXITIME_SEC_mvt'].to_numpy().astype(float)
def r(c): return np.sqrt(((g[c].to_numpy()-yy)**2).mean())
print('rule rows member RMSE: nn %.2f ple %.2f base_reg(fin1b) %.2f r_ctrl(premix reg with PLE) %.2f r_nn %.2f r_r2 %.2f'%(r('nn_member'),r('ple_member'),r('base_reg'),r('r_ctrl'),r('r_nn'),r('r_r2')))
print('rule rows candidate RMSE: pred_ctrl(PLE premix) %.3f pred_nn %.3f pred_r2 %.3f | chain joint %.3f ordered %.3f'%(r('pred_ctrl'),r('pred_nn_c'),r('pred_r2'),r('joint'),r('ordered')))
# correlation of nn member residual with base residual on rule rows (diversity)
rn=g['nn_member'].to_numpy()-yy; rb=g['base_reg'].to_numpy()-yy; rp=g['ple_member'].to_numpy()-yy
print('resid corr nn-base %.3f ple-base %.3f nn-ple %.3f'%(np.corrcoef(rn,rb)[0,1],np.corrcoef(rp,rb)[0,1],np.corrcoef(rn,rp)[0,1]))

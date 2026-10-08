import polars as pl, numpy as np
S='/private/tmp/claude-501/-Users-denyskolomiiets/237b9694-d169-4305-ab5c-7ff3144974ed/scratchpad'
d=pl.read_parquet(f'{S}/cv_rows_tok.parquet').filter(pl.col('evaluation_mask'))
y=pl.col('TAXITIME_SEC_mvt').cast(pl.Float64)
def se(c): return (pl.col(c)-y)**2
def trim(expr_a, expr_b, frac):
    # symmetric trimmed paired SSE: drop frac of rows at each end by |y - joint|? use trimming on paired diff? Standard in repo: trim by residual magnitude of control. Keep simple: trim rows with largest |y-control| (frac each side not applicable); we trim top frac of |y - control|.
    pass
rows=[]
def stats(g, label):
    n=g.height
    dj=(g['joint']-g['TAXITIME_SEC_mvt']).to_numpy().astype(float); do=(g['ordered']-g['TAXITIME_SEC_mvt']).to_numpy().astype(float)
    dsse=(do**2-dj**2); r_j=np.sqrt((dj**2).mean()); r_o=np.sqrt((do**2).mean())
    # trimmed: drop 3% of rows with largest |dj| (control residual) as repo trim3 analogue on decided rows
    k=int(0.03*n); keep=np.argsort(np.abs(dj))[:n-k] if k>0 else np.arange(n)
    nn=(g['nn_member']-g['TAXITIME_SEC_mvt']).to_numpy().astype(float); pl_=(g['ple_member']-g['TAXITIME_SEC_mvt']).to_numpy().astype(float); br=(g['base_reg']-g['TAXITIME_SEC_mvt']).to_numpy().astype(float)
    return dict(group=label, n=n, rmse_joint=round(r_j,2), rmse_ordered=round(r_o,2), d_rmse=round(r_o-r_j,3), d_sse=float(dsse.sum()), trim3_sse=float(dsse[keep].sum()),
                rmse_nn=round(np.sqrt((nn**2).mean()),2), rmse_ple=round(np.sqrt((pl_**2).mean()),2), rmse_base_reg=round(np.sqrt((br**2).mean()),2), nn_minus_ple_sse=float((nn**2-pl_**2).sum()))
nt=d.filter(pl.col('no_tok'))
out=[stats(nt,'ALL no-token')]
for f in range(1,7): out.append(stats(nt.filter(pl.col('fold')==f), f'fold {f}+{f+6} no-token'))
for ap in ['LIRF','EDDF','LTFM','LEMD','LFPG','EGLL','EHAM','EDDM','LEBL','LSZH']: out.append(stats(nt.filter(pl.col('ADEP_mvt')==ap), f'{ap} no-token'))
for m in range(1,13): out.append(stats(nt.filter((pl.col('ADEP_mvt')=='LIRF')&(pl.col('month')==m)), f'LIRF m{m} no-token'))
out.append(stats(nt.filter((pl.col('ADEP_mvt')=='LIRF')&(pl.col('month').is_in([6,7,8]))), 'LIRF Jun-Aug no-token'))
out.append(stats(nt.filter((pl.col('ADEP_mvt')=='EDDF')&(pl.col('month').is_in([1,2,3]))), 'EDDF Jan-Mar no-token'))
out.append(stats(d.filter(~pl.col('no_tok')),'ALL with tokens'))
out.append(stats(d.filter((~pl.col('no_tok'))&(pl.col('ADEP_mvt')=='LIRF')),'LIRF with tokens'))
t=pl.DataFrame(out)
with pl.Config(tbl_rows=100,tbl_cols=20,tbl_width_chars=260,fmt_str_lengths=30): print(t)
t.write_csv(f'{S}/natural_notok.csv')

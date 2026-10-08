import polars as pl, numpy as np, json
S='/private/tmp/claude-501/-Users-denyskolomiiets/237b9694-d169-4305-ab5c-7ff3144974ed/scratchpad'
d=pl.read_parquet(f'{S}/cv_rows_tok.parquet').join(pl.read_parquet(f'{S}/premix_candidates.parquet'),on='MVT_ID_mvt',how='left')
assert d['pred_r2'].null_count()==0 and (d['pred_nn_c']-d['nn_pred']).abs().max()<1e-9
d=d.with_columns(rule=pl.col('no_tok')&(~pl.col('visit')), body=pl.col('TAXITIME_SEC_mvt')<=7200,
                 decided=~((pl.col('ADEP_mvt')=='LIRF')&(~pl.col('has_nm'))&(pl.col('D')>=14400)))
ev=d.filter(pl.col('evaluation_mask'))
def paired(g,ctrl='pred_nn_c',cand='pred_r2'):
    y=g['TAXITIME_SEC_mvt'].to_numpy().astype(float); ec=(g[ctrl].to_numpy()-y)**2; ep=(g[cand].to_numpy()-y)**2
    dd=np.sort(ep-ec); n=len(dd)
    return dict(n=n, ctrl_rmse=float(np.sqrt(ec.mean())), cand_rmse=float(np.sqrt(ep.mean())), d_rmse=float(np.sqrt(ep.mean())-np.sqrt(ec.mean())),
                d_sse=float(dd.sum()), trim3=float(dd[3:-3].sum()) if n>6 else float('nan'), trim20=float(dd[20:-20].sum()) if n>40 else float('nan'))
res={}
for pop,mask in [('rule',pl.col('rule')),('rule_body',pl.col('rule')&pl.col('body')),('rule_decided',pl.col('rule')&pl.col('decided')),('no_tok_any_visit',pl.col('no_tok'))]:
    g=ev.filter(mask); folds=[paired(g.filter(pl.col('fold')==f)) for f in range(1,7)]
    pooled=paired(g); passed_folds=sum(x['d_sse']<0 for x in folds)
    gate=passed_folds>=4 and pooled['d_sse']<0 and sum(x['trim3'] for x in folds)<0 and sum(x['trim20'] for x in folds)<0
    res[pop]=dict(pooled=pooled,folds=folds,folds_negative=passed_folds,sum_trim3=sum(x['trim3'] for x in folds),sum_trim20=sum(x['trim20'] for x in folds),gate_pass=bool(gate))
    print(f"== {pop}: n={pooled['n']} ctrl {pooled['ctrl_rmse']:.3f} cand {pooled['cand_rmse']:.3f} dRMSE {pooled['d_rmse']:+.4f} dSSE {pooled['d_sse']:.4g} folds<0 {passed_folds}/6 sum_trim3 {res[pop]['sum_trim3']:.4g} sum_trim20 {res[pop]['sum_trim20']:.4g} GATE {'PASS' if gate else 'FAIL'}")
    for f,x in enumerate(folds,1): print(f"   fold {f}+{f+6}: n={x['n']} dRMSE {x['d_rmse']:+.4f} dSSE {x['d_sse']:.4g} trim3 {x['trim3']:.4g} trim20 {x['trim20']:.4g}")
# with-token rows unchanged?
wt=ev.filter(~pl.col('rule')); print('rows with tokens/visit: max|cand-ctrl| =', float((wt['pred_r2']-wt['pred_nn_c']).abs().max()))
# whole-file effect
allp=paired(ev); print(f"whole eval population: dRMSE {allp['d_rmse']:+.5f} (n={allp['n']})")
# per airport and LIRF months on rule rows
g=ev.filter(pl.col('rule'))
rows=[]
for ap in ['LIRF','EDDF','LTFM','LEMD','LFPG','EGLL','EHAM','EDDM','LEBL','LSZH']:
    x=paired(g.filter(pl.col('ADEP_mvt')==ap)); rows.append(dict(group=ap,**x))
for m in range(1,13):
    gg=g.filter((pl.col('ADEP_mvt')=='LIRF')&(pl.col('month')==m))
    if gg.height>40: rows.append(dict(group=f'LIRF m{m}',**paired(gg)))
rows.append(dict(group='LIRF Jun-Aug',**paired(g.filter((pl.col('ADEP_mvt')=='LIRF')&pl.col('month').is_in([6,7,8])))))
rows.append(dict(group='EDDF Jan-Mar',**paired(g.filter((pl.col('ADEP_mvt')=='EDDF')&pl.col('month').is_in([1,2,3])))))
t=pl.DataFrame(rows)
with pl.Config(tbl_rows=50,tbl_cols=12,tbl_width_chars=220,float_precision=3): print(t)
t.write_csv(f'{S}/gate_r2_groups.csv'); json.dump(res,open(f'{S}/gate_r2.json','w'),indent=1)

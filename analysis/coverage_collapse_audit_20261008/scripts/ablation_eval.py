import polars as pl, numpy as np, glob
S='/private/tmp/claude-501/-Users-denyskolomiiets/237b9694-d169-4305-ab5c-7ff3144974ed/scratchpad'
def mix(values,proba,reg,lobt):
    v=values.astype(np.float64).copy(); v[:,0]=reg; v[:,-1]=reg+86400.
    p=np.sum(proba.astype(np.float64)*np.nan_to_num(v),axis=1)
    return np.maximum(np.where(np.isnan(lobt),p,np.clip(p,lobt-3606.,lobt+3606.)),0.)
fr=[]
for f in range(1,7):
    n=f'fold_{f}_{f+6}'; a=np.load(f'{S}/ablation/ablation_{n}.npz'); old=f in (1,4)
    c=np.load(f"data/{'e197' if old else 'e202'}_inputs_20261007/{n}/control.npz")
    assert np.array_equal(c['mvt_id'],a['mvt_id']) and np.array_equal(a['member_real'],a['member_saved'])
    coef=c['mlp_coefficient']; r_real=c['reg']+coef*(a['member_real']-c['plain']); r_mask=c['reg']+coef*(a['member_masked']-c['plain'])
    base=c['pred'].astype(float)-mix(c['values'],c['proba'],c['reg'],c['lobt'])
    fr.append(pl.DataFrame({'MVT_ID_mvt':a['mvt_id'],'nn_masked':a['member_masked'],'pred_mask':base+mix(c['values'],c['proba'],r_mask,c['lobt']),
                            'seed_spread_mask':np.std(a['masked_seeds'],axis=0)*323.3}))
d=pl.read_parquet(f'{S}/cv_rows_tok.parquet').join(pl.read_parquet(f'{S}/premix_candidates.parquet'),on='MVT_ID_mvt').join(pl.concat(fr),on='MVT_ID_mvt').filter(pl.col('evaluation_mask'))
nt=d.filter(pl.col('no_tok')); print('natural no-token rows: max|masked-real| member =', float((nt['nn_masked']-nt['nn_member']).abs().max()), 'n', nt.height)
y=pl.col('TAXITIME_SEC_mvt').cast(pl.Float64)
def rm(c): return ((pl.col(c)-y)**2).mean().sqrt()
agg=lambda g: g.agg(n=pl.len(), nn_real=rm('nn_member'), nn_masked=rm('nn_masked'), ple=rm('ple_member'), base_reg=rm('base_reg'),
                    pred_real=rm('pred_nn_c'), pred_masked=rm('pred_mask'), pred_noNN=rm('pred_r2'), pred_ple=rm('pred_ctrl'),
                    d_mask_vs_real=rm('pred_mask')-rm('pred_nn_c'), d_mask_vs_noNN=rm('pred_mask')-rm('pred_r2'), d_mask_vs_ple=rm('pred_mask')-rm('pred_ctrl'),
                    shift_masked_minus_real=(pl.col('nn_masked')-pl.col('nn_member')).mean(), rms_shift=((pl.col('nn_masked')-pl.col('nn_member'))**2).mean().sqrt())
wt=d.filter(~pl.col('no_tok'))
with pl.Config(tbl_rows=80,tbl_cols=20,tbl_width_chars=300,float_precision=2):
    print('== rows WITH tokens, tokens masked at inference (stress test): by airport'); print(agg(wt.group_by('ADEP_mvt')).sort('ADEP_mvt'))
    print('== LIRF with tokens by month'); print(agg(wt.filter(pl.col('ADEP_mvt')=='LIRF').group_by('month')).sort('month'))
    print('== EDDF with tokens by month'); print(agg(wt.filter(pl.col('ADEP_mvt')=='EDDF').group_by('month')).sort('month'))
    print('== by fold, all with tokens'); print(agg(wt.group_by('fold')).sort('fold'))
    print('== LIRF Jun-Aug with tokens, by class'); 
    wt2=wt.with_columns(cls=pl.when(pl.col('coverage')>=2).then(pl.lit('own_track')).when(pl.col('vi_has_prev')==1).then(pl.lit('prev_visit')).otherwise(pl.lit('no_adsb')))
    print(agg(wt2.filter((pl.col('ADEP_mvt')=='LIRF')&pl.col('month').is_in([6,7,8])).group_by('cls')).sort('cls'))
    print('== ALL with tokens'); print(agg(wt.group_by(pl.lit(1))))
t=agg(wt.group_by('ADEP_mvt','month')).sort('ADEP_mvt','month'); t.write_csv(f'{S}/ablation_airport_month.csv')

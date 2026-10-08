import numpy as np, polars as pl
S='/private/tmp/claude-501/-Users-denyskolomiiets/237b9694-d169-4305-ab5c-7ff3144974ed/scratchpad'
def mix(values,proba,reg,lobt):
    v=values.astype(np.float64).copy(); v[:,0]=reg; v[:,-1]=reg+86400.
    p=np.sum(proba.astype(np.float64)*np.nan_to_num(v),axis=1)
    return np.maximum(np.where(np.isnan(lobt),p,np.clip(p,lobt-3606.,lobt+3606.)),0.)
frames=[]
for f in range(1,7):
    name=f'fold_{f}_{f+6}'; old=f in (1,4)
    inp=f"data/{'e197' if old else 'e202'}_inputs_20261007/{name}"
    c=dict(np.load(f'{inp}/control.npz',allow_pickle=False))
    z=np.load(f"data/{'e197' if old else 'e202'}_trajectory_20261007/{name}/ordered/prediction.npz")
    if f==1: print({k:(v.shape,v.dtype) for k,v in c.items()}); print('mlp_coefficient unique', np.unique(c['mlp_coefficient'])[:10], 'n unique', len(np.unique(c['mlp_coefficient'])))
    assert np.array_equal(c['mvt_id'],z['mvt_id'])
    coef=c['mlp_coefficient']; r_nn=c['reg']+coef*(z['member']-c['plain'])
    print(name,'reg exact',np.array_equal(r_nn,z['reg']),'max|diff|',float(np.abs(r_nn-z['reg']).max()))
    pred_nn=c['pred'].astype(float)+mix(c['values'],c['proba'],r_nn,c['lobt'])-mix(c['values'],c['proba'],c['reg'],c['lobt'])
    print(name,'pred exact',np.array_equal(pred_nn,z['pred']),'max|diff|',float(np.abs(pred_nn-z['pred']).max()))
    r_r2=(c['reg']-coef*c['plain'])/(1-coef)
    pred_r2=c['pred'].astype(float)+mix(c['values'],c['proba'],r_r2,c['lobt'])-mix(c['values'],c['proba'],c['reg'],c['lobt'])
    frames.append(pl.DataFrame({'MVT_ID_mvt':c['mvt_id'],'coef':coef,'plain':c['plain'],'r_ctrl':c['reg'],'pred_ctrl':c['pred'].astype(float),'r_nn':r_nn,'pred_nn_c':pred_nn,'r_r2':r_r2,'pred_r2':pred_r2,'lobt':c['lobt']}))
pl.concat(frames).write_parquet(f'{S}/premix_candidates.parquet'); print('saved')

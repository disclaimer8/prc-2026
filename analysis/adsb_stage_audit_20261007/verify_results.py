"""Independent arithmetic and artifact verification; no fit or new candidate."""
import os
os.environ['POLARS_MAX_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json,datetime
from pathlib import Path
import numpy as np
import polars as pl

B=Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def close(a,b):
    assert np.isclose(a,b,rtol=2e-12,atol=1e-7),(a,b)

result=read(B/'results/result.json');manifest=read(B/'portable_inputs/manifest.json')
assert result['registration_commit']=='f2fc89603be4a1521860a98b2adb641d392a3e09'
assert sha(B/'results/result.json')==read(B/'execution/completed.json')['ResultSHA']
assert sha(B/'portable_inputs/manifest.json')==result['input_manifest_sha256']
checked={}
for directory,inventory in [('results',result['outputs']),('portable_inputs',manifest['spec']['files'])]:
    for name,h in inventory.items():
        assert sha(B/directory/name)==h,name
        checked[directory+'/'+name]=h
d=pl.read_parquet(B/'portable_inputs/training.parquet')
a=pl.read_parquet(B/'results/rows.parquet')
rank=pl.read_parquet(B/'portable_inputs/ranking.parquet')
ID='MVT_ID_mvt';Y='TAXITIME_SEC_mvt'
assert d.height==a.height==2085047 and d[ID].n_unique()==d.height
assert rank.height==rank[ID].n_unique()==344841
assert d.select(ID,Y,'ADEP_mvt','month','coverage','has_nm','evaluation_mask','joint').equals(a.select(ID,Y,'ADEP_mvt','month','coverage','has_nm','evaluation_mask','joint'))
ids=d[ID].to_numpy();y=d[Y].to_numpy().astype(float);control=d['joint'].to_numpy().astype(float)
month=d['month'].to_numpy();ap=d['ADEP_mvt'].to_numpy();cov=d['coverage'].to_numpy();inc=d['evaluation_mask'].to_numpy()
assert inc.sum()==2084921 and (~inc).sum()==126
pop={'included':inc,'body':inc&(y<=7200),'decided':inc&~((ap=='LIRF')&~d['has_nm'].to_numpy()&(d['D'].to_numpy()>=14400))}
def groups(frame,keys):
    if not keys:return {():np.arange(frame.height)}
    return {tuple(row[:-1]):np.asarray(row[-1]) for row in frame.with_row_index('ix').group_by(keys).agg('ix').iter_rows()}
keys={'all':[],'airport_coverage':['ADEP_mvt','coverage'],'airport_coverage_nm':['ADEP_mvt','coverage','has_nm'],'month':['month'],'half':['half']}
dd=d.with_columns(pl.Series('half',np.where(month<=6,1,2)))
ixs={name:groups(dd,k) for name,k in keys.items()}
tables=read(B/'results/lag_tables.json');records=read(B/'results/comparisons.json');support=read(B/'results/support.json')
fields=['mvt_first_move','sw_mvt_first_move'];validated=0
for field in fields:
    x=d[field].to_numpy();elig=inc&(cov>=2)&np.isfinite(x)&(x>=0)&(x<=43200)
    assert np.array_equal(elig,a[field+'_eligible'].to_numpy())
    lag=a[field+'_lag'].to_numpy();covered=np.zeros(len(y),bool)
    for table in [t for t in tables if t['field']==field]:
        train=elig&((month<=6) if table['train_half']==1 else (month>6))
        test=elig&((month>6) if table['train_half']==1 else (month<=6))&(ap==table['airport'])&(cov==table['coverage'])
        fit=train&(ap==table['airport'])&(cov==table['coverage']);source='airport_coverage'
        if fit.sum()<100:fit=train&(cov==table['coverage']);source='coverage_fallback'
        if fit.sum()<100:fit=train;source='global_fallback'
        assert not (fit&test).any() and source==table['source']
        assert fit.sum()==table['train_rows'] and test.sum()==table['test_rows']
        assert hashlib.sha256(ids[fit].astype('<i8').tobytes()).hexdigest()==table['train_id_sha256']
        estimate=float(np.median(x[fit]-y[fit]));close(estimate,table['lag_seconds'])
        assert np.all(lag[test]==estimate)
        covered|=test
    assert np.array_equal(covered,elig)
    pred=control.copy();pred[elig]=np.maximum(x[elig]-lag[elig],0)
    assert np.array_equal(pred,a[field+'_proxy'].to_numpy())
    zero=control.copy();zero[elig]=x[elig]
    for row in [r for r in records if r['field']==field]:
        base=ixs[row['grouping']][tuple(row['group'][k] for k in keys[row['grouping']])]
        ii=base[pop[row['population']][base]];p=pred if row['arm']=='cross_half_median_lag' else zero
        ec=(control[ii]-y[ii])**2;ep=(p[ii]-y[ii])**2;delta=ep-ec;sort=np.sort(delta);use=elig[ii]
        assert len(ii)==row['rows'] and use.sum()==row['eligible_rows']
        expected={'rmse_control':np.sqrt(ec.mean()),'rmse_proxy':np.sqrt(ep.mean()),'delta_rmse':np.sqrt(ep.mean())-np.sqrt(ec.mean()),'delta_sse':delta.sum(),'mean_bias_control':np.mean(control[ii]-y[ii]),'mean_bias_proxy':np.mean(p[ii]-y[ii])}
        for k,v in expected.items():close(v,row[k])
        for k in (3,20):
            if len(ii)>2*k:close(sort[k:-k].sum(),row[f'trim{k}_sse'])
            else:assert row[f'trim{k}_sse'] is None
        if use.any():
            residual=x[ii][use]-y[ii][use];lag_oracle=residual.mean();total=ec[~use].sum()+np.square(residual-lag_oracle).sum()
            o=row['oracle_same_evaluation_cell_constant_lag'];assert o['label_leaking']
            close(lag_oracle,o['lag_seconds']);close(np.sqrt(total/len(ii)),o['rmse']);close(total-ec.sum(),o['delta_sse'])
            o=row['oracle_per_row_joint_or_fixed_proxy'];assert o['label_leaking']
            close(np.minimum(delta,0).sum(),o['delta_sse']);close(np.sqrt(np.minimum(ec,ep).mean()),o['rmse'])
            assert (delta<0).sum()==o['proxy_chosen_rows']
        validated+=1
    for name,frame,ok in [('training',d,inc),('ranking',rank,np.ones(rank.height,bool))]:
        xx=frame[field].to_numpy();cc=frame['coverage'].to_numpy();valid=ok&(cc>=2)&np.isfinite(xx)&(xx>=0)&(xx<=43200)
        lookup=groups(frame,['ADEP_mvt','coverage','has_nm'])
        for s in [s for s in support if s['field']==field and s['population']==name]:
            ii=lookup[(s['ADEP_mvt'],s['coverage'],s['has_nm'])]
            assert len(ii)==s['rows'] and ok[ii].sum()==s['evaluation_rows'] and valid[ii].sum()==s['eligible_rows']
    chosen=[r for r in records if r['field']==field and r['arm']=='cross_half_median_lag']
    halves=[r for r in chosen if r['grouping']=='half' and r['population']=='body']
    dec=next(r for r in chosen if r['grouping']=='all' and r['population']=='decided')
    assert result['decisions'][field]['observable_signal_for_separate_followup']==(all(r['delta_sse']<0 for r in halves) and dec['delta_sse']<0)

receipt={'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'registration':result['registration_commit'],'result_sha256':sha(B/'results/result.json'),'verification_source_sha256':sha(Path(__file__)),'artifact_hashes':checked,'unique_training_ids':d.height,'unique_ranking_ids':rank.height,'included':int(inc.sum()),'excluded':int((~inc).sum()),'lag_lookup_rows_verified':len(tables),'comparison_rows_verified':validated,'support_rows_verified':len(support),'checks':['Exact ID/y/month/airport/NM/control/mask equality','All output and compact input SHA256','Every training-half lookup median, count, fallback, disjoint heldout IDs, ID hash','Bit-exact observable predictions and untouched fallback','Every saved paired metric, trim, separately labelled oracle','Every coverage support cell and both diagnostic decisions'],'scope':'Independent arithmetic verification only; no model fit or candidate'}
(B/'verified_results.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='artifact_hashes'},indent=2))

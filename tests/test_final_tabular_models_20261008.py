"""Synthetic portability contracts; these tests do not open competition rows."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
FILE = ROOT / 'scripts/final_tabular_models_20261008.py'
if FILE.exists():
    spec = importlib.util.spec_from_file_location('finaltab', FILE)
    X = importlib.util.module_from_spec(spec); spec.loader.exec_module(X)
else:
    X = None


def proxy_frame():
    return pl.DataFrame({
        'ADEP_mvt': ['LIRF','LIRF','LIRF','LSZH','LSZH','LSZH'],
        'd_mvt_aobt': [500.,1000.,7000.,None,None,None],
        'd_mvt_lobt': [510.,1010.,500.,1200.,None,None],
        'd_mvt_eobt': [550.,1020.,550.,1250.,1400.,None],
        'TAXITIME_SEC_mvt': [450.,900.,480.,1100.,1000.,800.]})


class Contracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(X, 'final production adapter missing')

    def test_exact_seed_sets_never_partial_or_duplicate(self):
        self.assertEqual(X.expected_seeds('cat'), list(range(15)))
        self.assertEqual(X.expected_seeds('aobt'), list(range(5)))
        for seeds in ([0,1,2,3], [0,1,2,3,3], [0,1,2,3,5]):
            with self.assertRaises(ValueError): X.check_seed_set('raw', seeds)
        X.check_seed_set('raw', [4,3,2,1,0])

    def test_memory_ceiling_cat_24gib_lgb_32gib(self):
        # Measured 19.46 GiB cgroup peak in prepare() on prc-cloud (08.10.2026); 12 GiB would OOM every Cat seed.
        self.assertEqual(X.memory_limit('cat'), 24*1024**3)
        for kind in ('lgb','raw','lobt','sched','eobt','aobt','motion','base'):
            self.assertEqual(X.memory_limit(kind), 32*1024**3)
        with self.assertRaises(KeyError): X.MEMORY_LIMIT_GIB['raw']

    def test_prepare_releases_full_frame_before_encodings_and_training_frame_before_matrices(self):
        # Portability: with df/fit retained, prepare() peaked at 19.4-20.1 GiB (prc-cloud/hetzner probes 08.10.2026),
        # above the 12-14 GiB hosts. The release points must stay between the last use and the next allocation.
        import inspect
        src=inspect.getsource(X.prepare)
        self.assertLess(src.index("ordered '+split+' IDs changed"), src.index('del df'))
        self.assertLess(src.index('del df'), src.index('T.add_encodings'))
        self.assertLess(src.index('cats=categories('), src.index('del fit,lab'))
        self.assertLess(src.index('del fit,lab'), src.index('fit_proxy(ordinary)'))
        self.assertNotIn('fit[', src[src.index('del fit,lab'):])
        self.assertNotIn('df[', src[src.index('del df'):])

    def test_dynamic_population_rejects_final_targets_and_duplicate_ids(self):
        d=pl.DataFrame({'MVT_ID_mvt':[1.,2.,3.,4.], 'split':['train','train','rank','rank'],
                        'TAXITIME_SEC_mvt':[10.,20.,None,None]})
        X.validate_rows(d, 2, 2)
        with self.assertRaises(ValueError): X.validate_rows(d, 2, 344841)
        bad=d.with_columns(pl.when(pl.col('split')=='rank').then(8.).otherwise(pl.col(X.Y)).alias(X.Y))
        with self.assertRaises(ValueError): X.validate_rows(bad, 2, 2)
        with self.assertRaises(ValueError): X.validate_rows(d.with_columns(pl.lit(1.).alias(X.ID)),2,2)

    def test_proxy_json_roundtrip_matches_existing_recipe_with_missing_and_gate(self):
        X.configure(ROOT, 'unused', 1)
        import torch  # established OpenMP load order, before LightGBM
        import train as T
        f=proxy_frame(); p=f.drop(X.Y).reverse()
        expected_fit,expected_pred=T.proxy_base(f,p)
        state=json.loads(json.dumps(X.fit_proxy(f)))
        np.testing.assert_allclose(X.apply_proxy(f,state),expected_fit,rtol=0,atol=1e-10)
        np.testing.assert_allclose(X.apply_proxy(p,state),expected_pred,rtol=0,atol=1e-10)
        # Inference has no target column; changing held-out target cannot change output.
        np.testing.assert_array_equal(X.apply_proxy(p.with_columns(pl.lit(999999.).alias(X.Y)),state),X.apply_proxy(p,state))

    def test_category_unknown_and_null_keep_training_codes(self):
        f=pl.DataFrame({'ADEP_mvt':['LSZH','LIRF',None], 'n':[1.,2.,3.]})
        cats=X.categories(f,['ADEP_mvt'])
        p=pl.DataFrame({'ADEP_mvt':['LIRF','NEW',None,'LSZH'], 'n':[4.,5.,6.,7.]})
        m=X.pandas_matrix(p,['ADEP_mvt','n'],cats)
        self.assertEqual(m['ADEP_mvt'].cat.codes.to_list(),[0,-1,-1,1])
        with self.assertRaises(ValueError): X.pandas_matrix(p.drop('n'),['ADEP_mvt','n'],cats)

    def test_model_reload_lightgbm_and_catboost_exact(self):
        import torch
        import lightgbm as lgb
        from catboost import CatBoostRegressor
        f=pl.DataFrame({'ADEP_mvt':['LIRF','LSZH']*40,'n':np.arange(80,dtype=float)})
        cats=X.categories(f,['ADEP_mvt']); columns=['ADEP_mvt','n']
        y=np.arange(80,dtype=float)%9
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            matrix=X.pandas_matrix(f,columns,cats)
            model=lgb.train({'objective':'regression','verbose':-1,'num_threads':1,'min_data_in_leaf':2},lgb.Dataset(matrix,y,categorical_feature=['ADEP_mvt']),num_boost_round=4)
            model.save_model(str(path/'model.txt'))
            state={'engine':'lightgbm','columns':columns,'categories':cats,'kind':'raw','proxy':None}
            np.testing.assert_array_equal(model.predict(matrix),X.predict_saved(path,f,state))
            cp=X.cat_pool(f,columns,['ADEP_mvt'],y)
            cb=CatBoostRegressor(iterations=4,depth=2,verbose=False,thread_count=1,allow_writing_files=False).fit(cp)
            cb.save_model(str(path/'model.cbm'))
            state.update(engine='catboost',categorical_columns=['ADEP_mvt'])
            np.testing.assert_array_equal(cb.predict(cp),X.predict_saved(path,f,state))

    def test_output_publication_refuses_overwrite_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'file.json';X.save_json(p,{'ok':True})
            with self.assertRaises(FileExistsError): X.save_json(p,{'ok':False})
        with self.assertRaises(ValueError): X.validate_predictions(np.array([1.,np.inf]),2)

    def test_anchor_clips_and_never_reads_targets(self):
        p=pl.DataFrame({'d_mvt_aobt':[None,-1.,8000.,100.],X.Y:[1.,2.,3.,4.]})
        np.testing.assert_array_equal(X.anchor(p,'aobt'),[0.,0.,7200.,100.])
        np.testing.assert_array_equal(X.anchor(p,'raw'),np.zeros(4))

    def test_disabled_nonboolean_environment_is_removed(self):
        with patch.dict(os.environ,{'REG_OVERRIDE':'0','ADV_WEIGHT':'0'}):
            X.configure(ROOT,'unused',1)
            self.assertNotIn('REG_OVERRIDE',os.environ)
            self.assertNotIn('ADV_WEIGHT',os.environ)

    def test_merge_rejects_unbound_seed_predictions_and_absent_models(self):
        receipt={'commit':'a'*40};contract={'rank_rows':2,'features':'features',
            'files':{'features':'b'*64},'rank_ids_sha256':X.ids_sha([1.,2.])}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for seed in range(5):
                d=root/'models/final_tabular/raw'/f'seed_{seed}';d.mkdir(parents=True)
                np.savez(d/'predictions.npz',mvt_id=np.array([1.,2.]),reg=np.array([3.,4.]))
                X.save_json(d/'complete.json',{'protocol':X.PROTOCOL,'registration':'a'*40,'kind':'raw','seed':seed,
                    'reload_parity':True,'files':{},'feature_sha256':'b'*64})
            with patch.object(X,'admit',return_value=(receipt,contract)):
                with self.assertRaisesRegex(ValueError,'file binding'):X.merge(root,Path('unused'),'raw')
            self.assertFalse((root/'models/final_tabular/raw/merged').exists())


if __name__=='__main__': unittest.main()

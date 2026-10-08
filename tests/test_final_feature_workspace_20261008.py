import datetime as dt
import importlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR', 'load_as_storage')
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def module():
    assert importlib.util.find_spec('final_feature_workspace_20261008'), 'feature workspace adapter is absent'
    return importlib.import_module('final_feature_workspace_20261008')


def rows(year, months, start, target):
    times = [dt.datetime(year, m, 15, 12, tzinfo=dt.timezone.utc) for m in months]
    return pl.DataFrame({
        'MVT_ID_mvt': [float(start+i) for i in range(len(times))],
        'PHASE_mvt': ['DEP']*len(times), 'ADEP_mvt': ['EDDF']*len(times),
        'MVT_TIME_UTC_mvt': times, 'AOBT_3_flt': times,
        'BLOCK_TIME_UTC_mvt': [None]*len(times),
        'TAXITIME_SEC_mvt': [target]*len(times),
    }, schema_overrides={'TAXITIME_SEC_mvt': pl.Int32,
                        'BLOCK_TIME_UTC_mvt': pl.Datetime('us', 'UTC')})


class FinalFeatureTests(unittest.TestCase):
    def fixture(self, root):
        tr=rows(2025, list(range(1,13)), 1, 600)
        rk=rows(2026, [1,2,6,7], 100, None)
        tp=rk.select('MVT_ID_mvt').with_columns(pl.lit(None,dtype=pl.Int32).alias('TAXITIME_SEC_mvt'))
        return tr,rk,tp

    def test_challenge_boundary_rejects_hidden_final_labels_and_overlap(self):
        w=module();tr,rk,tp=self.fixture(None)
        result=w.challenge_contract(tr,rk,tp,expected_counts=(12,4))
        self.assertEqual(result['train_rows'],12)
        self.assertEqual(result['rank_rows'],4)
        for bad in [rk.with_columns(pl.lit(1,dtype=pl.Int32).alias('TAXITIME_SEC_mvt')),
                    rk.with_columns(pl.col('MVT_TIME_UTC_mvt').alias('BLOCK_TIME_UTC_mvt')),
                    rk.with_columns(pl.Series('MVT_ID_mvt',[1.,101.,102.,103.]))]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                w.challenge_contract(tr,bad,tp,expected_counts=(12,4))

    def test_challenge_schema_and_months_are_not_silently_coerced(self):
        w=module();tr,rk,tp=self.fixture(None)
        with self.assertRaisesRegex(ValueError,'schema'):
            w.challenge_contract(tr,rk.with_columns(pl.col('MVT_ID_mvt').cast(pl.Int64)),tp,expected_counts=(12,4))
        with self.assertRaisesRegex(ValueError,'month|period'):
            w.challenge_contract(tr,rows(2026,[1,3,6,7],100,None),tp,expected_counts=(12,4))

    def test_external_missing_file_and_missing_day_fail_explicitly(self):
        w=module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(FileNotFoundError,'EDDF'):
                w.weather_coverage(root,{'EDDF':['2026-02-01']})
            for kind in ('metar','metar_wind'):
                p=root/'data'/kind/'EDDF.csv';p.parent.mkdir(parents=True)
                p.write_text('station,valid,sknt\nEDDF,2026-02-02 12:00,1\n')
            with self.assertRaisesRegex(ValueError,'2026-02-01'):
                w.weather_coverage(root,{'EDDF':['2026-02-01']})
            for kind in ('metar','metar_wind'):
                (root/'data'/kind/'EDDF.csv').write_text('station,valid,sknt\nEDDF,2026-02-01 12:00,1\n')
            self.assertEqual(w.weather_coverage(root,{'EDDF':['2026-02-01']})['EDDF']['metar']['missing_days'],[])

    def test_stage_never_overwrites_or_links_historical_inputs(self):
        w=module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'source';root.mkdir();p=root/'input';p.write_bytes(b'immutable original')
            entry=w.file_record(p,'data/prc-2026-datasets/ranking.parquet',root)
            spec={'files':[entry]};dest=Path(tmp)/'work'
            w.materialize(root,dest,spec,disk_floor=0)
            q=dest/entry['path'];self.assertFalse(q.is_symlink())
            self.assertEqual(q.read_bytes(),b'immutable original')
            self.assertEqual(q.stat().st_mode & 0o222,0)
            self.assertEqual(p.read_bytes(),b'immutable original')
            self.assertTrue(p.stat().st_mode & stat.S_IWUSR)
            with self.assertRaises(FileExistsError):w.materialize(root,dest,spec,disk_floor=0)

    def test_manifest_drift_duplicate_and_escape_fail_before_copy(self):
        w=module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'input';p.write_text('first')
            e=w.file_record(p,'data/input',root)
            for entries in [[e,e],[dict(e,path='../outside')],[dict(e,source='../secret')]]:
                with self.assertRaises(ValueError):w.materialize(root,root/'work',{'files':entries},disk_floor=0)
                self.assertFalse((root/'work').exists())
            p.write_text('second')
            with self.assertRaisesRegex(ValueError,'drift'):
                w.materialize(root,root/'work',{'files':[e]},disk_floor=0)
            self.assertFalse((root/'work').exists())

    def test_e106_coalesce_preserves_every_other_column_and_order(self):
        w=module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'data').mkdir()
            f=pl.DataFrame({'MVT_ID_mvt':[3.,1.,2.], 'CALLSIGN_flt':['BAW1',None,None],
                            'FLIGHT_mvt':['XYZ2','DLH34','YY98'], 'airline_prefix':[None]*3,
                            'keep':[9,8,7]})
            f.write_parquet(root/'data/features_e77.parquet')
            w.e106(root)
            got=pl.read_parquet(root/'data/features_e106.parquet')
            self.assertEqual(got['airline_prefix'].to_list(),['BAW','DLH',None])
            self.assertTrue(got.drop('airline_prefix').equals(f.drop('airline_prefix')))

    def test_output_schema_order_id_and_label_checks(self):
        w=module();tr,rk,tp=self.fixture(None)
        c=w.challenge_contract(tr,rk,tp,expected_counts=(12,4))
        f=pl.concat([tr.with_columns(split=pl.lit('train')),rk.with_columns(split=pl.lit('rank'))])
        schema=w.schema_record(f.schema)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'out.parquet';f.write_parquet(p)
            self.assertEqual(w.check_output(p,schema,c)['rows'],16)
            f.select(list(reversed(f.columns))).write_parquet(p)
            with self.assertRaisesRegex(ValueError,'schema'):w.check_output(p,schema,c)
            f.with_columns(pl.when(pl.col('split')=='rank').then(5).otherwise(pl.col('TAXITIME_SEC_mvt')).cast(pl.Int32).alias('TAXITIME_SEC_mvt')).write_parquet(p)
            with self.assertRaisesRegex(ValueError,'label'):w.check_output(p,schema,c)
            f.with_columns(pl.when(pl.col('split')=='train').then(601).otherwise(pl.col('TAXITIME_SEC_mvt')).cast(pl.Int32).alias('TAXITIME_SEC_mvt')).write_parquet(p)
            with self.assertRaisesRegex(ValueError,'training label'):w.check_output(p,schema,c)

    def test_atfm_requires_every_calendar_day_but_not_every_destination(self):
        w=module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in w.external_paths():
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('static')
            for ap in w.AP:
                (root/f'analysis/sources/samples/osm_overpass/{ap}.json').write_text('{"elements":[{"id":1}]}')
                (root/f'data/metar/{ap}.csv').write_text('station,valid,tmpc,dwpc,sknt,gust,vsby,skyl1,wxcodes\n'+f'{ap},2026-02-01 12:00,0,0,0,0,0,0,\n')
                (root/f'data/metar_wind/{ap}.csv').write_text('station,valid,drct,sknt\n'+f'{ap},2026-02-01 12:00,0,0\n')
            pl.DataFrame({'ADEP_mvt':['EDDF'],'stand_key':['A1'],'RUNWAY_mvt':['01'],
                          'route_m':[1.],'route_detour':[1.]}).write_parquet(root/'data/taxi_route.parquet')
            (root/'data/ext/openflights_airlines.dat').write_text('1,Test,Alias,BA,BAW,Call,UK,Y\n')
            cols='APT_ICAO,FLT_DATE,FLT_ARR_1,DLY_APT_ARR_1,FLT_ARR_1_DLY,DLY_APT_ARR_W_1,DLY_APT_ARR_C_1,DLY_APT_ARR_G_1\n'
            for year in (2024,2025,2026):
                (root/f'data/ext_pru/apt_dly_{year}.csv').write_text(cols+f'EDDF,{year}-01-01,1,0,0,0,0,0\n')
            c={'weather_days':{},'mvt_days':['2026-02-01']}
            with self.assertRaisesRegex(ValueError,'ATFM.*2026-02-01'):w.external_coverage(root,c)
            (root/'data/ext_pru/apt_dly_2026.csv').write_text(cols+'EDDF,2026-02-01,1,0,0,0,0,0\n')
            self.assertEqual(w.external_coverage(root,c)['atfm_missing_days'],[])
            (root/'data/metar/EDDF.csv').write_text('station,valid\nEDDF,2026-02-01 12:00\n')
            with self.assertRaisesRegex(ValueError,'weather schema.*EDDF'):
                w.external_coverage(root,c)

    def test_runner_complete_dependency_order_with_real_subprocesses(self):
        w=module()
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)
            for d in ('src','data','logs'):(work/d).mkdir()
            # The literal source/output contract is independent of STEPS.
            plans=[('features.py',None,'features_final'),
                   ('e28_forward_neighbours.py','features_final','features_e28'),
                   ('e77_filed_twins.py','features_e28','features_e77'),
                   ('e113_prefix_lookup.py','features_e106','features_e113'),
                   ('e163_build_features.py','features_e113','features_e163'),
                   ('e51_add_route_features.py','features_e163','features_e106_e51'),
                   ('e157_wake_features.py','features_e106_e51','features_e157'),
                   ('e165_build_features.py','features_e157','features_e165'),
                   ('build_final_table.py','features_e165','features_fin1b'),
                   ('e185_motion_features.py','features_fin1b','e185_motion_features')]
            for name,prev,out in plans:
                body="import polars as pl\nfrom pathlib import Path\nimport os,sys\n"
                if prev:body+=f"f=pl.read_parquet('data/{prev}.parquet')\n"
                else:body+="f=pl.DataFrame({'CALLSIGN_flt':['BAW1'],'FLIGHT_mvt':['AAA2'],'airline_prefix':[None]})\nassert os.environ['X_FEATS']=='0'\n"
                if name=='build_final_table.py':body+="assert sys.argv[1:]==['data/features_fin1b.parquet','data/features_e163.parquet','data/features_e106_e51.parquet','data/features_e157.parquet','data/features_e165.parquet']\n"
                body+=f"f.write_parquet('data/{out}.parquet')\n"
                (work/'src'/name).write_text(body)
            w.execute_steps(work,sys.executable,disk_floor=0)
            self.assertEqual(pl.read_parquet(work/'data/e185_motion_features.parquet')['airline_prefix'].to_list(),['BAW'])
            self.assertEqual(len(list((work/'logs').glob('*.json'))),11)

    def test_ordered_runner_stops_before_downstream_on_producer_failure(self):
        w=module()
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);(work/'logs').mkdir();(work/'src').mkdir();(work/'data').mkdir()
            # Real subprocesses: downstream sees its predecessor's artifact,
            # and deliberately fails. No stage after that failure may execute.
            (work/'src/features.py').write_text("from pathlib import Path\nPath('data/features_final.parquet').write_text('first')\n")
            (work/'src/e28_forward_neighbours.py').write_text("from pathlib import Path\nassert Path('data/features_final.parquet').read_text()=='first'\nraise RuntimeError('deliberate second-stage failure')\n")
            with self.assertRaises(Exception):w.execute_steps(work,sys.executable,disk_floor=0)
            self.assertEqual(sorted(p.name for p in (work/'logs').glob('*.log')),
                             ['00_features.log','01_e28.log'])
            self.assertFalse((work/'data/features_e77.parquet').exists())

    def test_verify_requires_complete_inventory_and_rederived_identity(self):
        w=module();tr,rk,tp=self.fixture(None)
        contract=w.challenge_contract(tr,rk,tp,expected_counts=(12,4))
        f=pl.concat([tr.with_columns(split=pl.lit('train')),rk.with_columns(split=pl.lit('rank'))])
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);(work/'data').mkdir()
            for name in w.FINAL_OUTPUTS:f.write_parquet(work/name)
            spec={'protocol':w.PROTOCOL,'registration_commit':'a'*40,'contract':contract,
                  'expected_output_schemas':{n:w.schema_record(f.schema) for n in w.FINAL_OUTPUTS},
                  'files':[{'path':w.SELF,'sha256':w.sha(w.__file__)},
                           {'path':w.RAW+'submitting.parquet','sha256':'b'*64}]}
            w.save(work/'manifest.json',spec);manifest_sha=w.sha(work/'manifest.json')
            outputs={n:w.check_output(work/n,spec['expected_output_schemas'][n],contract) for n in w.FINAL_OUTPUTS}
            fin=outputs['data/features_fin1b.parquet']
            valid={'protocol':w.PROTOCOL,'registration_commit':'a'*40,'manifest_sha256':manifest_sha,
                   'outputs':outputs,'feature_path':'data/features_fin1b.parquet','feature_sha256':fin['sha256'],
                   'train_rows':12,'rank_rows':4,'train_ordered_id_sha256':fin['train_ordered_id_sha256'],
                   'rank_ordered_id_sha256':fin['rank_ordered_id_sha256'],'template_sha256':'b'*64,
                   'source_sha256':{w.SELF:w.sha(w.__file__)},'fits_performed':0,'seconds':0.1}
            argv=['feature','verify','--work',str(work),'--manifest',str(work/'manifest.json'),
                  '--manifest-sha256',manifest_sha]
            bads=[{'manifest_sha256':manifest_sha,'outputs':{}},
                  *[dict(valid,**{key:value}) for key,value in [('protocol','bad'),('registration_commit','c'*40),
                    ('feature_path','elsewhere'),('feature_sha256','c'*64),('train_rows',11),('rank_rows',3),
                    ('train_ordered_id_sha256','c'*64),('rank_ordered_id_sha256','c'*64),
                    ('template_sha256','c'*64),('source_sha256',{}),('fits_performed',1)]],
                  dict(valid,outputs={k:v for k,v in outputs.items() if k!=w.FINAL_OUTPUTS[0]})]
            with patch.object(w,'resource_guard',return_value={}),patch.object(w,'checked_manifest',return_value=spec),\
                 patch.object(w,'verify_files'),patch.object(sys,'argv',argv):
                for bad in bads:
                    (work/'feature_complete.json').write_text(json.dumps(bad))
                    with self.subTest(bad=bad),self.assertRaises(ValueError):w.main()
                (work/'feature_complete.json').write_text(json.dumps(valid));w.main()

    def test_seal_requires_journal_protocol_before_metadata(self):
        w=module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in (*w.SOURCES,w.SELF,w.DOC,w.TEST,w.SOURCE_CONTRACT):
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('exact source')
            journal=b'unrelated registration only'
            def git(args,**kwargs):
                name=args[-1].split(':',1)[1]
                return journal if name=='HYPOTHESES_LOG.md' else (root/name).read_bytes()
            with patch.object(w.subprocess,'check_output',side_effect=git),patch.object(w.pl,'read_parquet_schema') as read:
                with self.assertRaisesRegex(ValueError,'journal|preregistration'):
                    w.seal(root,root/'manifest.json','a'*40)
                read.assert_not_called()
                journal=w.PROTOCOL.encode();w.registered(root,'a'*40)


if __name__=='__main__':unittest.main()

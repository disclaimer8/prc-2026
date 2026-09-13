"""Match OSM aeroway=parking_position/gate refs to STAND_mvt; runway thresholds; prototype distance feature.
OSM data (c) OpenStreetMap contributors, ODbL 1.0. Read-only on data/."""
import json, re, glob, math, pathlib, collections
import pandas as pd, numpy as np
from scipy.stats import spearmanr
H = pathlib.Path.home()/'prc-2026'
OSM = H/'analysis/sources/samples/osm_overpass'
APTS = ['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']

def norm(s):
    s = str(s).upper().replace(' ', '').replace('-', '')
    m = re.fullmatch(r'([A-Z]*)0*(\d+)([A-Z]*)', s)
    return f'{m.group(1)}{int(m.group(2))}{m.group(3)}' if m else s
def norm_nosfx(s):
    s = norm(s); m = re.fullmatch(r'([A-Z]*\d+)[LRCABM]?', s)
    return m.group(1) if m else s

def center(e):
    if e['type'] == 'node': return e['lat'], e['lon']
    g = e.get('geometry') or []
    if not g:
        b = e.get('bounds');  return ((b['minlat']+b['maxlat'])/2, (b['minlon']+b['maxlon'])/2) if b else None
    return float(np.mean([p['lat'] for p in g])), float(np.mean([p['lon'] for p in g]))
def hav(a, b):
    R = 6371000; la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return 2*R*math.asin(math.sqrt(h))
def bearing(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    y = math.sin(lo2-lo1)*math.cos(la2); x = math.cos(la1)*math.sin(la2)-math.sin(la1)*math.cos(la2)*math.cos(lo2-lo1)
    return (math.degrees(math.atan2(y, x))+360) % 360

def load_osm(a):
    j = json.load(open(OSM/f'{a}.json'))
    stands = {'parking_position': {}, 'gate': {}}
    thr = {}; rwpts = {}
    for e in j['elements']:
        t = e.get('tags', {}); aw = t.get('aeroway')
        if aw in stands:
            c = center(e)
            if not c: continue
            for key in ('ref', 'name'):
                if t.get(key):
                    for v in [t[key]] + re.split(r'[;,]', t[key]) + re.split(r'[;,\s]+', t[key]):
                        v = v.strip()
                        if v: stands[aw].setdefault(key, {}).setdefault(v, c)
        elif aw == 'runway' and t.get('ref') and e['type'] == 'way' and e.get('geometry'):
            for rid in t['ref'].split('/'):
                rid = rid.strip()
                if re.match(r'\d+', rid): rwpts.setdefault(rid, []).extend((p['lat'], p['lon']) for p in e['geometry'])
    # runway may be split into several ways: departure start = extreme point opposite to heading
    for rid, pts in rwpts.items():
        hd = math.radians(int(re.match(r'(\d+)', rid).group(1))*10)
        lat0 = np.mean([p[0] for p in pts])
        proj = [p[0]*math.cos(hd) + p[1]*math.cos(math.radians(lat0))*math.sin(hd) for p in pts]
        thr[rid] = pts[int(np.argmin(proj))]
    return j['osm3s'].get('timestamp_osm_base'), stands, thr

cols = ['ADEP_mvt','ADES_mvt','PHASE_mvt','STAND_mvt','RUNWAY_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt']
tr = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(H/'data/prc-2026-datasets/training_2025-*.parquet')))])
rk = pd.read_parquet(H/'data/prc-2026-datasets/ranking.parquet')
rk_cols = [c for c in cols if c in rk.columns]; rk = rk[rk_cols]
for df in (tr, rk):
    df['APT'] = np.where(df.PHASE_mvt == 'DEP', df.ADEP_mvt, df.ADES_mvt)

rows = []; feat_rows = []
for a in APTS:
    ts, stands, thr = load_osm(a)
    g = tr[tr.APT == a]; vc = g.STAND_mvt.dropna().astype(str).value_counts()
    res = {'apt': a, 'osm_ts': ts, 'n_stands25': len(vc),
           'osm_pp_ref': len(stands['parking_position'].get('ref', {})), 'osm_gate_ref': len(stands['gate'].get('ref', {}))}
    def lookup_table(fn, sources):
        d = {}
        for aw, key in sources:
            for v, c in stands[aw].get(key, {}).items(): d.setdefault(fn(v), c)
        return d
    variants = {
        'pp_ref_exact': (lambda s: str(s).upper(), [('parking_position','ref')]),
        'pp_ref_norm': (norm, [('parking_position','ref')]),
        'pp+gate_ref_norm': (norm, [('parking_position','ref'), ('gate','ref')]),
        'pp+gate_ref+name_norm': (norm, [('parking_position','ref'), ('gate','ref'), ('parking_position','name'), ('gate','name')]),
        'all_norm_nosfx': (norm_nosfx, [('parking_position','ref'), ('gate','ref'), ('parking_position','name'), ('gate','name')]),
    }
    best = None
    for name, (fn, src) in variants.items():
        tbl = lookup_table(fn, src)
        hit = np.array([fn(s) in tbl for s in vc.index])
        res[f'{name}_uniq'] = round(hit.mean(), 3); res[f'{name}_mvts'] = round(vc[hit].sum()/vc.sum(), 3)
        if name == 'pp+gate_ref+name_norm': best = (fn, tbl)
    # 2026 stands absent from 2025
    r26 = rk[rk.APT == a].STAND_mvt.dropna().astype(str)
    new = set(r26.unique()) - set(vc.index)
    res['n_stands26'] = r26.nunique(); res['new26_uniq'] = len(new); res['new26_mvts_share'] = round(r26.isin(new).mean(), 4)
    fn, tbl = best
    res['r26_osm_mvts'] = round(r26.map(lambda s: fn(s) in tbl).mean(), 3)
    # runways
    rw = g[g.PHASE_mvt == 'DEP'].RUNWAY_mvt.dropna().astype(str).value_counts()
    res['rwy_ids_osm'] = ','.join(sorted(thr)); res['dep_rwy_match_mvts'] = round(rw[[r in thr for r in rw.index]].sum()/rw.sum(), 3)
    # prototype feature: stand -> departure threshold distance, Jan+Jul 2025 DEP
    dep = g[(g.PHASE_mvt == 'DEP') & g.MVT_TIME_UTC_mvt.dt.month.isin([1, 7])].copy()
    dep['taxi'] = (dep.MVT_TIME_UTC_mvt - dep.BLOCK_TIME_UTC_mvt).dt.total_seconds()
    dep = dep[(dep.taxi >= 60) & (dep.taxi <= 14400)]
    sc = dep.STAND_mvt.astype(str).map(lambda s: tbl.get(fn(s)))
    tc = dep.RUNWAY_mvt.astype(str).map(thr.get)
    ok = sc.notna() & tc.notna()
    dist = [hav(s, t) for s, t in zip(sc[ok], tc[ok])]
    dep = dep[ok].assign(dist=dist)
    res['feat_cov_dep'] = round(ok.mean(), 3)
    if len(dep) > 100:
        res['rho_dist_taxi'] = round(spearmanr(dep.dist, dep.taxi).statistic, 3)
        res['rho_within_rwy_mean'] = round(np.nanmean([spearmanr(x.dist, x.taxi).statistic for _, x in dep.groupby('RUNWAY_mvt') if len(x) > 500]), 3)
        res['dist_med_m'] = int(dep.dist.median())
    rows.append(res)
out = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50)
print(out.T.to_string())
out.to_csv(H/'analysis/sources/samples/osm_overpass/match_summary.csv', index=False)

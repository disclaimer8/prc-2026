"""Download OSM aeroway features for 10 PRC airports via Overpass API.
Data (c) OpenStreetMap contributors, ODbL 1.0."""
import csv, json, sys, time, urllib.request, urllib.parse, pathlib
APTS = ['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']
ROOT = pathlib.Path.home()/'prc-2026/analysis'
OUT = ROOT/'sources/samples/osm_overpass'; OUT.mkdir(parents=True, exist_ok=True)
ENDPOINTS = ['https://overpass-api.de/api/interpreter','https://overpass.kumi.systems/api/interpreter','https://maps.mail.ru/osm/tools/overpass/api/interpreter']
coords = {}
with open(ROOT/'traffic/ext/airports.csv', newline='', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        if r['ident'] in APTS: coords[r['ident']] = (float(r['latitude_deg']), float(r['longitude_deg']))
D = 0.05
DBIG = {'EHAM': 0.08}   # Polderbaan 18R/36L lies outside +-0.05 deg
import os
DATE = os.environ.get('OSM_DATE')   # e.g. 2026-01-01T00:00:00Z for attic query
if len(sys.argv) > 1: APTS = sys.argv[1:]
for a in APTS:
    lat, lon = coords[a]
    d = DBIG.get(a, D)
    bbox = f'{lat-d},{lon-d},{lat+d},{lon+d}'
    hdr = f'[out:json][timeout:180]' + (f'[date:"{DATE}"]' if DATE else '') + ';'
    q = hdr + f'(nwr["aeroway"~"^(parking_position|gate|holding_position|runway|taxiway|stopway|apron)$"]({bbox}););out body geom;'
    for ep in ENDPOINTS:
        try:
            req = urllib.request.Request(ep, data=urllib.parse.urlencode({'data': q}).encode(), headers={'User-Agent': 'prc-2026-research (non-commercial)'})
            js = urllib.request.urlopen(req, timeout=240).read()
            json.loads(js)
            (OUT/f'{a}{"_"+DATE[:10] if DATE else ""}.json').write_bytes(js)
            print(a, ep, len(js)); break
        except Exception as e:
            print(a, ep, 'ERR', e); time.sleep(5)
    time.sleep(3)

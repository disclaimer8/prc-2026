"""Count cloud-group CB omitted by the current wx_thunder expression.

Metadata only: no challenge flights, targets, model features or predictions.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

AIRPORTS = 'EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF LTFM LSZH'.split()
CLOUD_CB = re.compile(r'\b(?:FEW|SCT|BKN|OVC)\d{3}CB\b')
CURRENT = re.compile(r'TS|CB')


def inventory(root):
    rows = []
    sources = {}
    for airport in AIRPORTS:
        path = root / 'data/metar' / (airport + '.csv')
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        counts = {period: dict(reports=0, cloud_cb=0, cloud_cb_without_current_flag=0)
                  for period in ('all_local', 'training_2025', 'ranking_jan_jul_2026')}
        with path.open(newline='') as source:
            for row in csv.DictReader(source):
                valid = row['valid']
                groups = ['all_local']
                if valid.startswith('2025-'):
                    groups.append('training_2025')
                elif valid.startswith(('2026-01-', '2026-07-')):
                    groups.append('ranking_jan_jul_2026')
                cb = bool(CLOUD_CB.search(row.get('metar') or ''))
                current = bool(CURRENT.search(row.get('wxcodes') or ''))
                for group in groups:
                    counts[group]['reports'] += 1
                    counts[group]['cloud_cb'] += int(cb)
                    counts[group]['cloud_cb_without_current_flag'] += int(cb and not current)
        rows.extend(dict(airport=airport, period=period, **value) for period, value in counts.items())
    totals = {period: {name: sum(r[name] for r in rows if r['period'] == period)
                      for name in ('reports', 'cloud_cb', 'cloud_cb_without_current_flag')}
              for period in counts}
    return dict(source_sha256=sources, counts=rows, totals=totals,
                cloud_regex=CLOUD_CB.pattern, current_regex=CURRENT.pattern,
                meaning='Raw METAR records, including source duplicates if present; not joined flights or measured model gain.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(inventory(args.root), indent=2))

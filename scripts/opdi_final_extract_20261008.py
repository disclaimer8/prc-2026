"""OPDI v0.0.2 flight events for the two final months missing locally (Feb 2026: windows 20260209_20260219,
20260219_20260301; Jun 2026: 20260530_20260609, 20260609_20260619, 20260619_20260629), extracted with the
same filter as `scripts/get_opdi_events.sh` (journal 168) and reduced to what the LSZH stage (E159b) reads.

Runs on prc-cloud (network + disk), two Polars threads, nothing of the competition data is read:

    POLARS_MAX_THREADS=2 .venv/bin/python -I opdi_final_extract_20261008.py --out /root/opdi_final_20261008 \
        --windows 20260209_20260219 20260219_20260301 20260530_20260609 20260609_20260619 20260619_20260629 \
        --control-window 20260130_20260209 --control-sha b6e13779cc94e5a0b9cd6a050bb801bbd3a8dd7bbde15f27ade797ed93058065

Outputs under --out: `flight_list/` (monthly FLIGHT lists, intact), `raw/` (original 10-day EVENT files, intact),
`full/ev_<w>.parquet` (10 airports, all event types: the exact get_opdi_events.sh product), `lszh/ev_lszh_<w>.parquet`
(ADEP LSZH, the five event types E159b reads: the file copied to the Mac), `control/` (the already-held window
re-extracted with the 14 original flight lists, to prove the filter reproduces the existing extract) and
`manifest.json` with SHA-256, sizes and row counts of everything.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import polars as pl

BASE = 'https://www.eurocontrol.int/performance/data/download/OPDI/v002'
APS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']   # get_opdi_events.sh
KEEP = ['flight_id', 'icao24', 'flt_id', 'adep', 'ades', 'type', 't', 'longitude', 'latitude', 'altitude', 'source', 'version']
E159B_TYPES = ['exit-parking_position', 'entry-taxiway', 'take-off', 'first_seen', 'first-xing-fl50']  # src/e159_opdi_stage_cv.py TYPES
ORIGINAL_MONTHS = [f'2025{m:02d}' for m in range(1, 13)] + ['202601', '202607']                      # lists held on the Mac on 30.09
NEW_MONTHS = ['202602', '202603', '202605', '202606']


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def content_sha(df: pl.DataFrame) -> str:
    """Order-independent hash of the rows (parquet bytes depend on the writer)."""
    d = df.with_columns(pl.col(c).dt.strftime('%Y-%m-%dT%H:%M:%S%.6f') for c, t in df.schema.items() if isinstance(t, pl.Datetime))
    return hashlib.sha256(d.sort(d.columns).write_csv().encode()).hexdigest()


def download(url: str, dest: Path, log) -> dict:
    if dest.is_file() and dest.stat().st_size > 0:
        log(f'have {dest.name}')
    else:
        tmp = dest.with_suffix(dest.suffix + '.part')
        for attempt in range(1, 4):
            t0 = time.time()
            r = subprocess.run(['curl', '-sS', '--fail', '--retry', '3', '-A', 'Mozilla/5.0', '-o', str(tmp), url])
            if r.returncode == 0 and tmp.stat().st_size > 0:
                break
            log(f'download attempt {attempt} failed rc={r.returncode} {url}')
            time.sleep(30)
        else:
            raise RuntimeError('download failed: ' + url)
        os.replace(tmp, dest)
        log(f'got {dest.name} {dest.stat().st_size} bytes in {time.time() - t0:.0f} s')
    head = subprocess.run(['curl', '-sI', '-A', 'Mozilla/5.0', url], capture_output=True, text=True).stdout.lower()
    remote = next((int(l.split(':')[1]) for l in head.splitlines() if l.startswith('content-length:')), None)
    if remote is not None and remote != dest.stat().st_size:
        raise RuntimeError(f'{dest.name}: local {dest.stat().st_size} bytes, remote content-length {remote}')
    return dict(url=url, bytes=dest.stat().st_size, sha256=sha(dest), remote_content_length=remote)


def flight_index(list_dir: Path, months: list[str]) -> pl.DataFrame:
    """Exactly get_opdi_events.sh: all lists, id as string, first occurrence, our departures, stripped call sign."""
    fl = pl.concat([pl.read_parquet(list_dir / f'flight_list_{m}.parquet', columns=['id', 'icao24', 'flt_id', 'adep', 'ades'])
                    .with_columns(pl.col('id').cast(pl.String)) for m in sorted(months)]).unique('id')
    return fl.filter(pl.col('adep').is_in(APS)).with_columns(pl.col('flt_id').str.strip_chars())


def extract(raw: Path, fl: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    schema = pl.read_parquet_schema(raw)   # only the columns the product keeps (identical output, a third of the memory)
    ev = pl.read_parquet(raw, columns=[c for c in schema if c in KEEP or c == 'event_time']).with_columns(pl.col('flight_id').cast(pl.String))
    types = sorted(ev['type'].unique().to_list())
    ev = ev.join(fl, left_on='flight_id', right_on='id', how='inner')
    t = pl.col('event_time')
    ev = ev.with_columns(t=(t.str.to_datetime('%Y-%m-%d %H:%M:%S', time_zone='UTC', strict=False) if ev.schema['event_time'] == pl.String
                            else (t.dt.replace_time_zone('UTC') if ev.schema['event_time'].time_zone is None else t)))
    return ev.select([c for c in KEEP if c in ev.columns]), types


def describe(path: Path, df: pl.DataFrame) -> dict:
    ex = df.filter((pl.col('adep') == 'LSZH') & (pl.col('type') == 'exit-parking_position')).with_columns(day=pl.col('t').dt.date().cast(pl.String))
    return dict(bytes=path.stat().st_size, sha256=sha(path), content_sha256=content_sha(df), rows=df.height,
                flights=df['flight_id'].n_unique(), t_min=str(df['t'].min()), t_max=str(df['t'].max()),
                rows_by_adep=dict(sorted(df.group_by('adep').len().iter_rows())),
                rows_by_type=dict(sorted(df.group_by('type').len().iter_rows())),
                lszh_exit_parking_by_day=dict(sorted(ex.group_by('day').len().iter_rows())))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--windows', nargs='+', required=True)
    p.add_argument('--control-window')
    p.add_argument('--control-sha', help='SHA-256 of the existing Mac extract of the control window')
    p.add_argument('--control-content-sha', help='content hash of the existing Mac extract (same function as here)')
    a = p.parse_args()
    out = a.out
    for d in ('flight_list', 'raw', 'full', 'lszh', 'control'):
        (out / d).mkdir(parents=True, exist_ok=True)
    logf = open(out / 'extract.log', 'a')

    def log(msg):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}Z] {msg}'
        print(line, flush=True)
        logf.write(line + '\n'); logf.flush()

    man = dict(protocol='opdi-final-febjun2026-lszh-e159b-extract-20261008-v1', base_url=BASE, airports=APS, keep_columns=KEEP,
               e159b_types=E159B_TYPES, polars=pl.__version__, python=sys.version, threads=os.environ.get('POLARS_MAX_THREADS'),
               flight_lists={}, windows={}, control=None, started_unix=time.time())
    months = ORIGINAL_MONTHS + NEW_MONTHS
    for m in months:
        man['flight_lists'][m] = download(f'{BASE}/flight_list/flight_list_{m}.parquet', out / 'flight_list' / f'flight_list_{m}.parquet', log)
        man['flight_lists'][m]['rows'] = pl.read_parquet(out / 'flight_list' / f'flight_list_{m}.parquet', columns=['id']).height
    fl_all = flight_index(out / 'flight_list', months)
    log(f'flight index (all {len(months)} lists): {fl_all.height} departures of our airports')
    for w in a.windows:
        raw = out / 'raw' / f'flight_events_{w}.parquet'
        rec = dict(raw=download(f'{BASE}/flight_events/flight_events_{w}.parquet', raw, log), flight_lists=months)
        ev, rec['raw_types'] = extract(raw, fl_all)
        rec['raw_rows'] = pl.read_parquet(raw, columns=['flight_id']).height
        full = out / 'full' / f'ev_{w}.parquet'
        ev.write_parquet(full)
        rec['full'] = describe(full, ev)
        sub = ev.filter((pl.col('adep') == 'LSZH') & pl.col('type').is_in(E159B_TYPES))
        lszh = out / 'lszh' / f'ev_lszh_{w}.parquet'
        sub.write_parquet(lszh)
        rec['lszh'] = describe(lszh, sub)
        man['windows'][w] = rec
        log(f'{w}: raw {rec["raw_rows"]} rows -> full {ev.height} rows ({ev["flight_id"].n_unique()} flights) -> LSZH/E159b {sub.height} rows')
        json.dump(man, open(out / 'manifest.json', 'w'), indent=1)
    if a.control_window:
        w = a.control_window
        raw = out / 'raw' / f'flight_events_{w}.parquet'
        rec = dict(raw=download(f'{BASE}/flight_events/flight_events_{w}.parquet', raw, log), expected_sha256=a.control_sha,
                   expected_content_sha256=a.control_content_sha)
        ev14, _ = extract(raw, flight_index(out / 'flight_list', ORIGINAL_MONTHS))
        c14 = out / 'control' / f'ev_{w}.parquet'
        ev14.write_parquet(c14)
        rec['original_lists'] = describe(c14, ev14)
        rec['sha_matches_existing'] = (a.control_sha == rec['original_lists']['sha256']) if a.control_sha else None
        rec['content_matches_existing'] = (a.control_content_sha == rec['original_lists']['content_sha256']) if a.control_content_sha else None
        ev18, _ = extract(raw, fl_all)
        c18 = out / 'control' / f'ev_{w}_alllists.parquet'
        ev18.write_parquet(c18)
        rec['all_lists'] = describe(c18, ev18)
        rec['rows_gained_by_new_lists'] = ev18.height - ev14.height
        man['control'] = rec
        log(f'control {w}: 14 lists {ev14.height} rows sha {rec["original_lists"]["sha256"][:12]} (match {rec["sha_matches_existing"]}, '
            f'content match {rec["content_matches_existing"]}); 18 lists {ev18.height} rows (+{rec["rows_gained_by_new_lists"]})')
    man['finished_unix'] = time.time()
    man['complete'] = True
    json.dump(man, open(out / 'manifest.json', 'w'), indent=1)
    log('COMPLETE')


if __name__ == '__main__':
    main()

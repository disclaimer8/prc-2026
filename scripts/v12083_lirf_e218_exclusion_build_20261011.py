"""v12083: v12082 with the E218 PRU level shift removed on LIRF rows only (journal №460).

v12082 (231.1445) = the Jan/Jul rows of the E218 final bbae2cc5; v12080 (231.1805) = the Jan/Jul rows of the 09.10 final
76c32426, i.e. the same rows before the E218 stage. v12083 takes the v12080 value on every LIRF row (ADEP_mvt = LIRF in the
board ranking) and the v12082 value on every other row. It is one change against v12082 on fixed labels, so
v12083 - v12082 on the board is an exact measurement of the LIRF cells of E218 on Jan/Jul. It equals the Jan/Jul rows of
"bbae2cc5 with the LIRF rows of 76c32426", which `build` proves row by row. Zero fits, no labels, never uploads.

    .venv/bin/python scripts/v12083_lirf_e218_exclusion_build_20261011.py check    # everything except writing
    .venv/bin/python scripts/v12083_lirf_e218_exclusion_build_20261011.py build    # write-once candidate + build report
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('POLARS_MAX_THREADS', '4')
os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR', 'load_as_storage')
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SELF = 'scripts/v12083_lirf_e218_exclusion_build_20261011.py'
VERSION = 12083
PROTOCOL = 'v12083-v12082-minus-E218-on-LIRF-rows-zero-fit-20261011-v1'
CANDIDATE = f'submissions/gentle-igloo_v{VERSION}.parquet'
REPORT = f'submissions/gentle-igloo_v{VERSION}.manifest.json'
ID, Y = 'MVT_ID_mvt', 'TAXITIME_SEC_mvt'
EXCLUDE = 'LIRF'
PINS = {
    'data/prc-2026-datasets/submitting.parquet': '0d383408ed5ac27bfcfd12d6b961cee77c8a6364b10dad9fe3ec81115c79652b',
    'data/prc-2026-datasets/ranking.parquet': '9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86',
    'submissions/gentle-igloo_v12082.parquet': 'e87b79789ab342c6ad4ca6c64900f159ac7affd242eb284717ff321c122f5298',
    'submissions/gentle-igloo_v12080.parquet': 'ae1680cbab9b56a7748e7eac97afdf6359bbd38ad57545a763cdfa7868064b37',
    'data/final_tail_20261011/work/gentle-igloo_final.parquet': 'bbae2cc544989a0faf68abb85dbb4a9aab5ea38eec98d01fa9282c332b4234a1',
    'data/final_tail_20261009/work/gentle-igloo_final.parquet': '76c32426d750040690ff68bacd6ecbf3a7aa1e63340df5b9271a324b7064d554',
    'data/prc-2026-datasets/final_ranking.parquet': '4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f',
    'analysis/final_tail_20261011/pru_level_cells.csv': 'f1d8cb449f4582965c2d4853a10f8bad8356fdc49fb1802e85e899f671c8198f',
}
TEMPLATE, RANKING = 'data/prc-2026-datasets/submitting.parquet', 'data/prc-2026-datasets/ranking.parquet'
V82, V80 = 'submissions/gentle-igloo_v12082.parquet', 'submissions/gentle-igloo_v12080.parquet'
F_NEW, F_OLD = 'data/final_tail_20261011/work/gentle-igloo_final.parquet', 'data/final_tail_20261009/work/gentle-igloo_final.parquet'
CELLS, FRANK = 'analysis/final_tail_20261011/pru_level_cells.csv', 'data/prc-2026-datasets/final_ranking.parquet'
# The E218 stage on Jan/Jul rows of the final (journal №458, rows_changed_by_cell.csv): airport-month -> (shift, rows moved).
E218_JANJUL = {'EDDF-7': (1, 17534), 'EDDM-7': (-3, 13062), 'EGLL-1': (-1, 19005), 'EGLL-7': (-1, 20198), 'EHAM-7': (-3, 20350),
               'LEBL-7': (-4, 16766), 'LEMD-7': (-7, 16959), 'LFPG-7': (-5, 17492), 'LIRF-1': (9, 10549), 'LIRF-7': (11, 14468),
               'LSZH-7': (-3, 11948), 'LTFM-7': (-4, 23267)}
EXPECTED = dict(rows=344841, changed=25017, max_abs_diff=11, sum_sq=10549 * 81 + 14468 * 121)


def sha(path):
    h = hashlib.sha256()
    with open(ROOT / path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()


def stop(msg):
    raise SystemExit('STOP: ' + msg)


def moved_cells(frame, col):
    """{airport-month: (shift, rows)} of the nonzero entries of `col`; one shift per cell or STOP."""
    out = {}
    for r in frame.filter(pl.col(col) != 0).group_by('ADEP_mvt', 'month').agg(
            lo=pl.col(col).min(), hi=pl.col(col).max(), n=pl.len()).iter_rows(named=True):
        if r['lo'] != r['hi']:
            stop(f"{r['ADEP_mvt']}-{r['month']}: rows moved by different amounts")
        out[f"{r['ADEP_mvt']}-{r['month']}"] = (int(r['lo']), int(r['n']))
    return dict(sorted(out.items()))


def janjul(final, template):
    """Jan/Jul rows of a four-month final file in template order (= final_transport_20261009.derive_control)."""
    order = template.select(ID).with_row_index('_pos')
    out = order.join(final, on=ID, how='left').sort('_pos').select(template.columns)
    if out[Y].null_count():
        stop('template IDs missing from a final file')
    return out


def run(write):
    shas = {p: sha(p) for p in PINS}
    for p, h in PINS.items():
        if shas[p] != h:
            stop(f'pin mismatch {p}')
    if (ROOT / CANDIDATE).exists() or (ROOT / REPORT).exists():
        stop('v12083 already exists; write-once')
    template = pl.read_parquet(ROOT / TEMPLATE)
    v82, v80 = pl.read_parquet(ROOT / V82), pl.read_parquet(ROOT / V80)
    for name, f in (('v12082', v82), ('v12080', v80)):
        if f.schema != template.schema or not f[ID].equals(template[ID]) or f[Y].null_count():
            stop(f'{name}: schema/order differs from the template')
    rank = (pl.read_parquet(ROOT / RANKING, columns=[ID, 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt']).filter(pl.col('PHASE_mvt') == 'DEP')
            .select(pl.col(ID).cast(template.schema[ID]), 'ADEP_mvt', month=pl.col('MVT_TIME_UTC_mvt').dt.month().cast(pl.Int64)))
    x = template.select(ID).join(rank, on=ID, how='left', maintain_order='left')
    if x['ADEP_mvt'].null_count() or x.height != EXPECTED['rows'] or sorted(x['month'].unique().to_list()) != [1, 7]:
        stop('ranking does not cover the template exactly with Jan/Jul DEP rows')
    lirf = (x['ADEP_mvt'] == EXCLUDE).to_numpy()
    a82 = v82[Y].to_numpy().astype(np.int64)
    a80 = v80[Y].to_numpy().astype(np.int64)
    x = x.with_columns(d_e218=pl.Series(a82 - a80))
    # (1) v12082 - v12080 is exactly the E218 stage on Jan/Jul (12 cells, one integer shift each)
    got = moved_cells(x, 'd_e218')
    if got != E218_JANJUL:
        stop(f'v12082 - v12080 differs from the E218 Jan/Jul cells: {got}')
    cells = pl.read_csv(ROOT / CELLS)
    for key, (shift, _) in E218_JANJUL.items():
        ap, m = key.split('-')
        c = cells.filter((pl.col('ADEP_mvt') == ap) & (pl.col('month') == int(m)))['c'].item()
        if int(np.rint(c)) != shift:
            stop(f'{key}: shift differs from rint(c) of the E218 cell table')
    # (2) both control files are the Jan/Jul rows of their final files
    f_new, f_old = pl.read_parquet(ROOT / F_NEW), pl.read_parquet(ROOT / F_OLD)
    if not janjul(f_new, template).equals(v82) or not janjul(f_old, template).equals(v80):
        stop('control files are not the Jan/Jul rows of the two finals')
    # (3) the single change: LIRF rows take v12080, every other row keeps v12082
    a83 = np.where(lirf, a80, a82)
    cand = template.select(ID).with_columns(pl.Series(Y, a83).cast(template.schema[Y]))
    delta = a83 - a82
    x = x.with_columns(d_v83=pl.Series(delta))
    moved = moved_cells(x, 'd_v83')
    want = {k: (-s, n) for k, (s, n) in E218_JANJUL.items() if k.startswith(EXCLUDE + '-')}
    if moved != want:
        stop(f'v12083 - v12082 differs from minus the LIRF E218 cells: {moved}')
    if not (np.array_equal(a83[lirf], a80[lirf]) and np.array_equal(a83[~lirf], a82[~lirf])):
        stop('LIRF rows must equal v12080 and all other rows v12082')
    changed, sum_sq, max_abs = int(np.count_nonzero(delta)), int((delta ** 2).sum()), int(np.abs(delta).max())
    if (changed, sum_sq, max_abs) != (EXPECTED['changed'], EXPECTED['sum_sq'], EXPECTED['max_abs_diff']):
        stop('difference against v12082 differs from the registered single change')
    # (4) row-by-row: v12083 = Jan/Jul rows of (bbae2cc5 with the LIRF rows of 76c32426)
    frank = (pl.read_parquet(ROOT / FRANK, columns=[ID, 'PHASE_mvt', 'ADEP_mvt'])
             .filter(pl.col('PHASE_mvt') == 'DEP').select(pl.col(ID).cast(template.schema[ID]), 'ADEP_mvt'))
    fj = f_new.join(frank, on=ID, how='left', maintain_order='left').join(f_old.rename({Y: 'old'}), on=ID, how='left', maintain_order='left')
    if fj['ADEP_mvt'].null_count() or fj['old'].null_count() or fj.height != f_new.height:
        stop('final ranking/old final do not cover the E218 final')
    hybrid = fj.select(ID, pl.when(pl.col('ADEP_mvt') == EXCLUDE).then(pl.col('old')).otherwise(pl.col(Y)).cast(f_new.schema[Y]).alias(Y))
    if not janjul(hybrid, template).equals(cand):
        stop('v12083 is not the Jan/Jul rows of bbae2cc5 with LIRF rows from 76c32426')
    hyb = hybrid[Y].to_numpy().astype(np.int64) - f_new[Y].to_numpy().astype(np.int64)
    rms = float(np.sqrt(sum_sq / EXPECTED['rows']))
    report = dict(protocol=PROTOCOL, version=VERSION, candidate=CANDIDATE, builder=SELF, builder_sha256=sha(SELF),
                  inputs={p: shas[p] for p in PINS}, rows=EXPECTED['rows'], excluded_airport=EXCLUDE, fits_performed=0,
                  labels_read=False, uploaded=False, parent='submissions/gentle-igloo_v12082.parquet', parent_sha256=PINS[V82],
                  lirf_source='submissions/gentle-igloo_v12080.parquet', lirf_source_sha256=PINS[V80],
                  e218_janjul_cells={k: list(v) for k, v in E218_JANJUL.items()},
                  change=dict(changed=changed, sum_sq=sum_sq, rms_vs_v12082=rms, max_abs_diff=max_abs,
                              cells={k: list(v) for k, v in moved.items()}, lirf_rows=int(lirf.sum()),
                              lirf_rows_equal_v12080=True, other_rows_equal_v12082=True),
                  final_hybrid=dict(rows=int(fj.height), lirf_rows=int((fj['ADEP_mvt'] == EXCLUDE).sum()),
                                    changed_vs_bbae2cc5=int(np.count_nonzero(hyb)), janjul_equals_candidate=True),
                  unix=time.time())
    if write:
        tmp = ROOT / (CANDIDATE + '.tmp')
        cand.write_parquet(tmp)
        os.link(tmp, ROOT / CANDIDATE)
        tmp.unlink()
        if not pl.read_parquet(ROOT / CANDIDATE).equals(cand):
            stop('written candidate does not read back identically')
        report['sha256'] = sha(CANDIDATE)
        report['size_bytes'] = (ROOT / CANDIDATE).stat().st_size
        with open(ROOT / REPORT, 'x') as f:
            json.dump(report, f, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in report.items() if k != 'inputs'}, indent=1, sort_keys=True))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['check', 'build'])
    run(p.parse_args().action == 'build')


if __name__ == '__main__':
    main()

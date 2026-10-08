"""Breakthrough-review checks, journal №418 (three label-light checks, no model training).

(A) Exact-SSE low-parameter gate calibration budget at the premix level.
    Data: data/cv_preds_fin1b_hz/fold_k_{k+6}.npz (2025 production OOF; keys mvt_id, pred, reg, y,
    proba [N,6], values [N,6] with NaN = infeasible candidate). Regime order (src/train.py):
    0 regular, 1 schedule, 2 IOBT, 3 LOBT, 4 EOBT, 5 day shift (regular + 86400).
    Identity check: pred == sum_k p_k v_k (NaN -> 0); reported, not assumed.
    Perturbation p_i(theta) = p_i + B_i theta with 1^T B_i = 0, hence m_i(theta) = m_i + D_i theta is
    linear in theta and SSE(theta) - SSE0 = theta^T Q theta - 2 c^T theta exactly, Q = D^T D, c = D^T e,
    e = y - m. Fixed basis, J = 11 columns (registered before any outcome was computed):
      g_i = p_0 (v_0 - m_i)                      = d m / d z_0 of the softmax (shift of the regular logit),
            B = p_0 * (1 - p_0, -p_1, ..., -p_5)  (sums to zero)
      cols 0..3  g_i * 1[disagreement bucket b], buckets of Dis_i = sum_{k>=1} p_k |v_k - v_0| / (1 - p_0):
                 [0,600) [600,1800) [1800,3600) [3600,inf); rows with p_0 == 1 go to bucket 0
      col  4     g_i * 1[NM record missing]  (FLIGHT_ID_mvt is null in features_fin1b)
      col  5     g_i * 1[ADEP == LIRF]
      cols 6..10 h_ik = p_0 p_k (v_k - v_0), k = 1..5  (pairwise logit shift regular -> k; NaN -> 0)
    Ridge fixed in advance: lambda_main = (240 s / 0.1)^2 = 5.76e6 (Gaussian prior sd 0.1 on the logit
    shifts, noise sd 240 s); sensitivity lambda = 0 (pinv, rcond 1e-10) and 100 * lambda_main.
    Budgets (all pooled, premix level, dRMSE = sqrt((SSE0 + dSSE)/N) - sqrt(SSE0/N) over the evaluated
    rows): in-sample (12 months), leave-one-month-pair-out (train 10 months, apply the held-out pair,
    pairs k & k+6), half-year forward (Jan-Jun -> Jul-Dec), expanding forward (1..k-1 -> k, k = 4..12),
    one-month-gap forward (1..k-2 -> k, k = 4..12). Main rows: every OOF row. Secondary: body y <= 7200,
    and "serviced" = not (LIRF and NM missing). The simplex is NOT enforced in the exact computation;
    a clipped variant (p + B theta clipped to [0,1] and renormalised over feasible candidates) is reported
    as a sensitivity only. The chain pass-through of a premix shift is ~0.41-0.47 (E212, journal 407a):
    premix seconds are an upper bound on leaderboard seconds, not a forecast.
    Stop rule (registered): any forward leg (main lambda) with pooled premix gain > -0.05 s, or unstable
    sign (half-year and gap legs disagree, or < 6/9 months improved in the expanding leg) -> close.

(B) D-ATIS closures from the already-downloaded E208 texts only
    (analysis/e208_datis_20261008/execution/assets.jsonl, field records[].closures). Event = bulletins of
    one airport with the same set of closed objects whose validity windows (issued_at .. expires_at, the
    protocol's 60-minute freshness) overlap or are separated by <= 60 min; counted per hub and per day.
    2025 training days available in E208: 2025-07-01/08/15/22 (the four January 2025 dates were HTTP 404).
    Exposure (fixed rule): departures of training_2025-07 with ADEP == bulletin airport and
    issued_at <= MVT_TIME_UTC < expires_at, unique MVT_ID over all closure bulletins; also a stricter
    supersession variant where validity ends at the next accepted bulletin of the same airport and role.
    Stop rule (registered): < 10 independent closure events on 2025 training days or < 2 hubs -> close.
    Note: the raw record list (16 closure-bearing bulletins) was displayed while inspecting the file
    format before registration; the event/exposure counts were not computed before registration.

(C) Context count for the Schiphol taxiway A/B item: EHAM departures of final_ranking.parquet by month
    (Jan/Feb/Jun/Jul 2026) and the June rows with MVT <= 2026-06-26 (announced works 04.05-26.06.2026).
    Licence reading of the Schiphol pages is done with WebFetch outside this script.

Mac only, .venv/bin/python, <= 4 threads, read-only on data/. Output:
analysis/breakthrough_review_checks_20261008/result.json (+ run.log via the caller's redirection).
"""
from __future__ import annotations
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

for k, v in {'POLARS_MAX_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OMP_NUM_THREADS': '4',
             'POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR': 'load_as_storage'}.items():
    os.environ.setdefault(k, v)
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / 'data/cv_preds_fin1b_hz'
FEAT = ROOT / 'data/features_fin1b.parquet'
ASSETS = ROOT / 'analysis/e208_datis_20261008/execution/assets.jsonl'
TRAIN_JUL = ROOT / 'data/prc-2026-datasets/training_2025-07-01_2025-08-01.parquet'
FINAL = ROOT / 'data/prc-2026-datasets/final_ranking.parquet'
OUT = ROOT / 'analysis/breakthrough_review_checks_20261008'
PROTOCOL = 'breakthrough-review-checks-20261008-v1'
LAMBDA_MAIN = (240.0 / 0.1) ** 2
BUCKETS = np.array([600.0, 1800.0, 3600.0])
J = 11
MONTHS = list(range(1, 13))


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ----------------------------------------------------------------------------------------------- (A)
def load_oof():
    parts = []
    for k in range(1, 7):
        with np.load(NPZ / f'fold_{k}_{k + 6}.npz', allow_pickle=False) as z:
            d = {n: z[n] for n in ('mvt_id', 'pred', 'reg', 'y', 'proba', 'values')}
        d['fold'] = np.full(len(d['mvt_id']), k, np.int8)
        parts.append(d)
    out = {n: np.concatenate([p[n] for p in parts]) for n in parts[0]}
    ids = out['mvt_id']
    if len(np.unique(ids)) != len(ids):
        raise ValueError('duplicate mvt_id across folds')
    f = pl.read_parquet(FEAT, columns=['MVT_ID_mvt', 'split', 'month', 'ADEP_mvt', 'FLIGHT_ID_mvt', 'TAXITIME_SEC_mvt'])
    f = f.filter(pl.col('split') == 'train')
    key = pl.DataFrame({'MVT_ID_mvt': ids})
    m = key.join(f, on='MVT_ID_mvt', how='left', validate='1:1', maintain_order='left')
    if m['month'].null_count():
        raise ValueError('OOF ids not found in features_fin1b train split: %d' % m['month'].null_count())
    if not np.array_equal(m['TAXITIME_SEC_mvt'].to_numpy(), out['y']):
        raise ValueError('label mismatch between npz y and features')
    out['month'] = m['month'].to_numpy().astype(np.int8)
    out['lirf'] = (m['ADEP_mvt'] == 'LIRF').to_numpy()
    out['nm_missing'] = m['FLIGHT_ID_mvt'].is_null().to_numpy()
    exp_fold = ((out['month'] - 1) % 6) + 1
    if not np.array_equal(exp_fold, out['fold']):
        raise ValueError('fold/month layout differs from k & k+6')
    return out


def basis(o):
    p = o['proba'].astype(np.float64)
    v = o['values'].astype(np.float64)
    feas = np.isfinite(v)
    v0f = np.where(feas, v, 0.0)
    m_check = np.sum(p * v0f, axis=1)
    m = o['pred'].astype(np.float64)
    ident = dict(max_abs=float(np.max(np.abs(m_check - m))), rms=float(np.sqrt(np.mean((m_check - m) ** 2))),
                 rows_over_1s=int(np.sum(np.abs(m_check - m) > 1.0)), p_row_sum_max_dev=float(np.max(np.abs(p.sum(1) - 1))),
                 p_on_infeasible_max=float(np.max(np.where(feas, 0.0, p))), regular_always_feasible=bool(feas[:, 0].all()))
    p0 = p[:, 0]
    v0 = v[:, 0]
    g = p0 * (v0 - m)
    pn = 1.0 - p0
    dis = np.sum(p[:, 1:] * np.abs(v0f[:, 1:] - v0[:, None]) * feas[:, 1:], axis=1)
    dis = np.where(pn > 1e-9, dis / np.maximum(pn, 1e-9), 0.0)
    b = np.digitize(dis, BUCKETS)  # 0..3
    D = np.zeros((len(m), J))
    for j in range(4):
        D[:, j] = g * (b == j)
    D[:, 4] = g * o['nm_missing']
    D[:, 5] = g * o['lirf']
    for k in range(1, 6):
        D[:, 5 + k] = p0 * p[:, k] * np.where(feas[:, k], v0f[:, k] - v0, 0.0)
    e = o['y'].astype(np.float64) - m
    cols = ['g*bucket<600', 'g*bucket600-1800', 'g*bucket1800-3600', 'g*bucket>=3600', 'g*nm_missing', 'g*LIRF',
            'h_schedule', 'h_IOBT', 'h_LOBT', 'h_EOBT', 'h_dayshift']
    return D, e, ident, cols, dict(p=p, v=v0f, feas=feas, p0=p0, m=m, b=b, dis=dis)


def delta_p(theta, aux, o):
    """The probability perturbation B theta (N x 6); sums to zero per row."""
    p, p0, feas = aux['p'], aux['p0'], aux['feas']
    s = np.zeros(len(p0))
    for j in range(4):
        s += theta[j] * (aux['b'] == j)
    s += theta[4] * o['nm_missing'] + theta[5] * o['lirf']
    dp = -(s * p0)[:, None] * p
    dp[:, 0] += s * p0
    for k in range(1, 6):
        t = theta[5 + k] * p0 * p[:, k] * feas[:, k]
        dp[:, k] += t
        dp[:, 0] -= t
    return dp


def month_stats(D, e, month, masks):
    """Per month (and per secondary mask): Q, c, SSE0, N."""
    st = {}
    for name, mk in masks.items():
        st[name] = {}
        for mo in MONTHS:
            sel = (month == mo) & mk
            Dm = D[sel]
            em = e[sel]
            st[name][mo] = dict(Q=Dm.T @ Dm, c=Dm.T @ em, sse=float(em @ em), n=int(sel.sum()))
    return st


def agg(st, months):
    Q = sum(st[m]['Q'] for m in months)
    c = sum(st[m]['c'] for m in months)
    sse = sum(st[m]['sse'] for m in months)
    n = sum(st[m]['n'] for m in months)
    return Q, c, sse, n


def solve(Q, c, lam):
    if lam == 0:
        return np.linalg.pinv(Q, rcond=1e-10) @ c
    return np.linalg.solve(Q + lam * np.eye(len(c)), c)


def evaluate(theta, Q, c, sse, n):
    dsse = float(theta @ Q @ theta - 2 * c @ theta)
    r0 = np.sqrt(sse / n)
    r1 = np.sqrt(max(sse + dsse, 0.0) / n)
    return dict(dsse=dsse, rmse0=float(r0), rmse1=float(r1), drmse=float(r1 - r0), n=n)


def leg(st, train_months, eval_units, lam):
    """eval_units: list of month lists; theta fitted once on train_months, applied per unit and pooled."""
    Q, c, _, _ = agg(st, train_months)
    theta = solve(Q, c, lam)
    units = []
    for u in eval_units:
        Qe, ce, se, ne = agg(st, u)
        units.append(dict(months=u, **evaluate(theta, Qe, ce, se, ne)))
    Qe, ce, se, ne = agg(st, [m for u in eval_units for m in u])
    pooled = evaluate(theta, Qe, ce, se, ne)
    return dict(theta=theta.tolist(), train_months=train_months, units=units, pooled=pooled,
                units_improved=int(sum(x['dsse'] < 0 for x in units)), units_total=len(units))


def clipped_eval(theta, aux, o, D, e, month, eval_months, mask):
    """Actual SSE change when p + B theta is clipped to [0,1] and renormalised over feasible candidates."""
    sel = np.isin(month, eval_months) & mask
    dp = delta_p(theta, aux, o)[sel]
    p = aux['p'][sel]
    feas = aux['feas'][sel]
    v = aux['v'][sel]
    q = np.clip(p + dp, 0.0, 1.0) * feas
    q = q / np.maximum(q.sum(1, keepdims=True), 1e-12)
    m_new = np.sum(q * v, axis=1)
    m_lin = aux['m'][sel] + D[sel] @ theta
    y = o['y'][sel].astype(np.float64)
    sse0 = float(np.sum((y - aux['m'][sel]) ** 2))
    sse_lin = float(np.sum((y - m_lin) ** 2))
    sse_clip = float(np.sum((y - m_new) ** 2))
    n = int(sel.sum())
    r0 = np.sqrt(sse0 / n)
    return dict(n=n, drmse_linear=float(np.sqrt(sse_lin / n) - r0), drmse_clipped=float(np.sqrt(sse_clip / n) - r0),
                rows_clipped=int(np.sum(np.any((p + dp < 0) | (p + dp > 1), axis=1))),
                dp_row_sum_max=float(np.max(np.abs(dp.sum(1)))))


def part_a():
    log('A: loading OOF')
    o = load_oof()
    N = len(o['y'])
    log('A: rows', N)
    D, e, ident, cols, aux = basis(o)
    log('A: identity check', ident)
    masks = {'all': np.ones(N, bool), 'body_y<=7200': o['y'] <= 7200, 'serviced': ~(o['lirf'] & o['nm_missing'])}
    st = month_stats(D, e, o['month'], masks)
    res = dict(rows=N, identity=ident, columns=cols, lambda_main=LAMBDA_MAIN,
               column_rms_seconds={c: float(np.sqrt(np.mean(D[:, j] ** 2))) for j, c in enumerate(cols)},
               rmse0_all=float(np.sqrt(np.mean(e ** 2))), rmse0_body=float(np.sqrt(np.mean(e[masks['body_y<=7200']] ** 2))),
               bucket_counts=[int(np.sum(aux['b'] == j)) for j in range(4)], legs={})
    lam_set = {'main': LAMBDA_MAIN, 'zero': 0.0, 'x100': 100 * LAMBDA_MAIN}
    for mname in masks:
        res['legs'][mname] = {}
        for lname, lam in lam_set.items():
            if mname != 'all' and lname != 'main':
                continue
            s = st[mname]
            L = {}
            L['in_sample'] = leg(s, MONTHS, [MONTHS], lam)
            L['lopo_pairs'] = dict(units=[], theta=[])
            pooled_dsse = 0.0
            pooled_sse = 0.0
            pooled_n = 0
            for k in range(1, 7):
                pair = [k, k + 6]
                r = leg(s, [m for m in MONTHS if m not in pair], [pair], lam)
                L['lopo_pairs']['units'].append(dict(pair=pair, **r['pooled']))
                L['lopo_pairs']['theta'].append(r['theta'])
                pooled_dsse += r['pooled']['dsse']
                pooled_sse += r['pooled']['rmse0'] ** 2 * r['pooled']['n']
                pooled_n += r['pooled']['n']
            r0 = np.sqrt(pooled_sse / pooled_n)
            L['lopo_pairs']['pooled'] = dict(dsse=pooled_dsse, rmse0=float(r0), drmse=float(np.sqrt((pooled_sse + pooled_dsse) / pooled_n) - r0), n=pooled_n)
            L['lopo_pairs']['units_improved'] = int(sum(u['dsse'] < 0 for u in L['lopo_pairs']['units']))
            L['half_year_forward'] = leg(s, [1, 2, 3, 4, 5, 6], [[m] for m in range(7, 13)], lam)
            L['half_year_backward'] = leg(s, [7, 8, 9, 10, 11, 12], [[m] for m in range(1, 7)], lam)
            for name, gap in (('expanding_forward', 1), ('gap_forward', 2)):
                units = []
                dsse = 0.0
                sse = 0.0
                n = 0
                thetas = []
                for k in range(4, 13):
                    r = leg(s, list(range(1, k - gap + 1)), [[k]], lam)
                    units.append(dict(month=k, **r['pooled']))
                    thetas.append(r['theta'])
                    dsse += r['pooled']['dsse']
                    sse += r['pooled']['rmse0'] ** 2 * r['pooled']['n']
                    n += r['pooled']['n']
                r0 = np.sqrt(sse / n)
                L[name] = dict(units=units, theta_per_month=thetas, units_improved=int(sum(u['dsse'] < 0 for u in units)), units_total=len(units),
                               pooled=dict(dsse=dsse, rmse0=float(r0), drmse=float(np.sqrt((sse + dsse) / n) - r0), n=n))
            res['legs'][mname][lname] = L
            log(f'A: mask={mname} lambda={lname} in-sample {L["in_sample"]["pooled"]["drmse"]:+.4f} s, LOPO {L["lopo_pairs"]["pooled"]["drmse"]:+.4f} s '
                f'({L["lopo_pairs"]["units_improved"]}/6), half-year fwd {L["half_year_forward"]["pooled"]["drmse"]:+.4f} s '
                f'({L["half_year_forward"]["units_improved"]}/6), expanding {L["expanding_forward"]["pooled"]["drmse"]:+.4f} s '
                f'({L["expanding_forward"]["units_improved"]}/9), gap {L["gap_forward"]["pooled"]["drmse"]:+.4f} s ({L["gap_forward"]["units_improved"]}/9)')
    # clipped-simplex sensitivity on the main legs (mask all, lambda main)
    main = res['legs']['all']['main']
    sens = {}
    th = np.array(main['half_year_forward']['theta'])
    sens['half_year_forward'] = clipped_eval(th, aux, o, D, e, o['month'], list(range(7, 13)), masks['all'])
    th = np.array(main['in_sample']['theta'])
    sens['in_sample'] = clipped_eval(th, aux, o, D, e, o['month'], MONTHS, masks['all'])
    res['clipped_sensitivity'] = sens
    log('A: clipped sensitivity', sens)
    # gate-relevant verdict inputs
    fwd = main['half_year_forward']['pooled']['drmse'], main['expanding_forward']['pooled']['drmse'], main['gap_forward']['pooled']['drmse']
    res['stop_rule'] = dict(forward_drmse=dict(half_year=fwd[0], expanding=fwd[1], gap=fwd[2]),
                            any_forward_gain_worse_than_minus_0_05=bool(max(fwd) > -0.05),
                            sign_unstable=bool((np.sign(fwd[0]) != np.sign(fwd[2])) or main['expanding_forward']['units_improved'] < 6),
                            pass_through_e212='0.41-0.47 (journal 407a): LB seconds ~ 0.41-0.47 x premix seconds')
    res['stop_rule']['close'] = bool(res['stop_rule']['any_forward_gain_worse_than_minus_0_05'] or res['stop_rule']['sign_unstable'])
    return res


# ----------------------------------------------------------------------------------------------- (B)
def iso(s):
    return datetime.fromisoformat(s)


def part_b():
    log('B: reading E208 bulletins')
    rows = [json.loads(l) for l in open(ASSETS)]
    recs = [dict(day=r['day'], **rec) for r in rows for rec in r['records']]
    accepted = [r for r in recs if r['status'] == 'accepted']
    # dedupe identical texts (same bulletin assembled from several assets)
    uniq = {}
    for r in accepted:
        uniq.setdefault(r['text_sha256'], r)
    bull = sorted(uniq.values(), key=lambda r: (r['airport'], r['issued_at']))
    clos = [r for r in bull if r['closures']]
    for r in clos:
        r['_objs'] = tuple(sorted(f"{c['kind']}:{c['object']}:{c['state']}" for c in r['closures']))
        r['_issued'] = iso(r['issued_at'])
        r['_expires'] = iso(r['expires_at'])
    # supersession: next accepted bulletin of the same airport and role
    def superseded_end(r):
        end = r['_expires']
        for q in bull:
            if q['airport'] == r['airport'] and set(q['roles']) & set(r['roles']) and q['text_sha256'] != r['text_sha256']:
                t = iso(q['issued_at'])
                if r['_issued'] < t < end:
                    end = t
        return end
    for r in clos:
        r['_end_sup'] = superseded_end(r)
    # events: same airport + same object set, windows overlapping or within 60 min
    events = []
    for key in sorted({(r['airport'], r['_objs']) for r in clos}):
        grp = sorted([r for r in clos if (r['airport'], r['_objs']) == key], key=lambda r: r['_issued'])
        cur = None
        for r in grp:
            if cur is not None and r['_issued'] <= cur['end'] + timedelta(minutes=60):
                cur['end'] = max(cur['end'], r['_expires'])
                cur['bulletins'] += 1
                cur['end_superseded'] = max(cur['end_superseded'], r['_end_sup'])
            else:
                if cur:
                    events.append(cur)
                cur = dict(airport=key[0], objects=list(key[1]), roles=r['roles'], start=r['_issued'], end=r['_expires'],
                           end_superseded=r['_end_sup'], bulletins=1, day=r['day'])
        events.append(cur)
    for ev in events:
        ev['year'] = ev['start'].year
        ev['start'] = ev['start'].isoformat()
        ev['end'] = ev['end'].isoformat()
        ev['end_superseded'] = ev['end_superseded'].isoformat()
    ev2025 = [ev for ev in events if ev['year'] == 2025]
    per_hub_day = {}
    for ev in events:
        per_hub_day.setdefault(ev['airport'], {}).setdefault(ev['day'], 0)
        per_hub_day[ev['airport']][ev['day']] += 1
    # exposure of July-2025 training departures
    t = pl.read_parquet(TRAIN_JUL, columns=['MVT_ID_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt'])
    exposed_raw, exposed_sup = set(), set()
    per_event = []
    for ev in ev2025:
        s, e1, e2 = iso(ev['start']), iso(ev['end']), iso(ev['end_superseded'])
        sub = t.filter((pl.col('ADEP_mvt') == ev['airport']) & (pl.col('MVT_TIME_UTC_mvt') >= s) & (pl.col('MVT_TIME_UTC_mvt') < e1))
        sub2 = t.filter((pl.col('ADEP_mvt') == ev['airport']) & (pl.col('MVT_TIME_UTC_mvt') >= s) & (pl.col('MVT_TIME_UTC_mvt') < e2))
        a, b = set(sub['MVT_ID_mvt'].to_list()), set(sub2['MVT_ID_mvt'].to_list())
        exposed_raw |= a
        exposed_sup |= b
        per_event.append(dict(airport=ev['airport'], objects=ev['objects'], start=ev['start'], end=ev['end'], end_superseded=ev['end_superseded'],
                              exposed_departures=len(a), exposed_departures_superseded=len(b)))
    dep_days_2025 = t.filter(pl.col('MVT_TIME_UTC_mvt').dt.date().cast(pl.Utf8).is_in(['2025-07-01', '2025-07-08', '2025-07-15', '2025-07-22'])).height
    res = dict(accepted_records=len(accepted), unique_bulletins=len(bull), closure_bulletins_unique=len(clos), events_total=len(events),
               events_2025_training_days=len(ev2025), hubs_2025=sorted({ev['airport'] for ev in ev2025}),
               hubs_all=sorted({ev['airport'] for ev in events}), per_hub_day=per_hub_day, events=events,
               exposure_2025=dict(per_event=per_event, unique_departures_raw=len(exposed_raw), unique_departures_superseded=len(exposed_sup),
                                  training_departures_2025=2085047, departures_on_the_four_july_days=int(dep_days_2025)),
               stop_rule=dict(events_lt_10=len(ev2025) < 10, hubs_lt_2=len({ev['airport'] for ev in ev2025}) < 2))
    res['stop_rule']['close'] = bool(res['stop_rule']['events_lt_10'] or res['stop_rule']['hubs_lt_2'])
    log('B:', {k: res[k] for k in ('closure_bulletins_unique', 'events_total', 'events_2025_training_days', 'hubs_2025')}, res['exposure_2025']['unique_departures_raw'])
    return res


# ----------------------------------------------------------------------------------------------- (C)
def part_c():
    log('C: EHAM counts in final_ranking')
    f = pl.read_parquet(FINAL, columns=['MVT_ID_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt']).filter(pl.col('ADEP_mvt') == 'EHAM')
    by = f.group_by(pl.col('MVT_TIME_UTC_mvt').dt.strftime('%Y-%m').alias('ym')).len().sort('ym')
    june_in_works = f.filter((pl.col('MVT_TIME_UTC_mvt') >= datetime(2026, 6, 1, tzinfo=timezone.utc)) & (pl.col('MVT_TIME_UTC_mvt') < datetime(2026, 6, 27, tzinfo=timezone.utc))).height
    total = pl.read_parquet(FINAL, columns=['MVT_ID_mvt']).height
    res = dict(final_ranking_rows=total, eham_rows=f.height, eham_by_month={r['ym']: r['len'] for r in by.to_dicts()},
               eham_june_until_26_inclusive=int(june_in_works))
    log('C:', res)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ['A', 'B', 'C']
    res = dict(protocol=PROTOCOL, started_utc=datetime.now(timezone.utc).isoformat(), threads=4)
    if 'B' in which:
        res['B'] = part_b()
    if 'C' in which:
        res['C'] = part_c()
    if 'A' in which:
        res['A'] = part_a()
    res['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (OUT / 'result.json').write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    log('written', OUT / 'result.json')


if __name__ == '__main__':
    main()

"""Journal 136. Brute-force check of the E151 interval counts, with ties (the external review of 28.09: D3 = -1 when s_j == a_i)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e151_own_interval_flows import counts as counts_new  # noqa: E402

KMAX = 400


def counts_old(a, t, s, e):
    o = np.argsort(s, kind="stable"); s, e = s[o], e[o]
    es = np.sort(e)
    n_s_lt_a = np.searchsorted(s, a, "left"); n_e_lt_t = np.searchsorted(es, t, "left")
    start_in = np.searchsorted(s, t, "left") - np.searchsorted(s, a, "right")
    end_in = n_e_lt_t - np.searchsorted(es, a, "right")
    lo, hi = np.searchsorted(s, a, "right"), np.searchsorted(s, t, "left")
    nested = np.zeros(len(a))
    for k in range(min(KMAX, int((hi - lo).max(initial=0)))):
        j = lo + k; ok = j < hi; nested[ok] += e[j[ok]] < t[ok]
    return end_in - nested, nested, n_s_lt_a - n_e_lt_t + nested, start_in - nested


def brute(a, t, s, e, self_idx=None):
    out = np.zeros((4, len(a)))
    for i in range(len(a)):
        for j in range(len(s)):
            if self_idx is not None and j == i:
                continue
            if not (e[j] > a[i] and s[j] <= t[i]):
                continue
            if s[j] <= a[i]:
                out[0 if e[j] <= t[i] else 2, i] += 1
            else:
                out[1 if e[j] <= t[i] else 3, i] += 1
    return out


rng = np.random.default_rng(0)
bad_old = 0
for trial in range(300):
    n = rng.integers(2, 60)
    # minute-rounded starts (many ties), durations 0-40 min in whole minutes or seconds
    a = rng.integers(0, 30, n) * 60.0
    t = a + (rng.integers(0, 40, n) * (60 if trial % 2 else 1)).astype(float)
    b = brute(a, t, a, t, self_idx=True)
    new = np.array(counts_new(a, t, a.copy(), t.copy(), True))
    assert np.array_equal(new, b), (trial, a, t, new, b)
    old = np.array(counts_old(a, t, a.copy(), t.copy()))
    bad_old += int((old < 0).any(axis=0).sum())
    # arrivals: separate objects, ties with query bounds
    m = rng.integers(1, 40)
    s = rng.integers(0, 30, m) * 60.0 + rng.choice([0.0, 30.0], m)
    e = s + rng.integers(0, 30, m) * 60.0
    b2 = brute(a, t, s, e)
    new2 = np.array(counts_new(a, t, s, e, False))
    assert np.array_equal(new2, b2), (trial, "arr")
print(f"new counts == brute force on 300 groups (departures incl. self, arrivals); old gave negative counts on {bad_old} query rows")
# the reviewer's example: me 10:00-10:20, neighbour 10:00-10:10
a = np.array([0.0, 0.0]); t = np.array([1200.0, 600.0])
print("example old D1..D4 for me:", [x[0] for x in counts_old(a, t, a.copy(), t.copy())], "new:", [x[0] for x in counts_new(a, t, a.copy(), t.copy(), True)])

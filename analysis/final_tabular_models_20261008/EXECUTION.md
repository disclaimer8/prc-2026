# Final tabular fits — execution journal (Claude, 08.10.2026)

Protocol `final-four-month-tabular-fixed-recipe-save-models-v1`, journal №388,
registration commit `42a6f28aa02a59bfa24b720c10230cdd4bcac7ac`, receipt
`analysis/final_tabular_models_20261008/receipt_42a6f28.json` (attested on the Mac
13:52 WEST). Contract `CONTRACT.json` SHA `4b9dfddc…` binds WORK №385
(`data/final_feature_work_mac_20261008`: FIN1b `2bb90b76…`, motion `0c2a3ea1…`,
template `b3445d80…`, feature_complete `8494279d…`, ordered train/final ID hashes
`653698ad…`/`f7c0b43a…`, helpers train/features/target_encoding/e185_motion_view).
Recipe unchanged; two portability fixes with tests (Cat memory ceiling 12→24 GiB,
`del df`/`del fit,lab` in `prepare()`), 11/11 tests PASS on Mac, hetzner (9 native
tests; two `import torch` tests are not runnable in torch-less venvs by design).

## Admission probes (no model written; `models/` empty before the fits)

| Host | Unit / Invocation | Kind | cgroup peak | fit phase | speed |
|---|---|---|---|---|---|
| prc-cloud | `prc-final-tab-memprobe-cat` 7d32f3c9, 12:40–12:42Z | cat 8 thr | 19.46 GiB | 6.6 GiB | 0.563 s/it |
| prc-cloud | `prc-final-tab-memprobe-raw` 1d4fa5a1 | raw 8 thr | 19.35 GiB | 5.6 GiB | 0.692 s/round |
| hetzner | `prc-final-tab-memprobe-raw` cb5fd977 | raw 6 thr (cores 0,2,4,6,8,10) | 20.14 GiB | 5.5 GiB | 0.380 s/round |
| home-pc | user unit 406f8c4b, MemoryMax=12G | raw 8 thr | OOM-kill at 5 s (during `read_parquet`) | — | — |
| laptop 171 | user unit 1f21c62f, MemoryMax=13G | raw 8 thr | OOM-kill at 6 s (during `read_parquet`) | — | — |

Cross-node parity, 20-round raw LightGBM on the production matrix, 670,790 final
predictions: prc-cloud (ARM) vs hetzner (x86) max abs diff 7.96e-13, mean 4.3e-15,
IDs identical, not bitwise (summation order). Accepted as one recipe.
Consequence: Home (12 GiB cap) and the 14 GiB laptops cannot host this adapter's
full-frame path; all 52 fits run on prc-cloud and hetzner only.

## Production units (all `KillMode=control-group`, `MemorySwapMax=0`, `Nice=10`)

| Host | Unit | Invocation / MainPID | Work | Bounds | Log |
|---|---|---|---|---|---|
| prc-cloud | `prc-final-tab-cat-a2-20261008` | 8074d46b / 51501, started 13:00:35Z | Cat seeds 0–7, 8 threads | MemoryMax=24G, CPUQuota=800%, RuntimeMaxSec=43200 | `/root/prc-final-logs/cat_a.log` |
| prc-cloud | `prc-final-tab-cat-b2-20261008` | a29c3e9a / 51641, started 13:03:40Z | Cat seeds 8–14, 8 threads | same | `/root/prc-final-logs/cat_b.log` |
| hetzner | `prc-final-tab-lgb-hz-20261008` | ba8d78f0 / 78238, started 12:53:58Z | base:0, lgb:0–4, raw:0–4, lobt:0–4 (16 fits), 6 threads | MemoryMax=28G, CPUQuota=600%, AllowedCPUs=0,2,4,6,8,10, RuntimeMaxSec=50400 | `/srv/prc-final-logs/lgb_hz.log` |
| hetzner | `prc-final-tab-lgb-hz2-20261008` | 6ac9aa6a / 79808 (waits for queue 1) | sched:0–4, eobt:0–4 (10 fits) | same, RuntimeMaxSec=86400 | `/srv/prc-final-logs/lgb_hz2.log` |

Roots: prc-cloud `/root/prc-2026` (`.venv` Python 3.12.14, packages = Mac);
hetzner `/srv/prc-final-20261008` (tracked `src/` at HEAD, pinned files from
`42a6f28`, interpreter `/root/prc-2026/.venv/bin/python` Python 3.14.4 with the same
seven pinned package versions — same pattern as the qualified `.venv-cat-parity`
№308/313). Queue runner: `prc-final-logs/queue.sh` (sequential, `set -e`, no retry).

First Cat A launch (`prc-final-tab-cat-a-20261008`, c015d99d, 12:53:52Z) and Cat B
(`prc-final-tab-cat-b-20261008`, a8e50e61) were stopped at ~13:00Z only to add
`PYTHONUNBUFFERED=1`; their empty `cat/seed_0` and `cat/seed_8` directories were
removed (no model/state/prediction file existed). Recorded in `cat_a.log`.

Not yet assigned (10 fits): `motion:0–4`, `aobt:0–4` — to be started on prc-cloud
when a Cat shard finishes (same unit spelling, `--kind motion`/`aobt`, 8 threads,
MemoryMax=24G is enough for LGB: measured 19.4 GiB), or on hetzner as queue 3.

## Monitoring

```sh
ssh prc-cloud 'systemctl list-units "prc-final-tab-*" --all --no-pager; grep -E "COMPLETE|^[0-9]+:|Traceback" /root/prc-final-logs/cat_a.log /root/prc-final-logs/cat_b.log | tail; ls /root/prc-2026/models/final_tabular/cat'
ssh hetzner  'systemctl list-units "prc-final-tab-*" --all --no-pager; grep -E "START|DONE|COMPLETE|Traceback" /srv/prc-final-logs/lgb_hz.log /srv/prc-final-logs/lgb_hz2.log | tail; ls /srv/prc-final-20261008/models/final_tabular'
```
A seed is done when `models/final_tabular/<kind>/seed_<s>/complete.json` exists
(adapter prints `COMPLETE <kind> seed<s>`). A unit failure is `Result=` other than
`success` in `systemctl show`; partial seed directories must be removed by hand
before any relaunch (the adapter refuses to resume).

## Merge

`merge` needs, on one host under one root, all seed directories of a family plus the
same contract/receipt/WORK inputs; it does not call `resource_guard`, so it may run
on prc-cloud or on the Mac (`.venv/bin/python`). Copy hetzner's
`models/final_tabular/<kind>/seed_*` to prc-cloud (via the Mac; no cloud→hetzner key),
then per family:
```sh
python scripts/final_tabular_models_20261008.py merge --root ROOT --receipt ROOT/analysis/final_tabular_models_20261008/receipt_42a6f28.json --kind <cat|lgb|raw|lobt|sched|eobt|aobt|motion|base>
```
Output `models/final_tabular/<kind>/merged/predictions.npz` (`mvt_id`, `per_seed`,
`reg` = seed mean) in FIN1b final-row order (`ids_sha` `f7c0b43a…`), which differs from
the template order; the tail assembly must join by `MVT_ID_mvt`.

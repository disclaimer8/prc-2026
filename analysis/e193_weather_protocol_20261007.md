# E193: METAR history versus matched repeated-current weather

Status: implementation and synthetic qualification; no real-target fit before
root commits this protocol, worker and journal together. Worker:
`scripts/e193_weather_history.py`. Immutable references to existing FIN1b
E190 query tensors; existing environments and scientific helpers stay unchanged.

## Hypothesis and novelty

The time course of observed weather can distinguish recovery, persistent adverse
conditions and rapid changes that a single snapshot and existing six-report
snow/icing maxima do not represent. This is an input-history hypothesis, not a
claim of a new METAR source. Existing FIN1b already includes current weather;
E151 counts and E183 neighbor residual averages do not test weather sequences.
E192's negative event-relation screen is separate. No expected leaderboard gain
is asserted. Expert prior for history−current body RMSE: center 0 s, range
[−0.5,+0.5] s, uncalibrated and not an acceptance threshold.

## Measured source availability

All ten `data/metar/{ICAO}.csv` archives cover 25 December 2024 through 1 August
2026. Columns: `station, valid, tmpc, dwpc, sknt, gust, vsby, skyl1, wxcodes,
metar`. There are no duplicate station/time keys. Per station approximately
17,500 observations in 2025 and 2,970 in January/July 2026; LTFM exactly 17,514
and 2,970. Existing IEM archive files are reused, not downloaded or extended.
Their exact hashes are retained in the pack.

Raw-only coverage probe, before targets/models:

| Population | Queries | No current report within 3h | Empty history | All 12 bins observed |
|---|---:|---:|---:|---:|
| Training 2025 | 2,085,047 | 0 | 0 | 2,062,241 |
| Ranking Jan/Jul 2026 | 344,841 | 0 | 0 | 337,957 |
| LTFM training | 272,965 | 0 | 0 | 272,307 |
| LTFM ranking | 47,581 | 0 | 0 | 46,543 |

Mean observed bins 11.9867/11.9791 for train/rank. Missing gust is common
(95–99% depending on station); `skyl1` absent in 25–65%. Missing `wxcodes`
means no reported weather code, following the existing feature builder; its
five indicator values are zero, not an absent observation. Other numeric missing
values retain individual masks. `skyl1` is the first cloud-layer height, not a
validated meteorological ceiling.

Independent all-row current-weather equivalence passed before fitting: all
2,429,888 train+rank queries have exactly the same six scaled numeric values
and missing masks as the existing FIN1b weather columns (zero differing cells).

## Time, fields and control

Own-flight reference time: observed `AOBT_3_flt`, or `MVT_TIME_UTC_mvt − 900 s`
if null, matching existing FIN1b weather construction. These are query-observable
proxies, never BLOCK_TIME or a target-derived off-block time. Token construction
reads only the nine allowlisted structured METAR columns, not free-form METAR.

Twelve disjoint 30-minute intervals partition `(reference−6h, reference]`.
Each history bin uses its last actual report, with left-open/right-closed bounds.
An empty bin is zero with an explicit observation mask; no carry-forward and no
future report. Current weather is the latest report at/before the reference,
with the existing three-hour tolerance. Airport boundaries are isolated.

Eleven weather values: temperature, temperature minus dewpoint, wind speed,
gust speed, visibility, first cloud-layer height, snow/freezing/rain/thunder/fog
indicators using the existing code patterns. Six numeric values have fixed
physical scaling (no data fitting): temperature clipped ±50 divided by 50;
spread 0..50/50; wind/gust 0..100/100; visibility 0..20/20; cloud height
0..30,000/30,000. Five indicators remain 0/1. The eleven finite-value masks are
appended; missing values become zero. Clipping/scales are fixed before targets.

The **history** arm receives actual weather values/masks in the twelve bins.
The **current** arm replaces each observed bin's values/masks with the latest
current snapshot. Both arms receive exactly the same reporting cadence:
history-bin presence, age of the selected report divided by 6h, and fixed bin
position. The control therefore already knows when reports exist; the tested
difference is past weather content/missingness beyond this cadence. It is not
a control with zero knowledge of history availability.

Both additionally receive one common current-weather vector (22 values/masks
plus normalized current age and presence) in the fusion head. If a history bin
is empty, weather values/masks are zero in both arms. All-empty sequences have
the same position/missingness defaults and retain the query/current branch.
No neighbor labels, target/regime/residual estimates or label-dependent counters
enter weather inputs. Own-row labels are used only to recreate the unchanged
regular training population and for the existing training loss/evaluation.

## Matched architecture and training

Same modules, parameter shapes, initialization seeds and optimizer in both arms;
the arm switch only selects actual historical content versus repeated current
content. Existing PLE query backbone: PLE48/d12/version B/no embedding activation,
width 512, ending in its original 256-dimensional representation. Weather token
25→32 linear + SiLU; two 32→32 Conv1d layers, kernel 3, padding 1, each SiLU;
flatten 12×32 →64 linear + SiLU. Fusion: query256 + weather64 + current24
→256→1 with SiLU. No attention, additional framework or new package dependency.

All fits: seeds 0/1/2, eight epochs, batch 4096, AdamW lr .002, decay 1e−5,
OneCycle pct_start .1, original residual target/scaling and unchanged query
tensors. Same seed order and permutations; no early stopping or hyperparameter
search. The saved no-event E190 PLE is a descriptive additional baseline. It
does not have the same new fusion capacity, so current−PLE is not pure evidence
for new weather information. The primary contrast is history−current.

The E190 matched LGB5/pre-ADS-B mixture and its MLP coefficient remain exact,
including motion routing and LOBT mapping. This is not complete production-chain
CV: Cat15 and the full downstream production stages are absent.

## Frozen screen and conditional continuation

First train both arms on fixed pairs 1+7 and 4+10: **12 fits**. Publish complete
three-seed members, reconstruct exact means/mixture and evaluate. Report all,
body (`y<=7200`), decided, symmetric trim3/trim20 SSE differences, all/body/
decided RMSE, month SSE, NM-LTFM body and descriptive 1,000-resample paired
airport-day bootstrap interval using the inherited fixed seed 192.

**Only if history beats BOTH current and saved PLE in body, decided, trim3 and
trim20 SSE in BOTH screen pairs**, immediately continue the unchanged arms,
seeds, epochs, features and optimizer on 2+8, 3+9, 5+11 and 6+12: **24 more fits**.
Otherwise publish the negative screen and terminate. No window/architecture/
weight/seed search, fallback or retry is authorized. The conditional criterion
is programmatic, fixed before target fitting and cannot be bypassed by a better
pooled score. Bootstrap does not select continuation.

On continuation, the same per-pair criteria are reported for the remaining four
pairs; `all_six_pair_criteria_pass` requires all six to pass. Neither a screen
pass nor continuation completion authorizes ranking or submission: the artifact
always has `accepted_for_submission=false`. Root may separately register a full
production-chain comparison afterward.

## Integrity, resources and operational publication

Mac packs weather indices (~100 MB) and small weather tables, not duplicate query
tensors. Existing six fold bundles remain immutable references with original
registration/spec/file hashes. All-row eight-numeric-column alignment reproduces
original train and held-out tensor order; exact ID/y/month checks protect the
control and PLE predictions. Pack provenance includes weather CSV hashes,
FIN1b SHA, frozen helper/worker hashes and all packed array hashes. Every stage
checks this provenance again; exact seed mean and mixture are reconstructed.

Home `.venv-numemb`, Radeon RX7900XTX, CPUQuota200%, MemoryMax12GiB,
MemorySwapMax0, Nice10. No global source or environment changes. Full-batch
synthetic forward/backward on random-only data must pass before a real fit:
finite outputs/gradients, identical parameter counts, exact equal-input output
parity between the two arms, and empty-context handling. Measured throughput and
runtime hashes are retained in a smoke receipt tied to the final worker SHA.

Measured random-only full-shape GPU smoke passed on 7 October before targets:
4,096 queries × 12 weather bins, two warmup plus 12 timed forward/backward
batches per arm. Current/history 0.0163621/0.0163149 s per batch, peak GPU
allocation 810,587,136/810,451,968 bytes, equal 2,402,197 parameters in the
synthetic feature schema. Exact equal-input output parity and finite gradients
passed. Worker SHA `7a9af2109575a2f5260354ccb3f5cf6bcde94f22a837baf54912af3fdc725ce4`.
Receipt `data/e193_weather_smoke_20261007.json` on Mac and
`data/e193_smoke_20261007/result.json` on Home. Smoke service completed exit 0.
Pure synthetic compute projects about 10.3 minutes for the 12 screen fits;
allow 20–45 minutes with real bin preparation, I/O and checks. Conditional
24-fit continuation adds roughly twice that. Real first-fit timing supersedes
this estimate, without changing the recipe.

Expected initial screen duration below the completed E192 42-minute runtime
(12 rather than 64 small weather tokens; same query backbone), to be replaced by
the real first-fit estimate. Twelve screen fits and conditional 24-fit extension
have no deadline-driven seed reduction. Only one GPU experiment runs at a time;
Home CPU full-chain work keeps its separate allocation.

Stage directories publish atomically after all of that stage's fits and hashes.
`screen/evaluation.json` is written before any conditional continuation.
The outer `complete.json` records 12 or 36 fits and the exact evaluation hashes.
Failed staging is retained; there is no automatic resume/retry. Root collects
complete evidence and separately coordinates the next independent queued task.

## Commands after root registration

```sh
.venv/bin/python scripts/e193_weather_history.py pack --root . \
  --output data/e193_weather_inputs_20261007 --registration-commit SHA
# Transfer this small immutable pack; reuse original six query bundles at Home.
.venv-numemb/bin/python data/e193_weather_inputs_20261007/scripts/e193_weather_history.py run \
  --root . --input data/e193_weather_inputs_20261007 \
  --output data/e193_weather_20261007 \
  --smoke data/e193_smoke_20261007/result.json --registration-commit SHA
```

The `run` command belongs in the bounded user service above. A service/PID and a
real first epoch must be observed before reporting target training as started.

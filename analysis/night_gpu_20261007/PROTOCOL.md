# E204: two observation scales in the successful trajectory member

Protocol: `E204-trajectory-multiscale-d1d8-fixed-ordered-control-gated6pair-v1`.

Written before any E204 target fit or quality comparison. Root registers this
document, worker, tests and journal in one commit. No upload is authorized by
this worker. E201 stays cancelled; no environment changes are needed.

## Mechanism and novelty

The successful E197/E202 ordered member applies two message blocks to adjacent
selected trajectory points. Each final token can see offsets −2…+2. E204 keeps
block 1 at distance 1 and fixes block 2 at distance 8. It can see nine offsets
`−9,−8,−7,−1,0,1,7,8,9`, spanning 19 positions. This tests whether local motion
combined with a longer trajectory phase improves ordinary taxi-out prediction.
The distance is in selected actual points, not seconds or interpolated metres.
Irregular sampling and the fixed K64 view remain limitations.

Same 2×64 actual-point inputs, point masks, age/gap and other 12 channels, same
train-only normalization, same PLE query encoder, two message blocks, fusion
head, parameter shapes/count, initialization order, seeds 0/1/2, eight epochs,
batch 4096, MSE/AdamW/OneCycle parameters and 0.25 member weight. The frozen
`e197_trajectory_encoder.fit` is called through a process-local network adapter;
no frozen source is modified. Synthetic d1/d1 must reproduce original ordered
forward exactly. New d1/d8 initial parameter tensors must equal ordered tensors.

Registry search found no previous dilated/multiscale own-trajectory encoder.
This continues the successful own+previous trajectory member. It does not
revive failed event/spatial/MDN arms, E201, or prefix augmentation. There is one
fixed candidate, no dilation sweep or duration-pooling arm.

## Population and fixed comparator

The training population remains 2,085,047 2025 departures with frozen E190
ordinary residual targets. Own observed points: 1,251,417; previous-visit points:
1,345,381; previous-only: 322,974. Union observed: 1,574,391. All rows still have
the original query features, including rows with no points.

Parent E197 registration `bfe793c62f7cecf633c7cc31c747f48eb0676f2d`, spec
`93d90131cdfa32c04b8f8c3513f3e7f7cc19c9695d8f47d3fe98b4cc8c0a407d`.
Parent E202 registration `55fe9b68f7cc8405681f6c85d576979e067b55c3`, spec
`6d604eaaca830fa83c18ebb78986b028d338869123d5f8c10b2450dde1ec4309`.

The exact saved three-seed ordered predictions are the fixed comparator,
reconstructed from saved seeds at pack. No comparator refits. The stored
LGB5/pre-ADSB mixture coefficient/anchors/rounding rules are reused. This is a
component comparison, not full-current-chain OOF and not a leaderboard forecast.

All input IDs, months, labels and query permutations must match exactly. The
normalization receipt must name ordinary train rows and exclude heldout months.
No new extraction, label-bearing token, neighbour target or normalization fit.
Saved tensor/control/models/source hashes are sealed before fitting. Raw E194
observations are already verified and prepared; these tables are read in place.

## Preregistered continuation and rejection gates

First, fold pairs 1+7 and 4+10, each three seeds: six fits. For **each pair** all
four candidate-minus-ordered statistics must be strictly negative:

- Body SSE, target ≤7200 seconds.
- Decided SSE, using the existing immutable decided mask.
- Decided SSE after symmetrically removing three most negative and three most
  positive rowwise SSE differences within the fold.
- The same symmetric trim with twenty at each end.

Only screen PASS automatically opens pairs 2+8, 3+9, 5+11, 6+12: twelve more
fits. Six-pair PASS requires the screen gate, at least four of six negative
decided-SSE pairs, all four pooled statistics above negative, and all four sums
over **confirmation pairs alone** negative. Thus selected pairs cannot rescue a
negative confirmation aggregate. Trim sums remain within-fold; no global trim.

All-row trims, airport/day bootstrap, per-month SSE and body/decided RMSE deltas
are descriptive. They do not select a different dilation, model weight, airport
or population. Finite/ID/source/fit/resource failure is a technical failure,
not a scientific rejection, and never triggers automatic retries.

After a negative screen the queue stops after six fits. After a negative final
gate it stops after eighteen. On PASS it publishes verified component artifacts
for a separately registered comparison against the current v12075 chain. This
worker does **not** train ranking or build a candidate, even on component PASS.
Those require the additional current-chain gate with the successful trajectory
member as control; root owns that continuation.

The comparator was already selected on these months. The E204 first two pairs
select continuation; remaining four provide a limited new-candidate confirmation.
This is not unbiased fully nested selection. Prior component ΔbodyRMSE = 0 s,
expert range [−0.5,+0.5] s, uncalibrated and not a confidence interval. Expected
gain is uncertain; no leaderboard gain is promised.

## Resources and durable execution

Only Home `home-pc-lan`, `/home/a1/prc-2026`, existing `.venv-numemb`. Single
GPU process, CPUQuota 200%, MemoryMax 12G, MemorySwapMax 0, nice 10,
RuntimeMaxSec 21600, Restart=no, KillMode=control-group. PyTorch allocator is
limited to 20 GiB; synthetic full batch reserves exact resident-table bytes and
must stay below that limit. Existing K64 tables occupy about 9.2 GiB on GPU.
CPU mmap pages are released by the frozen loader. No data copies to Mac.

Expected full 18 fits roughly 1–2 hours from previous measured E197 runs,
plus verification; this is a planning estimate, not a progress claim. The new
full-size synthetic smoke records seconds/batch and peak VRAM. Maximum six-hour
service lifetime, no automatic retry/resume. Partial artifacts remain preserved.

Fresh output only; existing input/smoke/run paths cause refusal. systemd survives
Mac disconnect. `run/events.jsonl` has fsynced start/fit/fold/terminal records;
the journal has epoch losses from the actual child. Ready means a terminal
`result.json` plus successful independent reconstruction `audit.json` and a
successful exited systemd invocation, not an active waiting shell.

## Exact commands after root registration

On Mac, after the registration commit exists:

```sh
.venv/bin/python -m unittest tests.test_night_gpu_20261007
.venv/bin/python scripts/night_gpu_20261007.py pack --root . --registration-commit FULL_SHA
```

Copy the tests, protocol and sealed input to the same relative paths on Home.
The pack contains its own `input/scripts` with the new worker and exact frozen
helpers; no shared source on Home is overwritten. Source/data drift causes
refusal. Then start once:

```sh
ssh home-pc-lan 'systemd-run --user --unit=prc-e204-night-gpu-20261007 --property=WorkingDirectory=/home/a1/prc-2026 --property=CPUQuota=200% --property=MemoryMax=12G --property=MemorySwapMax=0 --property=Nice=10 --property=Restart=no --property=KillMode=control-group --property=RuntimeMaxSec=21600 /home/a1/prc-2026/.venv-numemb/bin/python -u /home/a1/prc-2026/analysis/night_gpu_20261007/input/scripts/night_gpu_20261007.py queue --root /home/a1/prc-2026 --registration-commit FULL_SHA'
```

Verify InvocationID/MainPID and actual epoch progress, then retrieve only small
run receipts/predictions/model artifacts needed for the next stage. Each phase
is a fresh process: qualified synthetic smoke → screen/conditional confirmation
→ independent audit. No external publication or submission.

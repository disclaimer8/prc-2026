# E204 closed: screen FAIL, artifact audit PASS

The fixed d1/d8 multiscale trajectory member fails its preregistered screen.
Both primary errors and both decided trims worsen in pair 1+7. Pair 4+10
improves, but the protocol requires all four statistics negative in **each**
pair. The pooled improvement does not override this rule.

| Evaluation pair | Rows | Body RMSE delta | Decided RMSE delta | Decided trim3 delta SSE | Decided trim20 delta SSE |
|---|---:|---:|---:|---:|---:|
| 1+7 | 344,419 | +0.092642 s | +0.096240 s | +18,062,406.56 s² | +18,630,568.43 s² |
| 4+10 | 360,962 | −0.203661 s | −0.203193 s | −27,227,914.94 s² | −26,713,345.86 s² |

Descriptive pooled results across 705,381 rows: body RMSE209.371704→209.328423
(−0.043281s;705,162body rows), decided242.201164→242.176590
(−0.024573s;705,324decided rows). These pooled figures were not the screen gate.
The prior was0s with uncalibrated expert range[−0.5,+0.5]s; both observed pair
body deltas fall within that range. This was a saved-LGB5/pre-ADSB component
comparison, not a current-chain or leaderboard result.

## Completed computation and durable terminal state

- Exactly **6 fits / 48 epochs**: two pairs × seeds0/1/2 × eight epochs.
- Total measured fit time1,366.616s (22min46.616s).
- systemd service wall time24min2.444s; CPU time47min13.119s.
- Actual fit peak GPU allocation11.656GiB; process peak RSS4.415GiB.
  The full-size synthetic smoke peaked at11.876GiB; the cgroup peak was12GiB,
  including its charged file cache. Limits remainedCPU200%/RAM12GiB/swap0,
  GPU allocator20GiB/nice10/no restart/six hours maximum.
- Last fit completed22:39:50.085UTC; screen FAIL saved22:39:51.938UTC.
- Independent reconstruction audit completed **22:39:58UTC**: statusPASS,
  scientific_gate_pass=false, fits6, folds2, ranking_authorized=false.
- Unit `prc-e204-night-gpu-20261007.service` is now inactive/dead,
  MainPID0, Resultsuccess, ExecMainStatus0. Its original invocation was
  `857a2b245c8546abb05022c57bcf57f2`; no automatic retry occurred.

The audit rechecked sealed sources/inputs and query bundles, every saved artifact
hash, exact three-seed means and mixture arithmetic, exact IDs/targets/months,
all saved metrics and the gate. The collected result's SHA was independently
compared with the audit on Mac; six complete eight-epoch fit receipts and both
fold reports were also checked locally. Large tensors, models and seed arrays
remain on Home. Only small receipts and logs were collected.

## Provenance and artifacts

- Registration `05a1b2eb7adfa5bd1abc47bc03a94095e0a39d4e`.
- Protocol `E204-trajectory-multiscale-d1d8-fixed-ordered-control-gated6pair-v1`.
- Input spec SHA256 `0a2069717bb5a8a85414c36c03d6284539f6eaf8de4ec967253752e9763fd712`.
- Worker SHA256 `a16d5b224e521255861b378170242607caaffbfd46f82f2ada0b356e9ed0746b`.
- Fixed controls: E197 registration`bfe793c62f7cecf633c7cc31c747f48eb0676f2d`,
  E202 registration`55fe9b68f7cc8405681f6c85d576979e067b55c3`; exact control
  prediction hashes are recorded separately for each pair in[result.json](result.json).
- Result SHA256 `0a6275eef8649e67bcb0e5855aac328d8750945fa30ddc1af6d44dccf2821609`.
- Audit SHA256 `15280bc4a5f66281a565a142d6f02439fd6836e749195f4e0babe2c6b66d9b0a`.

Local evidence: [audit](audit.json), [result](result.json), [summary and receipt
hashes](summary.json), [first pair](fold_1_7_complete.json),
[second pair](fold_4_10_complete.json), [worker log](worker.log),
[events](events.jsonl), [terminal state](terminal_state.txt),
[original launch evidence](execution.json), [synthetic smoke](smoke.json).

## Continuation decision

No four-pair extension, ranking fit, full candidate, upload or new GPU candidate
was launched. E206 remained an unregistered, unexecuted preparation draft;
its status is explicit in[STATUS.md](../night_trajectory_chain_20261007/STATUS.md).
Current bestv12075 is unchanged. This closes this exact fixed d1/d8 candidate;
it does not establish a general negative result for every trajectory resolution.

# E192 execution — 7 October 2026

Registration: bb80c3e345727175abf24d0270b5281577aa74d4 (journal323).
Worker SHA:427b3ab0e063f421f8cc3b150c06a4030d5870d0417e63d0b868a36ba4d0dff5.

Actual Mac preparation began around01:26UTC on7October; no E192 real-target fit
ran on6October. Log:data/e192_event_pack_20261007.log. All4,167,797 rawtraining
events packed.2,085,047 departure queries;66 have empty context;1,895,116 have a
full balanced64-token context. Both folds passed exact8numeric-column rowalignment
against original immutable MLP tensors; categorical vocabularies use fitmonths only.
Local and home full input verification agreed on spec SHA:
a1aa07494117157fc7f77b09ad2987f38ef6932bb9b7aa5f948cb335fa2abf6a.
Inputs:data/e192_event_inputs_20261006 (suffix is protocoldate, not fitdate).

Home service prc-e192-event-20261007.service started2026-10-07 01:28:13UTC
(02:28:13WEST), MainPID520502, invocation7334906eda294410974d9d254440efc4.
CPUQuota200%,MemoryMax12884901888,MemorySwapMax0,Nice10 verified.
Environment:.venv-numemb, unchanged. Cat service still active at launch,PID329700.
Log:data/e192_event_20261007.log. Final output:data/e192_event_20261006.
Only complete12fit output is published; hidden staging retained on failure.

The initial estimate remains1.2–2.5h until real epoch throughput is available.
No interim quality selection, codechange, seed reduction or upload authorized by
this execution record. Final measured results and completion evidence will follow.

Completion update, 7 October 2026: all 12 fits completed successfully at
02:10:29 UTC, 42 minutes 16.192 seconds after the 01:28:13 start. The registered
script atomically published the complete directory; it and the full log were
copied to Mac. Cgroup memory peak 8.6 GiB, RSS 4.447 GiB, peak GPU allocation
2,098,925,568 bytes. No retries or source/environment changes occurred.

Service-journal collection required SSH approval, which arrived at 06:40 UTC;
the registered local evaluator then completed with exit 0. Both continuation
flags are false. Final measured report and independent audit:
`analysis/e192_event_results_20261007.md` and `.json`. Original evaluation:
`data/e192_event_20261006/evaluation.json`. No ranking, expansion or upload.

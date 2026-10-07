# E202 execution status

## Post-reboot update, 14:45UTC

The Mac reboot around14:08UTC interrupted only its transfer controller; the original Home GPU unit remained alive. Registered post-reboot transport `5ef410c1974e5cb0f6e64e3e5bfdbf9a27254e38` started under day-controller `f9aa25cf58b8ba945ef4019a1f3b0d96e7525f58` at14:36UTC, PID33495. It verifies and reuses complete files, preserves incomplete files in a separate archive, and resumes missing transfers without restarting preparation or GPU fits.

- Fold3 admitted14:41:31UTC; admission SHA `2ba5c1cba1f3ad9f199b90aef4210929a6413bf4d901409440998a09deb6b44f`.
- Fold2 admitted14:42:33UTC; admission SHA `896fe0fdbe0defd1f614c4a6aae1f650687be5ce610e67f9eac17f2bd44fe93d`.
- At14:45UTC, **actual GPU training is verified**: `START fold_2_8 flat`, seed0 epoch3/8, all three tables resident on GPU. Same original PID626873 / invocation30d9322da32d4cfdb55ef5c7c3d5ecbf. CPU memory8.8GB, active.
- Folds5/6 are transferring concurrently. The original first-table wait deadline was met. No duplicated models or output replacement.

Current status/log: `data/e202_postreboot_transport_20261007.{json,jsonl}` and `data/prc_day_recovery_20261007/e202_transport.log`. Old time estimates and statuses below are historical; completion is not yet claimed.

Scientific registration350: `55fe9b68f7cc8405681f6c85d576979e067b55c3`.
Input spec: `6d604eaaca830fa83c18ebb78986b028d338869123d5f8c10b2450dde1ec4309`.
Unchanged24fit extension, flat/ordered, pairs2/3/5/6, three seeds, eight epochs.

Original controller PID78189 packed13:10:12UTC but stopped before its first
remote command because nested SSH inherited the filesystem/network sandbox.
Failed state is preserved as `data/e202_execution_20261007.json`.
Operational recovery commit `812e141e9196627529ed5a6f5cfb4449560062f1`
records the purpose-specific transport launcher; scientific inputs are unchanged.
The platform permission was actually granted much later: recovery PID79538
started **13:56:34UTC**, not13:10. No preparation or fitting occurred during
that approval delay.

Actual remote units observed active13:57:10–12UTC:

|Node|Unit|PID|Invocation|
|---|---|---:|---|
|171|prc-e202-prep-171-20261007.service|286096|6cd68e99f0384dbdbd95e5ba2d87be50|
|175|prc-e202-prep-175-20261007.service|116515|cb1cff50417a4f5899504f483648d6cb|
|Home|prc-e202-trajectory-20261007.service|626873|30d9322da32d4cfdb55ef5c7c3d5ecbf|

At13:58:46UTC, fold2 prepared in68.061s and fold3 in57.562s, each2 085 047IDs.
They stream directly laptop→Home via `scp -3`; later folds5/6 prepare next.
The Home unit is **waiting for first admitted table**, not yet training.
Laptop E201 waiters will release immediately after both local tables are
complete and the preparation unit exits successfully, independent of transfer.

At14:01:13UTC both laptop prep units were independently observed inactive,
MainPID0,Resultsuccess,ExecMainStatus0. Both pinned `two_prepared_complete`
markers exist with folds[2,5]/[3,6] and the exact registered spec. This releases
E201 CPU before either19GiB laptop transfer has completed; Home had received
approximately3.7GiB. Actual E201 fit progress is tracked by its separate owner.

Durable current status/log: `data/e202_transport_recovery_20261007.{json,jsonl,log}`.
Home fit log: `/home/a1/prc-2026/data/e202_gpu_20261007.log`.
Published output destination: `data/e202_trajectory_20261007` (complete24fit only).
No ranking or submission is authorized by this experiment.

Operational estimate at launch: first fit14:04–14:10UTC, full24fit15:45–16:10UTC;
these are estimates pending measured transfer/epoch throughput, not completion claims.

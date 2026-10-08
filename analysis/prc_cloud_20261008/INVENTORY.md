# PRC Cloud feature-build inventory — 8 October 2026

Status: operational host candidate; runtime and existing directory contents
await root's read-only inspection. This agent performed no SSH, installation,
transfer, feature build or production fit. The feature source/protocol remain
frozen after independent READY; this document does not authorize execution.

## Evidence and remaining admission

Parent-provided observations: SSH alias `prc-cloud`, root@2.29.61.190, 16 ARM
Neoverse N1 logical CPUs, approximately30GiB RAM and279GiB free disk, no GPU,
Python3.14.4, no system pip. `/root/prc-2026` exists; contents and virtualenv
availability are not established here. These are root's observations, not an
independent refresh by this agent. Root owns the one directory/runtime probe.

Local import inventory of all11 sealed feature producers establishes that this
stage needs NumPy, Polars and standard-library modules only. Geometry is included
in the sealed closure. No LightGBM, CatBoost, Torch, pandas or pyarrow import is
required by these producers. This does not qualify future model training on
ARM. Mac reference runtime observed here: Python3.12.14, NumPy2.5.3,
Polars1.44.2. `dependency_inventory.json` binds the11 producer SHA values and
frozen helper SHA33479b6d8e8e9648b0c5c6410cba157f940ce6ca0389cd031f8164326f56b1be.
Cloud Python3.14.4 is a different unqualified runtime; package availability and
any new isolated runtime are a separate root-owned setup boundary. No existing
environment should be modified.

Before an actual feature launch, root must confirm the chosen interpreter can
import the pinned package versions, exercise tiny timestamp/timezone/parquet
operations and run the12 synthetic feature tests. The tests cover schema and
label admission, the11-stage dependency order using synthetic producers, and
sealed input/output identity. They are runtime qualification, not proof of real
feature equality or measured full-build peak RAM. Root also needs live CPU/RAM/
disk admission and a working systemd manager with cgroupv2 CPU/memory/swap caps.
Use either a verified root system unit or a working root user manager; the
Home `--user` launch spelling is not automatically valid on a new root login.

## Minimal operational delta for preregistration385

Proposed wording (root owns the actual journal entry):

> Select `prc-cloud` as the Linux feature WORK host instead of Home. The sealed
> input list, source bytes, final periods, producer order, schemas, zero-fit
> scope and output admission are unchanged. Metadata seal remains on Mac with
> Git and the existing canonical inputs. Stage uses2CPU/3GiB/swap0/900s; build
> uses6CPU/24GiB/swap0/Nice10/7200s, KillMode=control-group and retained
> InvocationID. Run in a fresh dated `/root/prc-final-features-20261008` WORK,
> after verifying that path is absent. Pass the exact trusted manifest SHA.
> Input transport is a separately authorized root action. No automatic retry,
> reused partial WORK, full-repository transfer or historical output splice.

The helper is host-independent Linux code: it does not hardcode Home paths or
require Git for stage/build/verify. Thus this host change needs no source edit.
The existing protocol's wait for E207 is a Home resource condition; Cloud has no
E207 job, so root may explicitly remove that Home-only scheduling dependency
for Cloud after live admission. This does not change E207 jobs or compete for
its reserved Home RAM/GPU. Network occupancy still needs root's judgment.

ThirtyGiB physical RAM leaves approximately6GiB outside the24GiB cgroup cap;
this is capacity arithmetic, not a full-build memory measurement.279GiB free
exceeds the25GiB build admission observed here, but free space must be checked
at launch. Runtime7200s is the failure deadline, not a measured ARM ETA. A new
ARM host may finish more slowly; retain failure evidence rather than silently
raising the budget or retrying.

## Input and output boundary

Mac seal produces the exact64-file allowlist:14 challenge Parquets,35 external
inputs,11 producer sources, adapter, tests, protocol and source-contract JSON.
It stores metadata/hashes only. Root transports only absent allowlisted files
plus that manifest to the chosen Cloud source directory, after normal transfer
authorization. No `.git`, credentials, whole home directories, old feature
arrays, models or submissions are needed. Verify already-present files against
manifest bytes before treating them as reusable. Preserve unknown pre-existing
`/root/prc-2026` content; do not overwrite or recursively replace it.

`stage` makes read-only copies into the fresh WORK and retains original input
inodes. Build publishes all four expected tables and feature_complete.json only
after schema, ID, training-label-integrity and final-null-label checks. All
later trainers bind the completed feature SHA and ordered split IDs; this
inventory neither authorizes a fit nor changes their runtime qualifications.

## Concrete remaining responsibilities

- Root: report existing directory/interpreters and systemd state; choose an
  isolated qualified runtime if none exists, without altering existing venvs.
- Root: register the host/path/runtime operational delta385, seal on Mac, then
  authorize normal allowlist transport and run bounded stage/build units.
- Feature helper owner: keep reviewed source and protocol hashes frozen; review
  only a necessary source defect if actual runtime admission reveals one.

No data manifest, WORK, archive, copied input or remote process was created by
this inventory. Reproduction of old bestv12075 remains a separate task.

# v12075: private portable saved-component package

Protocol identifier: `v12075-private-allowlist-package-minimal-git-proof-zero-fit-v1`.

This authorizes one private package build and one **fresh materialized source/data checkout on the existing qualified external Mac runtime**. It does not authorize publication, communication with organisers, environment installation, model fitting, full-history cloning or a new raw-data retrain. Before registration only source/tests, metadata sealing and synthetic fixtures are permitted. Root owns preregistration and real execution. Existing source files, production outputs and environments remain unchanged.

## Exact closure and integrity

The old registered replay inventory is the authority: `analysis/reproduce_v12075_20261007/inventory_v2.json`, registration `70aff7199731c381945c45754ee70c7e3a964aa5`. `seal` verifies the original registered runner/spec/test hashes, then requires exact path/SHA equality between the old dependency report and three fixed groups:3711prediction/provenance files,7registered runner prerequisites,3comparison-only references. Their sum is5,232,081,827bytes. No root/home directory glob is used.

The new package manifest adds only this helper, its test, this protocol, GPL `LICENSE`, `pyproject.toml`, and `uv.lock`, each explicitly named and hashed. Runtime is documented, not copied. It also enumerates exact Git proof objects: original commits70aff719… and d45cb738…, trees along9required commit:path lookups, and their9source/journal blobs. Tree objects contain unchanged original directory entries but unrelated blobs and parent histories are absent. No `.git/config`, credential file, remote config, hook or arbitrary repository/history copy is allowed. Synthetic tests verify exact `git show` after reconstruction and absence of an unrelated blob. This deliberately partial object store is not a complete Git clone and is not claimed to pass full-history `git fsck`.

`bootstrap` first verifies the new helper/test/protocol/manifest against the root prereg commit and its exact journal protocol string. It writes a small fresh directory containing the manifest, package sources, exact Git object bytes and a registration receipt. It copies no model data. Home `stage` uses only declared paths: available local files are hash-checked and copied into fresh staging; missing paths are emitted as a plain exact transfer list. Original Home data is never overwritten. Every staged member must match its expected size and SHA before `build` succeeds.

The uncompressed tar has the exact manifest first, followed only by declared regular files in sorted order. UID/GID/mtime are normalized; no host paths or user/group names are injected. The whole archive is reopened and every member hash rechecked before a verified receipt is written. The streaming reader rejects absolute/traversal/noncanonical/control-character paths, `.git` overwrite, duplicate/extra/missing members, symlinks, hardlinks, devices, sparse files, wrong sizes and hashes, including members hidden after tar end padding. Extraction never invokes `extractall`. It uses a trusted preregistered local manifest SHA and creates only a new destination. Failure keeps partial evidence and has no automatic retry.

Comparisons remain a separate manifest group. The frozen runner generates the initial mixture before opening the two premix references; it seals the generated final file before opening v12075. The package introduces no comparison-derived prediction feature or shortcut. Target output SHA is `90d3cd0b5a68cc8008f2142f1801857752574b525b2ff046ad6bf4089a70aee2`.

## Execution, resources and evidence

Home is only a byte-staging/build host. `stage` and `build` refuse unbounded execution: Linux cgroup≤1CPU,≤1GiB RAM,swap0. Launch direct SSH systemd user units with nice10,RuntimeMaxSec1800,TimeoutStopSec30,KillMode=control-group and RemainAfterExit=yes; do not introduce another local SSH controller. Before staging reserve≥12GiB for≈5.3GB staging+≈5.3GB tar and receipts. Before archive creation the helper independently requires archive payload size+4GiB free. Stream buffers are8MiB; no Parquet/NPZ arrays or model libraries are loaded for bulk staging/build/verification.

At preparation Mac had≈15GiB free. Receive the archive as a stream into the new checkout, without storing a second5.2GB archive or installing/copying the1.5GB virtual environment. Extraction requires total payload bytes+4GiB initially and rechecks remaining file bytes+4GiB before every output file. Existing input paths are rejected. Mac metadata/extraction is single-process streaming, bounded by declared file sizes and the8MiB buffer; use no concurrent bulk transfer while E207 first prepared tables are awaiting delivery. Root schedules all large transfers after that transport priority.

The existing external interpreter must match Python `3.12.14 (main, Sep  1 2026, 14:09:38) [Clang 22.1.3 ]`, NumPy2.5.3,Polars1.44.2,LightGBM4.7.0 on macOS arm64, with caffeinate/memory_pressure/vm_stat present. `runtime-check` checks this and the unchanged historical registration guards. No environment mutation or downloading is allowed. Home/Linux Python3.12.15 is not qualified for this replay.

The new `replay` entrypoint verifies all materialized member hashes and runtime, then executes the **same code object** as the existing reviewed v2 supervisor with a private globals mapping that only tightens its disk floor to4GiB. The original module globals and file remain untouched. Existing6-thread,nice10,16GiB combined-RSS sampled guard,4-hour deadline,16GiB reclaimable-memory wait, exact child-group ownership and no-retry cleanup remain in force. Require≥5GiB free immediately before replay; the supervisor checks≥4GiB while it runs. Run under external `caffeinate -i`; no new scheduler is installed.

Prior replay measurements are101.19s and7.90GiB peak, with135MB existing output storage; these are planning evidence, not a timing claim for the new checkout. Fresh replay needs both original complete receipts. The independent final gate rechecks every packaged input, materialization lineage, supervisor→runner receipt hash,344841rows,42protected donors,0fits, and exact output/reference SHA90d3cd0…. `PORTABLE_REPLAY_VERIFIED.json` is required before claiming package replay success. A manifest/archive proof alone does not satisfy that claim.

All artifacts remain private. Availability/redistribution limitations in current project documentation are unchanged. The final proof must retain the qualification “existing external Mac runtime”; no clean-machine installation or cross-platform equivalence is claimed.

## Commands after root review and preregistration

Replace `REG` and `MANIFEST_SHA` with the reviewed40-/64-character values. Suggested fresh paths: Mac bootstrap `analysis/reproduction_bundle_20261008/bootstrap`; Home staging `/home/a1/prc-2026/data/v12075_portable_20261008/stage`; Home tar `/home/a1/prc-2026/data/v12075_portable_20261008/v12075.tar`; Mac checkout `/Users/denyskolomiiets/prc-v12075-materialized-20261008`. No path may already contain output from a prior attempt.

Metadata-only seal, before preregistration:

```sh
.venv/bin/python scripts/v12075_portable_bundle.py seal --root /Users/denyskolomiiets/prc-2026
```

After preregistration, generate the small bootstrap:

```sh
.venv/bin/python scripts/v12075_portable_bundle.py bootstrap --root /Users/denyskolomiiets/prc-2026 --registration REG --manifest-sha256 MANIFEST_SHA --out analysis/reproduction_bundle_20261008/bootstrap
```

Root transfers that directory directly to Home, then starts a bounded user unit whose command is:

```sh
/home/a1/prc-2026/.venv/bin/python -u /home/a1/prc-2026/data/v12075_portable_20261008/stage/scripts/v12075_portable_bundle.py stage --root /home/a1/prc-2026 --manifest /home/a1/prc-2026/data/v12075_portable_20261008/stage/BUNDLE-MANIFEST.json --registration REG --manifest-sha256 MANIFEST_SHA --out /home/a1/prc-2026/data/v12075_portable_20261008/stage
```

Fetch only `MISSING.txt`/`MISSING.json`; transfer precisely those regular paths from Mac into the fresh stage (e.g. direct rsync with `--files-from=MISSING.txt`). Source allowlist/SHA must match the sealed manifest. Do not run stage again: build independently verifies both local and transferred members. A separate bounded user unit builds:

```sh
/home/a1/prc-2026/.venv/bin/python -u /home/a1/prc-2026/data/v12075_portable_20261008/stage/scripts/v12075_portable_bundle.py build --root /home/a1/prc-2026/data/v12075_portable_20261008/stage --manifest /home/a1/prc-2026/data/v12075_portable_20261008/stage/BUNDLE-MANIFEST.json --registration REG --manifest-sha256 MANIFEST_SHA --out /home/a1/prc-2026/data/v12075_portable_20261008/v12075.tar
```

Root fetches the small `v12075.tar.verified.json` first. Stream only after its success:

```sh
ssh home-pc-lan cat /home/a1/prc-2026/data/v12075_portable_20261008/v12075.tar | /Users/denyskolomiiets/prc-2026/.venv/bin/python /Users/denyskolomiiets/prc-2026/scripts/v12075_portable_bundle.py materialize --root /Users/denyskolomiiets/prc-2026 --registration REG --manifest-sha256 MANIFEST_SHA --archive - --out /Users/denyskolomiiets/prc-v12075-materialized-20261008
```

Run with shell pipefail so a failed transport cannot be reported as success. The materializer verifies all expected members even if transport is truncated. In the new checkout use the unchanged qualified external interpreter:

```sh
/usr/bin/caffeinate -i /Users/denyskolomiiets/prc-2026/.venv/bin/python -u /Users/denyskolomiiets/prc-v12075-materialized-20261008/scripts/v12075_portable_bundle.py replay --root /Users/denyskolomiiets/prc-v12075-materialized-20261008 --manifest /Users/denyskolomiiets/prc-v12075-materialized-20261008/BUNDLE-MANIFEST.json --registration REG --manifest-sha256 MANIFEST_SHA
```

This is the only new-root replay in this protocol. No execution in the old completed output tree, no model refit, no retry of partial attempts, and no publication follow automatically.

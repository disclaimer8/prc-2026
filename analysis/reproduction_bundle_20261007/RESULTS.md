# v12075 portable replay archive: dependency gaps, 7 October 2026

**No archive has been built.** This is the requested bounded dependency and
runtime report. No model fit, replay, publication, environment change or Git
history change was performed. The current replay/source documentation is owned
by the replay agent and was not edited.

## Size and Home availability

The current `inventory_v2.json` has 3,711 pinned dependency paths, comprising
input data, frozen sources and provenance/control proofs. On Mac all are regular
files, all exist, and none resolves outside the repository or through a symlink.
The size/existence inspection did not replace the replay's full SHA validation.

| Set | Files | Logical bytes |
|---|---:|---:|
| Existing inventory dependencies | 3,711 | 5,187,588,486 (4.831GiB) |
| Missing runner/spec/test files listed below | 7 | 915,007 |
| Separate comparison-only references | 3 | 43,578,334 |
| Lower bound before Git/runtime material | 3,721 | 5,232,081,827 |

Home is `/home/a1/prc-2026`, with 663 GiB free at inspection. A low-priority
read-only hash check finished in 2.553 seconds:

- 1,311 existing dependency files, 2,502,715,791bytes, match expected SHA and size.
- 2,400 dependency files, 2,684,872,695bytes, are absent. No mismatched existing
  file and no unsafe existing path was found.
- All seven runner prerequisites and all three comparison references are absent.
- Largest missing groups: 44 OPDI files 1,532,598,644bytes; six nomonth OOF files
  258,554,612bytes; six E116prod OOF files 158,472,356bytes; 852 kt1 ADS-B files
  149,699,035bytes; saved regressor 118,773,552bytes. Exact paths are in JSON.

This amount is feasible to stage on Home. Mac need not hold a second heavy copy.
At least 2,729,366,036bytes would need copying before Git/runtime additions,
assuming the measured matching files remain unchanged.

## Exact closure missing from the inventory

These seven files are checked by the v2 registration guard but are **not** members
of `inventory_v2.json/files`:

```
scripts/reproduce_v12075_v2.py
scripts/reproduce_v12075_v2_launch_20261007.py
scripts/reproduce_v12075_lowmem_blend.py
analysis/reproduce_v12075_20261007/inventory_v2.json
analysis/reproduce_v12075_20261007/PROTOCOL_V2.md
tests/test_reproduce_v12075.py
tests/test_reproduce_v12075_v2.py
```

`scripts/reproduce_best.py`, its queue helper, E188 application source,
`docs/reproduction_manifest.json`, original setup and 207 frozen source files
are already pinned. Do not substitute current `src/` for the frozen source tree.

The v2 registration also executes `git show` against commit
`70aff7199731c381945c45754ee70c7e3a964aa5`: it needs the seven listed paths and
that commit's `HYPOTHESES_LOG.md` blob. E188 additionally requires
`d45cb738fb7671cf289ae8a400a30b5b46dfbb3d:scripts/e188_rank_20261006.py`.
Home has no `.git` directory. A plain tar of 3,711 files cannot satisfy these
guards. Required commits, tree IDs, blob IDs and SHA256 are recorded in
`closure.json`; a reviewed source release must preserve these Git objects or
provide a separately reviewed portable registration mechanism. No bypass or
history rewrite was attempted.

`pyproject.toml`, `uv.lock` and the registered journal blob are not in the data
inventory. Supplying only generic dependency metadata does not reproduce the
exact saved-cache runtime.

## Runtime boundary

All three saved tail stages compare their reconstructed training-input
signatures against the cached signature, including the full `sys.version`:

| Item | Required cached signature | Observed Home `.venv` |
|---|---|---|
| Python | `3.12.14 (main, Sep  1 2026, 14:09:38) [Clang 22.1.3 ]` | `3.12.15 (main, Oct  3 2026, 01:03:07) [Clang 22.1.3 ]` |
| NumPy | 2.5.3 | 2.5.3 |
| Polars | 1.44.2 | 1.44.2 |
| LightGBM | 4.7.0 | 4.7.0 |

Therefore the unchanged cache guard will reject Home's runtime even when all
files have arrived. The launcher also calls macOS `caffeinate`,
`memory_pressure`, and `vm_stat`; none exists on Home. Its resource contract is
six threads, 16 GiB monitored group RSS, 4 hours, 4 GiB initial free disk and 2.5 GiB
floor. Storage on Home is feasible; execution of the unchanged launcher on
Home is not qualified. No new environment or cross-platform exemption is
authorized by this report.

## Comparison references stay separate

The following files are verification references, never prediction inputs:

```
data/e203_chain_20261007/submissions/gentle-igloo_v940100.parquet
data/e203_chain_20261007/submissions/parts_v940100.npz
submissions/gentle-igloo_v12075.parquet
```

The first two compare the regenerated premix before the tail. The last is read
only after a newly assembled final output has been sealed. Their hashes and
sizes are a separate manifest group. v12075 reference SHA is
`90d3cd0b5a68cc8008f2142f1801857752574b525b2ff046ad6bf4089a70aee2`.

## Remaining packaging work

No existing reviewed archive builder/extraction verifier was found.
`prc_cold_archive_cleanup_20261007.py` verifies an unrelated 21-file cold copy
before deletion; it does not build a replay archive. Current documentation
explicitly leaves distribution/acquisition packaging unfinished.

A separate packaging step must close the source/Git/runtime requirements,
copy only missing pinned files into fresh Home staging, recheck every member,
and keep comparison references separate. An explicit allowlist must reject
absolute paths, `..`, link traversal and links outside authorized dependency
roots; credentials, personal files and arbitrary repository/home globs must
never enter the package. It must publish exact relative paths, SHA256 and byte
counts, then verify every archive member checksum or an isolated extraction
before claiming completion. No untested tar or improvised Git export was made.

Receipts: `dependency_manifest.json` (expected paths/hashes/sizes and separated
references), `home_check.json` (actual Home checks), `closure.json` (closure,
Git objects/runtime and missing-prefix totals), `check_home.py` (bounded
read-only checker). This report is the completed deliverable for the narrowed
task; archive construction and qualification have not been performed in this check.

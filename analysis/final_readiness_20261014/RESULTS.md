# Final submission readiness — 14 October 2026

Preparation completed before the 10:00 UTC deadline. No production model was
changed or refitted; no prediction was changed; no new PUT was performed.

## Submitted predictions

- Object: `gentle-igloo_final.parquet`, 670,790 departures, 2,145,576 bytes.
- SHA-256: `bbae2cc544989a0faf68abb85dbb4a9aab5ea38eec98d01fa9282c332b4234a1`.
- Uploaded once on 12 October at 08:44:59 UTC. The canonical ledger records
  `uploaded` and `closed`; regular uploads are closed.
- Fresh read-only STAT/GET from the competition bucket at 09:47:07 UTC on
  14 October returned exactly this SHA and size; see [remote_object.json](remote_object.json).
- Fresh local validator: every hard check passed. Reference comparison uses
  v12081 and its actual score 230.4478 s; see [validate.json](validate.json).
  The reference comparison is not a measured blind-final score.

## Review package

- README, GPL-3.0, source declarations, model design, validation, reproduction,
  final-submission documentation, Python lockfile and GPU requirements are present.
  [reviewer_files.json](reviewer_files.json) records 16 core file checksums.
- 481 relative links across 25 reviewer documents checked: no missing files and
  no links to untracked files; see [links.json](links.json).
- The tracked paper source, generated LaTeX and reading PDF include the 13 October
  audits and concurrent-work discussion. The PDF has 52 pages; its first page was
  rendered and visually checked. This is not a full visual re-review of all pages.
- The paper remains an optional JOAS draft, not a journal submission. Unknown
  blind-final results and the uncreated archive DOI remain explicitly marked.
- Full retraining from raw data in a clean clone has not been demonstrated.
  Component replay and its limitations remain documented in `docs/reproduction.md`.

## Verification

- 56 unittest modules run in separate processes: 548 tests, 546 passed and 2 skipped.
- The planning-triangulation module uses its own FunctionTestCase entry point:
  21 tests passed via `.venv/bin/python tests/test_planning_copy_triangulation_20261012.py`.
- Total: **567 passed, 2 skipped**. No dependencies were installed or changed.
  [tests_isolated.json](tests_isolated.json) retains the raw module results;
  its planning-module `NO TESTS RAN` discovery result is superseded by the direct
  21-test run recorded in [test_summary.json](test_summary.json).
- Monolithic discovery is not claimed to pass: it stalled at LightGBM model reload
  in an OpenMP barrier. The reload module passes alone (11 tests). A diagnostic
  monolithic run excluding that method also exposed shared `submission_guard.DAY`
  state left at 13 October while its own tests expect 5 October. All those guard
  tests pass in isolation. Production source, upload guards and frozen hashes were
  kept unchanged. Use module isolation for these checks.
- TruffleHog 3.99.2 scanned all Git history without online credential validation:
  seven detections, all manually classified as public GitHub URL substrings or
  a protocol SHA-256. No unresolved finding. The incremental scan after `f91b858`
  returned zero detections. See [secret_scan.json](secret_scan.json).
- `.env`, `.prc-upload-credentials.json` and competition datasets remain ignored.
  Existing history is retained as previously agreed; working notes and historical
  competitor analyses will become visible when the repository is made public.

## Remaining external actions

The repository was confirmed PRIVATE and GitHub recognised GPL-3.0 during this
preflight. Freeze and push the release commit before 10:00 UTC; record its exact
SHA in the local delivery package. Following the organiser's saved 8 October
instruction, make the repository public shortly after 10:00 UTC (11:00 Lisbon),
verify anonymous access, then send the prepared link and release SHA to
`challenge@opensky-network.org`. This preparation does not send that email or
publish the repository. A local source archive, prediction copy, checksum manifest
and completed email draft accompany the release handoff.

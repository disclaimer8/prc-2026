# E187 and artifact protection implementation plan

> For agentic workers: use subagent-driven-development with isolated file ownership, synthetic tests before implementation, and independent final review.

Goal: measure whether selection on decided-body improves the current six-member mixture, and prevent partial/overwritten artifacts from being accepted as complete results.

Spec: the user approved `analysis/codex_deep_audit_2026-10-04.md` in the main checkout and said “давай”. This plan implements its first two immediate actions; it does not alter E182's registered experiment.

Architecture: a new standalone E187 script reads existing OOF and selects weights; a new strict validator protects evaluation; ranking outputs gain isolated checkpoint and final-publication handling. The three implementation surfaces are independent. Root coordinates commits, preregistration, integration, and the real E187 run.

Tech stack: existing Python3.12, NumPy, Polars, unittest. No dependencies or environment changes.

## Global constraints

- Workspace: `/Users/denyskolomiiets/prc-2026-codex-e187`, branch `codex/e187-oof-validation`.
- Source data: `/Users/denyskolomiiets/prc-2026`; treat all existing arrays and submissions as immutable.
- Use `.venv/bin/python` only. Never `uv run`/`uv sync`/package installs.
- OMP/BLAS/Polars at most two threads; no base-model fitting or expensive data copies.
- No implementation agent commits, modifies journal, launches real selection, or starts training. Root registers journal+code in ONE commit BEFORE real E187.
- Do not modify `src/e172_members_cv.py`, `src/fold_metrics.py`, `src/gate.py`, running shell queues, or E182/E185 inputs.
- Synthetic tests may run before preregistration. Every new actual OOF output lives under a fresh E187 directory.
- Do not expose secrets, environment contents, or broad process command listings.
- Existing foreign untracked files stay untouched; explicit git paths only.

## Task 1: E187 objective comparison on existing predictions

Files owned: `src/e187_weight_objective.py`, `tests/test_e187_weight_objective.py`.

Input: FIN1b control `data/cv_preds_fin1b_hz`, members `raw,lobt,sched,eobt,aobt,mlpa` in `data/cv_e130<name>`, feature metadata FIN1b. Reference saved old selection: `data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlpa_f1b`. All paths resolve against explicit `--repo-root`; output specified separately.

Fixed grid: exact E172 grid step0.05, totalweight≤1.0, per-member box `0-0.1,0.05-0.2,0.05-0.15,0.05-0.15,0.1-0.2,0.15-0.35`, plus all-zero vector. Preserve deterministic original itertools ordering and ties.

Generate each prediction exactly as E172: reconstruct regular/day values, expected mixture, LOBT clip using current train.py constant, nonnegative clip, then preserve the original delta between stored control pred and rebuilt mixture. Select weights independently for held-out fold k using SSE summed over the OTHER five folds. Two objectives only: all SSE (control) versus decided AND y≤7200 SSE (candidate). Decided primary reproduces current fold_metrics mask; also report exact actual TabPFN mask `LIRF & FLIGHT_ID null & 14400<D<=108000` separately. Keep feature/mask definitions explicit.

Primary comparison: body-selected predictions minus all-selected predictions, for six pairs(1,7)..(6,12). Gate: six complete, decided<0 with at least4/6 lower, trimmed<0, y≤7200<0. Report full/all/tail and exact protected/nonprotected tail deltas; report Jan/Jul, airport/NM slices. No score tuning or new grids.

Secondary descriptive comparison: fixed production weights `[.033,.10,.067,.083,.10,.30]` on the SAME control/member OOF. Clearly label pre-tail, not v1139 (CatBoost/seeds/stages absent). Compare all-selected output against stored reference and fail on material discrepancy, with numerical tolerance stated before run (maxabs1e-6 seconds; SSE sums may have float rounding but selected weights must match or be explained before scoring).

Use vectorization/chunking to avoid a huge N×grid allocation. Exact clipped evaluation required; quadratic approximation may not replace it. Float64 sums. Process fold-by-fold, write separate new control/candidate NPZ with synchronized values, unique IDs/y/month checks, full-sixfold completeness, finite values. Source hashes and fixed config go in manifest/JSON. Do not silently overwrite a completed run; lock/no-clobber output. Runtime progress per fold. Default resource use two threads.

Tests first: synthetic examples where overwritten tail changes chosen weights; held-out y perturbation cannot affect its selected weights; exact clipped prediction equals scalar reference including NaN LOBT and nonzero stored delta; grid ordering/constraints; stale values are rebuilt; incomplete/duplicate/mismatched/nonfinite input rejects. Tests use fixtures and no real CV.

Deliver JSON with all fold weights, primary metrics, gate flags, secondary fixed-weight metrics, provenance, and a concise report that root can use for verdict. Exit status distinguish invalid execution from valid failed experimental gate.

## Task 2: Safe E130 ranking output

Files owned: `src/e130_rank.py`, new `src/rank_artifact.py`, `tests/test_rank_artifact.py`.

Preserve model mathematics and caller positional arguments. Add explicit `E130_OUT` output override. Refuse an existing final output before model training. Independent review found that in-place overwrite cannot safely distinguish a file replaced by an uncooperative legacy writer while fitting; therefore `E130_OVERWRITE=1` is rejected too, and a fresh output path is required. Never overwrite legacy output with a FIN1b run (fail early with useful error if FIN1b + implicit legacy path). Existing finished final remains unchanged on failure.

Save per-seed progress only to a separate clearly marked checkpoint; final output is published atomically after all requested seeds. Final NPZ carries backward-compatible mvt_id/reg/per_seed plus complete status, expected/completed seed counts, features path/hash, mode, rounds, source/config provenance. Reject zero/negative seeds/rounds, nonunique/nonfinite IDs/predictions, wrong row count for this ranking (344841). Use a per-output exclusive lock to prevent two publishers; atomic no-clobber publication even if another output appears during run. Helpers accept configurable expected_rows for fixture tests. Check final before expensive feature preparation.

Tests first: simulated interruption leaves old final intact; checkpoint remains separate; incomplete seeds cannot publish; successful final exact mean/count/IDs; collision does not overwrite; even explicit overwrite is rejected; output override + FIN1b guard; lock contention; nonfinite and invalid ID sets.

No actual LightGBM fits or real-output mutation; synthetic I/O tests only. Document the safe future E182 command using a distinct FIN1b filename.

## Task 3: Strict OOF validation and acceptance CLI

Files owned: new `src/strict_oof_gate.py`, `tests/test_strict_oof_gate.py`.

Do not alter legacy gate used by the running queue. Implement a new explicit CLI accepting control/candidate directories and `--features`. Before metric computation require exactly the expected six fold names, unique mathematically integer IDs in each and across folds (legacy float64 IDs accepted only if finite and exact integral within safe range), exact same ID sets and y, y also matching authoritative TAXITIME_SEC_mvt in features, finite y/pred, aligned months, complete coverage of training IDs in features. Reject extras, missing folds, mismatched shapes, duplicates, nonfinite values, and malformed npz. Restore no data silently. Values-consistency check may be optional because legitimate old E172 has stale candidates; report it if requested, don't mislabel pred invalid.

Compute standard decided/strict/all/body/trimmed/4of6 metrics consistent with legacy gate, print evidence and JSON option. Require globaldecided<0, at least4/6 lower andtrimmed<0; with `--require-body` also body<0. Distinct nonzero status for valid rejected experiment versus malformed inputs. Gate can run audit-only validation without demanding improvement for identical control/candidate. Must not fit or import heavy training.

Tests first: shuffled row alignment passes; missing sixth fold rejects; duplicate ID and incorrect candidate target reject; nonfinite reject; wrong month and training coverage reject; a valid poor candidate returns rejection, not success; a valid improvement passes; metric parity against hand-computed fixture. Existing production path untouched.

## Integration and review

- [x] Root reviews each diff and runs the owned tests; a fresh reviewer checks the whole patch before use. Two P2 findings fixed and independently re-reviewed; 61 synthetic tests pass.
- [x] Root writes E187 preregistration in HYPOTHESES_LOG (№256), forecast decided−0.02s, interval[−0.20,+0.20]; this is judgment, not data-derived. Forecast board0.00s, interval[−0.10,+0.10]. Journal+experiment+tests committed together as `47c1005` before real OOF selection (started 20:27Z).
- [x] Run E187 with source-data root and fresh output; save complete results and gate evidence. `data/e187_20261004_47c1005`, decided−0.004902s, 3/6 lower, rejected. Separate verdict №257, forecast inside interval; protocol unchanged.
- [x] Validate existing current control/reference with strict audit-only mode; run strict metrics on E187 outputs. Both valid; independent E187 gate reproduces rejection and exact metrics; reg/values consistent.
- [x] Integrate reviewed code into main checkout by fast-forward to `47c1005`; 67/67 tests there, including existing E186 tests. Queued E172/gate/metric/scripts unchanged. Keep isolation branch and saved evidence.
- [x] Update operational handoff with paths, commands, hashes/commit and result; Mac AOBT E182 continues, Hetzner E182 finished and E184 started. Three guard files copied to Hetzner after verifying/backing up its original e130_rank.py; SHA256 equal locally/remotely, remote py_compile and strict CLI --help pass.

## Review focus

1. Held-out labels must not select their own weights, including ties and aggregate subtraction.
2. The clipped mixture and preserved old delta must match E172 exactly; saved candidates must match new reg.
3. Final ranking must remain unchanged on exceptions, partial completion, collisions and competing writers.
4. Fold/ID/target correctness is required before a positive metric can count as success.
5. Actual PRC selection may run only after preregistration commit, and no changed code is fed to an already registered queue.

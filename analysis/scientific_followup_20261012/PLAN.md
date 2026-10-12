# Scientific follow-up implementation plan — 12 October 2026

**Goal:** Test residual dependence between departures and independently triangulate the interpretation of planning-time matches, then incorporate bounded findings into the existing article and documentation.

**Authorization:** Denys replied “давай” on 12 October after the proposal of these two analyses and correction of the pre-off-block claim. This authorizes local analysis, its code, and manuscript/documentation updates. The final submission has already been uploaded; no further submissions or changes to its recipe are part of this work.

**Architecture:** Two independent scripts read existing 2025 records and saved model/ADS-B outputs. Each has a protocol, input hashes, fail-closed alignment checks, synthetic statistical checks, machine-readable results and a report. A coordinating review checks the protocols before the registration commit and checks evidence before article integration.

**Environment:** Existing `.venv/bin/python`, NumPy/Polars and installed plotting tools. No environment changes, no new fits, no model inference. Local runs use bounded memory and at most two Polars threads each.

## Responsibilities and deliverables

- [x] Confirm current final-upload state and inspect project rules. The final is bbae2cc5, uploaded once on 12 October; journal ended at 462 when work began. Pre-existing unrelated change: `analysis/e197_trajectory_execution_20261007.md`.
- [x] Dependence analysis: `scripts/regime_dependence_audit_20261012.py`, its synthetic tests, and `analysis/regime_dependence_20261012/PROTOCOL.md`. Estimate conditional airport-day/hour residual dependence from FIN1b OOF predictions. Distinguish binary-regime uncertainty from RMSE uncertainty and common recording behavior from common model misspecification.
- [x] Triangulation analysis: `scripts/planning_copy_triangulation_20261012.py`, its synthetic tests, and `analysis/planning_copy_triangulation_20261012/PROTOCOL.md`. Prespecify tolerances, timestamp offsets that preserve the minute lattice, eligibility for independent ground episodes, populations and missing-data summaries. Treat ADS-B first movement as a fallible proxy.
- [x] Root reviews code/protocols and registers both with explicit outcome criteria and forecasts in journal entries 463 and 464, each together with its source code in its own local commit before outcome calculations.
- [x] Execute both bounded analyses; retain all primary and sensitivity results, including null or contradictory findings, input and code hashes, runtime and diagnostics.
- [x] Review results against source arrays and synthetic known-effect controls. Independently inspect exclusions, denominators, labels, uncertainty intervals and multiplicity handling.
- [x] Correct `docs/pru-view.md` to describe retrospective reconstruction, not pre-off-block forecasting or a proven irreducible error floor.
- [x] Integrate results and limits into `docs/data-quality-regimes.md`, an accessible follow-up report, and the editable article `paper/gentle-igloo-joas-draft.md`. Update the evidence outline and reproducibility description; regenerate `paper/main.tex` and the reading preview through the existing build.
- [x] Record verdicts in `HYPOTHESES_LOG.md`, update the current brief and handoff, rebuild the brief, and verify links/builds. Preserve all unrelated files and use only explicit paths in commits. No publication or outbound correspondence.

## Statistical review focus

1. All six OOF folds must map one-to-one to the correct 2025 departure IDs, with finite probabilities, correct class order and matching outcomes.
2. Permutation strata must preserve airport/season, missingness, probability and relevant time-of-day structure; exchangeability remains a working assumption, not established causality.
3. Previously observed schedule spikes and the LEBL summer episode are prior evidence. These analyses are prespecified follow-ups on already explored data, not fresh confirmatory discovery.
4. Negative-control offsets must preserve timestamp discretization; overlaps, missing candidate timestamps and absent ground coverage need explicit denominators.
5. A reduction after deleting rule-selected records is a sensitivity result. Neither rule agreement nor ADS-B proxy agreement establishes erroneous records, corrected congestion, or a lower bound on contamination.

## Completion evidence

The [combined report](RESULTS.md) states the supported temporal association, planning-time concentration, limited derived geometric support, and inconclusive strict raw evidence. Numerical claims link to saved reports and tables. The blind February/June competition result remains unknown.

- Dependence registration: `63ffe1b11f0f7f5c4677e3d895b405c449a05283`, journal 463; verdict 465. One run, 110.245 s, peak 1.412 GiB.
- Timestamp registration: `bf08d014000bd51ad65ae045be644ab0629f6c8f`, journal 464; verdict 466. One run, 20.20 s wall, peak 1.681 GiB. A separate label-blind, post hoc admission diagnostic explains the zero strict-prefix count; no thresholds or frozen outcomes changed.
- Seven dependence and 21 triangulation synthetic checks passed before outcome runs. Frozen source/test/protocol/preflight bytes were subsequently rechecked. Independent artifact audits reconcile the statistics, denominators, support and output hashes; their limits are explicit in each verification receipt.
- Article reading preview: 51 pages, 270-word abstract, 13 figures and 17 tables. Existing numeric tables retained; two incorrect narrative comparisons were corrected against the archived CSV. New methods/results and representative PDF pages inspected visually; whole-document contact sheets checked by the paper agent. No unresolved reference markers were found in extracted text; internal TeX warnings are not independently certified because the existing builder removes its successful compiler log.
- Brief build initially failed because Python `markdown` is absent. The renderer now falls back to the already installed Pandoc, without installing packages or changing `.venv`; the header deadline comes from Markdown and local links resolve from `review/`. Successful, idempotent build; all 118 sections, 116 tables and 3275 data cells retained. This small documentation-tool repair is outside the scientific estimands.
- 49 local Markdown links and newly introduced brief links verified. Final prediction SHA remains `bbae2cc544989a0faf68abb85dbb4a9aab5ea38eec98d01fa9282c332b4234a1`.
- Explicit-path local commit only; no push, publication or correspondence. The unrelated E197 change remains unstaged.

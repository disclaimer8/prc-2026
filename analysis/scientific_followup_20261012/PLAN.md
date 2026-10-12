# Scientific follow-up implementation plan — 12 October 2026

**Goal:** Test residual dependence between departures and independently triangulate the interpretation of planning-time matches, then incorporate bounded findings into the existing article and documentation.

**Authorization:** Denys replied “давай” on 12 October after the proposal of these two analyses and correction of the pre-off-block claim. This authorizes local analysis, its code, and manuscript/documentation updates. The final submission has already been uploaded; no further submissions or changes to its recipe are part of this work.

**Architecture:** Two independent scripts read existing 2025 records and saved model/ADS-B outputs. Each has a protocol, input hashes, fail-closed alignment checks, synthetic statistical checks, machine-readable results and a report. A coordinating review checks the protocols before the registration commit and checks evidence before article integration.

**Environment:** Existing `.venv/bin/python`, NumPy/Polars and installed plotting tools. No environment changes, no new fits, no model inference. Local runs use bounded memory and at most two Polars threads each.

## Responsibilities and deliverables

- [x] Confirm current final-upload state and inspect project rules. The final is bbae2cc5, uploaded once on 12 October; journal ends at 462. Pre-existing unrelated change: `analysis/e197_trajectory_execution_20261007.md`.
- [ ] Dependence analysis: `scripts/regime_dependence_audit_20261012.py`, its synthetic tests, and `analysis/regime_dependence_20261012/PROTOCOL.md`. Estimate conditional airport-day/hour residual dependence from FIN1b OOF predictions. Distinguish binary-regime uncertainty from RMSE uncertainty and common recording behavior from common model misspecification.
- [ ] Triangulation analysis: `scripts/planning_copy_triangulation_20261012.py`, its synthetic tests, and `analysis/planning_copy_triangulation_20261012/PROTOCOL.md`. Prespecify tolerances, timestamp offsets that preserve the minute lattice, eligibility for independent ground episodes, populations and missing-data summaries. Treat ADS-B first movement as a fallible proxy.
- [ ] Root reviews code/protocols and registers both with explicit outcome criteria and forecasts in journal entries 463 and 464, together with source code in one local commit before outcome calculations.
- [ ] Execute both bounded analyses; retain all primary and sensitivity results, including null or contradictory findings, input and code hashes, runtime and diagnostics.
- [ ] Review results against source arrays and synthetic known-effect controls. Independently inspect exclusions, denominators, labels, uncertainty intervals and multiplicity handling.
- [ ] Correct `docs/pru-view.md` to describe retrospective reconstruction, not pre-off-block forecasting or a proven irreducible error floor.
- [ ] Integrate results and limits into `docs/data-quality-regimes.md`, an accessible follow-up report, and the editable article `paper/gentle-igloo-joas-draft.md`. Update the evidence outline and reproducibility description; regenerate `paper/main.tex` and the reading preview through the existing build.
- [ ] Record verdicts in `HYPOTHESES_LOG.md`, update the current brief and handoff, rebuild the brief, and verify links/builds. Preserve all unrelated files and use only explicit paths in commits. No publication or outbound correspondence.

## Statistical review focus

1. All six OOF folds must map one-to-one to the correct 2025 departure IDs, with finite probabilities, correct class order and matching outcomes.
2. Permutation strata must preserve airport/season, missingness, probability and relevant time-of-day structure; exchangeability remains a working assumption, not established causality.
3. Previously observed schedule spikes and the LEBL summer episode are prior evidence. These analyses are prespecified follow-ups on already explored data, not fresh confirmatory discovery.
4. Negative-control offsets must preserve timestamp discretization; overlaps, missing candidate timestamps and absent ground coverage need explicit denominators.
5. A reduction after deleting rule-selected records is a sensitivity result. Neither rule agreement nor ADS-B proxy agreement establishes erroneous records, corrected congestion, or a lower bound on contamination.

## Completion evidence

The user-facing result will state which hypotheses gained support, which remained unresolved, and their practical implications. Numerical claims must be traceable to newly saved reports/tables. The blind February/June competition result remains unknown.

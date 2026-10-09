# JOAS paper: outline (state of 9 October 2026)

Companion to `gentle-igloo-joas-draft.md`. Every claim below names the file, figure or table that
carries its evidence; the draft must not say more than these sources say. The organiser stated on
9 October 2026 that a JOAS paper is not required but welcome; the draft is prepared on that basis.

## Working title (11 words)

**A Recording-Regime Mixture for Taxi-Out Time Reconstruction at Ten European Airports**

## Thesis (one sentence)

The challenge label is a *recorded* interval produced by several recording regimes, so a model of
how the label is produced (a probability mixture over regimes, anchored to the NM off-block proxy
and corrected by the aircraft's own ADS-B track) is worth more than any learner choice, and the
same discipline that found this (temporal folds, pre-registered gates, forecasts before every
upload, a label-free blend identity, full disclosure) is what makes the result reusable for
performance monitoring beyond the leaderboard.

## What the paper is and is not

The team finished 16th of 239 on the ordinary board (230.4812 s; third place 213.5212 s, a gap of
about 17 s). The contribution is therefore **not** rank. It is (a) the design, (b) the honest
uniform recipe for the blind final with its measured price, (c) the evidence discipline, (d) the
data-quality and PRU reading of the same model, and (e) documented negative results. The paper says
this in the introduction and the conclusion.

## Contributions (five)

1. **Recording regimes as a model of label production.** 6.76 % of 2025 labels equal take-off
   minus a planning time; at LIRF 83.7 % of labels above one hour do. A six-class mixture removed
   135 s of RMSE on day one; a calibrated detector of substituted records falls out of it.
   Evidence: `docs/data-insights.md`, `docs/figures/fig2_recording_regimes.png`, `docs/data-quality-regimes.md` §2.
2. **A chain that reads observations, not aggregates.** Residual to a per-airport calibrated NM
   proxy; coverage-gated surface stages from the flight's own adsb.lol track; an ordered-trajectory
   neural member whose weight was tuned at chain level on all twelve 2025 months (E212); TabPFN-2
   on the sparse Rome tail. Evidence: `docs/model-design.md`, `analysis/e212_nn_weight_10seed_20261008.md`,
   `docs/validation.md`.
3. **An honest blind-final recipe with its price measured.** One procedure over four months,
   leaderboard-specific parts removed (E81 blend, 42 frozen tail rows), prices measured by single
   uploads (canary +1.2546 s; E81 alone +0.5676 s; fresh tail rows pending), file sealed by SHA,
   validated, replayed byte-identically. Evidence: `docs/final-submission.md`, `analysis/v12090_upload_20261008/RESULTS.md`,
   `analysis/v12078_upload_20261009/RESULTS.md`, `analysis/final_tail_20261009/`.
4. **Evidence discipline and what transferred.** Temporal month-pair folds, pre-registered gates,
   forecasts before uploads, 38-pair CV-to-board calibration by type of change, the Gram identity
   for blends (and its limit, learned on v12078), and an eight-item leaderboard-use disclosure.
   Evidence: `docs/development-history.md` §3–§6, `docs/negative-results.md` lessons 1–12.
5. **A PRU reading and calibrated uncertainty.** The detector is calibrated for counting
   (9.46 % vs 9.58 %), the emulated additional-taxi-out indicator matches published 2025 values
   (MAE 3.3 s, r = 0.999, April–October), copies inflate the LIRF indicator by 42 s per flight;
   mixture intervals cover 51.0/80.1/89.7 % at 50/80/90 %. Evidence: `docs/data-quality-regimes.md`,
   `docs/pru-view.md`, `docs/uncertainty.md`, `docs/tables/*.csv`.

## Sections and the claims each must support

| § | Claims (2–4 each) | Evidence |
|---|---|---|
| Abstract | Task; three data properties; chain; 230.48 s, 16/239; what transferred; final recipe; PRU by-product | all of the below |
| 1 Introduction | Task is retrospective reconstruction (organiser 16/19 Sep); blind final announced 8 Oct; five contributions; rank is not the contribution | `docs/organiser-clarifications.md`, `README.md` |
| 2 Data | 2.2 regimes (6.76 %, LIRF 24.1 %, 83.7 % above 1 h; same-minute test); 2.3 proxy 384.9 s, medians −118…+296 s; 2.4 coverage drift incl. final months (EDDF Feb 32.2 %, LIRF Jun 0.7 %, Jul 6.2 %); 2.6 error budget (LIRF 30.8 %, no-NM 36.6 %) | `docs/data-insights.md`, fig2–fig5, `analysis/final_adsb_derivatives_20261008/RESULTS.md`, `docs/final-submission.md` §5 |
| 3 Method | 3.1 chain of v12077; 3.3 mixture + cost weights + rejected variants; 3.4 members incl. ten-seed NN at 0.35 chosen on 12 months (0.40 by rule, 0.35 only value better in all months); 3.5 stages and gate; 3.6 tail; 3.7 E81 and LSZH; **3.8 the uniform final recipe** (table of differences, stage order, served rows, OPDI coverage) | `docs/model-design.md`, `docs/final-submission.md` §2, `analysis/e212_…md` §4–§6, `analysis/final_tail_20261009/receipts/` |
| 4 Validation | Six folds; no fully nested CV (`quality_validated_on_full_chain=false`); Table 3 incl. E212 rows; CV→board by type (ratio 1.0 / 0.7 / 0.2–0.3 / ≤ 0) and the new observation that a mixture-weight shift transferred ≈ 3× its OOF estimate; Gram identity exact for complete files, not across stages after the blend (v12078 missed by +0.45 s) | `docs/validation.md`, `docs/development-history.md` §3–§4, `analysis/v12077_upload_20261008/RESULTS.md`, `analysis/v12078_upload_20261009/RESULTS.md` |
| 5 Results | 5.1 111 uploads, 63 bests, phases, gap to third; 5.2 single-change controls incl. v12076/v12077/v12078; 5.3 OOF by airport; 5.4 generalisation Feb+Jun vs Jan+Jul (247.3 vs 279.2; 0/120 worse than proxy); 5.5 uncertainty; 5.6 detector and PRU indicator; 5.7 the final file (SHA, validator, replay, control v12080 pending, ranking pending); 5.8 negative results of 8–9 Oct (E207 v3, E209–E215, NN protection) | fig1, `docs/development-history.md` §1, `docs/generalisation.md`, `docs/uncertainty.md`, `docs/data-quality-regimes.md`, `docs/pru-view.md`, `analysis/final_tail_20261009/MANIFEST.json`, `docs/negative-results.md` |
| 6 Discussion | What generalises (observation vs aggregates); dataset-specific rules; known risks of the final months (EDDF/LIRF coverage, EHAM works 18,194 of 20,881 June rows, input audit); leaderboard-use disclosure (8 items; v12076/77 add none; final removes items 1–2); limitations | `docs/final-submission.md` §5, `analysis/final_months_input_audit_20261009.md`, `analysis/coverage_collapse_audit_20261008.md`, `docs/development-history.md` §6, `docs/model-card.md` |
| 7 Open science | GPLv3; data terms table; four reproducibility levels with the status of each (replay exact; retrain not run end to end; new period = dated final pipeline with receipts) | `README.md`, `docs/reproduction.md`, `docs/final-submission.md` §6 |
| 8 Conclusion | Mixture > learner; observation transfers; by-product for PRU; evidence boundary stated | — |

## Figures

Existing in `docs/figures/` (all regenerated 8 Oct; fig1 after v12077):

| Paper no. | File | Use |
|---|---|---|
| 1 | `fig2_recording_regimes.png` | regime shares, long labels, SSE share |
| 2 | `fig3_nm_proxy_error.png` | proxy error by airport |
| 3 | `fig4_adsb_coverage_drift.png` | coverage 2025 vs 2026, Jan/Jul |
| 4 | `fig5_error_budget.png` | rows vs squared error |
| 5 | ASCII stage diagram (in text) | chain of v12077 / final |
| 6 | `fig1_leaderboard_progression.png` | 501.23 → 230.48 s |
| 7 | `fig_generalisation_months.png` | OOF RMSE by month vs baselines |
| 8 | `fig_generalisation_airport_month.png` | body RMSE and gain by airport × month |
| 9 | `fig_uncertainty_calibration.png` | coverage by airport, PIT |
| 10 | `fig6_regime_detector_calibration.png` | reliability of the detector |
| 11 | `fig7_regime_effect_on_kpi.png` | copies per month and effect on the indicator |

New figures wanted (not yet made; data exist):

- **Per-month ΔRMSE of the chain against the neural weight** (E212 grid 0.05–0.60, twelve
  months, three- and ten-seed member) from `analysis/e212_nn_weight_10seed_20261008/per_month.json`:
  shows why 0.35 and not the rule's 0.40. Until then the paper carries the table.
- **ADS-B coverage of the four final months vs the same months of 2025** by airport (table exists
  in `docs/final-submission.md` §5); could extend fig4.
- Optional: price decomposition of the final recipe on Jan/Jul (v12075 → v12076 → v12077 →
  v12078 → v12079 → v12080) as a waterfall once v12079/v12080 are scored.

## Tables

In document order (as numbered in the draft): 1 feature families; 2 regular-candidate weights
(v12075 vs v12077/final); 3 final recipe vs v12077; 4 six-fold comparison incl. E212; 5 CV vs board
by change; 6 phases; 7 single-change controls incl. v12076–v12078; 8 OOF by airport;
9 generalisation by month group; 10 neural weight by month (E212); 11 uncertainty coverage;
12 detector thresholds; 13 PRU indicator effect by airport; 14 Jan/Jul price of the final recipe
(canary, E81, fresh tail, control); 15 negative results of 8–9 October; 16 leaderboard-use
disclosure; 17 data terms. Final-month ADS-B coverage is in the prose of §2.4 (a table could be
added from `docs/final-submission.md` §5).

## Still missing and when it becomes available

| Item | Placeholder in draft | Available |
|---|---|---|
| Score of v12079 (fresh tail rows on v12077) | `[[RESULT: v12079]]` | when the organiser's scorer processes the file (stalled since 08:54 UTC, 9 Oct) |
| Score of control v12080 (Jan/Jul rows of the final file) | `[[RESULT: v12080]]`; forecast 231.7 [231.2; 232.3] pre-registered | after upload, planned once the scorer recovers (journal 439, 441) |
| Blind-final ranking (Jan+Jul, Feb+Jun, four months) | `[[RESULT: final ranking]]` ×3 | after 14 Oct 2026 10:00 UTC |
| Author ORCID, contact e-mail, funding, AI-assistance wording | front matter, back matter | author, before submission |
| Public archive (Zenodo) of the redistributable component archive | reproducibility statement | if created after the deadline |
| JOAS template check (`extra/joas.cls`, biber run, 12-word title, ≤ 5 keywords, ≤ 300-word abstract) | `paper/README.md` | class not bundled (no licence file); pdflatex compiled on 8 Oct, biber not yet run |
| Cross-airport (leave-one-airport-out) evidence | none; the draft states it was not run | would need a new experiment; not planned before the deadline |

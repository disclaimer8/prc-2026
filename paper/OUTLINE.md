# JOAS paper: outline (state of 11 October 2026)

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

The team stood 18th of 242 on the ordinary board on 11 October 2026 (best v12081, 230.4478 s;
third place 213.2898 s, a gap of about 17 s; 16th of 239 with v12077, 230.4812 s, on 8 October). The contribution is therefore **not** rank. It is (a) the design, (b) the honest
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
   uploads (canary +1.2546 s; E81 alone +0.5676 s, v12078; fresh tail rows alone +0.6958 s, v12079 =
   231.1770 s), file sealed by SHA, validated, replayed byte-identically; control v12080 = 231.1805 s,
   0.02 s below its pre-registered corridor [231.2; 232.3] and 0.56 s better than the measured
   expectation (H = 231.7429 s by the Gram identity), explained without labels (regime probabilities
   after refit + new inputs 55 %, neural member 23 %, tabular members 16 %; noise ±0.43–0.55 s, so the
   ±0.3 s corridor was about half as wide as it should have been). Evidence: `docs/final-submission.md`
   §4 and §7, `analysis/v12090_upload_20261008/RESULTS.md`, `analysis/v12078_upload_20261009/RESULTS.md`,
   `analysis/v12079_upload_20261009/RESULTS.md` with log 444, `analysis/v12080_control_explanation_20261009.md`
   (+ `results/gram.json`), `analysis/final_tail_20261009/`. **Last stage E218 (PRU level), disclosed
   as an input from the scored months:** per airport × month half the gap between the indicator
   implied by the predictions and EUROCONTROL's published additional taxi-out time, net of the 2025
   emulation offset, outside de-icing months; 2025 gate passed (6/6 pairs, −0.137 s pooled, forward
   legs, 12/12 months); v12081 = v12077 + E218 = 230.4478 s (−0.0334 s, 0.025 s above its corridor);
   final rebuilt `bbae2cc5…` (stages before E218 identical to 9 Oct; 415,697 rows, RMS 4.687 s);
   control v12082 = 231.1445 s (−0.0360 s vs v12080, inside corridor); v12083 (E218 removed on LIRF
   rows) = 231.1749 s ⇒ gain splits LIRF −0.0304 s (84 %) / other ten cells −0.0056 s; LIRF transfers
   ≈ 0.69 of its gap, others 0.37–0.48 (2025: 0.93); the reading "non-LIRF transfer, LIRF neutral" is
   refuted. Evidence: `docs/final-submission.md` §2.5, `analysis/e218_pru_level_recheck_20261010.md`,
   `analysis/v12081_upload_20261011/RESULTS.md`, `analysis/v12082_control_20261011/RESULTS.md`,
   `analysis/v12083_upload_20261011/RESULTS.md` (+ `verdict.json`), `analysis/final_tail_20261011/`.
4. **Evidence discipline and what transferred.** Temporal month-pair folds, pre-registered gates,
   forecasts before uploads, 38-pair CV-to-board calibration by type of change, the Gram identity
   for blends (and its limit, learned on v12078), and an eight-item leaderboard-use disclosure.
   Evidence: `docs/development-history.md` §3–§6, `docs/negative-results.md` lessons 1–12.
5. **A PRU reading and calibrated uncertainty.** The detector is calibrated for counting
   (9.46 % vs 9.58 %), the emulated additional-taxi-out indicator matches published 2025 values
   (MAE 3.3 s, r = 0.999, April–October), copies inflate the LIRF indicator by 42 s per flight;
   mixture intervals cover 51.0/80.1/89.7 % at 50/80/90 %. On 2026, the indicator from the final
   file's predictions matches the published values in June (MAE 9.2 s) and July (6.6 s) against
   6.6/8.2 s for the same procedure on 2025; flags: LIRF −20…−46 s in all four months, LTFM January
   and February (snow) beyond the de-icing allowance; regime shares stable (6.95 % vs 6.71 %).
   Evidence: `docs/data-quality-regimes.md`, `docs/pru-view.md`, `docs/uncertainty.md`,
   `docs/tables/*.csv`, `analysis/final_pru_and_regime_check_20261009.md` (+ `result.json`).

## Sections and the claims each must support

| § | Claims (2–4 each) | Evidence |
|---|---|---|
| Abstract | Task; three data properties; chain; 230.45 s, 18/242; what transferred; final recipe; E218 with its input disclosed; detector | all of the below |
| 1 Introduction | Task is retrospective reconstruction (organiser 16/19 Sep); blind final announced 8 Oct; five contributions; rank is not the contribution | `docs/organiser-clarifications.md`, `README.md` |
| 2 Data | 2.2 regimes (6.76 %, LIRF 24.1 %, 83.7 % above 1 h; same-minute test); 2.3 proxy 384.9 s, medians −118…+296 s; 2.4 coverage drift incl. final months (EDDF Feb 32.2 %, LIRF Jun 0.7 %, Jul 6.2 %); 2.6 error budget (LIRF 30.8 %, no-NM 36.6 %) | `docs/data-insights.md`, fig2–fig5, `analysis/final_adsb_derivatives_20261008/RESULTS.md`, `docs/final-submission.md` §5 |
| 3 Method | 3.1 chain of v12077; 3.3 mixture + cost weights + rejected variants; 3.4 members incl. ten-seed NN at 0.35 chosen on 12 months (0.40 by rule, 0.35 only value better in all months); 3.5 stages and gate; 3.6 tail; 3.7 E81 and LSZH; **3.8 the uniform final recipe** (table of differences, stage order, served rows, OPDI coverage); **3.9 E218** (rule, why E145 failed, 2025 gate, plain statement of the input) | `docs/model-design.md`, `docs/final-submission.md` §2, `analysis/e212_…md` §4–§6, `analysis/final_tail_20261009/receipts/` |
| 4 Validation | Six folds; no fully nested CV (`quality_validated_on_full_chain=false`); Table 4 incl. E212 rows; corridors tested: 4 of 7 uploads of 8–11 Oct inside; the control v12080 0.02 s below (lesson: ±0.5 s per σ for a refitted chain), v12081 0.025 s and v12083 0.0009 s above; E218 row in Table 5; CV→board by type (ratio 1.0 / 0.7 / 0.2–0.3 / ≤ 0) and a mixture-weight shift that transferred ≈ 3× its OOF estimate; Gram identity exact for complete files, not across stages after the blend (v12078 missed by +0.45 s), exact for H = v12078 + 42 fresh rows (231.7429 s) | `docs/validation.md`, `docs/development-history.md` §3–§4, `analysis/v12077_upload_20261008/RESULTS.md`, `analysis/v12078_upload_20261009/RESULTS.md` |
| 5 Results | 5.1 111 + 3 + 3 = 117 uploads, 64 bests (v12081 the last), eight phases, gap to third; 5.2 single-change controls incl. v12076–v12083 (E218 split LIRF / other); 5.3 OOF by airport; 5.4 generalisation Feb+Jun vs Jan+Jul (247.3 vs 279.2; 0/120 worse than proxy) and Figure 9 (NN weight by month); 5.5 uncertainty; 5.6 detector and PRU indicator; 5.7 recipe price (Table 14, Figure 13), the file of 9 Oct (SHA, validator, replay), control v12080 and its explanation, PRU 2026 check of the 9 Oct file and regime shares, E218 on the board (v12081–v12083) and the final file of 11 Oct `bbae2cc5…`, ranking pending; 5.8 negative results of 8–10 Oct (E207 v3, E209–E217, NN protection) | fig1, `docs/development-history.md` §1, `docs/generalisation.md`, `docs/uncertainty.md`, `docs/data-quality-regimes.md`, `docs/pru-view.md`, `analysis/final_tail_20261009/MANIFEST.json`, `analysis/v12080_control_explanation_20261009.md`, `analysis/final_pru_and_regime_check_20261009.md`, `analysis/e216_chain_scalars_20261009.md`, `analysis/e217_winter_nn_weight_20261010.md`, `analysis/final_tail_20261011/MANIFEST.json`, `analysis/v1208{1,2,3}_*_20261011/RESULTS.md`, `docs/negative-results.md` |
| 6 Discussion | What generalises (observation vs aggregates); dataset-specific rules; known risks of the final months (EDDF/LIRF coverage, EHAM works 18,194 of 20,881 June rows, input audit, LIRF level shift (now moved by E218; June +21 s unmeasured) and LTFM snow from the PRU 2026 check, ±0.5 s regime-probability lottery); leaderboard-use disclosure (8 items; v12076–v12080 add none; final removes items 1–2; E218 added to the final by board rules, input a published aggregate of the scored months); limitations incl. corridor width, E218, cross-platform replay | `docs/final-submission.md` §5, `analysis/final_months_input_audit_20261009.md`, `analysis/coverage_collapse_audit_20261008.md`, `docs/development-history.md` §6, `docs/model-card.md` |
| 7 Open science | GPLv3; data terms table (PRU monthly files: no open licence, E218 input, organiser's answers of 27 Sep / 8 Oct); four reproducibility levels with the status of each (replay exact on the build platform; prc-cloud replay limits; retrain not run end to end; new period = dated final pipeline with receipts) | `README.md`, `SOURCES.md`, `docs/reproduction.md` §2c, `docs/final-submission.md` §6, `analysis/final_replay_prc_cloud_20261010.md` |
| 8 Conclusion | Mixture > learner; observation transfers; by-product for PRU; evidence boundary stated | — |

## Figures

All figures used by the paper are in `paper/figures/` (JOAS expects one `figures/` folder next to
`main.tex`). `paper/make_paper_figures.py` copies the reused ones from `docs/figures/` unchanged
(regenerated there on 8 October; fig1 after v12077) and draws the two paper-only figures.

| Paper no. | File in `paper/figures/` | Use | Produced by |
|---|---|---|---|
| 1 | `fig2_recording_regimes.png` | regime shares, long labels, SSE share | `scripts/make_doc_figures.py` |
| 2 | `fig3_nm_proxy_error.png` | proxy error by airport | `scripts/make_doc_figures.py` |
| 3 | `fig4_adsb_coverage_drift.png` | coverage 2025 vs 2026, Jan/Jul | `scripts/make_doc_figures.py` |
| 4 | `fig5_error_budget.png` | rows vs squared error | `scripts/make_doc_figures.py` |
| 5 | ASCII stage diagram (in text; a float in `main.tex`) | chain of v12077 / final | — |
| 6 | `fig1_leaderboard_progression.png` | 501.23 → 230.48 s | `scripts/make_doc_figures.py` |
| 7 | `fig_generalisation_months.png` | OOF RMSE by month vs baselines | `scripts/generalisation_report.py` |
| 8 | `fig_generalisation_airport_month.png` | body RMSE and gain by airport × month | `scripts/generalisation_report.py` |
| 9 | `nn_weight_by_month.png` (**new**) | ΔRMSE vs neural weight, 12 months, 3 and 10 seeds; final months highlighted | `paper/make_paper_figures.py` from `analysis/e212_nn_weight_10seed_20261008/per_month.json` |
| 10 | `fig_uncertainty_calibration.png` | coverage by airport, PIT | `scripts/uncertainty_report.py` |
| 11 | `fig6_regime_detector_calibration.png` | reliability of the detector | `scripts/regime_quality_report.py` |
| 12 | `fig7_regime_effect_on_kpi.png` | copies per month and effect on the indicator | `scripts/regime_quality_report.py` |
| 13 | `final_recipe_price.png` (**new**) | (a) price of the recipe: E81 +0.568, fresh tail +0.696, sum +1.263, canary +1.255, control v12080 +0.699 against H +1.262 ± 1σ, control v12082 (with E218) +0.663; (b) v12078/v12079/v12090/v12080/v12081/v12082/v12083 against their corridors | `paper/make_paper_figures.py` from `analysis/v12080_control_explanation_20261009/results/{gram,sigma_piecewise,sigma_cluster}.json` and `analysis/v1208{1,2,3}_*_20261011/{score_response,forecast}.json`; corridors from `docs/final-submission.md` §4, §7, the canary report and the v12081 RESULTS |

Still optional (data exist, not made): ADS-B coverage of the four final months against the same
months of 2025 by airport (table in `docs/final-submission.md` §5; the paper gives it in the prose of
§2.4).

## Tables

In document order (as numbered in the draft; `build_tex.py` turns each into a captioned, labelled
longtable): 1 feature families; 2 regular-candidate weights (v12075 vs v12077/final); 3 final
recipe vs v12077 incl. the E218 row; 4 six-fold comparison incl. E212; 5 CV vs board by change incl. E218;
6 phases (eight); 7 single-change controls incl. v12076–v12083; 8 OOF by airport; 9 generalisation by month group;
10 neural weight by month (E212); 11 uncertainty coverage; 12 detector thresholds; 13 PRU indicator
effect by airport; 14 Jan/Jul price of the final recipe (canary, members, E81, fresh tail, H,
control) and the three E218 uploads; 15 negative results of 8–10 October incl. E216, E217;
16 leaderboard-use disclosure; 17 data terms incl. the PRU indicator files as the E218 input.

## Still missing and when it becomes available

| Item | Placeholder in draft | Available |
|---|---|---|
| Blind final ranking: upload time of the final file, RMSE and rank on Jan+Jul, Feb+Jun and four months; ordinary-board rank at close | `[[RESULT: final ranking — …]]` ×6 (abstract, §1, §5.1, §5.7, §6.3, §8) | after 14 Oct 2026 10:00 UTC |
| Zenodo DOI of the archived release / redistributable component archive | Open data statement | if created after the deadline |
| JOAS class build with the bibliography | — | `main.tex` compiled without errors against the official class on 10 Oct (40 pages, preprint layout, bibliography empty because biber is not installed); biber run still unverified |
| Cross-airport (leave-one-airport-out) evidence | none; the draft states it was not run | would need a new experiment; not planned |

Filled on 10 October: v12079 = 231.1770 s (+0.6958 s, log 444), control v12080 = 231.1805 s and its
explanation (log 445–447), the PRU 2026 check of the final file (log 448), E216 (log 442–443), the
funding statement and the AI-assistance statement.

Filled on 11 October: author ORCID 0000-0003-3212-3251 and contact e-mail denys@himaxym.com (given by the author); E217 (log 449–450); E218, its 2025 gate and data-use disclosure (log 451–452,
Section 3.9); the prc-cloud replay limits (log 453–454); v12081 = 230.4478 s (new best, 18/242),
the final rebuilt with E218 (`bbae2cc5…`), control v12082 = 231.1445 s and v12083 = 231.1749 s with
the exact LIRF / non-LIRF split (log 455–461); Figure 13 redrawn with the three E218 uploads.

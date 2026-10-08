# Negative results and lessons

This page summarises ideas that team gentle-igloo tested and did not keep, with
the measured effect and the reason. It is a selection of about 80 of the most
informative cases. The complete Russian register has every entry, including
accepted ideas and ideas closed without a run. Read this page together with the
[validation record](validation.md), which states what each kind of evidence can
and cannot establish.

## How the evidence was produced

**Data.** Training uses all 2025 departures (2,085,047 IDs in the evaluated
tables). The public ranking scores January and July 2026 (344,841 departures).

**Cross-validation.** The main scheme has six paired folds: months 1+7, 2+8,
3+9, 4+10, 5+11 and 6+12. Each fold holds out both months and trains on the
other ten. Each fold pairs a month from the first half of the year with the
same month of the second half, as January and July do in the ranking set. Where
boosting rounds are selected, the month after each held-out month is the inner
validation month. Candidate and control are compared on identical IDs. "x/6" is
the number of folds in which the candidate's squared error is lower.

**Subsets used in gates** (definitions in [validation.md](validation.md)):

- *decided*: rows the base model actually serves. It excludes LIRF departures
  without a Network Manager (NM) record and with take-off minus schedule ≥ 4 h,
  which a separate tail model handles. It became the main gate metric on 16 September.
- *body*: rows with observed target ≤ 7,200 s (written "y ≤ 7200" in the register).
- *trimmed*: the three (or twenty) largest and smallest paired differences removed in each fold.
- *after E14*: both arms scored after the LIRF flight-number rule E14. This gate
  leg was used from 23 September until E14 left the chain on 30 September.
- *LOMO*: twelve leave-one-month-out folds for the sparse LIRF tail. It reports
  the % change in SSE and "x/12" months improved.

**Units and signs.** Negative numbers are improvements. Most CV values in
seconds follow the historical convention of the strict gate: a paired SSE change
scaled by `6 * 2 * 252.43 * 344841`. This gives a first-order leaderboard-scale
value, not an exact RMSE difference. Where the source reports an actual RMSE
difference (mostly October experiments), the cell says **ΔRMSE**. The two
conventions are not comparable and should not be pooled. **LB** is the change in
public ranking RMSE between two scored files, both evaluated on all 344,841 rows.

**Pre-registration.** For each experiment the decision rule was written down
before the run. A typical early rule was "−0.30 s and at least 4/6 folds". Later
rules were "≥ 4/6, trimmed < 0, body < 0" or "both screen pairs improve". Before
each upload, a forecast range for the leaderboard change was also recorded. In
the October experiments, the protocol and code were committed before the first
model fit (for example, log No. 257, commit `47c1005`). Verdicts were applied as
written. A later change of mind is logged as a new entry and does not edit the
old one.

**Verdict symbols** (from the register): ✅ accepted, ❌ rejected after
measurement, ⛔ closed without a full run (already measured, not available, or
not permitted), 🟡 in progress. "❌ → ✅" marks a rejection that was later
reversed by a different test (see lesson 11).

**References.** "Log N" is entry N of [`HYPOTHESES_LOG.md`](../HYPOTHESES_LOG.md).
That file has 376 entries numbered 1–377 (No. 135 is unused) and runs from
23 September to 7 October 2026 (UTC). "Brief §X" is a section of
[`REVIEW_BRIEF.md`](../REVIEW_BRIEF.md). The brief holds the experiments from
before the log started (E1 to about E99) and, in §0, the registry of closed
ideas. "Val." is [validation.md](validation.md). Both registers are in Russian;
the numbers here are copied from them without re-rounding.

**Scope of a rejection.** A rejection closes the configuration that was
tested. It does not close every possible version of the idea. Several rejections
were later reversed (lesson 11).

## Rejected or neutral ideas

### External data sources

| Idea | How tested | Measured effect | Verdict | Reference |
|---|---|---|---|---|
| Other ADS-B receiver networks for Jan/Jul 2026 ground tracks (adsb.fi, ADSBHub, ADS-B Exchange, ADSB IQ, Internet Archive) | Archive and licence checks; ADS-B Exchange free sample of 1 Jan and 1 Jul 2026 | adsb.fi, ADSBHub: live state only, no archive. ADSB IQ: data from 29 Jun 2026, 0 ground points at all 10 airports. ADS-B Exchange: 52 aircraft on ground at LIRF, 36 at LTFM, 14 at LFPG in 14 shared instants (ours: 0 at LIRF and LTFM), but the archive is commercial | ⛔ not open data | Brief §0 |
| FlyItalyADSB (CC BY-SA 4.0) for Rome | Request to the network owners | Reply 16 Sep: no July 2026 archive ("we switched systems") | ⛔ | Brief §0 |
| Other copies of the adsb.lol archive (`staging-0`, `mlatonly-0`) to fill reception gaps (E49) | Ground-point counts | +0.5 % ground points, only where coverage already existed; 0 at LFPG, LIRF, LTFM. EDDF 30 Jan: 2,277 vs 2,430 points in the copy used | ❌ | Brief §0; Log 29; [coverage](../analysis/coverage_ceiling_2026-09-16.md) |
| ACARS/VDL2 "OUT" (off-block) messages | Full-day sweeps of airframes.io and other registries (2 Jul, 15 Jan 2026); labelled Dec 2025 match (E122); FRA/LHR/AMS on 15 Jul 2025 | OUT only at EGLL (25/669) and LSZH (18/439), 0 at the other eight hubs. E122 upper bound 0.008 s LB. FRA/LHR/AMS: OUT for 1 of 6,120 departures | ❌ | Log 9, 53, 55; [sweep](../analysis/acars_registry_sweep_2026-09-25.md) |
| OPDI exit-parking event as a second off-block source where we lack ground tracks (E159) | LOMO on LSZH 2025, then two LB slots | LSZH: −38.6 % RMSE on rows with the event in LOMO, but LB −0.131 s (forecast −1.2). Adding EDDF, which had no labels for calibration: LB +1.132 s | ✅ LSZH kept, ❌ EDDF | Brief §0; Log 215 |
| Other OPDI events (exit-apron etc.) combined as a stage (E168) | LOMO on LSZH 2025 | −0.4 % on tail rows | ❌ | Brief §0; Log 198 |
| Daily ATFM slot adherence from ansperformance.eu (H11) | Probe, half-year swap | Partial correlations ≤ 0.023; stage changes sign between half-years (+0.054 / −0.159) | ❌ | Brief §0; Log 167 |
| Calibration to EUROCONTROL PRU published taxi-out (airport × month, E145; runway × month) | 6-fold CV, then LB | Airport × month: CV −0.045 s (6/6), LB +0.114 s. Runway × month: CV +1.53 s, 0/6 | ❌ | Log 112, 114, 123 |
| Stand coordinates from X-Plane Gateway `apt.dat` (used by a top-3 team) | Stand-match coverage against OSM | OSM already matches the stand exactly for 96–99 % of departures (EDDM 59 % exact + 30 % by prefix, LEBL 90 %); ceiling below 0.1 s | ⛔ | Log 27 |
| Closed or commercial sources: FR24, FlightAware, airport A-CDM, DDR2/DPI, paid NOTAM, Navigraph | Licence and terms review; organiser guidance | Commercial, access by agreement, or game-only use. Organiser: "if their license applies as open data, ok". DPI would carry TOBT/TSAT | ⛔ | Brief §0; Log 28, 30 |

### Features

| Idea | How tested | Measured effect | Verdict | Reference |
|---|---|---|---|---|
| Six-hour METAR history instead of current weather (E193) | 2 month pairs × 3 seeds against an equal-capacity control | ΔRMSE body +0.015381 / +0.006401 s (pairs 1+7 / 4+10); trims worse | ❌ this encoding | Val.; [report](../analysis/e193_weather_results_20261007.md) |
| Cumulonimbus flags missed by the weather-code parser (E205) | 6 pairs, 12 fits | ΔRMSE vs existing weather: decided −0.009903 s (5/6). Against the current chain: decided +0.087097 s, 3/6, trims worse | ❌ | Val.; Log 371 |
| ERA5 hourly precipitation and snow cover instead of point METAR | Error-budget decomposition | Weather rows hold 16.21 of 108.66 s; 97 % of that (14.04 s) is in winter rows with no precipitation at the time; correctable biased part ≈ 0.003 s | ⛔ | Brief §0 (§21h) |
| "Winter memory": hours since snow or icing, 12/24/48-hour windows (H20) | Probe; winter months swapped (Jan+Feb ↔ Dec) | Winter months: worse in both directions (+0.51 / +0.34 s); the all-row gain came from a season indicator | ❌ | Log 190 |
| Runway × hourly runway-configuration interaction | Group-residual probe | Random half split: correlation +0.763, SSE −0.558 s. Month split (1–6 ↔ 7–12): correlation −0.041, SSE +0.973 s | ❌ | Brief §21i |
| Runway direction reversals: minutes since reversal, reversal during taxi | Bias correction by airport × bin, months 1–6 ↔ 7–12 | SSE +36 s² per row (worse). Rows near a reversal were more accurate (RMSE 115 at < 15 min vs 178 at > 180 min) | ❌ | Log 357 |
| Reference taxi time P10/P50 by stand × runway (E9); mean residual corrections by airport, runway, stand, type, hour (E7) | 6-fold CV | E9: required ≥ 0.5 s gain on NM rows in 0/6 folds (−0.12 to +0.29 s), mean full RMSE +0.68 s. E7: +0.03…+0.36 s | ❌ | Brief §7 |
| Route length on the OSM taxiway graph (E51) | 6-fold CV, twice | Old base: +0.01 s, 3/6. Re-test on the production feature set (U3): −0.165 s, 5/6 | ❌ → ✅ in final table | Brief §0; Log 139 |
| Queue at the gate: neighbours' push-back delay, overdue EOBT, mean lateness (E127, E137, E184) | 6-fold CV (E184 against the final feature table) | E137: decided +0.137 s, 2/6. E184 on the final table: decided −0.215059 s (5/6), trimmed +0.162186 s | ❌ family closed | Brief §0; Log 93, 280; [E184](../analysis/e184_verdict_2026-10-05.md) |
| Record order: MVT_ID / FLIGHT_ID deviation from neighbours (E27; rolling-median form E132) | 6-fold CV | E27: +4.37 s (fold 1+7 RMSE 391 → 413). E132: +0.025 s after E14, 3/6 | ❌ | Brief §0; Log 77 |
| Traffic counts inside the flight's own taxi interval [AOBT, ATOT] (E151; Yuan 2024, Wu 2026) | 6-fold CV, then LB | Production CV +0.034 s; LB +0.041 s. Against an older control it had shown −0.334 s (5/6) | ❌ | Brief §0; Log 138, 166 |
| ADS-B state of neighbouring aircraft for flights without own track (E183) | Probe on rows without track coverage | Screen +0.002 s (3/6). Oracle −3.8 s not reachable without labels: untracked rows and tracked neighbours are at different airports | ❌ | Log 253; [probe](../analysis/e183_probe_2026-10-04.md) |
| Spatial concentration of other ground aircraft instead of counts (E199) | 2 pairs × 3 seeds | ΔRMSE spatial − counts −0.022049 / +0.062688 s; registered requirements not all met | ❌ | Val.; [report](../analysis/e199_results_20261007.md) |
| Relational graph of observed airport events vs the same events without links (E192) | 2 pairs × 3 seeds | ΔRMSE graph − flat, body −0.010956 / +0.004960 s; in pair 4+10 all four decision SSE legs worse | ❌ one fixed graph layer | Log 325 |
| Turnaround: inbound delay, taxi-in and stand of the previous arrival (E162) | 6-fold CV | Decided +0.016 s, 3/6; trimmed −0.375 s | ❌ borderline | Log 196 |
| En-route ATFM delay by ANSP along the great-circle route (E170) | 6-fold CV against E163 | Decided +0.563 s, 3/6; trimmed +0.502; body +1.846 | ❌ | Log 217 |
| Daily arrival ATFM delay by cause, incl. de-icing (H12) | Probe | De-icing cause is zero at all airports; stage −0.01…−0.05 s | ❌ | Brief §0; Log 168 |
| Delay wave at the destination from flights of other hubs (H25) | 6-fold CV against the final table | Decided +0.079 s (4/6), trimmed −0.036, body +0.893 | ❌ added capacity fails body leg | Log 241 |
| Disruption-day features: day share without NM, day median of MVT − AOBT (E3) | 6-fold CV | ≥ 3 s gain on no-NM rows in only 3/6 folds; full RMSE +2.9 and +4.5 s in two folds | ❌ | Brief §7 |
| Schedule-substitution habit of a flight number on Rome rows with NM (E133) | 6-fold CV, then LB | CV −0.72 s (5/6); LB +0.36 s | ❌ | Brief §0; Log 70 |
| Restore NM fields for rows without NM (E108); clean airline prefix with a lookup table (E113) | 6-fold CV | E108 +0.20 s (after E14 +0.31). E113 +0.30 s (after E14 +0.26) | ❌ | Brief §0; Log 24, 25 |
| Remove more location identities: stand alone (E105), residual target encodings by stand × runway and operator `ter_*` (E110), `stand_prefix` (E119), `ap_runway` (E120) | LB slots | +1.45, +0.40, +1.17, +0.43 s. (Removing stand identity together with all 2025 target encodings, E112, was accepted: −1.06 s) | ❌ | Brief §0; Log 22, 41 |

### Regime classifier variants

The target is a recorded interval. Many 2025 labels equal take-off minus a
scheduled or filed time. A classifier assigns probabilities to these recording
regimes, and the prediction is the probability mixture.

| Idea | How tested | Measured effect | Verdict | Reference |
|---|---|---|---|---|
| Choose classifier boosting rounds by mixture SSE instead of log-loss (E23) | Nested 6-fold CV | +1.21·10⁹ SSE ≈ +1.07 s; 4/6 lower but the losses were larger | ❌ | Brief §7 |
| Hard regime choice when p ≥ τ instead of the mixture | Out-of-fold check | τ = 0.5: +29 s. τ = 0.35: −8.4 s, all from LIRF tail rows that TabPFN already handles; worse elsewhere | ❌ | Brief §0; [review](../analysis/gemini_deep_research_review_2026-09-15.md) |
| Per-row choice within SCHED / IOBT / LOBT / EOBT (partial-label, E107) | 6-fold CV | +0.35 s, worse in 6/6 | ❌ | Brief §0; Log 11 |
| Neural-network regime classifier (E104) | First fold | Log-loss better (0.190 vs 0.197); mixture worse (384 vs 359 s) | ❌ | Brief §0 |
| Train the classifier on mixture SSE with a cross-fitted regressor (E111) | Fold probe | +0.14 s, after E14 +0.19 (implementation with a ±1,800 s clip) | ❌ this form | Brief §0; Log 23 |
| Give the classifier the regressors' out-of-fold predictions (stacking, E158) | 6-fold CV | Decided +0.329 s, 2/6 | ❌ | Brief §0; Log 195 |
| Sample weights for the 2025 → 2026 shift from adversarial validation (E103) | LB slot | +1.77 s | ❌ | Brief §0 |
| Remove the schedule regime when the aircraft could not physically be there (flight-number pairs; same-aircraft landing from ADS-B; hard ceiling ŷ ≤ MVT − previous in-block) | Measurements on 2025 | Flight-number pairs exist for 39 of 1,149,981 departures. Schedule share 3.17 % where the constraint fires vs 3.14 % elsewhere. Hard ceiling +5.56 s: the truth exceeds it in 1,323 rows (0.10 %) | ❌ family closed | Brief §0 (§21h) |

### Sparse LIRF tail (Rome departures without NM record)

| Idea | How tested | Measured effect | Verdict | Reference |
|---|---|---|---|---|
| Different priors for the empirical regime table (E2) | LOMO | Dir(0.5) +1.0 % SSE; empirical Bayes K = 1.5 +0.3 %, K = 3 −2.1 % (gate −5 %) | ❌ | Brief §7 |
| TabPFN also for delays below 4 h (E12, E12b, E136) | Jan+Jul 2025 holdout; LB | E12: SSE +497 %. E12b: −14.1 % but January worse. E136: LB +2.179 s against v316 | ❌ | Brief §7, §21cd |
| Flight identity derived from the ranking month (E15, E16a, E16b) | 6-fold CV; LOMO | E15: +1.2 s LB-equivalent (1/6). E16a: −2.7 % (gate −5 %), 6/12 months. E16b: +62.5 %, 2/12 | ❌ | Brief §7 |
| Separate model or TabPFN for no-NM rows at the other nine airports (E13) | Jan+Jul 2025 holdout | SSE +2.6 %, without top-3 rows +26.1 %, worse in both months | ❌ | Brief §7 |
| TabPFN regressor instead of the three-regime classifier (E177) | LOMO, 126 rows | On y: +161.9 % (0/12). On y − D: +70.3 % (4/12). Mixtures: +48.5 %, +18.5 % | ❌ | Brief §0; Log 228 |
| Airline prefix in the E14 flight-number logit, with wider coverage (E153) | 6-fold probe against the old control, then against production predictions | Old control: −1.33 s (4/6). Production: +4.76 s (1/6), trimmed +4.47 (0/6) | ❌ withdrawn before upload | Brief §0; Log 141 |
| Hand rules: flight-number rule at D ≥ 4 h; recalibrated P(schedule) at D 1–2 h; garbled callsign with D > 12 h forced to "next day" | Checks on 2025 | +0.085 s; +5.61 s (0/6); 0/5 next-day cases at D ≤ 13.7 h | ❌ | Brief §0; Log 155, 201 |

### ADS-B and trajectory stages

| Idea | How tested | Measured effect | Verdict | Reference |
|---|---|---|---|---|
| Match a flight to a track by take-off time without callsign (E31); by climb timing; by joint assignment (Hungarian, E139) | 6-fold CV; 2025 measurements | E31: −0.05 s, worse at EGLL/LEMD/LFPG (wrong aircraft). Climb timing agrees with the callsign in 46 % (367/797 unambiguous windows); 14,545 ambiguous windows. E139 (time only): links 904 (4.8 %) of 18,913 target rows | ❌ / ⛔ | Brief §0; Log 91 |
| Neighbours' ADS-B taxi times for flights without own track (E35 → E60) | 6-fold CV | Decided −0.09 s (gate −0.3), body +0.07. 0.28 of the 0.37 s all-row gain was in rows rewritten by the tail table | ❌ | Brief §0 |
| CatBoost stage instead of, or mixed with, LightGBM (E41, E148) | 6-fold CV | E41: +0.95 s, 0/6. 50:50 mix: after E14 +0.39 s, 0/6 | ❌ | Brief §0; Log 121 |
| Reception-quality features inside the stage (E44) | 6-fold CV | −0.03 s (gate −0.3), 4/6 | ❌ | Brief §0 |
| Wider stage target clip ±3,600 s instead of ±1,800 s (E150d) | 6-fold CV | Body +0.370 s; after E14 trimmed +0.070 | ❌ | Brief §0; Log 129 |
| Monotone constraints on first-movement features (E161) | 6-fold CV | Decided +0.026 s, 3/6; all legs worse | ❌ | Brief §0; Log 199 |
| Soft attenuation of the correction by regime uncertainty instead of the hard gate E152 (E179) | 6-fold CV | Decided −0.002 s, 3/6 | ❌ | Brief §0; Log 237 |
| Replace the learned stage by a fixed physical estimate: take-off minus first movement (or stand-walk), lag fitted on the other half-year | Registered audit against the current chain | ΔRMSE body +20.520 s, decided +18.266 s; stand-walk +23.597 / +21.040 s. A perfect per-row selector would gain at most −0.689 / −0.862 s | ❌ | Log 362; [audit](../analysis/adsb_stage_audit_20261007/RESULTS.md) |
| Second, wider trajectory scale in the sequence encoder (E204) | 2 pairs, 6 fits; both pairs must improve | ΔRMSE pair 1+7 body +0.092642, decided +0.096240 s; pair 4+10 −0.203661 / −0.203193 s | ❌ | Val.; [report](../analysis/night_gpu_20261007/RESULTS.md) |

### Model families and blend members

| Idea | How tested | Measured effect | Verdict | Reference |
|---|---|---|---|---|
| XGBoost as an additional regressor | NNLS blend weights | Weight 0 on both halves of the sample | ❌ | Brief §5 |
| Separate models per airport (E38) | 6-fold CV, then LB | CV −10.98 s (6/6), but body +0.05 s; LB +3.79 s | ❌ | Brief §7 |
| Learned features built from 2025 aggregates: ADS-B features inside the base model (E32), regime rates by group (E50) | 6-fold CV, then LB | E32: decided −1.41 s (6/6), LB +0.33 s. E50: decided −2.79 s, strict −0.79 s, LB +5.78 s | ❌ | Brief §7 |
| Regressor trained only on rows it serves (a competitor's −5.99 s step, E73) | CV, LB with a written forecast | CV −0.01 s; LB +0.010 s (forecast range −0.3…+0.2) | ❌ | Brief §0 (§21o) |
| Strong LightGBM regularisation: `min_data_in_leaf` 500, `min_gain_to_split` 0.35 (E155) | 6-fold CV against production predictions | +0.733 s, 0/6 | ❌ | Log 146 |
| Out-of-fold Huber corrector on confident regular rows (E154) | 6-fold CV against production predictions | +0.513 s, 0/6 | ❌ | Log 142 |
| Extra members that differ only in hyperparameters or loss (E174), or that drop whole feature families (E180) | Member weight grid on out-of-fold predictions | E174: +0.014 (2/6) and +0.046 (0/6). E180: weight 0 in 5/6, +0.014 | ❌ | Log 221, 240 |
| k-nearest-neighbour member, K = 50 (E178) | Member weight grid | Weight 0 in all folds; standalone RMSE 227–260 s vs LightGBM 186–212 s; correlation 0.90–0.91 | ❌ | Log 239 |
| More MLP members: 3 seeds (E101); second MLP without 2025 encodings (E176c); wider and longer (E181b) | Member weight grid | E101: −0.646 vs −0.657 with one seed. E176c: +0.034 (2/6), correlation with first MLP 0.982–0.986. E181b: +0.129, trimmed +0.109 | ❌ | Brief §0; Log 230, 246 |
| TabM-style BatchEnsemble network (E186) | 6-fold CV, fixed weight 0.15 | Decided +0.123132 s, 1/6; body +0.126584; trimmed +0.121079 | ❌ | Log 263; [report](../analysis/e186_verdict_2026-10-04.md) |
| Conditional density heads: Gaussian (E196); mixture density network in the current chain (E200) | 2-pair screen; 6-pair current-chain comparison | Gaussian worse on all decision legs. MDN: ΔRMSE body −0.200152 s, decided +0.054472 s, 3/6 (≥ 4 required) | ❌ | Val.; Log 340, 347 |
| TabICLv2 for LTFM rows without NM (E189) | 2 pairs × 4,096 rows | ΔRMSE on the LTFM sample +1.433814 / +3.439552 s | ❌ this protocol | Log 295 |

### Post-processing, blending and calibration

| Idea | How tested | Measured effect | Verdict | Reference |
|---|---|---|---|---|
| Clip predictions to airport × runway × stand group bounds (2024-edition protocol, E45) | 6-fold CV | +65 s on all rows; +1.6 s even on confident regular rows, worse in 6/6 | ❌ | Brief §0 |
| Isotonic proxy calibration instead of one multiplier (E55) | 6-fold CV | Proxy RMSE 378 → 346, final +0.46 s, 1/6 | ❌ | Brief §0 |
| Clip regressor target at 3,600 s instead of 7,200 s (E56) | 6-fold CV | +1.33 s, worse in 6/6 | ❌ | Brief §0 |
| Intercept in the proxy calibration (E57); refit the ADS-B stage on the base it corrects (E59) | 6-fold CV | +0.10 s (4/6); +0.14 s (0/6) | ❌ (E59 later reversed as E169) | Brief §0 (§21f) |
| Prediction floor (MVT − AOBT_3) − 3,600 s (a competitor's idea, E124) | 6-fold CV | +0.19 s, worse in 6/6; the truth violates the floor in 1.2 % of eligible rows | ❌ | Brief §0; Log 61 |
| Tail blending: member weight by base-prediction bin, or fixed "physics" weights (E160) | 6-fold CV | Bins vs one weight: +0.029 s, worse in 5/6. Fixed weights: +11.30 s, 0/6 | ❌ | Log 176 |
| Greedy NNLS blend over a pool of saved out-of-fold files | Out-of-fold check | −15.1 s was an artefact: weight 0.80 went to a file made before the early-stopping leak was fixed | ❌ | Log 191 |
| Blend weights chosen on decided-body SSE instead of all-row SSE (E187) | 6-fold CV, 1,593-point grid | Decided −0.004902 s; 3/6 lower, 3/6 identical (≥ 4 required) | ❌ | Log 257; [report](../analysis/e187_verdict_2026-10-04.md) |

### Validation schemes

| Idea | How tested | Measured effect | Verdict | Reference |
|---|---|---|---|---|
| Gate on all-row SSE | Re-scored on decided rows and compared with LB | E28: all-row −6.24 s (5/6, accepted), decided +0.37 s (2/6), LB −0.01 s | Replaced by decided metric on 16 Sep | Brief §7 |
| Early-stopping leak as the reason per-airport models won in CV (E43) | Nested round selection on separate inner months | −10.55 s without leak vs −10.98 s with it: explains 0.4 of 11 s | ❌ as explanation | Brief §7 |
| LIRF tail gate "better in ≥ 8 of 12 months" | LB slots for variants it had rejected | Airline prefix in TabPFN (E118, −5.0 %, 6/12): LB −3.60 s. Local Rome date (E26a, −6.2 %, 5/12): LB −0.46 s | Rule removed | Log 56 |
| Fixed −0.30 s CV threshold | Review of where the threshold came from | Not derived from data. E141 (−0.137 s, 6/6) was uploaded anyway: LB −0.040 s | Rule removed | Log 56, 98, 123 |
| Gate leg "after E14" | Re-check after E14 left the chain | Wake-ladder features E157: −0.133 → −0.062 s on decided rows without E14 | Replaced by decided | Log 192 |

## Methodological lessons

**1. The type of change predicted transfer to the leaderboard better than the
size of the CV gain.** By 17 September, changes that added no learned capacity
had transferred, and some transferred more strongly than in CV. A calibration
of the filed-time proxy (E46) went from CV −0.40 to LB −1.00. ADS-B stages based
on each flight's own track went from CV −9.5 / −1.4 / −3.0 to LB −15.2 / −2.0 /
−3.7 (E25, E30, E36). Learned capacity on 2025 aggregates did the opposite:
E32, E38 and E50 went from CV −1.41 / −10.98 / −2.79 to LB +0.33 / +3.79 / +5.78.
Later cases followed the same pattern: E133 went from CV −0.72 to LB +0.36, and
E145 from CV −0.045 (6/6) to LB +0.114. On 4 October, 38 single-change CV/LB
pairs were reviewed, 15 of them re-scored from out-of-fold predictions. The
ratio of LB change to decided-row CV change was about 1.0 for ADS-B stages and
about 0.7 for blend members. It was 0.2–0.3 for new base features, and 0 or
negative for rules and calibrations fitted on 2025. The sign of the CV change
matched the leaderboard in only 60–70 % of pairs.
(Brief §7; Log 252; [report](../analysis/board_vs_cv_slices_2026-10-04.md))

**2. For removing training-period capacity, CV was not a reliable judge in
either direction.** Removing the 2025 target encodings (E99) was worse in CV
(+0.94) but improved the leaderboard (−0.167). Removing `month` (E24) gave
−0.19 on decided rows (3/6) and −0.87 on the leaderboard. Removing the E14
flight-number rule was harmful against production predictions (+0.376, 3/6)
and gave LB −0.227. Other removals were worse on the leaderboard: E105 +1.45,
E110 +0.40, E119 +1.17, E120 +0.43. The 4 October re-scoring also found no
usable CV-to-LB ratio for this class. Removals were therefore decided by a
leaderboard slot with a forecast recorded in advance. (Brief §0, §7; Log 41,
141, 166, 252)

**3. Component-level gains shrank or reversed in the full chain.** A mixture
density network (MDN) tested as a component (E198) improved body RMSE by 0.399541
and decided RMSE by 0.142199 over six pairs (5/6). Inside the current chain
(E200) it gave body −0.200152 but decided +0.054472, 3/6, and was rejected. The
ordered-trajectory encoder (E197) beat the previous neural member (PLE) by
4.465932 / 7.113443 s as a component. Against an equal-capacity unordered
control it won only 0.249545 / 0.126464 s. In the full chain it gave body
−0.614049 (E203), and the leaderboard gain was 0.5051 s. OPDI on LSZH gave −38.6 % RMSE on event rows
in LOMO; the forecast was −1.2, and the leaderboard gave −0.131. The wake-ladder
features (E157) gave −1.47 s all-row RMSE on fold 1, then −0.133 over six folds,
then −0.062 after the chain changed. (Val.; Log 160, 192, 215)

**4. The control must be the production model's out-of-fold predictions.**
E153 looked like −1.33 (4/6) against an older control and was +4.76 (1/6)
against production predictions. It was withdrawn before upload. E151 was −0.334
(5/6) against an older control, +0.034 against production, and +0.041 on the
leaderboard. E14 itself had been worth −7.9 against the 14 September base but
was +0.376 (3/6) against the production base. Removing it gave LB −0.227. A
stronger base absorbs later stages. (Log 138, 141, 166)

**5. One wrong label can outweigh a gate.** On 2,084,920 decided out-of-fold (21 September chain)
rows (RMSE 228.2 s), the ten worst rows held 24.3 % of the SSE. In training,
34 of 2.09 million departures have labels above 12 h (LIRF 31, LFPG 2, LSZH 1).
These are mostly time at the stand carried over from the previous day. The
worst single row was worth 7.47 s on the leaderboard scale. That is 25 times
the 0.30 s gate and more than the 2.73 s of E77, the best accepted change that
month. Since then, if the decided and trimmed measures differ by more than a
factor of two, the trimmed one decides. E82 (all rows −2.94, trimmed −0.84) and
E84 (decided −0.15, trimmed +0.14) were both rejected correctly by this rule.
(Brief §21y)

**6. A subset selected by the true value rejects regime shifts by
construction.** The body subset (y ≤ 7,200 s) is defined by the label. A change
that moves probability towards long regimes on ambiguous rows must look worse
on this subset. E89 changed the classifier weights only for errors beyond one
hour. In CV it gave decided −1.02, all −9.37, trimmed −8.65 and body +11.27,
and it was rejected under the registered rule. Uploaded as a separate slot, it
improved the leaderboard by 1.18 s against v97 (v115, 243.3072). For added
learned capacity, the body leg remained informative. Per-airport models (E38)
won only on 0.02–0.04 % of rows, were +0.05 on the body and +3.79 on the
leaderboard. H25 was rejected on body +0.893. (Brief §7, §21ag, §21ai; Log 241)

**7. Probes for group effects must be split by month, as the CV is.** For the
runway × configuration interaction, a random half split gave a split-half
correlation of +0.763 and an apparent SSE reduction of 0.558 s. A control group
behaved normally under this split, so the probe looked sound. Training on months
1–6 and testing on 7–12 (and the reverse) gave a correlation of −0.041 and an
SSE increase of 0.973 s. Rows from the same day and hour fell into both random
halves, so the "configuration effect" was the effect of particular hours, which
does not carry to other months. (Brief §21i)

**8. Blends of complete submissions are predictable by the Gram identity;
blends with untested components are not.** All files are scored on the same
344,841 rows. So `score(w)² = wᵀGw` with
`G_ij = (s_i² + s_j² − mean((P_i − P_j)²)) / 2` gives the score of any blend
from known scores and pairwise distances
([`scripts/blend_score_identity.py`](../scripts/blend_score_identity.py)).
Blend E76 (v89) was forecast at 246.9202 and scored 246.9205. A fixed 50:50
blend of v12061 and v12070 was forecast at 231.608594 before rounding and scored
231.6095. E78 added a stage-level partner with no score of its own. It was
forecast at 246.70 and scored 247.3631. The identity then gave the partner's own
leaderboard RMSE as 338.46 s, against 247.7501 for the base file, although in CV
the partner had trailed the control by only 7 %. Blend weights came from CV
only. For E76, leaderboard-optimal weights would have given −1.004 s against
v82 instead of the −0.8296 s achieved; they were not used, because fitting
weights to the leaderboard is probing. (Brief §21q, §21s)

**9. A model that learns a residual absorbs any input change it can express,
so a better intermediate quantity is not a better result.** On six folds, a
change in the level of the proxy was absorbed by 95–99 % and a change in its
shape by 65–68 % ([`src/e57_absorption.py`](../src/e57_absorption.py)). An
intercept (E57) and a refit on the right base (E59) therefore gave +0.10 and
+0.14 s. An isotonic calibration made the proxy more accurate (RMSE 378 → 346)
and the final prediction worse (+0.46 s, 1/6). (Brief §0, §21f)

**10. A large error pocket is a target only if its error is biased.** Weather
rows held 16.21 of 108.66 s of error. 97 % of that was in winter rows with no
precipitation at the time. The correctable biased part of the precipitation
rows was about 0.003 s, so ERA5 was not downloaded. Rare aircraft types had RMSE
235 vs 167 s, but a bias of +23 s; the ceiling for a full correction was about
0.07 s. (Brief §0, §21h; Log 357)

**11. A rejection depends on the base, the measurement and the threshold.**
Five closures were later reversed:

- adsb.lol ground tracks were closed on 13 September because ground state was
  read only from the `ground` flag, which marks 0.02 % of EDDM points. Reopened
  as E25, they gave LB −15.22 (273.41 → 258.19).
- The "8 of 12 months" tail rule had rejected E118. Uploaded later, E118 gave
  LB −3.60.
- Route length on the taxiway graph (E51) failed on the old base and passed as
  U3 (−0.165, 5/6).
- Refitting the ADS-B stage on the production base (E59 → E169) passed with
  −0.110 (5/6).
- The arrival-link rule E21 improved 6/6 folds but was rejected because its
  gain without the top-3 rows (0.35) was below a 0.5 threshold. Re-tested as
  E167 on production predictions, it was confirmed in 11/12 months.

(Brief §0; Log 56, 192)

**12. A blend member needs both diversity and accuracy.** The k-nearest-neighbour
member (E178) was the least correlated member (0.90–0.91). Its standalone RMSE
(227–260 s vs 186–212 s for LightGBM) gave it weight 0 in every fold. The MLP
member (E176) had a correlation of 0.957–0.973 and similar accuracy. It gave
decided −0.620 (6/6) and LB −0.775. A second MLP correlated 0.982–0.986 with the
first and added +0.034 (2/6). (Log 223, 230, 239, 241)

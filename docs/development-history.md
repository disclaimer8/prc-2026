# Development history

This page shows how the model went from 501.23 s to 230.48 s RMSE, which design decisions
moved the leaderboard score, and how far cross-validation could be trusted along the way.
The [model design](model-design.md) describes the final chain; [validation](validation.md)
gives the evidence for its current members; the [glossary](glossary.md) names every
experiment ID.

**Conventions**
- Scores are leaderboard RMSE in seconds on the 344,841 January and July 2026 departures;
  negative changes are improvements. *Gain* is the change against the previous best score.
  Where an upload bundled several changes, the row also gives the effect against the closest
  scored single-change control.
- Evidence: **B §x** is a section of [REVIEW_BRIEF.md][B], the working brief (Russian; §2–§7
  cover 13–17 September, the dated §21 sections run to 4 October). **L №n** is an entry of
  [HYPOTHESES_LOG.md][L], the experiment register (Russian; starts on 23 September). All 111
  scores are in [`tables/leaderboard_gentle-igloo.csv`](tables/leaderboard_gentle-igloo.csv).
- *CV* is the six month-pair folds of 2025 (see [validation](validation.md)). Most CV
  numbers in the brief are on a "leaderboard-equivalent" scale: the paired change in squared
  error, converted to seconds of board RMSE. They are comparable with board changes in sign
  and rough size, not exactly.

## 1. Score progression

111 uploads between 13 September and 8 October 2026 produced 63 new best scores. The 31
steps with a gain of about 0.3 s or more account for 267.9 s of the total 270.7 s; the other
31 add 2.9 s together. Step 27 (−0.299 s) is in the first table because it is a separate
design decision; step 13 is measured against v28, the best file before it.

### Steps of 0.3 s or more

| Step | UTC | File | RMSE | Gain | What changed | ID | Evidence |
|---:|---|---|---:|---:|---|---|---|
| 1 | 09-13 10:58 | v1 | 501.23 | — | One LightGBM regressor on the target, clipped at 7,200 s. January+July 2025 holdout: 531.8 | — | B §2 |
| 2 | 09-13 11:07 | v2 | 366.22 | −135.01 | Recording-regime mixture: a classifier for "recorded off-block = scheduled time" plus the regressor; the prediction is the probability-weighted mean of the two candidates. Holdout 404.7 | — | B §2, §3 |
| 3 | 09-13 11:40 | v3 | 354.20 | −12.03 | Six regimes (regular, schedule, IOBT, LOBT, EOBT, one-day shift); METAR weather; absolute timestamps removed from the features. Holdout 402.4 | — | B §2 |
| 4 | 09-13 12:56 | v4 | 344.06 | −10.14 | Stand-to-runway geometry from OpenStreetMap, runway configuration per hour, target encoding. The holdout got worse (406.3) | — | B §2, §5 |
| 5 | 09-13 13:46 | v5 | 339.21 | −4.84 | Regressor = 0.65 CatBoost + 0.35 LightGBM | — | B §2, §5 |
| 6 | 09-14 00:06 | v8 | 279.62 | −59.59 | Two changes. (a) v7, scored 306.58 (−32.6): the regressor learns the residual to the NM proxy (take-off minus `AOBT_3`, else LOBT, else EOBT); neighbour features from other departures by airport, runway and stand; predictions kept inside the LOBT matching window; CatBoost on the new features; five LightGBM seeds. (b) An empirical table of regime probabilities for LIRF departures without an NM match, 4–30 h after schedule, replaces 42 ranking predictions: −27.0 vs v7 | К1 (finding of the first review), E1 | B §2, §5, §7 row E1 |
| 8 | 09-14 18:02 | v13 | 274.93 | −4.63 | TabPFN-2 replaces the table for the same 42 rows (leave-one-month-out squared error −41.5 %, better in 11 of 12 months). −4.70 vs v8 | E11 | B §7 row E11 |
| 9 | 09-15 09:06 | v18 | 274.28 | −0.65 | Flight-number history: probability of the schedule regime from the flight number's frequency in other months, for LIRF without NM and 0–4 h after schedule, only with ≥ 3 past rows (43 ranking rows). −0.84 vs v14, a clean rebuild of v13 (+0.19) | E14 | B §2, §7 row E14 |
| 10 | 09-15 13:14 | v22 | 273.41 | −0.87 | Full rebuild without the `month` feature | E24 | B §7 row E24 |
| 11 | 09-15 20:00 | v27 | 258.19 | −15.22 | First ADS-B surface stage. adsb.lol ground tracks give the first ground fix, the first movement after the stand and the end of the last stop; a small LightGBM corrects the base residual on 222,517 covered rows. CV −9.53, better in 6 of 6 folds. The technique was seen in team merry-quicksand's public code; the implementation is our own | E25 | B §0, §2, §7 row E25, §21a |
| 13 | 09-16 00:05 | v37 | 256.16 | −2.02 | Extended stage features (path length, stops, last stop, speed) and a larger stage model, trained per airport. The features carry the gain: the pooled version (v29, step 14) scored 256.14, so the per-airport split added +0.02 | E30, E40 | B §2, §7 rows E30, E40 |
| 15 | 09-16 00:06 | v30 | 252.43 | −3.71 | Stand geometry in the stage: distances of track points to the assigned stand, time of leaving a 150 m zone. CV −2.98 | E36 | B §0, §7 |
| 16 | 09-17 00:06 | v55 | 251.42 | −1.00 | NM proxy scaled per airport × timestamp source (least squares, shrunk towards a global value). Scales for LOBT 0.42–0.69, for `AOBT_3` 0.80–1.16. CV −0.40. Team elegant-alligator had reported similar LOBT scales | E46 | B §7 row E46 and table of 17.09, §21b |
| 18 | 09-18 11:15 | v78 | 248.77 | −2.48 | Previous-visit features of the same aircraft added to the E36 stage. They come from airborne points (within 7 km, below 2,500 ft) that the stored files held but the extractor had not used. v76 (step 17) had added a separate previous-visit stage for rows without a ground segment. The pair: CV −2.37, board −2.65 | E64, E65 | B §21i |
| 19 | 09-19 00:06 | v80 | 248.33 | −0.44 | Three small stage additions together: the track extended backwards through the stand, where the aircraft was first seen and how far it was from the runway threshold, and a mid-taxi reception gap read as the aircraft's turnaround. Alone each failed the 0.3 s gate (CV −0.05, −0.15, −0.26); together CV −0.38 | E70 (E63, E66, E69) | B §21hb, §21i, §21j |
| 20 | 09-19 16:46 | v82 | 247.75 | −0.58 | Movement threshold in the track extraction lowered from 3 kt to 1 kt (median error against known block times 302.6 s → 193.5 s on 18,050 departures). CV −0.37 | E74 (E71) | B §21j |
| 21 | 09-20 08:01 | v89 | 246.92 | −0.83 | Whole-file blend of v82 with two earlier complete submissions: v44 (per-airport models) and v60 (ADS-B features inside the main model, with E46). Weights 0.134 and 0.053 from CV. Predicted before upload: 246.9202 | E76 | B §21q |
| 22 | 09-20 11:50 | v97 | 244.49 | −2.43 | Neighbour features on filed times: other departures' take-off minus LOBT and minus schedule (28 columns). −3.26 vs v82, the same chain on the old features; CV −2.73 | E77 | B §21r, §21t |
| 23 | 09-20 14:19 | v98 | 243.83 | −0.66 | The E76 blend refitted against the stage output of the new chain: v44 0.1214, v60 0.0270. Predicted 243.8342 | E81 | B §21v |
| 24 | 09-22 09:28 | v115 | 243.31 | −0.53 | Cost weights in the regime classifier: rows whose candidate values spread more than 3,600 s get weight (spread/3,600)², capped at 400. It failed its CV gate and was uploaded as a test with a written forecast (245.0, range 241.5–248.5). −1.18 vs v97 | E89 | B §4, §21ag–§21ai |
| 25 | 09-22 09:29 | v116 | 242.66 | −0.65 | E81 blend on top of v115. Predicted 242.6554 | E81 | B §21ai |
| 27 | 09-23 12:50 | v163 | 242.26 | −0.30 | 2025 target encodings removed from the regressor (v155: −0.17 vs v115, although CV said +0.94), then the E81 blend. Predicted 242.2622 | E99, E81 | B §21ba, §21be |
| 28 | 09-24 02:38 | v195 | 241.86 | −0.40 | Airline prefix parsed from the flight number for rows without NM; the operator was unknown for all of them. −1.28 vs v155, the same chain without the blend | E106 | B §21bg–§21bh, §21bq; L №6 |
| 29 | 09-24 02:38 | v225 | 241.20 | −0.66 | Stand identity and the remaining stand×runway and operator residual encodings (`ter_*`) removed together. Removing `ter_*` alone (v215) was +0.40; adding the stand removal gave −1.06 vs v215 | E112 (E110) | B §21bq |
| 30 | 09-24 02:41 | v226 | 240.14 | −1.07 | E81 blend on v225. Predicted 240.1387 | E81 | B §21bq |
| 33 | 09-26 00:09 | v316 | 236.32 | −3.60 | Airline prefix as a feature of the TabPFN tail model for LIRF without NM. Only the 42 tail rows change (RMS change over the file 19.5 s, largest 6,352 s). The variant had been rejected by a "better in ≥ 8 of 12 months" rule (better in 6 of 12); forecast −0.27 | E118 (built as E123) | B §21br, §21bv, §21ca |
| 34 | 09-27 00:05 | v366 | 235.52 | −0.80 | CatBoost averaged over five seeds, combined with the v316 tail. Five seeds alone gave −0.51 (v285 vs v265) | E121b, E135 | B §0, §21cd; L №42, №76, №92 |
| 45 | 10-01 00:12 | v808 | 233.59 | −0.98 | Anchored members: LightGBM regressors on y − clip(take-off − T) for T = LOBT and schedule, plus one on raw y, mixed into the regular candidate at 2/3 of their CV weights (0.17, 0.13, 0.13). CV about −1.24; forecast −0.39 | E130, E171–E173 | B §21cm; L №215 |
| 47 | 10-02 00:13 | v899 | 233.04 | −0.40 | MLP member (category embeddings, residual to the proxy) at weight 0.25 next to five anchored members. −0.78 vs v859, the same file without the MLP; CV −0.62 | E176 | B §0, §21cn; L №224, №241 |
| 59 | 10-07 07:01 | v12070 | 231.44 | −0.54 | Neural member rebuilt with piecewise-linear numerical embeddings (three seeds), replacing the MLP at weight 0.25; −0.54 vs v12061, the same chain with the old MLP. Uploaded before a full-chain CV at the owner's request; forecast 0 (range ±1). The later retrospective comparison gave −0.34 on the `decided` rows, 6 of 6 pairs | E190 | L №299, №306, №318, №321; [validation](validation.md); [results](../analysis/submission_results_20261007.md) |
| 61 | 10-07 21:53 | v12075 | 230.94 | −0.51 | Ordered-trajectory neural member: the network also reads up to 64 ADS-B observations of the current and of the previous visit of the aircraft, in time order. It replaces E190 at weight 0.25. Pre-E81 comparison −0.55 on `decided`, 6 of 6 pairs; forecast −0.16 | E197, E202, E203 | L №367; [validation](validation.md); [upload record](../analysis/e203_upload_20261007/RESULTS.md) |
| 63 | 10-08 20:14 | v12077 | 230.48 | −0.34 | Weight of the neural member in the regular candidate 0.25 → 0.35, the five anchored members scaled by 0.65/0.75, the tree pair keeps the remainder; nothing else changed against v12076 (zero fits). The weight was chosen before the upload on 2025 out-of-fold predictions of the exact chain (E212): the registered leave-one-pair-out rule picked 0.40 (−0.228 s pooled on `decided`, 6 of 6 pairs), but 0.40 was worse than 0.25 in August and a January–June choice did not hold on July–December, so the coordinator took 0.35, the only grid value below 0.25 in all twelve months at three and at ten seeds. January + July 2025 estimate −0.11 s; forecast −0.08 with corridor [230.62; 230.87]; the result fell 0.14 s below the corridor on the favourable side | E212 | L №407a, №410, №425, №426, №427; [E212 report](../analysis/e212_nn_weight_10seed_20261008.md); [upload record](../analysis/v12077_upload_20261008/RESULTS.md) |

### Smaller steps

| Step | UTC | File | RMSE | Gain | What changed | ID | Evidence |
|---:|---|---|---:|---:|---|---|---|
| 7 | 09-14 00:07 | v9 | 279.56 | −0.07 | ×1.3 on the regular candidate for 1,136 rows without NM on January 2026 NM-outage days; idea closed | E4 | B §2, §7 row E4 |
| 12 | 09-15 20:23 | v28 | 258.18 | −0.01 | Neighbour windows after the flight's own take-off, rebuilt under E25 | E28 | B §7 |
| 14 | 09-16 00:06 | v29 | 256.14 | −0.02 | E30 stage pooled over airports (see step 13) | E30 | B §2 |
| 17 | 09-18 11:06 | v76 | 251.25 | −0.18 | Previous-visit stage for rows without a ground segment | E64 | B §21i |
| 26 | 09-23 11:03 | v162 | 242.56 | −0.10 | Blend of v115 with v44, v60 and an MLP partner (CV weights 0.010, 0.064, 0.132). Predicted 242.5613 | E100 | B §21bc |
| 31 | 09-25 00:02 | v227 | 240.13 | −0.01 | Predictions raised near the lower edge of the LOBT window (CV −0.14) | — | B §21br |
| 32 | 09-25 04:16 | v247 | 239.92 | −0.21 | Runway-configuration features removed (−0.20 alone, v245 vs v225), with E81 and the edge lift | E116 | B §21br |
| 35 | 09-28 00:05 | v404 | 235.48 | −0.04 | Stage features of the first observed part of the track (where it starts, initial speed); kinematic estimate of the unseen taxi start | E134, E141 | B §21bz, §21cj |
| 36 | 09-28 00:07 | v414 | 235.43 | −0.05 | E70 stage averaged over five seeds | E142 | B §21cj |
| 37 | 09-28 00:10 | v447 | 235.35 | −0.08 | CatBoost 10 seeds; E64 stage five seeds | E144, E143 | B §21cj |
| 38 | 09-29 00:06 | v467 | 235.34 | −0.01 | CatBoost 15 seeds | E144 | L №138 |
| 39 | 09-29 00:08 | v477 | 235.28 | −0.06 | Stages without the reception-density features that drifted between 2025 and 2026 | E147 | L №138 |
| 40 | 09-29 00:15 | v507 | 235.17 | −0.11 | E70 stage trained on all rows, including rows where the record contradicts the track | E150 | L №138 |
| 41 | 09-30 00:05 | v538 | 235.02 | −0.15 | 25 CatBoost seeds; E64 stage on all rows; both stages without drift features | E144, E150c | L №138, №166 |
| 42 | 09-30 00:08 | v558 | 234.80 | −0.22 | Gate: surface stages leave a row unchanged when a planning regime has probability ≥ 0.95 (forecast −0.02) | E152 | L №166 |
| 43 | 09-30 00:12 | v588 | 234.80 | −0.01 | E14 removed (−0.23 vs v538): the base model had absorbed it | — | L №141, №166 |
| 44 | 10-01 00:05 | v655 | 234.57 | −0.23 | E152 and the removal of E14 combined | — | B §21cm; L №215 |
| 46 | 10-02 00:06 | v819 | 233.44 | −0.15 | Destination-airport arrival ATFM delay in the base (E163); LOBT-window regime rules (E167) | E163, E167 | L №241 |
| 48 | 10-03 00:08 | v949 | 232.95 | −0.09 | LightGBM members at 2/3 of CV weights, MLP kept at 0.25 (−0.10 vs MLP at 2/3) | — | L №245 |
| 49 | 10-03 00:11 | v959 | 232.83 | −0.12 | Base model on the final feature table FIN1b (−0.22 vs the same weights on the old base) | FIN1b | L №245 |
| 50 | 10-03 00:16 | v979 | 232.75 | −0.08 | CatBoost on FIN1b, 10 seeds | — | L №245 |
| 51 | 10-04 16:43 | v1109 | 232.67 | −0.09 | MLP member retrained on FIN1b, 2/3 weight | E181a | B §21cn |
| 52 | 10-04 16:45 | v1119 | 232.62 | −0.04 | E181a at full weight | E181a | B §21cn |
| 53 | 10-04 16:50 | v1139 | 232.62 | −0.00 | 15 CatBoost seeds; stage trained on the production base | E169 | B §21cn |
| 54 | 10-05 06:57 | v12008 | 232.36 | −0.26 | Movement-only member for rows without LOBT outside LIRF (4,907 ranking rows), weight 0.35 × 2/3. CV −0.05; forecast −0.06 | E185 | L №255, №260, №262, №267 |
| 55 | 10-05 10:12 | v12048 | 232.24 | −0.12 | Five anchored members retrained on FIN1b, with E185 | E182 | L №265, №273, №278 |
| 56 | 10-05 16:42 | v12060 | 232.10 | −0.14 | OPDI exit-parking stage at LSZH, full weight | E159b/c | L №285, №288 |
| 57 | 10-05 16:42 | v12061 | 231.99 | −0.11 | 50:50 blend of v12048 and v12060 (half the LSZH correction), fixed before either was scored | E159 | L №285, №288 |
| 58 | 10-06 21:19 | v12073 | 231.98 | −0.01 | 50:50 blend of v12061 with v12072 (previous-visit stage retrained for LIRF rows with NM) | E188 | L №321, №324 |
| 60 | 10-07 07:02 | v12074 | 231.44 | −0.00 | E190 and E188 in one chain | E190, E188 | [results](../analysis/submission_results_20261007.md) |
| — | 10-08 13:20 | v12090 | 232.19 | (+1.25) | Technical canary of the final recipe on January/July: E81 removed and the 42 frozen tail rows replaced by a fresh ten-seed TabPFN-2 ensemble, old members and weights (three-seed neural member at 0.25). Forecast 231.6, corridor [231.1; 232.3]; not a candidate for best | — | L №389–391; [canary record](../analysis/v12090_upload_20261008/RESULTS.md) |
| 62 | 10-08 20:05 | v12076 | 230.82 | −0.12 | Neural member as the mean of ten seeds instead of three, the only change against v12075 (zero-fit replay). Seeds 3–9 were fitted on the frozen E203 ranking pack (the final-view seeds cannot be applied to the January/July queries); every checkpoint replayed exactly. On 2025 out-of-fold data the ten-seed mean at the same weight was neutral (+0.0065 s on `decided`); forecast −0.03 with corridor [230.74; 230.99] | E203 | L №410, №416, №417; [upload record](../analysis/v12076_upload_20261008/RESULTS.md) |

## 2. Phases

### 2.1 From one regressor to a recording-regime mixture (13 Sep; v1–v5; 501.2 → 339.2; 5 uploads)

**Observed.** The target is a recorded interval, not always a physical one. Among 4,126
departures of 2025 with a target above one hour, the target matched take-off minus the
scheduled time (±120 s) in 45.5 % of cases, take-off minus IOBT in 10.9 %, LOBT in 10.8 %
and EOBT in 6.6 %. At LIRF this explains 87 % of the long values at this ±120 s tolerance (83.7 % at the
model's ±6 s tolerance, [data insights](data-insights.md) Figure 2b); 14 labels are shifted by
exactly one day (B §3, item 2). In the January+July 2025 holdout, 45 rows above four hours
carried 52 % of the sum of squared deviations (B §3, item 1).

**Response.** Model the recording process. A classifier gives the probability of each
regime, each regime has a candidate value, and the prediction is the probability-weighted
mean (the squared-error optimum if the probabilities are calibrated). The two-class version
alone removed 135 s (v2). Six classes, weather, geometry and a CatBoost/LightGBM blend
removed 27 s more (v3–v5).

**Cost.** Regime labels need dataset rules: a ±6 s tolerance, and no planning regimes at
EDDF, EHAM and LSZH, whose labels have second resolution ([model design](model-design.md)).
At v4 the 2025 holdout stopped tracking the board (402.4 → 406.3 while the board went 354.2
→ 344.1). Runway use in the ranking months differs from the holdout months: the EGLL 09R
share ranged 11–71 % by month in 2025 and was 46 % and 38 % in January and July 2026 (B §3,
item 6). From then on, features that depend on 2026 conditions were judged on the board (B
§2).

### 2.2 Residuals to NM timestamps and the sparse Rome tail (14–15 Sep; v7–v22; 339.2 → 273.4; 8 uploads)

**Observed.** The NM actual off-block `AOBT_3` is close to the label (median `AOBT_3` −
block −0.85 min, 97.6 % within 15 min) but rounded and noisy. On all 2,062,577 departures of
2025 with a LOBT, the block time lies within 3,606 s of LOBT (B §3, items 4–5). Departures
without an NM match were 1.6 % of rows but 67 % of holdout squared error, 44 % of it at LIRF
(B §3, item 1). For LIRF without NM and 4–30 h late, 61–100 % of labels per delay band
equalled the scheduled time, and 39 % of rows 12–30 h late were shifted by one day; the
classifier gave that shift almost zero probability (B §3, item 3).

**Response.** The regressor learns the residual to the NM proxy and uses neighbour features;
predictions stay inside the LOBT window (v7, −32.6). The Rome tail gets its own treatment:
an empirical regime table (v8, −27.0), then TabPFN-2 (v13, −4.7), then flight-number history
for the 0–4 h band (v18, −0.84 vs v14). Removing the `month` feature gave −0.87 (v22).

**Cost.** The tail gains rest on 42–43 ranking rows. A reviewer found that 65.1 % of the
table's holdout gain came from three rows (B §11). TabPFN outputs differ between machines
(39 of the 42 rows differed between x86 and Apple Silicon, B §21bv), so the final chain
copies these 42 predictions from a fixed donor file ([model design](model-design.md)).

### 2.3 Own-aircraft ADS-B surface stages (15–19 Sep; v27–v82; 273.4 → 247.8; 16 uploads)

**Observed.** The challenge tables do not show when the aircraft started to move. A probe of
adsb.lol on 13 September concluded there was no ground coverage. That was an error: it found
ground points only by the `ground` flag, which is set on 0.02 % of points at EDDM (B §0). On
15 September a public competitor reported −14 s from adsb.lol ground tracks (B §21a), and
the source was reopened.

**Response.** A second-stage LightGBM corrects the base residual from the flight's own
track: first ground fix, first movement, stops, stand distances (E25, E30, E36). On 18
September the team found that the stored surface files also held airborne points of the same
aircraft; previous-visit features followed (E64, E65), then a 1 kt movement threshold (E74)
and three small additions (E70). Separately, a per-airport scale for the NM proxy (E46) gave
−1.00. Together −25.7 s; the stages repeatedly transferred at full size or more (section 3).

**Cost.** Coverage limits the stages. Full ground coverage would be worth about −28 s on the
ranking set, but LFPG, LIRF in July and LTFM are almost unobserved in any open archive (B
§21b, [coverage ceiling](../analysis/coverage_ceiling_2026-09-16.md)). 56.9 % of the error
on the rows the base model decides sits in rows with coverage 0–1 (B §21h). The daily
archives (about 3.5–4 GB) are processed as a stream, about two minutes per day (B §21a, §7
row E25).

### 2.4 Filed-time neighbours, classifier costs and whole-file blends (20–23 Sep; v88–v175; 247.8 → 242.3; 13 uploads)

**Observed.** Two complete pipelines that had lost as replacements, v44 and v60, had
residuals correlated 0.939 and 0.964 with the accepted chain, so each carried some
independent information (B §21q). At LTFM the neighbour features built on `AOBT_3` were
nearly constant: 96.3 % of 2025 departures fall into three 60 s bins of take-off minus
`AOBT_3`. Filed times still varied: corr(y, take-off − LOBT) 0.560 against 0.157 for
`AOBT_3` (B §21r).

**Response.** Whole-file blends with weights from CV, priced before upload with the Gram
identity (section 4): E76, then E81 on each new chain. Neighbour features on filed times
(E77, −2.43). Cost-weighted regime classifier (E89, −1.18 against its comparator). Removal
of 2025 target encodings (E99).

**Cost.** The final file depends on two old complete pipelines (v44, v60) whose exact
out-of-fold counterparts are not available; this limits full-chain validation
([validation](validation.md)). One slot was lost when a model without its own board score
was blended in (E78, section 4). Dropping the E81 blend and the LOBT edge lift later cost
+0.69 (v314 vs v316, B §21cj), so both stayed.

### 2.5 Removing 2025-specific capacity; recording rules (24–27 Sep; v185–v396; 242.3 → 235.5; 20 uploads)

**Observed.** Features that encode 2025 averages (target encodings by stand and runway,
stand identity, runway configuration) fit 2025 and did not carry over to 2026 (section 3).
For LIRF rows without NM, whether the airport records the scheduled time depends on the
airline, but the operator was unknown for all of them (L №6).

**Response.** Remove capacity one step at a time and test each removal on the board: target
encodings (E99), stand identity with residual encodings (E112), runway configuration (E116).
Add the airline prefix from the flight number (E106), later also in the TabPFN tail model
(E118). Average CatBoost over five seeds (E121). Together −6.74 s; the largest step (v316,
−3.60) changed only the 42 Rome tail rows.

**Cost.** Several removals made the score worse: residual encodings alone +0.40, stand
identity with the encodings kept +1.45, `stand_prefix` +1.17, `ap_runway` +0.43 (B §21bq,
§21bs, §21ca). The v316 step shows how strongly the score depends on a few tail rows.

### 2.6 Stage refinements and model members (28 Sep – 4 Oct; v404–v1019; 235.5 → 232.6; 35 uploads)

**Observed.** Many new features failed the CV gates (B §0). Members that learn the same
target from different anchors correlate 0.986–0.99 with the main regressor (B §21cm).

**Response.** More seeds (CatBoost 10, 15, then 25; stages 5). Stage training on all rows
(E150) and without drifting reception features (E147). The planning-regime gate (E152). E14
dropped once the base had absorbed it. Anchored LightGBM members (E171–E173), an MLP member
(E176, later on FIN1b as E181a), the destination ATFM delay (E163) and the final feature
table FIN1b. Together −2.90 s; two steps were large: anchored members (v808, −0.98) and the
MLP member (v899, −0.40; −0.78 against its control).

**Cost.** Most uploads tested changes worth 0.01–0.2 s. LightGBM member weights were scaled
to 2/3 and the MLP kept at full weight after comparing complete files on the board (L №241,
№245). Two external-data stages transferred weakly: OPDI at LSZH gave −0.13 against a
leave-one-month-out forecast of −1.2; at EDDF it gave +1.13 and was dropped (L №215).

### 2.7 Final members and the ordered-trajectory network (5–8 Oct; v12008–v12077; 232.6 → 230.5; 14 uploads, one of them the final-recipe canary)

**Observed.** The neural member saw only tabular features. The ADS-B track entered the model
only through hand-made stage features. Its weight, 0.25, had been inherited from the MLP
member E181a and was never chosen at the level of the full chain.

**Response.** A movement-only member for rows without LOBT outside LIRF (E185, −0.26),
anchored members on FIN1b (E182, −0.12), the LSZH OPDI correction at half weight (−0.25 in
two steps), numerical embeddings in the neural member (E190, −0.54), and the
ordered-trajectory member (E203, −0.51). On 8 October the member was averaged over ten
seeds (v12076, −0.12) and its weight raised to 0.35 with the anchors scaled by 0.65/0.75
(v12077, −0.34). The weight came from E212, an exact recompute of the 2025 out-of-fold chain
at twelve grid values with the three-seed and the ten-seed member: the pooled optimum was
0.40 for both members, 0.35 was the only value below 0.25 in every month, and the ten seeds
alone changed nothing at the old weight (+0.0065 s). Both uploads were forecast before
upload; v12077 came out 0.26 s better than its central forecast.

**Cost.** GPU training (AMD RX 7900 XTX, about 269 s per seed fit;
[model design](model-design.md)). E190 was uploaded before a full-chain CV (L №321), and
there is no fully nested CV of the final chain ([validation](validation.md)). A sequence
model on the raw ground track had been closed on 23 September because most stage error sits
where the track starts after pushback (L №13); on 26 September the team recorded that this
did not follow, since the rest of a track can still carry information (B §21bz). E197 put
the sequence inside the regular-duration member, before the surface stages.

## 3. Cross-validation against the leaderboard

Most uploads from 20 September on carried a forecast range written before upload (for
example B §21ah; L №138, №166). The table compares CV and board effects of single changes.

| Change (board pair) | Kind | CV | Board | Evidence |
|---|---|---:|---:|---|
| ADS-B stage E25 (v27 vs v22) | observation of the flight | −9.53 | −15.22 | B §7 |
| Stand geometry E36 (v30 vs v29) | observation of the flight | −2.98 | −3.71 | B §7 |
| Previous visits E64+E65 (v78 vs v55) | observation of the flight | −2.37 | −2.65 | B §21i |
| Filed-time neighbours E77 (v97 vs v82) | observation of other flights | −2.73 | −3.26 | B §21t |
| Proxy scale E46 (v55 vs v30) | calibration, no new capacity | −0.40 | −1.00 | B §7 |
| Airline prefix E106 (v195 vs v155) | recording rule | −0.58 | −1.28 | B §21bg–§21bh, §21bq |
| Classifier cost weights E89 (v115 vs v97) | loss weighting | −1.02, but +11.27 on y ≤ 7,200 | −1.18 | B §21ag, §21ai |
| Ordered trajectories E203 (v12075 vs v12074) | neural member | −0.55 (before E81) | −0.51 | [validation](validation.md) |
| Neural member: ten seeds instead of three (v12076 vs v12075) | seed averaging, no new capacity | +0.0065 (decided, pooled 2025, before E81) | −0.12 | L №410, №417 |
| Neural weight 0.25 → 0.35, anchors × 0.65/0.75, E212 (v12077 vs v12076) | mixture weight, no new capacity | −0.19 (ten-seed member, decided, pooled 2025: 217.723 → 217.535); about −0.11 on January + July 2025 | −0.34 | L №410, №425, №427; [E212 report §4](../analysis/e212_nn_weight_10seed_20261008.md) |
| `month` removed, E24 (v22 vs v18) | capacity removed | −2.18 | −0.87 | B §7 |
| Target encodings removed, E99 (v155 vs v115) | capacity removed | +0.94, 0 of 6 folds better | −0.17 | B §21ba, §21be |
| Per-airport models E38 (v43 vs v30) | learned capacity | −10.98 | +3.79 | B §7 |
| Regime shares by group E50 (v65 vs v30) | 2025 aggregates | −2.79 | +5.78 | B §7 |
| Flight-number habit, Rome with NM, E133 (v249 vs v247) | 2025 aggregates | −0.72 | +0.36 | B §0, §21ca |
| MLP partner E100 (v161 vs v115) | other model family | −0.66 | −0.26 | B §0, §21bc |
| Anchored members E171–E173 (v808 vs v655) | members, weights × 2/3 | about −1.24 | −0.98 | B §21cm |
| Destination ATFM delay E163 (v778 vs v655) | external aggregate | −0.76 | −0.17 | B §21cm |
| Monthly PRU calibration E145 (v448 vs v447) | calibration to external data | better in 6 of 6 folds | +0.11 | B §21cj |
| Airline prefix in TabPFN tail E118 (v316 vs v247) | 42 tail rows | forecast −0.27 | −3.60 | B §21bv |

**Rules the team derived**

1. *Observation transfers; training-period aggregates do not* (B §7, 17 September). Changes
   that read what the specific flight did (ADS-B stages) or what other flights filed (E77)
   transferred at full size or more. Learned capacity on 2025 aggregates won in CV and lost
   on the board (E38, E50, later E133; also E32, ADS-B features inside the main model: CV
   −1.41, board +0.33).
2. *Capacity-free changes transfer, even through a failed gate* (B §21ai). E89 failed the y
   ≤ 7,200 leg because that subset is selected by the true label, so any shift of
   probability towards long regimes looks worse there by construction. On the board it
   gained 1.18 s.
3. *Removing 2025 capacity can transfer against CV* (B §21be). E99 was worse in all six
   folds and better on the board. CV measures 2025 against 2025 and cannot see the shift, so
   for removals CV served only as a cost filter (decided ≤ +1.0 s, B §21ba). This is not
   universal: four other removals hurt (section 2.5).
4. *Measure only rows the base model decides* (B §21b, 16 September). Later stages overwrite
   10–45 LIRF tail rows per fold, which carried up to 62 % of a fold's squared error. With
   them excluded, the per-airport models' CV gain fell from −10.55 to −1.82 s; an
   early-stopping leak explained only 0.4 s of it.
5. *Small-sample month rules mislead.* The "better in ≥ 8 of 12 months" rule for the tail
   counted months with 4–6 rows like July with 39 rows. It rejected E118, which then gave
   −3.60; the rule was withdrawn (B §21bv).

On 4 October the team calibrated this on 38 single-change pairs, 15 of them recomputed from
out-of-fold files (L №252). The board change followed the CV change on the `decided` rows
(Pearson 0.89, slope 1.06) and on its y ≤ 7,200 part (0.90), but not on the y > 7,200 tail
(0.09). The sign agreed in only 60–70 % of pairs. Board/CV ratios by type: ADS-B stages
about 1.0; ensemble members about 0.7; new base features 0.2–0.3; rules and calibrations
fitted on 2025 zero or negative (the earlier E46 proxy scale was an exception); tail models
and capacity removal not scaled by CV at all. Later forecasts used CV × type ratio.

## 4. Predicting blends of submitted files: the Gram identity

Every board score is an unweighted RMSE over the same 344,841 rows (B §21q). For two scored
files with predictions P₁, P₂, scores s₁, s₂ and RMS difference d = sqrt(mean((P₁ − P₂)²)),
write eᵢ = Pᵢ − y. Then mean(e₁e₂) = (s₁² + s₂² − d²)/2, and the blend wP₁ + (1 − w)P₂
scores

```text
RMSE² = w·s₁² + (1 − w)·s₂² − w·(1 − w)·d²
50:50:  RMSE = sqrt((s₁² + s₂²)/2 − d²/4)
```

No labels are needed: two scores and two files suffice. Implementations:
[`src/blend_submissions.py`](../src/blend_submissions.py) (general weights) and
[`scripts/blend_score_identity.py`](../scripts/blend_score_identity.py) (fixed two-file
blend with rounding bounds).

| Upload | Predicted before upload | Board | Difference | Evidence |
|---|---:|---:|---:|---|
| v89 (E76) | 246.9202 | 246.9205 | 0.0003 | B §21q |
| v98 (E81) | 243.8342 | 243.8347 | 0.0005 | B §21v |
| v116 (E81) | 242.6554 | 242.6556 | 0.0002 | B §21ai |
| v162 (E100 blend) | 242.5613 | 242.5609 | 0.0004 | B §21bc |
| v163 (E81) | 242.2622 | 242.2622 | < 0.0001 | B §21be |
| v226 (E81) | 240.1387 | 240.1391 | 0.0004 | B §21bq |
| v12071 (50:50 v12061 + v12070) | 231.6086 | 231.6095 | 0.0009 | B, "Read first" |

For v12071: s₁ = 231.9882 (v12061), s₂ = 231.4449 (v12070), d = 14.155 s, recomputed for
this page from the two files. The uploaded file is rounded to whole seconds (at most 0.5 s,
RMS 0.345 s from the unrounded blend), which explains the residual 0.0009 s.

Uses. (1) *Price a blend before spending a slot*; the weights came from CV and were fixed
before the score was computed. (2) *Refuse a bad blend*: weights fitted against the bare
base model promised −9.68 s in CV for v97; the identity priced them at 249.6954, 5.20 s
worse than v97, and nothing was uploaded (B §21t). (3) *Expose a hidden component*: v91
blended a model without its own board score (E78) and scored 247.3631 against a forecast of
246.70. Solving the identity gave 338.46 s for that component, which had trailed the control
by only 7 % in CV. Rule: blend only complete, scored submissions (B §21s). The identity
holds on the ranking rows only and says nothing about another period.

## 5. A note on the late phase

After 26 September progress was slow. From v316 (236.32 s, 26 September) to v12077 (230.48
s, 8 October) the best score improved by 5.84 s over 57 further uploads. On 26 September the
team was 8th of 192 and 3.14 s behind third place (B §21ca). On 8 October in the morning it
was 14th of 237, and third place was 215.62 s, 15.32 s ahead ([team-best
snapshot](tables/leaderboard_team_best_20261008.csv)); at 20:15 UTC, after v12077, it was
16th of 239 with third place at 213.5212 s, 16.96 s ahead
([upload record](../analysis/v12077_upload_20261008/RESULTS.md)). Leading teams improved
much more in the same period. Late gains came from members, seeds and small stage changes
worth 0.01–1.0 s each. No submission was tuned row by row against the leaderboard. Blend weights were fixed
from cross-validation; weights that would have been optimal on the board (−1.004 s for E76)
were rejected (B §21q), and the working rule was not to probe the ranking labels (B §21bw; L
№262). The board was still used to choose between complete, pre-built variants, for example
the TabPFN tail variant kept in v316 and the 2/3 scaling of LightGBM member weights (B
§21bv; L №241, №245). The final score therefore includes selection effects on the January
and July 2026 rows.

## 6. Leaderboard use disclosure

The ranking months were used to choose between complete, pre-built variants. The table lists
every design choice in the v12075 chain that was decided, or kept, because of a leaderboard
result rather than cross-validation. Nothing in the chain was fitted to individual ranking rows,
and no blend weight was optimised on the board.

| # | Choice in v12075 | How it was chosen | Evidence |
|---:|---|---|---|
| 1 | 42 LIRF tail predictions frozen from v361 and protected from later stages | The airline-prefix TabPFN variant failed its leave-one-month-out rule (6 of 12 months) but gained −3.60 s on the board; its single TabPFN draw was then kept, and a seed ensemble was rejected on the board (+0.86 s) | B §21bv; L №118, №133 |
| 2 | Whole-file blend with v44 and v60 (E81) | Weights from cross-validation (NNLS on 2025 OOF), not from the board; keeping the blend was confirmed by a board comparison (+0.69 s without it and the LOBT lift) | B §21q, §21v, §21cj |
| 3 | LightGBM anchored members at 2/3 of their CV weights; neural member at full weight | Two board comparisons of complete files (×1 worse by 0.18 s; ×2/3 for the MLP worse by 0.10 s) | L №241, №245 |
| 4 | Movement-only member (E185) at 0.35 × 2/3 | CV weight scaled by the same 2/3 rule | L №262 |
| 5 | Removal of `month`, 2025 target encodings, stand identity, runway configuration and the E14 rule | Each removal was tested with one upload and a forecast written beforehand, because CV on 2025 cannot see the 2025→2026 shift; four other removals made the board worse and were not kept | L №41, №141, №166; B §21ba, §21bq, §21bs |
| 6 | Cost-weighted regime classifier (E89) | Failed the y ≤ 7,200 s leg of its gate, uploaded as a test with a forecast, kept after −1.18 s | B §21ag–§21ai |
| 7 | LSZH OPDI correction at 50 % | Weight fixed before scoring as a cautious transfer; the same correction at EDDF was rejected on the board (+1.13 s) | L №215, №285 |
| 8 | Uploads without a full-chain CV (E190 and five fixed files on 6 October; E203 after a pre-E81 comparison) | Owner's decision; the best of the scored files became the current chain | L №318, №321, №366 |

Items 1 and 2 depend on files that exist only for January and July 2026 and cannot be carried to
a new period. How the final submission on the four-month dataset treats them is described in the
final-submission documentation.

The two changes of 8 October (v12076, v12077) add no item to this table. The ten-seed mean
was decided before the extra seeds were trained (L №394, №416), and the weight 0.35 was
chosen on 2025 out-of-fold predictions and written into the final assembler
(`WEIGHTS['final']`, L №425) before v12077 was built; the grid, the selection rule and the
forecast were registered in advance (L №407a, №426). The uploads confirmed the choice on the
board; they did not make it. The one weight decision that departed from the registered rule
(0.35 instead of the rule's 0.40) was taken on 2025 months, not on the ranking months, and
in the direction of the smaller change.

[B]: ../REVIEW_BRIEF.md
[L]: ../HYPOTHESES_LOG.md

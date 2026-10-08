---
title: "A Recording-Regime Mixture for Taxi-Out Time Reconstruction at Ten European Airports"
subtitle: "Team gentle-igloo in the EUROCONTROL PRC Data Challenge 2026"
author:
  - name: Denys Kolomiiets
    affiliation: Independent researcher (team gentle-igloo)
    orcid: "[[RESULT: ORCID]]"
    email: "[[RESULT: contact e-mail]]"
keywords: [taxi-out time, recording regimes, ADS-B surface tracks, gradient boosting, open aviation data]
status: "DRAFT, not submitted. Numbers marked [[RESULT: ...]] are not yet available."
target: "Journal of Open Aviation Science (JOAS), research article"
---

> **Draft status.** This manuscript describes the state of the solution on 8 October 2026, after
> 108 scored submissions. Every number is copied from the repository documentation
> (`README.md`, `docs/*.md`, `docs/tables/*.csv`). Items marked `[[RESULT: …]]` depend on results
> that did not exist when the draft was written (the final four-month blind ranking, the
> February/June generalisation study, the uncertainty calibration and the data-quality study) and
> must be filled from `docs/generalisation.md`, `docs/uncertainty.md`, `docs/data-quality-regimes.md`
> and `docs/pru-view.md` before submission. The draft is not an accepted article and must not be
> cited as one.

# Abstract

The EUROCONTROL Performance Review Commission (PRC) Data Challenge 2026 asked participants to
predict the recorded taxi-out time of departures at ten European airports, training on all 2025
departures (2,085,047 rows) and scored by root-mean-square error (RMSE) on 344,841 departures of
January and July 2026. We describe the solution of team gentle-igloo, which reached 230.94 s RMSE
from a first submission of 501.23 s and ranked 14th of 237 teams on 8 October 2026
[[RESULT: final Jan/Jul rank and four-month blind-ranking result]]. The design follows three
properties of the data. First, the label is a recorded interval that follows several recording
regimes: 6.8 % of 2025 labels coincide within 6 s with take-off minus a planning timestamp, and at
Rome Fiumicino (LIRF) the planning time replaces the real off-block time for 83.7 % of labels longer
than one hour. A multiclass classifier therefore weights a regular-duration estimate against five
alternative candidates, and the prediction is the probability-weighted mean. Second, the Network
Manager (NM) minute-rounded actual off-block time is a strong but biased anchor (384.9 s RMSE alone,
median error from −118 s to +296 s by airport); gradient-boosted trees learn the residual to a
per-airport calibrated proxy. Third, the aircraft's own ADS-B ground track (adsb.lol, ODbL) shows
when it started moving; coverage-gated correction stages and an ordered-trajectory neural member use
it. A TabPFN-2 classifier handles the sparse Rome tail. Validation used six temporal month-pair
folds of 2025 with pre-registered decision rules. We report which classes of change transferred
from cross-validation to the leaderboard (observational stages: fully; learned capacity on 2025
aggregates: not at all), a label-free identity that predicts the score of blends of scored files,
and a full disclosure of leaderboard use. Code is GPLv3; data licences are documented; the best
file is replayed byte-exactly from archived components.

# 1 Introduction

Taxi-out time, the interval between an aircraft leaving its stand and taking off, is one of the
operational quantities that the EUROCONTROL Performance Review Unit (PRU) monitors at European
airports. The PRU's *additional taxi-out time* indicator compares the observed taxi-out time with a
reference time for the stand and runway used, and reads the difference as an approximate measure of
departure-queue time during congested periods [@eurocontrol2023atxot]. The indicator is computed
from the Airport Operator Data Flow combined with Network Manager (NM) trajectory data
[@eurocontrol2023atxot]. Taxi-out time is therefore both an operational variable and a recorded
one: what is monitored is the interval between two recorded timestamps, and the quality of the
recording is part of the quantity.

The PRC Data Challenge 2026 [@prc2026site] made this quantity the target of an open modelling
competition. Participants received all 2025 departures and arrivals at ten airports (EDDF, EDDM,
EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM) with NM flight records, and were asked to predict
the recorded taxi-out time `TAXITIME_SEC_mvt` of 344,841 departures in January and July 2026, scored
by RMSE on a public leaderboard with at most five uploads per UTC day. The organiser required GPLv3
source code, documented open external data and reproduction documentation, and announced that a
large part of the prize assessment would rest on model design and documentation rather than on
the ranking alone [@prc2026site]; the organiser's dated statements are quoted verbatim in the
repository. On 8 October 2026 a blind final phase was added: one submission for January, February, June and July
2026 (670,790 departures), ranked separately on the initial two months, the two new months and all
four [[RESULT: outcome of the final ranking]]. Previous editions of the challenge, on take-off weight
[@spinielli2025atow] and fuel burn [@sun2026fuel], were followed by solution papers in this journal
[@murca2025atow; @grigoriou2026fuel; @kacar2026fuel]; this paper follows that practice.

An important feature of the task is what remains visible in the ranking file. Only the departure
off-block time and the target are hidden; the take-off time, the NM timestamps of the flight
(scheduled, initial, latest and estimated off-block, and the minute-rounded actual off-block
`AOBT_3`), the stand, the runway and all arrivals remain visible. The organiser confirmed on
16 September 2026 that "the rest in the same row and other rows is available to your model", and
on 19 September that "the model is for post-ops, not for tactical use". The task is therefore a
**retrospective reconstruction of when the aircraft left the stand**, not a pre-departure forecast.
We state this at the outset because it shapes every design decision and bounds every claim.

This paper makes four contributions.

1. We document a *recording-regime* phenomenon in the challenge labels: a substantial fraction of
   recorded off-block times are copies of planning timestamps, concentrated at one airport, and these
   rows dominate the squared error. We show that modelling the recording process as a probability
   mixture is worth more than any learner choice (Section 2.2, Section 5).
2. We describe a modular chain (Section 3) in which gradient-boosted trees learn the residual to a
   calibrated NM proxy, coverage-gated stages correct the estimate from the flight's own ADS-B
   surface track, a neural member reads the ordered track, and a small tabular foundation model
   handles a 42-row sparse tail.
3. We report a validation protocol of temporal month-pair folds with pre-registered decision rules,
   and an empirical account of which classes of change transferred from cross-validation to the
   leaderboard and which did not (Section 4).
4. We disclose every design choice that was taken or kept because of a leaderboard result, and we
   state the reproducibility level actually achieved for each part of the submission (Sections 6–7).

# 2 Data

## 2.1 Challenge data

The training set contains 2,085,047 departures of 2025 at the ten airports, each with the movement
record (`_mvt` columns, sourced from the Airport Operator Data Flow) and, for 98.9 % of departures,
a matched NM flight record (`_flt` columns) with the scheduled off-block time, the initial, latest
and estimated off-block times (IOBT, LOBT, EOBT_1) and the actual off-block time `AOBT_3`. The target
`TAXITIME_SEC_mvt` is take-off time minus recorded off-block time. The ranking set holds 344,841
departures of January and July 2026 with the same columns except the hidden off-block time and
target. Arrivals are included in both sets with their block times.

Timestamps are mostly minute-resolved. 97.7 % of `AOBT_3` values fall on a whole minute. At EDDM,
EGLL, LEBL, LEMD, LFPG, LIRF and LTFM at least 99.95 % of block times sit within ±6 s of a whole
minute; at EDDF, EHAM and LSZH only 21.7–24.9 % do, close to the 13/60 expected from uniformly
distributed seconds, so these three airports record block time to the second. A label at the seven
minute-resolved airports is therefore a difference of two minute-resolution times with a few seconds
of jitter. Label ranges also differ by airport: LEBL and LEMD labels stop at 3,604 s and 3,598 s,
while the other airports reach 6,899–131,167 s; 369 labels are negative.

## 2.2 Recording regimes

The first property that shaped the model is that the label does not always describe a physical
taxi. In 2025, 140,993 labels (6.76 %) equal take-off minus the scheduled time, IOBT, LOBT or EOBT_1
within ±6 s, or lie between one day and one day plus two hours after a planning time (a one-day
shift). We call these *recording regimes*: the recorded off-block time is a copy of a planning
timestamp rather than an observation. Figure 1 shows the shares per airport.

![Figure 1. Recording regimes in the 2025 labels. (a) Per airport, share of labels equal to take-off minus a planning time (LIRF 24.1 %; 6.9–9.3 % at the other six labelled airports). (b) Among labels above one hour, 83.7 % at LIRF match a planning time against 3.1–5.6 % elsewhere. (c) The schedule class is 4.4 % of rows but 28.0 % of the label's squared deviation from its mean; the 14 one-day-shift labels hold 16.8 %.](../docs/figures/fig2_recording_regimes.png)

Two tests separate substitution from coincidence. First, a same-minute test: at LIRF 18.1 % of
block times fall in the scheduled minute, against 3.7 % in the minute before and 1.3 % in the minute
after. At EDDM, EGLL, LEBL, LEMD, LFPG and LTFM the scheduled minute holds 4.1–5.6 % against
3.5–5.3 % in its neighbours, which is consistent with on-time pushbacks whose planning-time
candidate equals the physical taxi time anyway. Second, the long-label test: among the 1,967 LIRF
labels above one hour, 83.7 % match a planning time, against 3.1–5.6 % at the other airports with
at least 20 such labels. EDDF, EHAM and LSZH, which record block time to the second, show no
planned-regime labels and are excluded from regime labelling in the model.

Panel (c) shows why this matters for RMSE. The schedule class is 4.4 % of rows but holds 28.0 % of
the squared deviation of the label from its mean; 14 one-day-shift labels hold a further 16.8 %. In
January and July 2025, 45 of 344,419 labels exceed four hours and hold 52.4 % of the label's sum of
squared deviations. A single regressor trained on the physical taxi fails exactly where the metric
is decided.

The regime labels used for training come from explicit comparisons with the target (`regime_label`
in `src/train.py`): ±6 s tolerance, six classes (regular, schedule, IOBT, LOBT, EOBT, one-day
shift), no planned regimes at EDDF, EHAM and LSZH. These are dataset-specific heuristics; they
must be reassessed for another source of movement records.

## 2.3 The NM off-block proxy

The second property is that the NM actual off-block time `AOBT_3` is a strong but noisy anchor
(Figure 2). It is present for 98.92 % of 2025 departures and rounded to the minute. Used directly,
take-off minus `AOBT_3` gives an RMSE of 384.9 s against the label, where a constant prediction
gives 546.4 s. Of all proxy errors, 39.4 % are within two minutes and 74.6 % within five. The error
has an airport-specific offset: the median is −118 s at LIRF, +87 s at EDDF, +107 s at EHAM and
+296 s at LTFM, where only 16.9 % of departures are within two minutes. The median absolute error
is 121–184 s at nine airports and 356 s at LTFM. On all 2,062,577 departures of 2025 with a LOBT,
the recorded block time lies within 3,606 s of the LOBT, which defines a feasibility window used
throughout the chain.

![Figure 2. Signed error of take-off minus AOBT_3 as a taxi-out estimate, per airport (medians from −118 s at LIRF to +296 s at LTFM) and the share of departures by error size (16.9 % within two minutes at LTFM, 33–49 % elsewhere).](../docs/figures/fig3_nm_proxy_error.png)

Departures without an NM match are 1.08 % of 2025 rows (1.56 % in January and July 2025) and 1.53 %
of the ranking set, reaching 4.96 % at EHAM in January 2026. They have no `AOBT_3`, LOBT, IOBT or
EOBT_1. In out-of-fold predictions these rows hold 36.6 % of the squared error (Section 2.6).

## 2.4 ADS-B surface coverage and its drift

The third property is that the aircraft's own ADS-B ground track shows when it actually started
moving, but coverage is uneven and changed between the training and ranking months (Figure 3). We
used the adsb.lol `globe_history` archives for 2025 and 2026, published under ODbL 1.0
[@adsblol2025; @adsblol2026]; the organiser confirmed on 17 September 2026 that this source "can be
considered open data for the challenge". An *own surface track* means that the departing aircraft
has an adsb.lol surface segment before take-off (coverage class ≥ 2 in our extraction).

![Figure 3. Share of departures with an own ADS-B surface track in January and July, 2025 against 2026, per airport. LIRF goes from 0.6 % to 78.5 % in January and from 93.9 % to 6.2 % in July; LEMD January from 2.3 % to 96.3 %; EGLL July from 35.9 % to 96.3 %; LTFM stays near zero.](../docs/figures/fig4_adsb_coverage_drift.png)

Over January and July the share of departures with a track rises from 53.8 % in 2025 to 64.9 % in
2026, but single airport-months move much more: LIRF from 0.6 % to 78.5 % in January and from
93.9 % to 6.2 % in July; LEMD January from 2.3 % to 96.3 %; EGLL July from 35.9 % to 96.3 %; EDDF
down by 16–17 points in both months. LTFM has practically no surface track in either year and LFPG
stays between 5 % and 25 %. A model that learned *where aircraft are heard*, rather than *how they
move*, would not transfer. This drove the decision to use ADS-B as coverage-gated correction
stages rather than as base features (Section 3.5), and to drop reception-density features whose
behaviour did not transfer.

An early probe of adsb.lol on 13 September 2026 concluded that there was no ground coverage; it had
read ground state only from the `ground` flag, which is set on 0.02 % of points at EDDM. The source
was reopened on 15 September after a public competitor reported a gain from the same archive. We
record this because a rejection depends on the measurement used (Section 4.3).

## 2.5 Other external data

The remaining external inputs are OpenStreetMap stand, runway and taxiway geometry (ODbL 1.0,
retrieved through Overpass) [@osm]; the Iowa Environmental Mesonet METAR archive (public domain
per the IEM statement) [@iem]; daily arrival ATFM delay per destination airport from the
EUROCONTROL Aviation Intelligence Portal (portal terms: free with attribution, non-commercial;
accepted as open data for the challenge by the organiser on 17 September 2026); and Open
Performance Data Initiative (OPDI) v0.0.2 exit-parking and taxiway events [@opdi], used for a
correction at LSZH only (accepted as open by the organiser on 4 October 2026). The TabPFN-2
classifier weights are used under the Prior Labs License 1.1 and are not redistributed. The
challenge data themselves are provided by the organiser to all teams and are not redistributed.
Section 7 lists each source with its terms; the organiser's statements are quoted with timestamps
in the repository.

## 2.6 Error budget

Figure 4 decomposes the 2025 out-of-fold squared error of the chain before its final blend
(E195 `joint` arm, 2,084,921 rows, RMSE 218.3 s; Section 4.1 defines the out-of-fold predictions).
LIRF holds 7.7 % of rows and 30.8 % of the squared error, LFPG 11.5 % and 21.6 %. EDDF, EDDM and
EHAM together hold 31.0 % of rows and 6.5 % of the error. Three one-day-shift rows hold 13.5 %, and
11 labels above four hours hold 25.4 %. The 1.07 % of rows without an NM record hold 36.6 %. Rows
without an own surface track are 39.5 % of the data, hold 57.1 % of the error and have an RMSE of
262.3 s against 183.8 s with a track (229.2 s against 145.4 s on labels up to 7,200 s). The ten
largest errors make up 25.8 % of the total and the largest 1 % of rows 62.0 %.

![Figure 4. Share of rows against share of squared error in 2025 out-of-fold predictions, by airport, recording regime, label size, NM match and ADS-B track.](../docs/figures/fig5_error_budget.png)

The chain has a dedicated component wherever the error concentrates: the LIRF tail without NM has
its own model, rows without LOBT outside LIRF get a movement-only member, and the ADS-B stages
correct the start of taxi where a track exists. On labels up to 7,200 s the largest airport errors
after LIRF (363.5 s) are LTFM (224.2 s) and LFPG (211.9 s): one has no surface tracks and the most
biased proxy, the other has tracks for at most a quarter of its departures. These remain the open
problems.

# 3 Method

## 3.1 Overview

Figure 5 shows the chain of the best submission, v12075. A feature table (FIN1b) feeds a regime
classifier and a regular-duration candidate; the two combine into a probability-weighted mixture;
surface stages correct the mixture where a track exists, subject to a gate; final controls blend
with two earlier pipelines, enforce the LOBT window, protect 42 frozen tail predictions and apply a
half-weight OPDI correction at LSZH.

```
Challenge data (movements + NM timestamps) --+
METAR, OSM geometry, ATFM aggregates --------+--> Feature table FIN1b (170 tree / 189 neural columns)
                                             |        |                      |
                                             |        v                      v
                                             |  Regime classifier      Regular-duration candidate:
                                             |  LightGBM, 6 classes    CatBoost + LightGBM residual to NM proxy
                                             |                         + 5 anchored LightGBM members
adsb.lol surface tracks ---------------------+------------------------ + ordered-trajectory neural member (0.25)
                                             |        |                      |
                                             |        +----------+-----------+
                                             |                   v
                                             |   Probability-weighted mixture (regular / SCHED / IOBT / LOBT / EOBT / +1 day)
                                             |                   v
                                             +--> Surface stages: own track (E141), previous visit (E64, E188)
                                                                 v
                                                  Gate: keep pre-surface value if a planning regime has p >= 0.95 (E152)
                                                                 v
                                                  Final controls: blend with v44/v60 (E81), LOBT window, frozen 42 tail rows,
                                                  LSZH OPDI correction at 50 %
                                                                 v
                                                             Submission
```

*Figure 5. Stage order of the submitted chain (v12075). Experiment identifiers (E-numbers) are those of the repository's experiment register.*

## 3.2 Features

The final feature table FIN1b has 170 columns for the tree models and 189 for the neural member
(Table 1). Most are built from the challenge data: the planning-time intervals (take-off minus
each NM timestamp, `AOBT_3` minus EOBT_1, a flag for an `AOBT_3` more than 3,606 s from the LOBT);
airport, runway, aircraft and flight identity, including the airline prefix parsed from the
flight number; traffic counts at the airport and on the runway around the flight; neighbour NM taxi
proxies (other departures' take-off minus `AOBT_3` by airport, runway and stand); neighbour features
ordered by filed times (take-off minus LOBT and minus schedule of other departures); stand
turnaround; forward neighbours (departures in the next 15–60 min); planned-arrival residuals; and
runway wake-separation features. External features are METAR weather, stand-to-runway geometry and
taxi-route length on the OpenStreetMap taxiway graph, and the destination airport's daily arrival
ATFM delay.

**Table 1. Feature families of FIN1b and their share of the total split gain in the saved regime classifier and LightGBM regressor.**

| Family | Source | Columns (trees / neural) | Gain share: classifier / regressor |
|---|---|---:|---:|
| Planning-time intervals | Challenge data: movement and NM times | 11 / 11 | 40.4 % / 36.7 % |
| Airport, runway, aircraft, flight identity | Challenge data | 12 / 15 | 45.5 % / 23.2 % |
| Local time | Challenge data | 2 / 2 | 0.2 % / 0.4 % |
| Airport traffic counts | Challenge data: all movements | 28 / 28 | 1.2 % / 3.3 % |
| Runway traffic and configuration | Challenge data | 12 / 12 | 0.4 % / 1.5 % |
| Neighbour NM taxi proxies | Other departures' take-off − `AOBT_3` | 23 / 23 | 3.6 % / 11.7 % |
| Stand turnaround | Arrivals on the same stand | 4 / 4 | 1.0 % / 3.4 % |
| Forward neighbours | Departures in the next 15–60 min | 14 / 14 | 1.0 % / 2.9 % |
| Filed-time neighbours | Other departures' take-off − LOBT or schedule | 27 / 27 | 3.9 % / 8.7 % |
| Planned-arrival residuals | NM planned arrival | 2 / 2 | 0.3 % / 0.7 % |
| Wake separation | Runway sequence and wake category | 7 / 7 | 0.5 % / 1.2 % |
| Weather (METAR) | Iowa Environmental Mesonet | 15 / 15 | 0.6 % / 1.9 % |
| Stand-to-runway geometry | OpenStreetMap | 5 / 5 | 0.6 % / 1.8 % |
| Taxi-route length | OpenStreetMap taxiway graph | 3 / 3 | 0.6 % / 2.2 % |
| Destination arrival ATFM delay | EUROCONTROL portal aggregates | 5 / 5 | 0.2 % / 0.4 % |
| Target encodings (month-out) | 2025 labels | 0 / 16 | — |
| Ordered ADS-B observations | adsb.lol | 0 / 2 streams × 64 tokens × 12 channels | — |
| **Total tabular columns** | | **170 / 189** | **100 % / 100 %** |

All models drop the airport-day context, runway head- and crosswind, `month`, day of year and
absolute timestamps. The tree models also drop stand identity, hourly runway configuration and all
target encodings; each of these removals was tested on the leaderboard and improved it (Section 5.2).
The neural member keeps these three groups; its target encodings exclude the row's own calendar
month. Gain share measures training-loss reduction inside the boosters and is not the RMSE change
from removing a feature; high-cardinality categoricals and correlated columns share credit.

## 3.3 Recording-regime mixture

Let $r(x)$ be the regular-duration prediction for a departure with features $x$. The candidate
vector contains $r(x)$, the four take-off-minus-planning-time intervals (schedule, IOBT, LOBT,
EOBT_1) and $r(x) + 86\,400$ for the one-day shift. A multiclass LightGBM [@ke2017lightgbm]
estimates regime probabilities $p_k(x)$; infeasible or missing candidates are masked and the
remaining probabilities renormalised. Before later corrections the prediction is

$$\hat y(x) = \sum_k p_k(x)\, c_k(x),$$

the probability-weighted mean of the candidates. Under squared-error loss this is the optimal point
prediction when the probabilities and the candidates' conditional means are correct; we do not
claim that the fitted mixture is perfectly calibrated [[RESULT: calibration figures from
docs/uncertainty.md]]. The two-class version of this idea (schedule copy versus regular) removed
135.0 s of leaderboard RMSE on the first day; six classes, weather and geometry removed 27.0 s more
(Section 5.1).

The classifier uses cost weighting when the feasible candidates span more than 3,600 s: the row
weight is (spread / 3,600)², capped at 400; rows with smaller spread keep weight one. This
emphasises regime errors that are expensive under RMSE and is an approximation to minimising the
final mixture's error. Explicit LOBT-window rules (E167) rule out regimes that are impossible given
the LOBT, for example a schedule-copy label more than an hour away from the LOBT.

Several alternatives were measured and rejected: a hard regime choice at a probability threshold
(+29 s at τ = 0.5 in out-of-fold error), per-row choice among the four planning times, a neural
classifier (better log-loss, worse mixture), training the classifier on mixture error, stacking the
regressor's out-of-fold predictions into the classifier, and adversarial-validation sample weights
(+1.77 s on the leaderboard). The repository's negative-results document lists them with effects.

## 3.4 Regular-duration candidate

**Residual to a calibrated NM proxy.** The ordinary regressors do not learn the label directly.
They learn the residual to an NM proxy: take-off minus `AOBT_3` where it exists and lies within
0–7,200 s, else take-off minus LOBT, else minus EOBT_1. The proxy is scaled by a least-squares
coefficient per airport and timestamp source, shrunk towards a global value (E46); the fitted
`AOBT_3` scales are 0.80–1.16 and the LOBT scales 0.42–0.69. An airport median is the fallback when
no proxy exists. The regressor's training target is clipped to 0–7,200 s; longer labels are left to
the regime mixture and the tail model. Learning the residual to the proxy, together with neighbour
features, gave −32.6 s on the leaderboard (v7 against v5).

A consequence that we measured and used throughout is that a residual model absorbs any input
change it can express: on six folds a change in the *level* of the proxy was absorbed by 95–99 %
and a change in its *shape* by 65–68 %. An isotonic calibration made the proxy more accurate (RMSE
378 → 346 s) and the final prediction worse (+0.46 s). A better intermediate quantity is not a
better result.

**Members.** The regular candidate is a fixed-weight combination (Table 2). The core is
0.65 × CatBoost [@prokhorenkova2018catboost] (15 seeds) + 0.35 × LightGBM (5 seeds), both on the
proxy residual. Five further LightGBM members add diversity: one learns the raw target and four
learn it relative to clipped take-off-minus-T anchors for T = LOBT, schedule, EOBT_1 and `AOBT_3`.
Each uses a five-seed average. Their weights were fitted on 2025 out-of-fold predictions and then
scaled to 2/3 after a comparison of complete files on the leaderboard (Section 6.3). For rows
without a LOBT outside LIRF (4,907 ranking rows), a movement-only member built from ADS-B motion
features replaces 7/30 of the regular candidate (E185).

**Table 2. Fixed weights inside the regular-duration candidate of v12075.**

| Component | Weight |
|---|---:|
| Residual blend 0.65 × CatBoost (15 seeds) + 0.35 × LightGBM (5 seeds) | 19/60 |
| Raw-target LightGBM member | 1/30 |
| LOBT-anchored LightGBM member | 0.10 |
| Schedule-anchored LightGBM member | 0.10 |
| EOBT-anchored LightGBM member | 0.10 |
| AOBT-anchored LightGBM member | 0.10 |
| Ordered-trajectory neural member (mean of three seeds) | 0.25 |

**Neural member with ordered trajectories.** The last member is a multilayer perceptron with
piecewise-linear numerical embeddings [@gorishniy2022embeddings] (48-bin training quantiles, 12
embedding dimensions, width 512; `rtdl_num_embeddings` 0.0.12, version-B preset), trained for eight
epochs with AdamW, learning rate 0.002, weight decay 10⁻⁵, OneCycleLR and seeds 0, 1 and 2. It reads
189 tabular columns and, in addition, up to 64 ordered ADS-B observations of the current visit and
of the previous visit of the same aircraft to the airport: time to take-off, position, altitude,
speed, heading and on-ground flag, twelve channels with missingness masks, processed by message
blocks over previous, current and next tokens (E197/E203). It is not a graph of interacting
flights. Each seed has 1,908,437 parameters and took about 269 s to fit on an AMD Radeon RX 7900 XTX.
The member replaced an earlier tabular-only version (E190) at the same weight and gained 0.51 s on
the leaderboard. Its component-level advantage over the previous member (4.5–7.1 s on two
month-pair screens) was much larger than its full-chain effect; against an equal-capacity control
without ordering it won only 0.25 and 0.13 s on the same pairs, so most of the component-level gain
reflects changed inputs and capacity, not ordering alone.

## 3.5 Surface stages from the aircraft's own track

Where the flight's own adsb.lol ground track exists, gradient-boosted stages correct the mixture
output. The extraction, run day by day over the archive (about 3.5–4 GB per day, processed as a
stream), yields the first ground fix, the first movement after the stand at a 1-knot threshold, the
stops and their ends, the path length, distances of track points to the assigned stand, the time of
leaving a 150 m zone around it, and a kinematic estimate of the unseen start of the taxi from the
distance at the first fix and the initial speed. Lowering the movement threshold from 3 kt to 1 kt
reduced the median error against known block times from 302.6 s to 193.5 s on 18,050 departures.
The stored surface files also hold airborne points of the same aircraft within 7 km and below
2,500 ft; these supply *previous-visit* features, the aircraft's earlier arrival at the airport.

Two populations are corrected by disjoint stages. The own-track stage (E141) corrects rows with an
own surface track; the previous-visit stage (E64) corrects rows without one that have a previous
visit. Both are small LightGBM models on the residual of the mixture, trained on all rows including
those where the record contradicts the track, averaged over seeds 42, 1, 2, 3 and 4, with their
target clipped to ±1,800 s. Seven reception-density features are dropped from E141 and three from
E64 because their behaviour did not transfer between 2025 and 2026. A third stage (E188) retrains the
previous-visit correction for LIRF rows that have an NM record, no own track and a previous visit;
its incremental evidence is small and mixed (Section 4.1). A gate (E152) restores the pre-surface
prediction when the maximum probability of the four planning-time classes is at least 0.95, because
physical movement cannot explain a label that is a copy of a planning time. The gate is based on
model confidence, not on knowledge of the true regime.

The first surface stage gave −15.2 s on the leaderboard; stand geometry −3.7 s; previous visits
−2.5 s; the 1-knot threshold −0.6 s. We note that the technique of extracting ground tracks from
adsb.lol had been seen in another team's public code; the implementation is ours. A replacement of
the learned stages by a fixed physical estimate (take-off minus first movement, with a lag fitted on
the other half-year) was measured and rejected: +20.5 s on the body subset and +18.3 s on decided
rows, and even a perfect per-row selector between the learned and the physical estimate would have
gained at most 0.69–0.86 s. Coverage bounds the stages: full ground coverage would be worth about
28 s on the ranking set, but LFPG, LIRF in July and LTFM are almost unobserved in any open archive.

## 3.6 Sparse Rome tail

LIRF departures without an NM match and 4–30 h after schedule have few examples and very different
possible values (minutes or hours; for rows 12–30 h late, 39 % were shifted by one day). Their regime
probabilities come from a TabPFN-2 classifier [@hollmann2025tabpfn] with the airline prefix and the
local date as features, replacing an empirical regime table (−27.0 s) and then improving on it
(−4.7 s; a leave-one-month-out squared error −41.5 %, better in 11 of 12 months). Adding the airline
prefix to the tail model changed only 42 ranking rows and gained 3.60 s on the leaderboard,
although it had failed a "better in at least 8 of 12 months" rule (6 of 12); the rule counted months
with 4–6 rows like months with 39 and was withdrawn. A reviewer found that 65.1 % of the tail
table's holdout gain came from three rows. The tail is where the score is most fragile.

TabPFN inference is not bit-identical across runs and machines: 39 of the 42 rows differed between
an x86 and an Apple-Silicon run. The submitted chain therefore copies these 42 predictions from a
fixed earlier submission and restores them after all later stages. This preserves the output across
builds but makes the donor file an explicit dependency, and it is not a rule for choosing rows in a
new period [[RESULT: tail treatment in the final four-month submission, from docs/generalisation.md]].

## 3.7 Final controls

After the gate, the whole file is blended with two earlier complete pipelines that had lost as
replacements but carried partly independent information: v44 (per-airport classifiers and regressors,
256.34 s on the leaderboard) and v60 (an earlier global chain with ADS-B features inside the base,
252.10 s). Their 2025 cross-validation residuals correlated with the accepted chain at 0.939 and
0.964. Under squared error a small weight on a partner lowers the error when ρ·σ_partner < σ_main.
The weights were fitted by non-negative least squares on 2025 out-of-fold predictions against the
chain of 20 September and fixed since:

$$p_{\mathrm{E81}} = 0.8516\,p_{\mathrm{current}} + 0.1214\,p_{\mathrm{v44}} + 0.0270\,p_{\mathrm{v60}}.$$

Complete out-of-fold counterparts of v44 and v60 were not kept, so both donors enter only as
archived ranking files; this is the main obstacle to a fully nested validation (Section 4.1). An
adjustment near the lower edge of the LOBT window and a clip to the feasibility window follow, then
the 42 protected tail values are restored. The last stage fits a small LightGBM residual model on
OPDI exit-parking and taxiway events at LSZH only (inputs: OPDI duration, its difference from the
prediction, the prediction, the exit-to-taxiway interval) and blends the corrected and uncorrected
files 50:50. The same correction at EDDF, which had no labels for calibration, worsened the
leaderboard by 1.13 s and was dropped; at LSZH it gave −0.13 s against a leave-one-month-out forecast
of −1.2 s.

# 4 Validation

## 4.1 Temporal month-pair cross-validation

The main scheme has six outer folds of 2025: months 1+7, 2+8, 3+9, 4+10, 5+11 and 6+12. Each fold
holds out both months and trains on the other ten, mirroring the January + July ranking set. Where
boosting rounds are selected, the month after each held-out month serves as the inner validation
month and the selected iteration count is used for a refit on the outer training months. Candidate
and control are compared on identical rows by paired squared error. An integrity checker verifies
fold inventory, unique IDs within and across folds, complete coverage of training IDs, targets
against the feature table, held-out months, shapes and finite values; later evaluators also bind
input hashes, seed receipts and stage outputs. These checks establish artifact consistency; they
cannot establish that every upstream model was independently cross-fitted.

Three subsets are reported. *Decided* excludes the LIRF rows without an NM record and 4 h or more
after schedule, which a separate tail model serves (2,084,920 of 2,085,047 rows). *Body* is rows with
observed target ≤ 7,200 s (2,084,448 rows). *Trimmed* removes the three or twenty largest and smallest
paired differences in each fold. Body is defined by the label and is a diagnostic, not a routing
condition (Section 4.3).

A caveat applies to all cross-validation results of the final chain. Some later stages were trained
on historical out-of-fold predictions of earlier ones, the stored classifier probabilities are
already feasibility-masked, and exact out-of-fold donors for the E81 blend are unavailable. The
retrospective comparison of the current members (E195) therefore runs through the gate but **before
the E81 blend, the LOBT adjustments, the tail protection and the LSZH stage**, and is
*selection-informed* validation: the production manifest records
`quality_validated_on_full_chain = false`. Table 3 gives the comparison for the last two accepted
members.

**Table 3. Retrospective six-fold comparison of the chain before the E81 blend (2025 out-of-fold RMSE, seconds).**

| Variant | Decided RMSE | Change | Body RMSE | Change | Pairs improved |
|---|---:|---:|---:|---:|---:|
| Control (previous MLP member, no E188) | 218.610 | — | 183.541 | — | — |
| E190 neural member (PLE) | 218.270 | −0.340 | 183.174 | −0.366 | 6/6 |
| E188 LIRF previous-visit stage | 218.603 | −0.007 | 183.536 | −0.005 | 3/6 |
| E190 + E188 (v12074) | 218.263 | −0.347 | 183.169 | −0.372 | 6/6 |
| E203 ordered-trajectory member (v12075) | 217.717 | −0.546 vs v12074 | 182.555 | −0.614 vs v12074 | 6/6 |

The 2025 out-of-fold RMSE of the chain before the blend is 218.3 s; its leaderboard RMSE is 230.9 s.
The two numbers cover different rows and stages: the out-of-fold figure spans twelve months,
excludes the 126 LIRF tail rows (which hold 111 of the 122 labels above four hours) and stops before
the blend and the LSZH stage; on January and July 2025 alone the out-of-fold RMSE is 279.2 s, and four
labels above four hours hold 38.2 % of that squared error. Inputs also shift between years (ADS-B
coverage; the share of departures without an NM record, 1.53 % against 1.08 %), and the chain was
developed by selecting changes on the same 2025 data, so its out-of-fold error is optimistic. We have
not measured how much of the gap each factor explains.

## 4.2 Pre-registration and gates

Before each experiment the decision rule was written down. Early rules were "−0.30 s and at least
4 of 6 folds"; later rules were "at least 4 of 6, trimmed < 0, body < 0" or "both screen pairs
improve". From 20 September most uploads also carried a leaderboard forecast range recorded before
the upload. In the October experiments the protocol and code were committed before the first model
fit. Verdicts were applied as written; a later change of mind is logged as a new entry and does not
edit the old one. The register holds 376 numbered entries from 23 September to 7 October 2026 and
the working brief holds the earlier experiments; the most informative rejections are summarised in
English in the repository.

Some rules were themselves revised when they failed. The fixed −0.30 s threshold was not derived
from data and was removed; a stage change worth −0.137 s (6 of 6 folds) was uploaded anyway and gave
−0.04 s. The "8 of 12 months" tail rule rejected a change that then gained 3.60 s. If the decided and
trimmed measures differ by more than a factor of two, the trimmed one decides, because on the
21 September chain the ten worst out-of-fold rows held 24.3 % of the squared error and the worst
single row was worth 7.47 s on the leaderboard scale, 25 times the gate.

## 4.3 Transfer from cross-validation to the leaderboard

Most uploads from 20 September on carried a written forecast, which lets us compare the
cross-validation effect of a single change with its leaderboard effect (Table 4). On 4 October this
was calibrated on 38 single-change pairs, 15 of them recomputed from out-of-fold files. The
leaderboard change followed the decided-row cross-validation change (Pearson 0.89, slope 1.06) and
its body part (0.90), but not the y > 7,200 s tail (0.09); the sign agreed in only 60–70 % of pairs.

**Table 4. Cross-validation against leaderboard effect of single changes (seconds; negative is better). CV values before October are on a leaderboard-equivalent scale, comparable in sign and rough size.**

| Change (leaderboard pair) | Kind | CV | Leaderboard |
|---|---|---:|---:|
| ADS-B own-track stage E25 (v27 vs v22) | observation of the flight | −9.53 | −15.22 |
| Stand geometry in the stage E36 (v30 vs v29) | observation of the flight | −2.98 | −3.71 |
| Previous visits E64+E65 (v78 vs v55) | observation of the flight | −2.37 | −2.65 |
| Filed-time neighbours E77 (v97 vs v82) | observation of other flights | −2.73 | −3.26 |
| Proxy scale E46 (v55 vs v30) | calibration, no new capacity | −0.40 | −1.00 |
| Airline prefix E106 (v195 vs v155) | recording rule | −0.58 | −1.28 |
| Classifier cost weights E89 (v115 vs v97) | loss weighting | −1.02 (body +11.27) | −1.18 |
| Ordered trajectories E203 (v12075 vs v12074) | neural member | −0.55 (before E81) | −0.51 |
| `month` removed E24 (v22 vs v18) | capacity removed | −2.18 | −0.87 |
| Target encodings removed E99 (v155 vs v115) | capacity removed | +0.94, 0 of 6 | −0.17 |
| Per-airport models E38 (v43 vs v30) | learned capacity | −10.98 | +3.79 |
| Regime shares by group E50 (v65 vs v30) | 2025 aggregates | −2.79 | +5.78 |
| Flight-number habit at LIRF with NM E133 (v249 vs v247) | 2025 aggregates | −0.72 | +0.36 |
| Anchored members E171–E173 (v808 vs v655) | members at 2/3 weight | about −1.24 | −0.98 |
| Destination ATFM delay E163 (v778 vs v655) | external aggregate | −0.76 | −0.17 |
| Monthly PRU calibration E145 (v448 vs v447) | calibration to external data | better in 6 of 6 | +0.11 |
| Airline prefix in TabPFN tail E118 (v316 vs v247) | 42 tail rows | forecast −0.27 | −3.60 |

The ratio of leaderboard to decided-row cross-validation change by type of change was about 1.0 for
ADS-B stages, about 0.7 for blend members, 0.2–0.3 for new base features, and zero or negative for
rules and calibrations fitted on 2025 (the proxy scale was an exception); tail models and capacity
removals were not scaled by cross-validation at all. Later forecasts used the cross-validation
change times the type ratio. Five rules were derived from these observations.

1. *Observation transfers; training-period aggregates do not.* Changes that read what the flight did
   (ADS-B stages) or what other flights filed (filed-time neighbours) transferred at full size or
   more. Learned capacity on 2025 aggregates won in cross-validation and lost on the leaderboard
   (per-airport models, regime shares by group, ADS-B features inside the base model: CV −1.41,
   leaderboard +0.33).
2. *Capacity-free changes transfer even through a failed gate.* The cost-weighted classifier failed
   the body leg because that subset is selected by the true label: any shift of probability towards
   long regimes looks worse there by construction. It gained 1.18 s on the leaderboard.
3. *Removing 2025 capacity can transfer against cross-validation.* Removing target encodings was
   worse in all six folds and better on the leaderboard; cross-validation on 2025 cannot see the
   2025→2026 shift, so for removals it served only as a cost filter. This is not universal: four
   other removals made the leaderboard worse (+0.40 to +1.45 s).
4. *Measure only rows the base model decides.* Later stages overwrite 10–45 LIRF tail rows per fold,
   which carried up to 62 % of a fold's squared error; with them excluded, the per-airport models'
   cross-validation gain fell from −10.55 to −1.82 s.
5. *Probes for group effects must be split by month.* A runway × configuration interaction showed a
   split-half correlation of +0.763 under a random split and −0.041 under a month split, because
   rows from the same hour fell into both random halves.

Component-level gains also shrank in the full chain: a mixture-density head improved the body RMSE by
0.40 s as a component and worsened the decided RMSE by 0.05 s inside the chain; the wake-separation
features went from −1.47 s on one fold to −0.13 s over six and −0.06 s after the chain changed. The
control must be the production model's out-of-fold predictions: one change looked like −1.33 s
against an older control and was +4.76 s against production predictions.

## 4.4 Predicting blends of scored files without labels

Every leaderboard score is an unweighted RMSE over the same 344,841 rows. For two scored files with
predictions $P_1$, $P_2$, scores $s_1$, $s_2$ and root-mean-square difference
$d = \sqrt{\mathrm{mean}((P_1 - P_2)^2)}$, the errors $e_i = P_i - y$ satisfy
$\mathrm{mean}(e_1 e_2) = (s_1^2 + s_2^2 - d^2)/2$, so the blend $wP_1 + (1-w)P_2$ scores

$$\mathrm{RMSE}^2 = w\,s_1^2 + (1-w)\,s_2^2 - w(1-w)\,d^2 .$$

No labels are needed. Seven blends were priced before upload and scored within 0.0009 s of the
forecast (the residual is the rounding of the uploaded file to whole seconds). The identity served
three purposes: to price a blend before spending a slot; to refuse a blend whose cross-validation
weights promised −9.68 s but which the identity priced 5.20 s worse than the base file; and to
expose a hidden component, when a blend with an unscored partner missed its forecast by 0.66 s and
the identity gave the partner's own leaderboard RMSE as 338.46 s although it had trailed the control
by only 7 % in cross-validation. The rule that followed is to blend only complete, scored
submissions. Leaderboard-optimal blend weights (−1.004 s instead of the −0.83 s achieved for the
first blend) were computed and not used, because fitting weights to the leaderboard is probing.

# 5 Results

## 5.1 Leaderboard progression

108 uploads between 13 September and 7 October 2026 produced 61 new best scores (Figure 6). The 30
steps with a gain of about 0.3 s or more account for 267.5 s of the total 270.3 s improvement.
Table 5 groups them into seven phases; each gain is against the previous best file, and where an
upload bundled changes the repository's development history gives the single-change control.

![Figure 6. Best leaderboard RMSE over time, from 501.23 s (v1, 13 September) to 230.94 s (v12075, 7 October), with labelled milestones.](../docs/figures/fig1_leaderboard_progression.png)

**Table 5. Phases of development and the design changes that moved the leaderboard score.**

| Phase | Dates | Best RMSE (s) | Changes that moved the score (leaderboard gain, s) |
|---|---|---|---|
| 1. Recording-regime mixture | 13 Sep | 501.2 → 339.2 | Two-regime mixture (−135.0); six regimes and METAR (−12.0); OSM geometry, hourly runway configuration, target encoding (−10.1); CatBoost + LightGBM (−4.8) |
| 2. NM residuals and the Rome tail | 14–15 Sep | 339.2 → 273.4 | Residual to the NM proxy with neighbour features (−32.6); regime table for LIRF without NM (−27.0), then TabPFN-2 (−4.7); `month` removed (−0.9) |
| 3. Own-aircraft ADS-B stages | 15–19 Sep | 273.4 → 247.8 | First surface stage (−15.2); extended features (−2.0); stand geometry (−3.7); per-airport proxy scale (−1.0); previous visit (−2.5); 1-knot threshold (−0.6) |
| 4. Filed-time neighbours and blends | 20–23 Sep | 247.8 → 242.3 | Filed-time neighbours (−2.4); blends with v44 and v60 (−0.8, −0.7); cost-weighted classifier (−0.5) |
| 5. Removing 2025-specific capacity | 24–27 Sep | 242.3 → 235.5 | Airline prefix for rows without NM (−0.4); stand identity and residual encodings removed (−0.7); airline prefix in the TabPFN tail (−3.6); CatBoost seed averaging (−0.8) |
| 6. Stage refinements and members | 28 Sep – 4 Oct | 235.5 → 232.6 | Anchored members (−1.0); MLP member (−0.4); planning-regime gate (−0.2); final feature table; several steps of 0.01–0.2 |
| 7. Neural members with trajectories | 5–7 Oct | 232.6 → 230.9 | Movement-only member (−0.26); LSZH OPDI correction at half weight (−0.25 in two steps); piecewise-linear embeddings (−0.54); ordered ADS-B observations (−0.51) |

The largest gains came from describing the recording process (−135.0 s, then −27.0 s for the Rome
tail) and from a physical sensor (−15.2 s for the first surface stage), not from learner tuning. After
26 September progress slowed: from 236.32 s to 230.94 s over 54 uploads. On 26 September the team was
8th of 192 and 3.14 s behind third place; on 8 October it was 14th of 237 and 15.32 s behind third
place (215.62 s). Leading teams improved much more in the same period. [[RESULT: final Jan/Jul rank at
the close of the ranking on 14 October 2026.]]

## 5.2 Single-change controls (ablations)

Because each upload was a complete file scored on the same rows, the leaderboard provides
single-change controls that are not available from cross-validation alone. Table 6 collects the ones
that define the submitted chain.

**Table 6. Single-change leaderboard controls (RMSE change in seconds; negative is better).**

| Component | Control pair | Effect |
|---|---|---:|
| Two-regime mixture instead of one regressor | v2 vs v1 | −135.01 |
| Residual to NM proxy + neighbour features | v7 vs v5 | −32.6 |
| Empirical regime table for LIRF without NM (42 rows) | v8 vs v7 | −27.0 |
| TabPFN-2 instead of the table (same rows) | v13 vs v8 | −4.70 |
| First ADS-B surface stage | v27 vs v22 | −15.22 |
| Stand geometry in the stage | v30 vs v29 | −3.71 |
| Per-airport × source proxy scale | v55 vs v30 | −1.00 |
| Previous-visit features | v78 vs v55 | −2.65 |
| 1-kt movement threshold | v82 vs v80 | −0.58 |
| Filed-time neighbour features | v97 vs v82 | −3.26 |
| Cost-weighted regime classifier | v115 vs v97 | −1.18 |
| 2025 target encodings removed from the regressor | v155 vs v115 | −0.17 |
| Airline prefix for rows without NM | v195 vs v155 | −1.28 |
| Stand identity + residual encodings removed | v225 vs v215 | −1.06 |
| Runway-configuration features removed | v245 vs v225 | −0.20 |
| Airline prefix in the TabPFN tail (42 rows) | v316 vs v247 | −3.60 |
| CatBoost five seeds | v285 vs v265 | −0.51 |
| Anchored members at 2/3 weight | v808 vs v655 | −0.98 |
| MLP member at 0.25 | v899 vs v859 | −0.78 |
| E81 blend and LOBT edge lift (dropping both) | v314 vs v316 | +0.69 |
| Movement-only member for rows without LOBT | v12008 vs v1139 | −0.26 |
| LSZH OPDI correction, full weight | v12060 vs v12048 | −0.14 |
| Piecewise-linear embeddings replacing the MLP | v12070 vs v12061 | −0.54 |
| Ordered-trajectory member replacing E190 | v12075 vs v12074 | −0.51 |

Changes that were tested the same way and made the leaderboard worse are part of the record: per-airport
models (+3.79), regime shares by group (+5.78), ADS-B features inside the base model (+0.33),
adversarial-validation weights (+1.77), the OPDI correction at EDDF (+1.13), a seed ensemble of the
TabPFN tail (+0.86), monthly calibration to published PRU taxi-out times (+0.11), removal of the
`ap_runway` feature (+0.43) or of stand identity while keeping the encodings (+1.45).

## 5.3 Out-of-fold error by airport

Table 7 gives the 2025 out-of-fold RMSE by airport for the chain before the E81 blend (E195 `joint`
arm, 2,084,921 rows; the 126 LIRF tail rows are excluded). LIRF, LFPG and LTFM have the largest
errors on labels up to 7,200 s. At LSZH six labels above 7,200 s hold 85 % of the airport's squared
error, which explains the gap between its two columns.

**Table 7. 2025 out-of-fold RMSE by airport, chain before the E81 blend.**

| Airport | Evaluated rows | Share of squared error | RMSE, all rows (s) | RMSE, labels ≤ 7,200 s (s) |
|---|---:|---:|---:|---:|
| EDDF | 230,141 | 2.5 % | 103.9 | 103.9 |
| EDDM | 167,334 | 1.4 % | 92.7 | 92.7 |
| EGLL | 239,546 | 9.3 % | 196.0 | 188.6 |
| EHAM | 247,951 | 2.6 % | 101.7 | 98.5 |
| LEBL | 179,705 | 4.0 % | 149.0 | 149.0 |
| LEMD | 212,242 | 4.1 % | 138.3 | 138.3 |
| LFPG | 239,552 | 21.6 % | 298.9 | 211.9 |
| LIRF | 160,578 | 30.8 % | 436.4 | 363.5 |
| LSZH | 134,907 | 9.5 % | 263.8 | 102.1 |
| LTFM | 272,965 | 14.3 % | 228.2 | 224.2 |
| **All** | **2,084,921** | **100 %** | **218.26** | **183.17** |

## 5.4 Final four-month ranking

[[RESULT: RMSE and rank of `gentle-igloo_final.parquet` on January+July, February+June and all four
months, from the organiser's final ranking; recipe differences from v12075 (treatment of the E81
donors and the 42 frozen tail rows, TabPFN seeds), from `docs/generalisation.md`.]]

[[RESULT: February/June generalisation analysis: per-month and per-airport error, ADS-B coverage of
the two new months, share of departures without an NM record, from `docs/generalisation.md`.]]

# 6 Discussion

## 6.1 What generalises

The evidence of Section 4.3 separates the chain into parts with different expected transfer.

*Expected to transfer to other periods at these airports:* the recording-regime mixture as a
structure, the residual-to-proxy formulation with per-airport calibration, the coverage-gated
surface stages, and the filed-time neighbour features. Each reads an observation of the specific
flight or of flights around it, and each transferred from 2025 cross-validation to the 2026
leaderboard at full size or more.

*Dataset-specific:* the regime labelling rules (±6 s tolerance; no planned regimes at the three
second-resolution airports), the list of airports whose recording substitutes the planning time, the
LOBT feasibility window of 3,606 s, and the per-airport proxy scales. These must be re-estimated for
another source of movement records or another set of airports.

*Period-specific and not transferable as they stand:* the 42 frozen TabPFN predictions and the two
archived blend donors, which exist only for January and July 2026. [[RESULT: how the final
four-month submission treated them and what that cost, from `docs/generalisation.md`.]]

*Not expected to transfer:* learned capacity on 2025 aggregates (per-airport models, target
encodings, stand identity, runway configuration, regime shares by group, flight-number habits). All
of these won in 2025 cross-validation and lost on the 2026 leaderboard, and removing them improved it.

A separate question is transfer to other airports. The chain has not been validated on an unseen
airport; the recording-regime phenomenon is an empirical property of the supplied records and may
be absent, or take a different form, elsewhere. [[RESULT: any cross-airport leave-one-out evidence
from `docs/generalisation.md`.]]

## 6.2 A data-quality view

The regime classifier is, in effect, a detector of substituted off-block records. Its probabilities
flag rows where the recorded block time is unlikely to be an observation, and its per-airport
pattern (Figure 1) identifies where this happens. For a performance-monitoring unit the detector is
arguably more useful than the point prediction: additional taxi-out time computed from a block time
that is a copy of the schedule measures the schedule, not the queue. [[RESULT: detector precision and
recall against the regime labels, per airport and month, and the PRU-indicator sensitivity analysis,
from `docs/data-quality-regimes.md` and `docs/pru-view.md`.]]

The ADS-B stages suggest a second data-quality use. Where the aircraft's first movement is observed,
the difference between the recorded off-block time and the observed movement onset is an independent
check on the record. We do not claim that the track is correct where the two disagree: a first
received message is not automatically an off-block event, and the recorded interval is what the
indicator is defined on.

## 6.3 Leaderboard use disclosure

The ranking months were used to choose between complete, pre-built variants. Nothing in the chain
was fitted to individual ranking rows, no blend weight was optimised on the leaderboard, and
leaderboard-optimal weights that were computed were rejected. Table 8 lists every design choice in
v12075 that was decided or kept because of a leaderboard result rather than cross-validation. The
final score therefore includes selection effects on the January and July 2026 rows, and the blind
four-month ranking is the first measurement free of them for February and June
[[RESULT: compare]].

**Table 8. Design choices in v12075 that depended on a leaderboard result.**

| # | Choice | How it was chosen |
|---:|---|---|
| 1 | 42 LIRF tail predictions frozen from one TabPFN draw and protected from later stages | The airline-prefix tail variant failed its leave-one-month-out rule (6 of 12) but gained −3.60 s on the leaderboard; a seed ensemble was then rejected on the leaderboard (+0.86 s) |
| 2 | Whole-file blend with v44 and v60 | Weights from non-negative least squares on 2025 out-of-fold predictions, not from the leaderboard; keeping the blend was confirmed by a leaderboard comparison (+0.69 s without it and the LOBT lift) |
| 3 | LightGBM anchored members at 2/3 of their CV weights; neural member at full weight | Two leaderboard comparisons of complete files (×1 worse by 0.18 s; ×2/3 for the neural member worse by 0.10 s) |
| 4 | Movement-only member at 0.35 × 2/3 | CV weight scaled by the same 2/3 rule |
| 5 | Removal of `month`, 2025 target encodings, stand identity, runway configuration and the flight-number rule | Each removal tested with one upload and a forecast written beforehand; four other removals made the leaderboard worse and were not kept |
| 6 | Cost-weighted regime classifier | Failed the body leg of its gate, uploaded as a test with a forecast, kept after −1.18 s |
| 7 | LSZH OPDI correction at 50 % | Weight fixed before scoring as a cautious transfer; the same correction at EDDF was rejected on the leaderboard (+1.13 s) |
| 8 | Uploads without a full-chain CV (the E190 member and five fixed files on 6 October; E203 after a pre-blend comparison) | Owner's decision under time pressure; the best of the scored files became the current chain |

## 6.4 Limitations

- The model reconstructs a *recorded* off-block time after the fact. It uses the take-off time and
  the flight's own post-pushback track and is not a pre-departure forecast; it has not been
  validated for tactical use, unseen airports or a changed NM schema.
- There is no fully nested cross-validation of the exact final chain. Later stages were trained on
  historical out-of-fold predictions of earlier ones; exact out-of-fold donors for the blend do not
  exist. The retrospective comparison stops before the blend and is selection-informed.
- A few extreme labels dominate RMSE. Eleven labels above four hours hold 25.4 % of the 2025
  out-of-fold squared error; one wrong tail row can outweigh a gate many times over. Small changes
  in how such rows are hedged move the score more than a model improvement, and the 42-row tail
  rests on a single non-deterministic TabPFN draw.
- Results depend on ADS-B ground coverage, which differs by airport and year and is near zero at
  LTFM and low at LFPG. Removing reception-density features does not remove all sensitivity to
  changes in the receiver network.
- The out-of-fold RMSE (218.3 s) and the leaderboard RMSE (230.9 s) are not directly comparable, and
  we have not decomposed the gap.
- We do not claim an irreducible error floor. Proxy residual variance, correlations with block time
  and mixture uncertainty are diagnostics, not independent error components.
- [[RESULT: uncertainty: calibration of the regime probabilities and of any predictive interval, from
  `docs/uncertainty.md`.]]

# 7 Open science

**Code.** The repository is licensed under the GNU GPLv3 [@kolomiiets2026repo]. Most of the roughly
560 files in `src/` and `scripts/` are individual experiments kept for traceability; their identifiers
match the experiment register. 386 unit tests of the build, upload and validation helpers run in
about 5 s without any data.

**Data.** Table 9 lists each input with its terms. The code licence, a data licence and the
organiser's permission to use a source in the challenge are three different things, and we record
them separately. We do not redistribute the challenge data, the EUROCONTROL portal files or the OPDI
extracts; we give acquisition instructions instead. If either EUROCONTROL-portal input were set
aside, the affected parts are the destination ATFM-delay features (leaderboard effect when added:
−0.17 s) and the half-weight LSZH correction (−0.25 s over two steps); the rest of the chain does not
use them.

**Table 9. Data inputs of v12075 and their terms.**

| Input | Licence or terms | Status for the challenge |
|---|---|---|
| Challenge movements and NM records | Provided by the organiser to all teams | Challenge data |
| adsb.lol `globe_history` 2025 and 2026 | ODbL 1.0 | Open licence; accepted by the organiser on 17 Sep 2026 |
| OpenStreetMap (via Overpass) | ODbL 1.0, © OpenStreetMap contributors | Open licence |
| Iowa Environmental Mesonet METAR archive | Public domain (IEM statement) | Open |
| ansperformance.eu daily arrival ATFM delay by airport | Portal terms: free with attribution, non-commercial | Accepted as open data for the challenge on 17 Sep 2026 |
| OPDI v0.0.2 events (LSZH correction only) | No separate dataset licence published | Accepted as open by the organiser on 4 Oct 2026 |
| TabPFN-2 classifier weights | Prior Labs License 1.1 (Apache 2.0 plus attribution) | Used for 42 tail predictions; weights not redistributed |

**Reproducibility levels.** We distinguish four levels and state which was achieved.

1. *Verify the submitted file.* SHA-256 `90d3cd0b…aee2`, 344,841 unique IDs, schema matching the
   organiser's template. Achieved; a script is provided.
2. *Replay the best submission from archived components.* Done independently on 7 October 2026: all
   344,841 predictions and the Parquet bytes identical, no model refits, 101 s and 7.9 GiB peak
   memory. The replay recomputes the classifier probabilities, feasibility masks, mixture and every
   tail stage from frozen code, and reuses the saved models and member predictions after checking
   them against the receipts written at training time. It shows that the recipe runs as recorded,
   not how accurate it is. The archive is about 5.2 GB in 3,711 hashed files; part of it (challenge
   data, OPDI extracts, tables that embed challenge labels) cannot be redistributed, so the complete
   archive is available to the organisers on request.
3. *Retrain everything from raw data.* The trainers, extractors and stage builders exist and are
   listed in order in the reproduction guide with measured or estimated wall-clock where known (for
   example about 1 h 45 min per CatBoost seed on six cores; about 269 s per neural seed on the GPU).
   A clean end-to-end run of the current recipe has **not** been completed.
4. *Apply to a new period.* Not available as one command: the 42 frozen tail predictions and the two
   archived blend donors are specific to the ranking period. [[RESULT: final-submission pipeline
   status for the four-month file.]]

**Environment.** The CPU stack (Python ≥ 3.12; Polars 1.44.2, LightGBM 4.7.0, CatBoost 1.2.10,
scikit-learn 1.9.1, TabPFN 8.5.0) is locked in `uv.lock`. The neural member was trained with
PyTorch 2.13.0 + ROCm and `rtdl_num_embeddings` 0.0.12, pinned separately. Replaying the archived
submission needs neither a GPU nor PyTorch.

**Provenance.** Every modelling decision, gate and upload is recorded with its evidence in the
experiment register; organiser statements on data use and evaluation are quoted verbatim with UTC
timestamps. Where an idea came from elsewhere we say so: the first ADS-B surface stage followed a
technique seen in another team's public code (our implementation); per-airport scaling of the NM
proxy was also reported by another team; the numerical embeddings follow Gorishniy et al.
[@gorishniy2022embeddings] and the tail classifier is TabPFN-2 [@hollmann2025tabpfn].

# 8 Conclusion

The taxi-out label of the PRC Data Challenge 2026 is a recorded interval that follows several
recording regimes, and the regimes decide the metric: a probability mixture over how the off-block
time was recorded was worth 135 s of RMSE on the first day, more than every later change combined.
The remaining gains came from anchoring the regular taxi to a calibrated NM proxy, from the aircraft's
own ADS-B surface track where it exists, and from a small foundation model on a 42-row tail. What
transferred from 2025 cross-validation to the 2026 leaderboard was what read an observation of the
flight; what did not transfer was learned capacity on 2025 aggregates, and removing it helped. We
have tried to state the evidence boundary of each result: the best file is replayed byte-exactly,
but the chain has no fully nested cross-validation, the ranking months informed design choices that
we list, and the tail rests on a non-deterministic draw. [[RESULT: one sentence on the blind
four-month result.]] For a performance-monitoring reader, the most useful by-product may not be the
point prediction but the detector of substituted records, which identifies where additional
taxi-out time would measure the schedule rather than the queue.

# Acknowledgements

EUROCONTROL Performance Review Commission Data Challenge 2026, supported by the OpenSky Network
[@schafer2014opensky]; adsb.lol and its contributors (ODbL 1.0); © OpenStreetMap contributors (ODbL
1.0); Iowa Environmental Mesonet, Iowa State University; EUROCONTROL Performance Review Unit and the
Open Performance Data Initiative. TabPFN-2 weights by Prior Labs ("Built with PriorLabs-TabPFN").
The team is one developer working with AI coding assistants; all modelling decisions, gates and
uploads are the author's and are recorded with their evidence in the repository.
[[RESULT: the author confirms the AI-assistance disclosure wording against the journal's policy at
submission time.]]

# Funding statement

No external funding was received. [[RESULT: author to confirm.]]

# Open data statement

The challenge data (2025 training and 2026 ranking movements and NM records) are distributed by the
organiser to registered teams and are not redistributed here. All other inputs are open or public:
adsb.lol `globe_history` 2025 and 2026 (ODbL 1.0), OpenStreetMap (ODbL 1.0), Iowa Environmental
Mesonet METAR (public domain), EUROCONTROL Aviation Intelligence Portal daily airport files (portal
terms; accepted for the challenge) and OPDI v0.0.2 (accepted for the challenge). Acquisition scripts
and the dated terms of each source are in `SOURCES.md` of the repository. Derived tables that embed
challenge rows or labels are not published.

# Reproducibility statement

Source code is at https://github.com/disclaimer8/prc-2026 under GPLv3 [@kolomiiets2026repo]. The
submitted file is verifiable by hash; its exact replay from archived components (about 5.2 GB, 3,711
hashed files) has been run independently and is available to the organisers on request; retraining
from raw data is documented step by step but has not been run end to end; application to a new
period is not a single command. `docs/reproduction.md` states these levels and the environment
(`uv.lock`, `requirements-gpu.txt`). [[RESULT: public archive location (e.g. Zenodo DOI) for the
redistributable part of the component archive, if created.]]

# AI-assistance statement

[[RESULT: Author to finalise. Draft: The author used AI coding assistants for code drafting, log
keeping and documentation drafting during the challenge. All experiments, decisions and the final
text were reviewed by the author, who takes responsibility for the content.]]

# References

::: {#refs}
:::

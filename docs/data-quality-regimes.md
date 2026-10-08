# Recording regimes as a data-quality signal for airport taxi-out statistics

Some departure records in the challenge data carry a block time that is not a physical
off-block but a copy of a planning timestamp (schedule, IOBT, LOBT, EOBT_1) or a time dated
one day early. The regime classifier inside our model (`src/train.py`, described in
[model design](model-design.md#recording-regime-mixture)) estimates, per departure, the
probability that a record is such a copy. This page turns that classifier into a
data-quality tool: how well it detects the copies, how many there are per airport and month,
and what they do to monthly taxi-out statistics and to a PRU-style "additional taxi-out time"
indicator computed from the same records.

Everything here is computed by [`scripts/regime_quality_report.py`](../scripts/regime_quality_report.py)
from the production out-of-fold predictions of 2025 (`data/cv_preds_fin1b_hz`, six
month-pair folds, 2,085,047 departures; file hashes in
[`regime_quality_run.json`](tables/regime_quality_run.json)). No model is trained by the
script. Numbers in the text are rounded from the tables it writes.

```
POLARS_MAX_THREADS=2 .venv/bin/python scripts/regime_quality_report.py --data-root data
```

## 1. What is being detected, and against which reference

- **Label (the +-6 s rule).** `regime_label` in `src/train.py`: a record is a planning copy
  when take-off minus the recorded block time equals take-off minus SCHED, IOBT, LOBT or
  EOBT_1 within +-6 s (first match wins, in that order), or when the label lies between one
  day and one day plus two hours (one-day shift). EDDF, EHAM and LSZH record block time to
  the second and show no plateau at the planning times, so planned regimes are not labelled
  there (only the one-day shift is; it occurs once, at LSZH). In 2025 the rule marks 140,993
  of 2,085,047 departures (6.76 %); the share is 24.1 % at LIRF and 6.9-9.3 % at the other six
  labelled airports.
- **Detector.** `P(planning copy) = 1 - P(regular)` from the six-class classifier, out of fold
  (each month predicted by a model that never saw it). Class order: regular, sched, iobt,
  lobt, eobt, dayshift.
- **The label is a rule, not ground truth.** An on-time departure whose physical off-block
  happens to fall in the scheduled minute is labelled a copy although nothing was copied.
  [Data insights](data-insights.md#2-recording-regimes) shows that outside LIRF the
  scheduled minute holds 4.1-5.6 % of block times against 3.5-5.3 % in the neighbouring
  minutes, i.e. most non-LIRF labels are coincidences of this kind; at LIRF the scheduled
  minute holds 18.1 % against 1.3-3.7 % next to it. All precision and recall figures below
  are measured against this rule and inherit its ambiguity.

## 2. Calibration and detection power

![Reliability diagrams of the detector, one panel for the seven labelled airports together and one per airport; the curve follows the diagonal up to P about 0.4 at every airport and up to 1.0 at LIRF, the only airport where probabilities above 0.5 occur in volume.](figures/fig6_regime_detector_calibration.png)

Tables: [`regime_detector_summary.csv`](tables/regime_detector_summary.csv),
[`regime_detector_calibration.csv`](tables/regime_detector_calibration.csv),
[`regime_detector_thresholds.csv`](tables/regime_detector_thresholds.csv),
[`regime_detector_class_agreement.csv`](tables/regime_detector_class_agreement.csv).

**Counts are calibrated.** Over the seven labelled airports the mean probability is 9.46 %
against a label share of 9.58 %; the expected number of copies (sum of probabilities) is
139,319 against 140,992 labels (-1.2 %). Per airport the two agree within 0.1 percentage
point (EDDM 7.76 % vs 7.84 %, EGLL 8.39 % vs 8.48 %, LEBL 9.15 % vs 9.26 %, LEMD 7.60 % vs
7.69 %, LFPG 6.93 % vs 7.04 %, LIRF 23.91 % vs 24.14 %, LTFM 6.83 % vs 6.93 %). The
reliability curve follows the diagonal across the whole range: in the bin [0.10, 0.20) the
mean probability is 0.143 and the label rate 0.147; in [0.50, 0.60) 0.548 vs 0.547; in
[0.90, 0.95) 0.929 vs 0.949; in [0.99, 1] 0.994 vs 0.999. At the three airports without
planned-regime labels the detector is quiet: 7.6, 8.9 and 5.1 expected copies in 230,141,
247,951 and 134,907 departures. Brier score 0.075 (labelled airports), 0.053 (all).

**Individual flags are confident only at LIRF.** AUC against the rule is 0.795 over the
labelled airports (0.74-0.81 per airport). Precision and recall of the flag `P >= t`:

| threshold | flagged | precision | recall (all labelled) | recall at LIRF |
|---|---:|---:|---:|---:|
| 0.50 | 17,424 | 71.1 % | 8.8 % | 31.9 % |
| 0.80 | 4,768 | 92.8 % | 3.1 % | 11.4 % |
| 0.95 | 1,871 | 99.2 % | 1.3 % | 4.8 % |

17,361 of the 17,424 flags at 0.5 are at LIRF; the other six labelled airports contribute
0-22 flags each. The detector does not try to call the coincidental copies: the classifier
is trained to minimise the squared error of the mixture prediction, and a coincidental copy
costs nothing (its candidate equals the physical taxi time), so the probability stays near
the base rate there. Where the copy is costly, a long delay recorded as a short taxi, the
probability rises, and at 0.95 the flag is 99.2 % precise. Among flagged labels the most
probable non-regular class matches the label's class in 99.3 % of schedule copies (11,699 of
11,785 at 0.5), 50 % of IOBT copies (156 of 309) and 13 % of EOBT_1 copies (37 of 279); the 12
flagged one-day shifts are all assigned to another class.

The two uses are therefore different: **the expected count is a usable monthly statistic
at every labelled airport; the per-flight flag is usable at LIRF, and only above about 0.8
elsewhere, where it almost never fires.**

## 3. Share of planning-copy records by airport and month

![Top row: monthly share of planning-copy departures per airport, label rule and detector; bottom row: effect of those records on the monthly PRU-style additional taxi-out time, per airport with its own scale.](figures/fig7_regime_effect_on_kpi.png)

Table: [`regime_effect_by_airport_month.csv`](tables/regime_effect_by_airport_month.csv)
(one row per airport and UTC month: counts, label share, expected share, mean and median
taxi-out under three treatments, the PRU-style indicator under three treatments, and the
published PRU values for the same month). Airport-year totals:
[`regime_effect_by_airport.csv`](tables/regime_effect_by_airport.csv).

The label share and the expected share move together through the year at every airport,
with two exceptions worth PRU's attention:

- **LEBL, July-September 2025.** The label share rises from 8-9 % to 11.9 % (July), 12.3 %
  (August) and 10.4 % (September) while the detector stays at 8-9 %. The extra labels are
  long: 226 labelled rows with a recorded taxi-out above 30 minutes in July against 20
  expected by the detector (August 142 vs 17, September 135 vs 14; January-June 5-47 vs
  1-8). This looks like a change in how LEBL records delayed departures in summer 2025, not
  like coincidence; the detector, trained on the other months, misses it.
- **LTFM, February 2025.** 115 labelled rows above 30 minutes against 33-100 in other
  months; here the detector agrees (73 expected against 13-48 in other months).

## 4. Effect on monthly taxi-out statistics and on a PRU-style indicator

### 4.1 How the indicator is emulated

PRU's additional taxi-out time (ATXOT indicator documentation, ed. 01.00, March 2023,
linked from [ansperformance.eu/methodology/additional-taxi-out-time](https://ansperformance.eu/methodology/additional-taxi-out-time/);
our reading in [`analysis/eurocontrol_methodology_2026-09-30.md`](../analysis/eurocontrol_methodology_2026-09-30.md))
is taxi-out (ATOT - AOBT) minus an unimpeded reference, where the reference is the 10th
percentile of taxi-out per stand x departure-runway group over 12 months, valid when at
least 10 flights of the group lie at or below it; flights without stand, runway or AOBT,
taxi-out above 120 minutes, helicopters and de-icing after AOBT are excluded; the airport
indicator is the mean over flights with a valid reference. The script follows this with the
following deviations, all forced by what the challenge data contain:

| PRU | Here | Why |
|---|---|---|
| rolling 12 months ending with the analysed month | calendar year 2025 for every month | only 2025 is labelled |
| month by local take-off time | UTC month of take-off | no change of substance |
| de-icing after AOBT excluded (APDF flag) | not excluded | flag not in the challenge data |
| helicopters excluded | 41 common ICAO helicopter designators excluded | no helicopter flag |
| percentile method unspecified | inverted empirical CDF, no interpolation | |
| negative taxi-out not mentioned | excluded (369 rows) | impossible values |

Filters remove 3,289 of 2,085,047 departures (0.16 %); 1,928,605 (92.5 %) have a valid
reference; per airport 82 % (LFPG, 2,380 groups) to 99 % (EGLL)
([`pru_reference_groups.csv`](tables/pru_reference_groups.csv)).

**The emulation matches PRU's published 2025 indicator where it can.** Against
`taxi_out_additional_time_2025.csv` from ansperformance.eu, April-October 2025 (70
airport-months): mean absolute difference 3.3 s, median 1.8 s, maximum 14.5 s, correlation
0.999; monthly flight counts differ by at most 160. November-March the difference grows to
25.7 s on average with a bias of +23.5 s and a maximum of 154 s (EDDM, December), which is the
de-icing exclusion we cannot apply: PRU removes 20-31 % of EDDM's winter departures. At LIRF,
where de-icing is rare, the 12 months agree within 11.6 s (bias -4.0 s). **This also shows
that the published indicator is computed from the same records, planning copies included.**

### 4.2 Three treatments

For each airport and month the indicator is computed three ways; the reference is
recomputed under each treatment.

- **As recorded (A):** all records, as PRU would see them.
- **Rule-cleaned (B):** records labelled by the +-6 s rule removed. Removes every copy the
  rule can see, including coincidences, which have ordinary taxi times and should move the
  indicator little.
- **Probability-cleaned (C):** every record weighted by P(regular). Removes only what the
  detector sees. Within a probability bin the true copies are the records with the longer
  recorded taxi-out, which the detector cannot know from the features, so C removes less of
  the inflation than B. **A - C is a lower bound on the effect of copies, A - B is the full
  rule-based effect including the neutral coincidences.**

Airport-year effects, seconds per flight (positive = the as-recorded value is higher):

| airport | copies (rule / detector) | mean taxi-out, raw (A - B) | mean taxi-out, PRU filters (A - B) | indicator A | A - B | A - C | largest monthly A - B |
|---|---:|---:|---:|---:|---:|---:|---:|
| EDDM | 7.8 % / 7.8 % | -2.3 | -2.3 | 232.0 | -1.7 | -4.1 | +0.1 |
| EGLL | 8.5 % / 8.4 % | -3.1 | -3.1 | 388.5 | -2.8 | -5.3 | -1.1 |
| LEBL | 9.3 % / 9.2 % | +4.4 | +4.4 | 230.9 | +7.3 | -2.8 | +26.3 (July) |
| LEMD | 7.7 % / 7.6 % | +2.0 | +2.0 | 217.6 | +0.8 | -2.3 | +2.1 |
| LFPG | 7.0 % / 6.9 % | -1.9 | -1.8 | 271.9 | -2.0 | -4.4 | +0.7 |
| LIRF | 24.1 % / 23.9 % | +89.1 | +40.0 | 400.6 | +42.1 | +6.2 | +66.7 (July) |
| LTFM | 6.9 % / 6.8 % | -1.2 | -1.2 | 294.2 | +1.1 | -1.3 | +15.5 (February) |

At EDDF, EHAM and LSZH the rule changes nothing by construction; the probability treatment
moves the indicator by 0.2-0.7 s, which is the noise floor of re-estimating a 10th percentile
on minute-resolution data.

**Outside LIRF the copies do not inflate the statistics.** At EDDM, EGLL, LFPG, LEMD and
LTFM the annual effect on the indicator is between -2.8 and +1.1 s per flight by the rule
and between -5.3 and -1.3 s by probability (negative: the labelled records have slightly
shorter taxi-out than the rest, as on-time departures do). The exceptions are the two
episodes of section 3: LEBL July-September (+26, +19, +15 s per flight by the rule) and LTFM
February (+15 s by the rule, +20 s by probability).

**At LIRF the copies are the dominant data-quality issue.** The raw monthly mean taxi-out
is 55-175 s higher with the copies than without; after PRU's 120-minute filter still
23-63 s; the median 3-45 s. The PRU-style indicator is 25-67 s per flight higher by the
rule, with the largest effects in June-August (58-67 s), and 1.5-13 s higher by the
probability lower bound. For an indicator whose annual value at LIRF is 401 s, the rule-based
effect is about 10 %. The reference itself is barely affected (mean absolute difference
between the as-recorded and the rule-cleaned reference 7.2 s at LIRF, below 5 s elsewhere);
removing the copies with the as-recorded reference kept gives +45.4 s instead of +42.1 s.

## 5. Per-flight table

`docs/tables/regime_probabilities_2025.parquet` (17.7 MB, zstd; not stored in Git, regenerated
in about 7 s by `scripts/regime_quality_report.py --data-root data`): one row per 2025 departure with `MVT_ID_mvt` (int64), `airport`, `fold` (the two
held-out months the row was predicted in), `p_planning_copy` and the six class
probabilities `p_regular, p_sched, p_iobt, p_lobt, p_eobt, p_dayshift` (float32, four
decimals). No other challenge column is included; the table can be joined to the challenge
data on `MVT_ID_mvt`. 17,431 rows have `p_planning_copy >= 0.5` and 1,873 have `>= 0.95`.

**2026.** The final model fits the same classifier on all of 2025 and applies it to the
2026 submission months; the submission builder (`src/blend.py`) saves `mvt_id` and `proba`
in `submissions/parts_vN.npz`. The same table for 2026 is then

```
.venv/bin/python scripts/regime_quality_report.py export \
    --parts submissions/parts_v<N>.npz \
    --mvt data/prc-2026-datasets/final_ranking.parquet \
    --out docs/tables/regime_probabilities_2026.parquet
```

(tested on the January/July 2026 parts of an earlier submission: 344,841 rows, 3.0 MB,
mean P 6.9 %). 2026 has no labels, so the calibration of section 2 cannot be re-checked
there; the counts should be read with the LEBL episode in mind, which shows that the detector
can miss a change in recording practice it has not seen.

## 6. Limitations

- The +-6 s rule is the only reference; it has no ground truth behind it, labels
  coincidences as copies, and cannot see copies at EDDF, EHAM and LSZH, where block time
  is recorded to the second. Precision, recall and the "rule-cleaned" effect are statements
  about the rule.
- The detector is a by-product of a regression model: it is calibrated for counting and for
  confident flags, not for recalling coincidental copies, and it reacts to recording changes
  only after they appear in training data (LEBL, summer 2025).
- The PRU emulation uses the calendar year as reference window and cannot apply the
  de-icing exclusion; its November-March values differ from PRU's published figures for that
  reason, by up to 154 s at EDDM.
- Months are UTC months; the challenge data start on 1 January 2025, so no earlier reference
  window exists.

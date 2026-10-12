# Predictions in PRU terms: unimpeded reference plus additional time

PRU reports taxi-out performance as *additional taxi-out time*: actual taxi-out minus an
unimpeded reference per stand x departure-runway group (10th percentile over 12 months).
A taxi-out prediction can be read the same way: `prediction = reference + predicted
additional time`. This page decomposes our 2025 out-of-fold predictions like that and asks
two questions: how well do we predict the additional time of an individual departure, and
how close is a PRU-style monthly indicator computed from predictions to the one computed
from the labels.

These are **retrospective reconstructions** using post-operations inputs, including the
take-off time and NM actual off-block reports. The results below describe held-out 2025
month pairs; they do not evaluate forecasts made before off-block.

Computed by [`scripts/regime_quality_report.py`](../scripts/regime_quality_report.py)
(same run as [data-quality-regimes.md](data-quality-regimes.md), which also describes the
reference emulation and its deviations from PRU's method). Tables:
[`pru_view_additional_by_airport.csv`](tables/pru_view_additional_by_airport.csv),
[`pru_view_monthly_kpi.csv`](tables/pru_view_monthly_kpi.csv),
[`pru_view_monthly_kpi_summary.csv`](tables/pru_view_monthly_kpi_summary.csv).

## 1. Setup

- **Predictions.** Production out-of-fold predictions for 2025 (`data/cv_preds_fin1b_hz`):
  `pred` is the mixture output (regular regressor weighted against the planning-time
  candidates by the regime probabilities), `reg` is the regular-regime regressor alone.
- **Reference.** 10th percentile of taxi-out per airport x stand x runway over 2025, computed
  from records in the regular regime (label rule of `src/train.py`), valid with at least 10
  flights at or below it, after PRU's filters (stand, runway, 0-7,200 s, no helicopters).
- **Rows.** Regular-regime records with a valid reference: 1,788,520 departures (85.8 % of
  2025). Records matched by the +-6 s planning-time rule are excluded to define this
  subset; they are treated in the data-quality page. The rule includes coincidental
  matches and does not establish an individual flight's physical taxi-out time.
- `additional_actual = y - reference`, `additional_pred = pred - reference`. Because the
  reference is the same on both sides, the RMSE of the additional time equals the RMSE of
  the taxi-out itself on these rows; what the decomposition adds is the scale of the
  quantity (its standard deviation) and a PRU-style baseline to compare against.

## 2. Per-departure additional time

| airport | n | reference, s | additional: mean / sd, s | RMSE pred | bias pred | RMSE reg | bias reg | RMSE of airport-month mean | skill vs that baseline | corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 1,788,520 | 706 | 262 / 301 | 188.3 | +5.6 | 186.2 | +0.1 | 292.4 | 0.59 | 0.78 |
| EDDF | 224,173 | 639 | 222 / 242 | 158.3 | +0.1 | 158.4 | +0.1 | 241.8 | 0.57 | 0.76 |
| EDDM | 144,230 | 582 | 234 / 278 | 156.3 | +5.0 | 155.9 | +0.4 | 273.5 | 0.67 | 0.83 |
| EGLL | 215,273 | 978 | 391 / 384 | 214.1 | +4.6 | 213.7 | +1.2 | 382.9 | 0.69 | 0.83 |
| EHAM | 231,013 | 586 | 196 / 229 | 147.8 | +0.8 | 147.9 | +0.8 | 227.6 | 0.58 | 0.76 |
| LEBL | 155,435 | 723 | 224 / 235 | 168.5 | +10.6 | 167.2 | -0.1 | 234.0 | 0.48 | 0.70 |
| LEMD | 171,902 | 789 | 217 / 222 | 163.9 | +5.2 | 163.5 | 0.0 | 221.5 | 0.45 | 0.68 |
| LFPG | 181,067 | 712 | 274 / 330 | 211.7 | +5.9 | 212.0 | +0.5 | 327.8 | 0.58 | 0.77 |
| LIRF | 113,936 | 726 | 359 / 391 | 279.9 | +36.8 | 258.9 | -0.7 | 386.2 | 0.47 | 0.71 |
| LSZH | 125,149 | 518 | 214 / 235 | 151.6 | +0.5 | 151.7 | +0.5 | 234.7 | 0.58 | 0.76 |
| LTFM | 226,342 | 729 | 293 / 335 | 214.4 | +1.4 | 214.1 | -1.9 | 333.4 | 0.59 | 0.77 |

"RMSE of airport-month mean" is the error of predicting every departure's additional time
by the mean additional time of its airport and month (an in-sample oracle of the monthly
indicator); "skill" is `1 - MSE(pred) / MSE(that baseline)`.

What the table says:

- The unimpeded reference is 518-978 s depending on the airport; the additional time on
  top of it averages 196-391 s with a standard deviation of 222-391 s. Our prediction
  explains 45-69 % of the variance of the additional time beyond the airport-month level
  (correlation 0.68-0.83). The remaining 150-280 s RMSE is the observed error of this fitted
  model on held-out records. It does not establish an irreducible error floor or isolate
  unpredictable queueing from model error and recording differences.
- The **regular regressor has small observed mean bias** on the rule-defined regular subset
  at every airport (|bias| <= 1.9 s). The **mixture has positive mean error on that subset**:
  +36.8 s at LIRF, +10.6 s at LEBL, +4.6 to +5.9 s at EDDM, EGLL, LEMD and LFPG. The mixture
  also assigns probability to planning-time candidates; moving towards a longer candidate
  raises the prediction even when the realised label belongs to the regular regime.
  Thus `reg` gives a less biased monthly summary of this selected subset in the reported
  evaluation. That does not establish its accuracy for physical taxi-out, or make it a
  replacement for the mixture when the target is the recorded interval across all regimes.

## 3. Monthly indicator from predictions and from labels

For each airport and month the PRU-style indicator (mean of taxi-out minus the reference
over flights with a valid reference) is computed from the labels and from the predictions,
on the same rows with the same reference. Three variants, 120 airport-months each:

| variant | rows | reference | MAE, s | bias, s | max abs, s | corr | sd of the label indicator, s |
|---|---|---|---:|---:|---:|---:|---:|
| as recorded | all records | from all records | 6.4 | +0.4 | 33.2 | 0.994 | 77 |
| regular regime, mixture `pred` | rule-regular records | from regular records | 10.5 | +6.8 | 64.5 | 0.985 | 71 |
| regular regime, regressor `reg` | rule-regular records | from regular records | 6.4 | 0.0 | 32.4 | 0.993 | 71 |

The as-recorded mixture and the regular-subset regressor reproduce their respective
label-based indicators within 6.4 s on average, against a month-to-month spread of 71-77 s;
using the mixture on the regular subset instead gives 10.5 s. The as-recorded variant uses
our emulation of PRU's statistic and has a bias of +0.4 s. Its largest monthly differences
are at LTFM (March -33 s, December +26 s, February -26 s) and LEMD (March -26 s); per airport
the mean absolute difference is 3.0-5.5 s at EDDF, EDDM, EGLL, EHAM, LEBL, LFPG and LSZH,
7.6 s at LIRF, 9.7 s at LEMD and 14.6 s at LTFM. As in section 2, the mixture output
overstates the regular-regime indicator at LIRF (+35.5 s on average), the regressor does not
(-0.7 s).

For PRU, these results support a **retrospective monthly reconstruction** of the recorded
indicator on the evaluated 2025 data. The decomposition separates a stand-runway reference
estimated from recorded taxi-out times from the additional recorded duration. The latter
is relevant to performance monitoring, but it is not a direct measurement of queueing:
recording practices, de-icing and other operational factors can also affect it. The choice
between `pred` and `reg` must follow the population and quantity being summarised.

## 4. Limitations

- The reference window is the calendar year 2025 rather than PRU's rolling 12 months, and
  the de-icing exclusion cannot be applied; see the deviation table in
  [data-quality-regimes.md](data-quality-regimes.md#41-how-the-indicator-is-emulated).
- Out-of-fold predictions are evaluated on the same year they are trained around (each
  month pair held out against the other ten months); the 2026 ranking months are a different year
  and the leaderboard RMSE, not these numbers, measures transfer.
- The reference is estimated from the full calendar year, including the evaluation months.
  Using the same reference for predictions and labels makes this a comparison of recorded
  indicators; it does not test a reference available prospectively for an unseen period.
- "Regular regime" is defined by the +-6 s label rule, which has no ground truth; records
  labelled as copies are excluded from the regular-subset comparisons rather than modelled.
- Per-departure RMSE and the monthly indicator are affected by the right tail of recorded
  durations; close monthly agreement does not imply that individual long delays are
  reconstructed accurately or that the records measure physical queueing.

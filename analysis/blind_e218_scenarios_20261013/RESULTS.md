# E218 scenario bounds for the blind February/June months (journal 467 → 469)

Label-free scenario audit, 0 fits, 0 inferences, 0 uploads. Script:
`scripts/blind_e218_scenarios_20261013.py`; machine output: `results.json` in
this directory. Inputs are only the sealed cell table
(`analysis/final_tail_20261011/rows_changed_by_cell.csv`: changed rows and the
rounded shift per airport × month) and the published board scores of
v12080/v12082/v12083 (`analysis/v12083_upload_20261011/verdict.json`). The
final file `bbae2cc5…` is unchanged under every scenario; this audit prices
its E218 stage on the two months nobody has scored.

## Model and cross-checks

For a cell with rounded shift `c` applied to `n` rows, the registered model
(journal 467) takes the mean pre-shift residual to be `−2βc`, where β is the
share of the published indicator gap that is real bias, so
`ΔSSE(β) = Σ n·c²·(1 − 4β)`. β = 0.25 is break-even and β = 0 is the case
where the gap is pure noise. Two checks were asserted in the script before
any scenario was printed:

- `Σ n·c²` over the February/June LIRF cells is 6,977,098, exactly the
  β = 0 value in `verdict.json`.
- Solving β from the measured January/July board split in this
  parametrisation gives β_LIRF = 0.715 and β_nonLIRF = 0.344, and both
  reproduce the measured split (−0.0304 s and −0.0056 s on 344,841 rows)
  to within 0.002 s. (The verdict's 0.74 / 0.37 use the unrounded shifts
  and, for the non-LIRF part, the v12077 parametrisation; the values here
  use the rounded shifts that were actually applied.)

`Σ n·c²` is 4,978,893 on January/July and 9,757,542 on February/June, of
which the two LIRF cells carry 6,977,098 (71.5 %). The single largest cell
of the final is LIRF June: `c = +21` s on 14,362 rows, `n·c² = 6,333,642`,
which is 73 % of June's total and 65 % of the February/June total.

## Scenario bounds

ΔRMSE is shown at a reference RMSE of 231 s (`sqrt(231² + ΔMSE) − 231`);
the model-exact quantity is ΔMSE in `results.json`. Positive is worse
than the same file without the E218 stage.

| Scenario | February | June | Feb + Jun | Four months* |
|---|---:|---:|---:|---:|
| β = 0 (gap is pure noise) | +0.0158 | +0.1036 | +0.0648 | +0.0130 |
| β = 0.25 (break-even) | 0.0000 | 0.0000 | 0.0000 | −0.0185 |
| Measured Jan/Jul split (LIRF 0.715, others 0.344) | −0.0203 | −0.1510 | −0.0932 | −0.0638 |
| β = 0.516 uniform (v12082 reading) | −0.0168 | −0.1103 | −0.0690 | −0.0520 |
| β = 0.93 uniform (2025 relation) | −0.0429 | −0.2822 | −0.1763 | −0.1042 |
| β = 1.0 uniform | −0.0473 | −0.3112 | −0.1945 | −0.1130 |

\* Four months adds the measured January/July ΔSSE (−5,739,430 over
344,841 rows) to the February/June scenario over 670,790 rows; this is
why the β = 0.25 row is not zero there.

## Reading

- The worst registered case is bounded and asymmetric by month: if the
  published gaps were pure noise everywhere, the stage would cost
  +0.065 s on February/June, +0.104 s on June alone and +0.013 s on the
  four-month ranking. This replaces the looser +0.066 s February/June
  worst case of log 452, which it matches, and adds the per-month split
  that number hid: June carries almost all of the risk because of the
  +21 s LIRF cell.
- At the transfer actually measured on January/July, split by LIRF and
  the rest, the stage would gain −0.093 s on February/June and −0.064 s
  on the four-month ranking. The full 2025 relation (β = 0.93) would
  give −0.176 s and −0.104 s; neither is claimed, both are scenarios.
- Nothing here is a measurement of February/June: those labels are held
  by the organiser. The audit's content is that the bet is now priced
  per subset and per cell, with its break-even at β = 0.25 in every
  cell, well below every transfer measured so far (0.34–0.72 on
  January/July in this parametrisation, 0.93 on 2025 out-of-fold data).

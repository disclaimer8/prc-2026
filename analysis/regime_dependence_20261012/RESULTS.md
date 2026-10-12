# Regime residual dependence audit

Protocol: `regime-dependence-FIN1b-conditioned-day-hour-v1-20261012`.

Retrospective analysis on previously explored 2025 OOF; no model changes, fitting, model inference or submissions.

One outcome run under registration `63ffe1b11f0f7f5c4677e3d895b405c449a05283` completed successfully in **110.245 s**, peak RSS **1,516,339,200 bytes (1.412 GiB)**. All source, protocol, tests and preflight bytes remain identical to the registration.

The primary population retains **1,434,335 of 1,472,048** rows at the seven eligible airports; **37,713** rows were removed by the fixed sparse-stratum rule. There are **6,728 retained primary-population strata** and **2,548 airport-days**. The three airports whose planning labels are structurally disabled remain excluded from the primary analysis. Full support and checks appear in `result.json`.

Inference uses **199 permutations** (minimum attainable Monte Carlo p-value **0.005**, not an exact p-value below 0.005) and **999 date bootstraps**. Secondary and subgroup p-values below are descriptive and unadjusted. Covariance is in squared probability-residual units; the normalised value divides by the residual second moment.

| Population | n | excess day covariance | normalised excess | day permutation p | mean-residual cluster/iid ratio |
|---|---:|---:|---:|---:|---:|
| primary_seven | 1434335 | 5.346857e-05 | 0.000717447 | 0.0050 | 1.35007 |
| half_1 | 694848 | 3.0506801e-05 | 0.000393022 | 0.0050 | 1.15202 |
| half_2 | 739487 | 7.4011313e-05 | 0.00103342 | 0.0050 | 1.55177 |
| airport_EDDM | 163493 | 7.2228141e-06 | 0.000110903 | 0.4450 | 0.965984 |
| leave_out_EDDM | 1270842 | 5.7980838e-05 | 0.000765572 | 0.0050 | 1.39256 |
| airport_EGLL | 234945 | 9.2047333e-06 | 0.000125912 | 0.2800 | 1.00504 |
| leave_out_EGLL | 1199390 | 6.3174125e-05 | 0.00084452 | 0.0050 | 1.41612 |
| airport_LEBL | 173731 | 0.00037221439 | 0.00478217 | 0.0050 | 3.24697 |
| leave_out_LEBL | 1260604 | 1.8263459e-05 | 0.000246569 | 0.0050 | 1.07536 |
| airport_LEMD | 207199 | 3.0281673e-05 | 0.000470925 | 0.0050 | 1.19519 |
| leave_out_LEMD | 1227136 | 5.7239057e-05 | 0.000750652 | 0.0050 | 1.37212 |
| airport_LFPG | 234620 | -5.0983035e-06 | -8.51522e-05 | 0.4800 | 0.876054 |
| leave_out_LFPG | 1199715 | 6.6305186e-05 | 0.000856746 | 0.0050 | 1.42179 |
| airport_LIRF | 151102 | 0.00010723459 | 0.000748147 | 0.0050 | 1.24167 |
| leave_out_LIRF | 1283233 | 4.9082025e-05 | 0.000738919 | 0.0050 | 1.37761 |
| airport_LTFM | 269245 | 1.1581664e-05 | 0.000188745 | 0.0450 | 1.07378 |
| leave_out_LTFM | 1165090 | 6.6459811e-05 | 0.000856788 | 0.0050 | 1.40058 |

## Primary temporal scales

| Pair type | observed covariance | exact finite-shuffle null | excess | date-bootstrap 95% interval |
|---|---:|---:|---:|---|
| airport_day | 4.4246743e-05 | -9.2218273e-06 | 5.346857e-05 | [4.1944975e-05, 6.6326916e-05] |
| airport_hour | 0.00020157085 | -4.7691009e-05 | 0.00024926185 | [0.00020879296, 0.0002911906] |
| same_day_different_hours | 3.4757914e-05 | -6.9016011e-06 | 4.1659515e-05 | [3.0763301e-05, 5.4456781e-05] |

Gate: `{"positive_primary_with_ci_and_nominal_p": true, "positive_both_halves": true, "positive_every_leave_airport_out": true, "distributed_temporal_association": true}`.

The fixed gate supports positive temporal association after the stated conditioning. Its wording does not imply homogeneous effects: LEBL has the largest normalised day excess; EDDM, EGLL and LFPG do not show individually significant day association at the nominal level. Removing LEBL still leaves positive pooled day excess, **1.826346e-5** (descriptive p=0.005).

For the **mean centred binary regime residual only**, the uncorrected cluster/iid variance proxy is **1.35007**, conditional date-bootstrap interval **[1.25920, 1.45196]**; its exact finite-shuffle reference is **0.927039**, not one. These quantities condition on fitted OOF models and empirically estimated stratum means. The shuffle fixes the overall centred mean, so this is not its actual randomisation variance. No RMSE uncertainty or effective number of flights is inferred.

Artifact verification checked metric algebra and day = hour + different-hour pair accounting for all 17 reported populations, recomputed all three primary permutation p-values from the 199 saved draws, and verified support, gates and the finite-null variance ratio. Bootstrap interval integrity was checked; bootstrap draws were not retained, so their quantiles were not independently recomputed. See `verification.json`; the immutable `result.json` SHA256 is `533d653aacd405c3eb87103e54361d43113165eb184cba2e36ce88ab656f66a5`.

## Limits

- Previously explored, selection-informed 2025 OOF; not an independent confirmatory test.
- Residual exchangeability within coarse strata is an assumption; unmeasured composition remains.
- Common monthly model misspecification or temporal omitted features can explain association.
- No physical/recording mechanism, causal effect, Bayes floor, RMSE standard error or score gain established.
- Cluster/iid ratio concerns a mean binary regime residual under independent-cluster assumptions only.
- Bootstrap is conditional on OOF fits/stratum means; serial dependence across dates is not modelled.
- Secondary and subgroup p-values are descriptive and not multiplicity-corrected.

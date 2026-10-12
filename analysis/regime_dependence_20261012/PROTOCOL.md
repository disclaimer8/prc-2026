# Regime residual dependence: registered descriptive analysis

Protocol: `regime-dependence-FIN1b-conditioned-day-hour-v1-20261012`.

Status: source/protocol preparation only. The coordinator must register this exact protocol, source, tests and preflight receipt in one commit before `run`. These are previously explored data and historically selected OOF models; this is a preregistered analysis of a retrospective hypothesis, **not an independent confirmatory discovery**.

## Question and limits

Do errors in FIN1b's probability of a nonregular recorded-time regime co-vary within an airport-day, after conditioning on broad observable flight composition? Is dependence restricted to the same clock hour, or also present between different hours of a day?

A positive result is evidence of residual temporal association. A shared recording process is one explanation, alongside omitted operations, changing composition within coarse strata and common misspecification of the monthly OOF model. The analysis cannot identify these mechanisms, prove a Bayes error floor, improve a forecast, or estimate the uncertainty of leaderboard RMSE. A negative result bounds this specific conditional association, not every latent-state model.

## Inputs, integrity and population

- `data/features_fin1b.parquet`: 2,429,888 rows, 970,626,280 bytes at inventory; select only `split == train`, expected **2,085,047** DEP rows in 2025. No ranking outcomes or predictions read.
- The exact six files `data/cv_preds_fin1b_hz/fold_{a}_{a+6}.npz`, a=1,...,6: read `mvt_id`, `y`, `proba` only. Total about 158 MB. Class order: regular, sched, iobt, lobt, eobt, dayshift.
- `data/adsblol_kt1/offblock2_2025-*.parquet`: exact 364 days January 1–December 30, 32,393,238 bytes. December 31 is absent and remains missing. Read ID/coverage only; no extraction or inferred replacement.
- `src/train.py`: execute only its existing `regime_label` function AST, with explicit recipe globals TOL=6, SCHED_TOL_LONG=0, DAY=86400, planned classes excluded at EDDF/EHAM/LSZH. No import of the fitting module and no fitting/inference. Dayshift precedence and first-match ordering remain the existing implementation.
- Preflight checks: each file and pooled OOF IDs finite, integral and unique; all six folds exist exactly once; exact feature/OOF ID-set and target equality; fold membership equals month pairs; all selected timestamps are 2025; all ten expected airports present; probability shape N×6, finite, [0,1], row sum within 2e-6; coverage unique and matched to train IDs. Duplicate/unmatched IDs or mismatched years/labels stop, without results.
- A SHA256/size manifest pins every input, source `train.py`, and ordered train IDs. Preflight computes integrity facts only, not regime labels, residuals, covariance or inference. Execution verifies source, tests, protocol and receipt byte-for-byte against the registration commit, exact protocol text in row 463 of that commit's journal, and all input hashes.

Primary population: the seven airports where planned regimes are labelled (EDDM, EGLL, LEBL, LEMD, LFPG, LIRF, LTFM), including all NM statuses, all target magnitudes and the historical 126 tail rows. This is a classifier-residual question, not the downstream `decided` RMSE population. The three airports with structurally disabled planning labels are excluded from the primary question, and their counts are explicitly reported. No filtering by observed error or correctness.

## Residual and conditioning

For row i, `z_i = 1{regime_label_i > 0} - (1 - proba_i,regular)`.

Strata are the Cartesian product of:

1. airport;
2. month (also holds the fitted OOF classifier constant);
3. NM record present (`FLIGHT_ID_mvt` non-null);
4. coverage class: -1 missing, 0 recorded coverage 0/1, 2, 3;
5. p_nonregular bins: [0,.01), [.01,.05), [.05,.20), [.20,.50), [.50,.80), [.80,.95), [.95,.99), [.99,1];
6. UTC hour band: 00–03, 04–07, 08–11, 12–15, 16–19, 20–23;
7. weekday versus weekend, by UTC MVT date.

Keep strata with at least **20 rows spanning at least four distinct UTC dates**. No adaptive bin merging. Report excluded rows per airport and full support. Retain fixed rows for every comparison. Set `u_i = z_i - mean(z in its stratum)`. This descriptive sample centring removes coarse miscalibration; it is estimated using all retained 2025 rows and is not a deployable transformation or OOF-trained calibration. Shuffling whole residual values preserves their stratum membership, distribution and mean.

The bins do not establish conditional independence from every measured or unmeasured variable. Residual exchangeability inside a stratum is the permutation null's assumption, not a demonstrated property of the data.

## Statistics and exact null

Three fixed pair populations: (1) distinct flights within airport×UTC-date (primary), (2) distinct flights within airport×UTC-date×hour (secondary), (3) different clock hours within the same airport×UTC-date (secondary). Never pair a row with itself. Pairs are oriented; using unordered pairs yields the same mean.

For group g, numerator `A_g = (sum u_g)^2 - sum(u_g^2)` and denominator `P_g = n_g(n_g-1)`. Pair-weighted covariance is `sum A_g / sum P_g`. Between-hour contributions are day minus hour contributions. No quadratic pair enumeration is necessary. Report pair counts and the residual second moment `Q/N`; normalised excess covariance divides the excess by `Q/N`. This normalisation is descriptive, not a fitted exchangeable ICC.

Finite-stratum correction is mandatory. Under a uniform shuffle of centred residuals within stratum s, a pair from the same stratum has expected product `-Q_s/[n_s(n_s-1)]`; a cross-stratum pair has expectation zero. Sum this using counts per stratum×group to obtain exact null numerator B. Report observed covariance, null covariance `B/P` and **excess `(A-B)/P`** separately. In particular, zero is not the exact finite-shuffle null for the raw covariance.

For the *mean of this centred binary residual contrast only*, report `R_day = sum_g(sum u_g)^2 / Q = 1 + A_day/Q` and the exact shuffle baseline `R_null = 1 + B_day/Q`. This is an **uncorrected conditional cluster/iid variance proxy**, not a general sample-size multiplier: both variance proxies would divide by N²; no small-G correction is applied. It assumes independent airport-day clusters for that hypothetical mean, conditions on estimated stratum means and fixed OOF models, and ignores cross-day/airport dependence. The constrained shuffle fixes the overall centred mean, so R is not its actual randomisation variance. No effective number of flights, uncertainty of prediction RMSE, or generalisation confidence interval is derived.

## Permutation and descriptive robustness

- **199** uniform permutations of the whole u values independently within immutable strata, NumPy `default_rng(20261012)`. Random-key ordering within each stratum generates the permutation. No probability or label is refitted.
- Two-sided Monte Carlo p-value for excess: `(1 + count(|A_perm-B| >= |A_observed-B|))/(199+1)`, using the same fixed denominator. Minimum p=.005; Monte Carlo resolution must be stated.
- Primary analysis is the seven-airport pooled day statistic. Report the three temporal scales and the same metrics for January–June, July–December, each primary airport and each leave-one-airport-out population. Strata do not cross these subpopulations, so subpopulation permutations remain valid. Secondary/subgroup p-values are **unadjusted descriptive values**; no secondary discovery claim is made from them.
- **999 date bootstraps**, seed 20261013: resample UTC dates within each month, carrying every retained airport-day contribution on a selected date jointly. Report percentile 95% intervals for primary excess covariance at the three temporal scales and R_day. These intervals are conditional on the fixed OOF models and estimated stratum means; centring is not re-estimated. They preserve contemporaneous cross-airport association but do not model serial dependence between dates, or sampling of different years/airports.

## Forecast and interpretation gate (before outcomes)

Dependence is plausible; its sign and size are unknown. No numerical gain or narrow effect-size corridor is forecast.

Call **positive primary temporal association** only if day excess >0, its two-sided permutation p≤.05, and its date-bootstrap lower limit >0. Call it **distributed temporal association** only if the primary condition holds and day excess is positive in both calendar halves and in every leave-one-airport-out population. A significant pooled result that fails these robustness conditions is reported as localised or temporally unstable. A failure is a useful negative result, not a reason to change strata, seed, exclusions, repetitions or statistic. No fitted rule or score-optimisation stage follows this audit.

## Execution boundaries and outputs

- Existing `.venv/bin/python` only; no package changes or uv. Mac CPU only: OMP=2, BLAS=1, Polars=2, nice +5; no GPU/remote work.
- Hard wall-clock alarm **900 seconds**, resident-memory limit **6 GiB** checked after phases/permutation batches. Expected minutes and roughly 1–2 GiB (estimates, not measured run results). Input hashing about 1.16 GB; only required Parquet columns read.
- Allowed outputs only in `analysis/regime_dependence_20261012/`: this protocol, `preflight.json`, later immutable `result.json` and `RESULTS.md`. No production artifact changes. Existing result files stop execution.
- Stage 1: schema/integrity preflight and synthetic tests, then coordinator review/freeze. Stage 2: one registered outcome run. If a technical correction is needed, preserve failed evidence and require a new registration before outcomes.

```sh
.venv/bin/python -m unittest discover -s tests -p test_regime_dependence_audit_20261012.py -v
.venv/bin/python scripts/regime_dependence_audit_20261012.py preflight
# Coordinator freezes source + protocol + tests + receipt + journal, then:
.venv/bin/python scripts/regime_dependence_audit_20261012.py run --registration-commit FULL_40_CHARACTER_SHA
```

Implementation plan: (1) independent synthetic pair enumeration, exhaustive tiny-null, permutation-stratum, injected-dependence and alignment failures; (2) minimal aggregate implementation and input guard; (3) read-only preflight; (4) coordinator review and immutable registration; (5) run once and report every planned statistic and limitation. Synthetic tests do not invoke the real outcome analysis.

# E196: conditional density before a diffusion model

Protocol `E196-conditional-density-mean-gaussian-mdn4-2pair-3seed-8epoch-v1`.
Worker `scripts/e196_conditional_density.py`. Code and journal must share a
registration commit before target fitting. This is a Gaussian/mixture baseline
for the user's distribution-learning hypothesis, not an implemented diffusion model.

The mechanism is whether learning conditional residual dispersion/multimodality
changes the conditional mean beneficially, with exactly the existing observable
inputs. It supplies no new weather/trajectory information. Prior for body RMSE
difference versus fresh mean: center0s, uncalibrated range[-1,+1]s; no guaranteed
leaderboard gain and no numerical improvement threshold for acceptance.

## Fixed experiment

Original immutable E190 query bundles for outer pairs1+7 and4+10. Same original
ordinary-regime training population, residual proxy target/clipping±7200 and
standardization; existing planned/day branches stay fixed. All arms use the
original PLE48/d12/versionB/no activation, width512 backbone and train-only bin
construction. Seeds0/1/2,8epochs,batch4096, AdamW .002/weight_decay1e-5 and
OneCycle pct_start .1. Floor-sized batches and fixed seed permutations match
the original recipe. No early stopping, tuning or automatic retry.

Three heads/objectives:

* `mean`: unchanged scalar head and squared error.
* `gaussian`: mean/log-sigma, Gaussian negative log likelihood, log-sigma
  clamped[-4,3]. Mean head inherits the original initialization, sigma starts1.
* `mdn4`: four logits, means and log-sigmas with stable log-sum-exp mixture
  likelihood. Equal initial weights, sigma1, original mean plus fixed offsets
  [-.75,-.25,.25,.75] in standardized residual units. Same sigma clamp.

The backbone is identical; head sizes1/2/12 have explicitly different parameter
counts. Extra-head initialization consumes random values, so dropout draws need
not be identical across objectives; three fixed seeds mitigate, not eliminate,
that difference. The experiment tests these fixed density recipes, not every
possible distribution architecture. The fresh mean arm checks reproduction.

Final prediction is the analytic conditional mean: mu for Gaussian and
sum(softmax(logits)*mu) for mixture. It is transformed back to seconds before the
existing fixed mixture. There is no Monte Carlo sampling or median/quantile
substitution. Original MLP contribution is replaced through the saved exact
row-specific coefficient; no blend weights are selected.

Total18fits=2pairs×3arms×3seeds. Existing controls/evaluation metadata/saved PLE
are copied from the SHA-verified E192 input pack; query tensors remain immutable
references, not duplicated. Exact IDs/y/months, source/runtime/input hashes,
all18 fits, every three-seed mean and final mixture arithmetic must verify.
All original seed predictions and model weights are retained; publication is
atomic only after the full experiment. A failed staging directory remains.

## Fixed rejection gates

Report all/body/decided RMSE, body/decided/symmetric trim3/trim20 SSE, month SSE,
NM-LTFM body and the inherited descriptive paired airport-day bootstrap.
Gaussian must beat BOTH fresh mean and saved E190 PLE in body, decided, trim3,
trim20 SSE in BOTH pairs. MDN4 must beat fresh mean, Gaussian AND saved PLE on
all those legs in BOTH pairs. No selection by pooled score alone. Always
accepted_for_submission=false. Any later six-pair/current-chain fit requires a
separate registration; failures do not trigger a different density configuration.

This is a two-pair LGB5/pre-ADS-B mixture screen. It does not establish current
Cat15/full-chain CV, distribution calibration, diffusion quality or leaderboard
transfer. The historical upstream OOF limitation remains explicit.

## Resource and qualification

Home Radeon RX7900XTX in existing `.venv-numemb`; CPUQuota200%, MemoryMax12GiB,
MemorySwapMax0, nice10. No environment change. Queued after E193 completes,
including its prespecified conditional continuation; no concurrent full GPU fit.
The successor waits for process completion, not merely systemd start ordering.
A separate Home CPU job has12CPU/16GiB. Smoke was random-only,4096-row full
shape,finite outputs and gradients in all three arms; Gaussian formula checked
against standard-normal NLL. Receipt `data/e196_smoke_20261007.json`, worker SHA
`20c75c893703591fb7bd465a987a587a266b79c9f32526cfeb37135109b2baa3`.
Synthetic parameters mean2282197/Gaussian2282454/MDN2285024; peakGPU<754MB.
This short smoke is not a reliable runtime benchmark; first real fit gives ETA.
The first attempt failed before fitting on a missing frozen helper import; after
copying that unchanged helper the fresh v2 smoke service exited0 at07:11:09UTC.
No real target was fitted by either smoke attempt.

After registration REG:

```
.venv/bin/python scripts/e196_conditional_density.py pack --root . --output data/e196_density_inputs_20261007 --registration-commit REG --smoke data/e196_smoke_20261007.json
.venv-numemb/bin/python data/e196_density_inputs_20261007/scripts/e196_conditional_density.py run --root . --input data/e196_density_inputs_20261007 --output data/e196_density_20261007 --registration-commit REG
```

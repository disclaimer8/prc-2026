# E198: fixed four-pair confirmation of E196 MDN4

Protocol `E198-mdn4-remaining4pair-confirmation-mean-control-3seed8epoch-v1`.
Worker `scripts/e198_density_extension.py`. Root must register worker, this
protocol and journal in one commit before fitting. E196 stays immutable.

## Reason and fixed scope

E196 MDN4 passed its prespecified body/decided/trim3/trim20 criteria against fresh
mean, Gaussian and saved PLE on pairs 1+7 and 4+10. Its fresh mean differs slightly
from saved PLE (+0.000094/+0.010642 s all-RMSE); exact reproduction is not claimed.
E198 therefore keeps BOTH fresh mean and saved PLE comparisons.

Those two pairs selected MDN4 and are explicitly labelled `selected_screen`.
The remaining pairs **2+8, 3+9, 5+11, 6+12** are labelled
`remaining_confirmation`. Report them separately; combining all six does not
turn the selected two into independent validation. Fixed expert prior for
MDN4−fresh-mean body on remaining pairs: center −0.2 s, uncalibrated interval
[−1,+0.5] s; no leaderboard forecast or numerical gain threshold.

Exactly **24 new fits = 4 pairs × mean/MDN4 × seeds 0/1/2**. Eight epochs each.
No further Gaussian fits, mixture-size search, architecture/weight changes,
early stopping, altered seed count, automatic retry or ranking continuation.

## Exact inherited recipe

The wrapper calls **unchanged `e196_conditional_density.fit`** and its network
and prediction/loss helpers. It does not copy or replace the trainer. Original
E190 PLE48/d12/version B/no activation, width512, train-only numerical bins;
batch4096, AdamW .002/decay1e−5, OneCycle pct_start .1, eight epochs, original
regular-row population, clipped/standardized residual targets and permutations.

Mean uses the original scalar MSE head. MDN4 uses four logits, four means and
four log-sigmas; log-sigma clamp [−4,3], equal initial weights, sigma1, fixed
mean offsets [−.75,−.25,+.25,+.75] in standardized residual units. Its output
is the analytic softmax-weighted conditional mean, converted back to seconds.
No Monte Carlo, diffusion or quantile substitution. The extra MDN head has more
parameters, exactly as in E196. Extra-head initialization can affect dropout
draws; this remains a fixed-recipe comparison, not identical stochastic traces.

MLP slot coefficient and mixture/LOBT/motion arithmetic are unchanged from the
E190 screen. The saved current production mixture is not refitted or reweighted.
This is still LGB5/pre-ADS-B component CV, without current Cat15/full-chain
downstream stages. A pass can motivate a separately registered full-chain check;
it is never automatic submission acceptance.

## Inputs and source proof

Use E193's immutable all-six-fold metadata/control/PLE pack only as a verified
container; **no weather inputs are used**. Parent registration
`fc7107de145a87a5a1ff93314f318b440b1de825`, spec SHA
`2b8f65e7212c91523d87a166ceb955376784223ee1dfb9cf1b403fa2f88a0f70`.
For the four new folds, verify selected file hashes, exact ID/y/month against
original query metadata, query registration/spec/array hashes. Query tensors
remain references under `data/mlp_split_20261005/inputs`; no duplication.

Before packing, revalidate E196 registration/runtime/source and all 18 original
fit receipts, prediction/model hashes, three-seed means, mixture arithmetic and
all original comparisons. Recompute the original MDN screen gate. Copy only
verified old mean/MDN4/PLE predictions and evaluation metadata needed for the
six-pair report; retain original screen input/complete/evaluation hashes in the
new manifest. Input files and active helper sources are hashed again before and
after fitting. Original E196 source SHA:
`20c75c893703591fb7bd465a987a587a266b79c9f32526cfeb37135109b2baa3`.

## Fixed gate and reporting

Per all six pairs, report MDN4−fresh-mean and MDN4−saved-PLE all/body/decided RMSE,
body/decided/symmetric trim3/trim20 SSE, each month's SSE, NM-LTFM body and
the inherited descriptive paired airport-day bootstrap. Also report fresh
mean−PLE to preserve the observed baseline difference.

The eventual six-pair screen passes only if, **against each of fresh mean and
saved PLE separately**:

- decided SSE improves in at least **4 of 6** pairs;
- summed body SSE is negative;
- summed trim3 and summed trim20 SSE are both negative.

Trimming follows the repository gate convention: sort paired squared-error
differences **within each fold**, remove three/twenty extremes from each side,
then sum the six fold values. It is not global trimming after concatenation.
Report pooled decided SSE too; the registered gate uses the fold-count condition
above. Report the four remaining pairs separately with their decided improvement
count and pooled SSE, without inventing an additional post-result cutoff.
Bootstrap and per-month results do not select another model or endpoint.

`six_pair_gate_pass` is an aggregate component-screen result containing selected
pairs. `accepted_for_submission=false` remains fixed, including on a pass.
Full current-chain acceptance and ranking require new root registration.

## Resources and execution

Home Radeon RX7900XTX, existing `.venv-numemb`, CPUQuota200%, MemoryMax12GiB,
MemorySwapMax0, nice10. No environment/helper changes. Existing E196 synthetic
and all 18 real fits qualified the exact imported network/fit/runtime; this
wrapper introduces no new tensor operations in the model, so it needs no new
synthetic architecture benchmark. New selftests cover fold disjointness,
absence of Gaussian continuation and fixed gate boundaries. Pack and actual
run check the same runtime receipt again.

E196 measured 25.3–26.7 s per fit: roughly 10.5 minutes training for 24 fits;
allow 12–25 minutes including preparation, hashing, predictions and evaluation.
This is an estimate, not a deadline-based seed/epoch adjustment.

One GPU job at a time. Unit `prc-e198-density-20261007.service`; output
`data/e198_density_20261007`. E197 waits for observed E198 start and completion
before fitting; unit absence before launch is not completion. A failure retains
hidden staging, produces no ready marker and receives no automatic retry.
All 24 fits publish atomically only after complete verification; evaluation
follows. Verify exact seed means, final mixture, finite values and full metadata
before reporting completion. Collect to Mac and repeat registered evaluation.

After root registration SHA:

```sh
.venv/bin/python scripts/e198_density_extension.py pack --root . \
  --output data/e198_density_inputs_20261007 --registration-commit SHA
# Transfer small pack; existing query bundles remain in place.
.venv-numemb/bin/python data/e198_density_inputs_20261007/scripts/e198_density_extension.py run \
  --root . --input data/e198_density_inputs_20261007 \
  --output data/e198_density_20261007 --registration-commit SHA
```

Root owns journal and submission decisions. No user confirmation or new fitting
scope is inferred from a positive score.

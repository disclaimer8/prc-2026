# Draft protocol: fill unused E194 token slots

Status: ready for implementation/review, NOT a preregistration and NOT permission to fit. Root must commit reviewed code + journal + immutable protocol before any target-conditioned test. No experiment number reserved here.

## Hypothesis and exact contrasts

E194 maps 64 uniform-time grid positions to the next actual raw observation and deduplicates the indices. Most raw visits with >64 observations therefore provide <64 tokens. Filling existing padding can expose local motion context that neither E197 nor E204 saw. It does not increase neural parameter count. It increases observed-token count and changes adjacency/pooling; those effects must be stated.

Primary screen has O/F only; T is a separately registered follow-up after F. Described arms have the same 2×64 slots, existing 12 channels, missingness/presence masks, E197 ordered d1/d1 encoder, training IDs, MSE target, tabular inputs, seeds 0/1/2, eight epochs and .25 regular-candidate slot:

- **O:** saved verified E197/E202 ordered predictions, unchanged original sampler; no refit.
- **F:** preserve every original selected raw index, fill unused slots with deterministic temporal coverage.
- **T:** preserve every original selected raw index, first fill with observed speed-transition brackets, then the same temporal-coverage rule as F.

The first registered experiment tests F−O: additional unused observations under the same capacity. If F passes, a separate preregistration can test T−F at exactly matched cardinality; T−O is also required. A positive T−O alone does not establish that transition selection helped. T is not part of the minimal initial screen.

## Deterministic sampler specification

Keep identity resolution, day boundary, 12h/+600s E64 visit definition and final t≤takeoff cutoff exactly as E194. Inputs are the full raw slice already pinned by raw_ranges, not regenerated trajectories. If raw_count≤64, all arms are byte-identical to original. For raw_count>64:

1. Start with original `sample_time_indices(t)` indices, including both endpoints. Never drop an old index. Let remaining=64−len(selected).
2. F filler, fixed before any target evaluation: `unselected=setdiff1d(arange(n),old)`; `need=64-len(old)`; select `unselected[rint(linspace(0,len(unselected)-1,need)).astype(int)]`. Concatenate with old and sort indices. This is O(n) sampling spread across the remaining RAW INDICES. It retains actual rows, including duplicates, without inventing observations; duplicates/stale remain marked by existing channels. Handle need=0 without calling linspace. All lengths must be exactly min(64,n).
3. T priority list: adjacent raw rows with finite gs on both endpoints, dt in (0,30] seconds, and a crossing of gs≤1 vs gs>1 in either direction. Sort crossing pairs chronologically, then by raw index. Visit each pair in that order; append its missing left endpoint then missing right endpoint until budget is exhausted. Fill any remaining slots using step 2. The one-knot threshold is a fixed metadata event definition, not an off-block label or estimated target. Do not tune it after results.
4. Sort selected raw indices before encoding. Preserve original 12-channel formulas, including gap to previous RAW point. That channel is not a bug: elapsed ages already encode selected-token intervals.
5. Assert exactly min(64,n) distinct sorted raw indices, old_indices⊆new_indices, original visit membership, sorted timestamps, all raw output points≤takeoff, exact original channel values/known masks at retained indices, unchanged PAD semantics and exact old output when raw_count≤64. A later F/T contrast also requires identical cardinality.

Do not add airport reception-count features or auxiliary targets. Do not swap in E204 dilation. No choice conditioned on y, OOF residual or submission score.

## Preparation and integrity

Use fresh mechanism-specific outputs. Preserve all E194/E197/E202 artifacts. Pin source/helper SHAs, E194 complete manifests, each raw input SHA, target/query IDs, existing seeds/models and original comparator. Validate the sampler using synthetic irregular timestamps, duplicate timestamps, missing gs, consecutive crossings and slots exhausted midway through a pair.

Normalization: same E197 rule, first six channels only, train-month-only statistics, clipping ±8, float16 values/uint8 masks. New samplers can change the fitted statistics; save both sets explicitly. The matched F/T test uses the same rule, not a claim that exact means/stds are unchanged. Record how often source points are duplicate timestamps/stale/unknown-stale and whether F/T spend slots on those; this is metadata, not a reason for post-hoc exclusion.

## Fixed six-pair test and falsification

Root selected the full six-pair F-only test before any target evaluation: 1+7,2+8,3+9,4+10,5+11,6+12. One new arm ×3 seeds×6 pairs×8epochs =18fits; O uses saved verified E197/E202 ordered fits. No two-pair screening, no choice of favorable folds, no event arm or sampler tuning after results.

Report each pair separately and pooled: all/decided/body(y≤7200)/within-fold trim3/trim20, exact same protected-ID policy; descriptive own/previous-only/no-points/airport diagnostics cannot choose a new routing rule.

Primary gate: ≥4/6 decided pairs improved AND pooled decided SSE<0, pooled body SSE<0, within-fold trim3sum<0 and trim20sum<0. Failure of any criterion stops this fixed experiment. Existing O was historically selected using these months and other pipeline pieces are not fully nested; this is selection-informed six-pair evidence, not an untouched confirmatory estimate.

T requires separate code+protocol registration and a matched F control if pursued later; it is not an automatic fallback on an F failure.

Then an independently registered current-chain comparison against saved E203 ordered preE81, with each branch's own surface refits and exact current control, must pass body<0, decided<0, ≥4/6, within-fold trims<0, with all six pair results reported. E200/E205 establish why a component pass is insufficient. E81/LOBT/protect42/halfLSZH complete-tail OOF remains unavailable; report historical upstream/selection-informed status and do not revive E201.

No ranking fit, upload or changed production file follows merely from this draft. Register the full candidate separately after current-chain PASS.

## Resource envelope and prior

Home RX7900XTX, existing `.venv-numemb`, same 2 CPU/12GiB RAM/20GiB VRAM limit/nice10/swap0 envelope as E204, no environment mutation. Laptop171/175 prepare each screen fold and send tensors directly to Home; avoid two9.2GiB table copies on Mac. Single GPU trainer. Selected six-pair18-fit run estimate≈72minutes by scaling measured E2046fits/24min; plan75minutes, not a measured runtime for new sampler/preparation. Historical optional budgets: F-only two-pair6fits≈24min or jointF/Ttwo-pair12fits≈48min; neither is the selected protocol. Wall allowance6h for preparation+18fits, with own no-auto-retry terminal receipts. Current-chain stage is separate. CPU current-chain stage is separate and starts only after registered PASS.

No measured quality delta or empirically calibrated forecast exists for this change. Prior centre0; provisional expert component-body range[−1,+1]s is not a CI, cutoff or promise. Learned-input effects have uncertain transfer; any board forecast should be stated only after current-chain evidence under AGENTS rules.

Counterexample that kills the story: added rows are mostly stale/near-duplicate observations and distort mean/max pooling or adjacency; F/T then lose to O despite fuller tensors. Another: transitions are speed jitter and existing firstmovement/stops/stand summaries already contain their usable information; T loses to F. Either is an informative negative, not a reason to try thresholds on the same folds.

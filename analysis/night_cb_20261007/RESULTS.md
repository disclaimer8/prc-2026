# E205 result — 7 October 2026

Registered as journal370, commit `05a1b2eb7adfa5bd1abc47bc03a94095e0a39d4e`.
All12 fixed LightGBM fits finished on Hetzner and laptops171/175. Sixfold
collection independently recomputed metrics and verified ID/y/month/control,
input and output SHA, and unchanged126 excluded predictions. Units now have
MainPID0; none is still training. No ranking fit, candidate or upload.

| Contrast | Body delta RMSE, s | Decided delta RMSE, s | Decided folds improved | Within-fold decided trim3/20 SSE | Gate |
|---|---:|---:|---:|---:|---|
| CB versus matched existing-weather residual model | −0.012586095 | −0.009903176 | 5/6 | −9,137,660.818 / −8,782,682.023 | PASS |
| CB versus current E203 ordered control | −0.017653960 | +0.087097186 | 3/6 | +68,931,967.839 / +57,951,667.517 | FAIL |
| Existing-weather residual versus E203, descriptive | −0.005067865 | +0.097000361 | 2/6 | +77,962,963.325 / +67,846,897.904 | — |

**The registered experiment fails:** both primary contrasts were required.
The four CB features show a small incremental signal in the matched residual
comparison, but this fixed residual deployment harms the current chain's
decided population and trimmed scores. This does not prove that CB has no
useful information in another separately registered mechanism. No model,
weight, airport slice or threshold was chosen from these results.

Both primary body results lie within the preregistered expert range
[-0.5,+0.5]s (corridor passed). That range was not a confidence interval.
The comparison is selection-informed historical E203 OOF ending before E81,
not fully nested or full production-tail CV and not a leaderboard estimate.

Label-free flight coverage:40,154 training and4,513 ranking departures have
cloud CB omitted by current `wx_thunder`. Existing thunder and report-age
features matched on all2,429,888 joined flights. Detailed airport/month
coverage is in `input_receipt.json`.

Evidence: `execution/evaluation.json`, `execution/complete.json`, per-node
terminal receipts/models/predictions. Input spec:
`09cbfb8a5053ab03117d2948d19b07987bdb94b38fd90d60933ea8658c7c934b`.
Ten focused contracts and compilation passed before registration. No existing
environment or production feature builder was changed.

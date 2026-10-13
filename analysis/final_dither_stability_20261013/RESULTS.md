# Stability of the final file to a fresh dither draw, including the neural member (journal 468 → 470)

Label-free scenario audit, 0 fits, 0 inferences, 0 uploads. Script:
`scripts/final_dither_stability_20261013.py`; machine output:
`results.json` in this directory. The question is the one left open in
journal 423: if `final_ranking.parquet` had carried a different draw of
the provider's ±12 s dithering, how much would the submitted file's
predictions — and therefore its score — have moved, now that the premix
contains the ordered-trajectory neural member (E203) at weight 0.35?

## What is measured and what is scenario

Measured inputs, each traceable to a saved record:

- On the v870 premix (the final's tabular ancestors without the neural
  member), rebuilding the January/July features on the re-dithered file
  changed predictions by RMS 16.20 s outside the 43 LIRF tail rows, and
  the realised board deviation of such a change was estimated at
  ±0.08 s (1σ) on 344,841 rows (`analysis/final_jitter_sensitivity_20261008.md` §3).
- In the final premix the tabular members carry weight 0.65 and the
  neural member 0.35 (`WEIGHTS['final']`, log 425).
- Stages after the premix passed a premix shift through at 0.76
  (v12076: premix RMS shift 6.3 s → file RMS shift 4.849 s,
  `analysis/v12076_upload_20261008/RESULTS.md`).
- The tail ensemble's spread across re-draws was measured on the 2025
  tail in E210: averaging over ±6 s input shifts cut the per-row spread
  from 65.9 s to 27.3 s without changing squared error (log 397b, 403).

The neural member's own response to a shifted take-off anchor is **not
measured** here: its prepared tensors (19 GB) and OOF checkpoints are on
the Home machine, so no re-inference under shifted token times was run
on the Mac. It enters as registered scenarios (journal 468). The
submitted file itself is fixed; every number below is counterfactual.

## Results

Premix responses are combined as independent components,
`sqrt((0.65·16.20)² + (0.35·RMS_NN)²)`, passed through at 0.76, and the
1σ band is scaled from the measured ±0.08 s in proportion to the
final-file RMS change and from 344,841 to 670,790 rows.

| Neural-member response (RMS, scenario) | Premix change | Final-file change | Board band (1σ) |
|---|---:|---:|---:|
| 0 s (member ignores the anchor shift) | 10.53 s | 8.00 s | ±0.028 s |
| 8 s | 10.90 s | 8.28 s | ±0.029 s |
| 16 s (like the tabular members) | 11.93 s | 9.06 s | ±0.032 s |
| 32 s | 15.37 s | 11.68 s | ±0.041 s |
| 64 s | 24.75 s | 18.81 s | ±0.067 s |

The band reaches 0.05 s only if the neural member's response to a
±12 s re-draw were RMS 43.7 s — about 2.7 times the measured response
of the whole tabular premix and of the same order as the member's
pairwise seed spread (70.6 s) — while its inputs shift by at most 12 s
in one time channel. The 62 served tail rows are negligible at their
measured spread: at 27.3–65.9 s per row they add 0.0002–0.001 s of
counterfactual ΔRMSE over 670,790 rows (scenario table in
`results.json`; even an implausible 700 s per row would add 0.098 s).

## Reading

Under every registered scenario, including a neural response four
times the tabular one, a different dither draw would have moved the
final ranking by at most about ±0.07 s at 1σ, and by about ±0.03 s if
the neural member responds like the tabular members it is blended
with. This extends the ±0.08 s January/July estimate of 8 October to
the four-month final file and closes the E203 gap left in journal 423
as a bounded scenario statement, not as a measurement. A direct
measurement remains possible on the Home machine (re-tokenise one OOF
fold under shifted take-off times and re-run the saved checkpoints);
it is recorded here as the one step this audit deliberately did not
take, because it cannot change the submitted file and the bound above
already sits below the noise the control upload measured for refits
(about ±0.29 s on January/July, log 445–447).

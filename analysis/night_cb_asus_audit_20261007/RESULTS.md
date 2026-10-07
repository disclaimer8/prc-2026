# ASUS independent CB join: PASS

Source committed before execution:
`e22be494a7df45bb72c582f7ae787a73782bbf15`.
Actual Windows process7372, PowerShell7.6.6, existing `.venv-prc`.
Terminal07October2026 22:23:34UTC, exit0; process no longer exists.
No scheduled task/future trigger was created. ASUS is now free.

- All2,429,888 unique flight IDs and four independently reconstructed CB arrays
  match exactly, including NaNs; old thunder/age and every coverage group match.
- Canonical row digest:
  `89c832f0446906fc03f7c06b715051145e3b1febf3294d7a92ae4a23380edcd4`.
- Elapsed5.791989seconds, peak RSS713,482,240bytes (~680.4MiB).
- Actual affinity[0,2], below-normal priority16384, hard kernel6GiB cap,
  AC checks and300second deadline; no environment changes or target fitting.
- Result SHA verified after collection:
  `5a2266a16a38d3df0d082ee85f1dee021a3f33f26c9b513bf7a0796a7ed26e9f`.
- Portable manifest SHA:
  `f1e40899755dc9c34bed78c4cd26c160fdc7fa2b7adc36d83ac76ff27fe8cfa9`.

Evidence: `results/result.json`, `results/started.json`, `terminal.json`,
`worker.log`, and `inputs/manifest.json`. Mac rechecked result SHA, source commit,
full coverage and row digest. The synthetic qualification separately checked
duplicate last-row handling, exact report boundaries, TCU exclusion, no-CB NaNs,
and detection of a one-cell digest mutation.

This verifies input reconstruction. It does not change E205's negative
current-chain gate or authorize a candidate/upload.

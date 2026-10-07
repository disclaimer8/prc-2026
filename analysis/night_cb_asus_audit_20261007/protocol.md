# Independent ASUS CB flight join audit

Protocol `E205-ASUS-independent-current-CB-join-metadata-audit-v1`.
This is a target-free parity audit of the E205 input join, not model fitting,
CV comparison or a new candidate. Code/protocol/PowerShell launcher are committed
before pack and execution. Existing frozen E205 files are read only.

The portable package contains exactly eight flight query columns: MVT_ID_mvt,
split, month, ADEP_mvt, AOBT_3_flt, MVT_TIME_UTC_mvt, wx_thunder, metar_age_min.
It also contains the ten archived METAR CSVs, four expected CB arrays and IDs,
expected row digest/airport-month coverage, immutable audit scripts and hashes.
No target, block time, OOF, residual, model or ranking prediction is transferred.

Mac pack verifies the original E205 METAR/FIN1b input hashes and target-free
coverage. It saves the existing E205 join as the expected result. ASUS runs a
separate parser and NumPy backward-asof implementation. It verifies all2,429,888
unique IDs and all four CB columns exactly including NaN positions, recomputes
the canonical little-endian per-row digest, and compares all coverage groups.
It also independently reproduces old current thunder and report age.

Same public data and semantics: latest report at/before observed AOBT_3, missing
fallback MVT−900seconds, inclusive3h tolerance, airport isolation, last source
row for duplicate timestamps, FEW/SCT/BKN/OVC+three-digit-height+CB cloud groups.
No CB yields flags0 and base/amountNaN; absent raw report yields fourNaNs.

Windows ASUS176 only: existing `.venv-prc`, PowerShell7.6.6, psutil AC guard,
affinity CPUs0/2, below-normal priority. Reuse unchanged verified
`adsb_stage_audit_20261007.windows_limit` for kernel hard6GiB committed memory.
At least3GiB free at start; battery checked each airport. Temporary sleep request
is released at exit. Python hard deadline300seconds; PowerShell reports exit
and durable result SHA. Fresh destinations only, all portable files/source SHA
verified before/after, no environment changes, no retry and no future trigger.

This job may finish in seconds. Completion is a durable result with exact-array
PASS; neither a prepared script nor an idle machine is described as training.

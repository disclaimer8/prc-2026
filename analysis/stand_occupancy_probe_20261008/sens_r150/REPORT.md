# Stand-occupancy probe — POST-HOC SENSITIVITY R=150 m (not pre-registered), otherwise identical

Rows files: ['rows_2025-02-06.parquet', 'rows_2025-06-11.parquet', 'rows_2025-07-03.parquet', 'rows_2026-02-05.parquet', 'rows_2026-06-10.parquet', 'rows_2026-07-02.parquet']

## Pool and candidates per airport-month (one day each)

| month | airport | pool (cov<2) | stand coords | base | dwell≥1 | moved≥1 | ≥1 gated | unique (accepted) | unique % of pool | unique (NM-proxy gate) | same call sign |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-02 | EDDM | 8 | 8 | 8 | 3 | 3 | 1 | 1 | 12.5 | 0 | 1 |
| 2025-02 | EGLL | 145 | 145 | 145 | 34 | 29 | 3 | 3 | 2.1 | 2 | 3 |
| 2025-02 | LEBL | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0.0 | 0 | 0 |
| 2025-02 | LEMD | 528 | 526 | 528 | 0 | 0 | 0 | 0 | 0.0 | 0 | 0 |
| 2025-02 | LSZH | 9 | 8 | 9 | 7 | 6 | 1 | 1 | 11.1 | 1 | 1 |
| 2025-06 | EDDM | 12 | 9 | 12 | 5 | 5 | 2 | 2 | 16.7 | 2 | 1 |
| 2025-06 | EGLL | 563 | 554 | 563 | 42 | 2 | 1 | 1 | 0.2 | 1 | 1 |
| 2025-06 | LEBL | 11 | 10 | 11 | 7 | 7 | 2 | 2 | 18.2 | 1 | 2 |
| 2025-06 | LEMD | 38 | 36 | 38 | 4 | 4 | 1 | 1 | 2.6 | 1 | 0 |
| 2025-06 | LSZH | 33 | 30 | 33 | 6 | 6 | 0 | 0 | 0.0 | 0 | 0 |
| 2025-07 | EDDM | 470 | 400 | 470 | 0 | 0 | 0 | 0 | 0.0 | 0 | 0 |
| 2025-07 | EGLL | 435 | 425 | 435 | 27 | 3 | 0 | 0 | 0.0 | 0 | 0 |
| 2025-07 | LEBL | 52 | 45 | 52 | 18 | 12 | 0 | 0 | 0.0 | 0 | 0 |
| 2025-07 | LEMD | 372 | 362 | 372 | 14 | 13 | 6 | 6 | 1.6 | 6 | 6 |
| 2025-07 | LSZH | 45 | 41 | 45 | 11 | 10 | 1 | 1 | 2.2 | 0 | 0 |
| 2026-02 | EDDM | 14 | 12 | 14 | 8 | 7 | 0 | 0 | 0.0 | 0 | 0 |
| 2026-02 | EGLL | 46 | 44 | 45 | 15 | 12 | 0 | 0 | 0.0 | 0 | 0 |
| 2026-02 | LEBL | 6 | 5 | 6 | 1 | 1 | 0 | 0 | 0.0 | 0 | 0 |
| 2026-02 | LEMD | 25 | 24 | 24 | 3 | 2 | 1 | 1 | 4.0 | 1 | 1 |
| 2026-02 | LSZH | 18 | 15 | 12 | 9 | 9 | 1 | 1 | 5.6 | 1 | 1 |
| 2026-06 | EDDM | 14 | 10 | 14 | 6 | 6 | 2 | 1 | 7.1 | 1 | 1 |
| 2026-06 | EGLL | 33 | 32 | 33 | 19 | 15 | 1 | 1 | 3.0 | 1 | 1 |
| 2026-06 | LEBL | 28 | 25 | 26 | 6 | 5 | 1 | 0 | 0.0 | 0 | 0 |
| 2026-06 | LEMD | 88 | 87 | 88 | 8 | 5 | 2 | 2 | 2.3 | 2 | 2 |
| 2026-06 | LSZH | 237 | 229 | 231 | 12 | 4 | 0 | 0 | 0.0 | 0 | 0 |
| 2026-07 | EDDM | 7 | 7 | 7 | 4 | 4 | 1 | 1 | 14.3 | 1 | 1 |
| 2026-07 | EGLL | 32 | 32 | 32 | 23 | 22 | 3 | 3 | 9.4 | 3 | 3 |
| 2026-07 | LEBL | 33 | 27 | 33 | 11 | 11 | 0 | 0 | 0.0 | 0 | 0 |
| 2026-07 | LEMD | 144 | 140 | 144 | 10 | 10 | 3 | 3 | 2.1 | 3 | 2 |
| 2026-07 | LSZH | 15 | 9 | 15 | 4 | 4 | 1 | 1 | 6.7 | 0 | 1 |

## Totals per year

| year | pool | stand coords | ≥1 gated | unique | unique % | unique (NM gate) | same call sign |
|---|---|---|---|---|---|---|---|
| 2025 | 2722 | 2600 | 18 | 18 | 0.7 | 14 | 15 |
| 2026 | 740 | 698 | 16 | 14 | 1.9 | 13 | 13 |

## 2025 accepted candidates against the label

n = 18; candidate implied taxi: median |err| 164 s, within 60 s 16.7 %, median signed -164 s, RMSE 222.9; chain (joint) on the same rows: median |err| 66 s, within 60 s 50.0 %, RMSE 181.0. SSE candidate 8.940e+05 vs chain 5.900e+05 (½-blend 6.325e+05); candidate closer on 27.8 % of rows; same call sign on 15 rows.

| airport | n | cand median abs err | chain median abs err | cand within 60 s | cand SSE | chain SSE |
|---|---|---|---|---|---|---|
| EDDM | 3 | 304 | 59 | 0.0 % | 2.864e+05 | 2.020e+05 |
| EGLL | 4 | 187 | 73 | 0.0 % | 2.516e+05 | 3.810e+04 |
| LEBL | 2 | 325 | 304 | 0.0 % | 2.175e+05 | 2.216e+05 |
| LEMD | 7 | 64 | 45 | 42.9 % | 5.919e+04 | 8.737e+04 |
| LSZH | 2 | 173 | 112 | 0.0 % | 7.927e+04 | 4.088e+04 |

# Final months against 2025 training coverage: label-free input audit (journal 437)

Team gentle-igloo, PRC Data Challenge 2026. Journal 435 item I3. Diagnostic only: no model change, no fit, no upload.
Inputs: the sealed FIN1b feature workspace `data/final_feature_work_mac_20261008/data/features_fin1b.parquet`
(split `train` = 2025, 2,085,047 departures; split `rank` = January, February, June, July 2026, 670,790 departures)
and `data/prc-2026-datasets/final_ranking.parquet` (670,790 departure rows; the file also holds 670,407 arrival
rows). Script: `scripts/final_months_input_audit_20261009.py` (Mac `.venv`, 4 threads, peak RSS 5.6 GB, 2 s after load).
The ADS-B coverage side is in `analysis/final_adsb_derivatives_20261008/RESULTS.md` and is not repeated here.

## Summary

1. **Runways.** No departure runway label of the final months is new at its airport (4 LFPG rows carry `NA`). Runways
   with fewer than 100 departures in 2025 hold at most 0.1 % of any airport-month. Runway *configurations* (the
   `rwy_config` string: departure set | arrival set in the hour) are the only categorical input with real novelty, and
   only at two airports: EHAM 5.8–11.9 % of rows in configurations with < 100 departures in 2025 (3.7–7.9 % never seen
   in 2025), LTFM January/February 2.4–3.5 %; all other airport-months are at or below 3.5 %. Against the same calendar
   month of 2025 the EHAM share rises to 29–53 %, LTFM January/February to 24–27 %.
2. **Context.** The destination ATFM features and the hourly traffic windows stay inside the 2025 1st–99th percentile
   band (at most 1.3 % and 3.3 % outside). Two features leave it: `day_movements` and `wx_temp`. Daily traffic in
   June/July 2026 is above the whole 2025 range on most days at LEMD (+7 % movements/day; 51–69 % of rows above the
   2025 P99), LEBL (+5 %; 32–40 %), LIRF July (+5 %; 44 %) and LSZH July (+5 %; 37 %); January 2026 is below the 2025
   P1 at EHAM (−12 % departures/day; 28 % of rows), EDDM (24 %), LEBL (21 %), LIRF (16 %), EDDF (15 %). June 2026 is
   hotter than any 2025 day at the northern airports (P99 2026 against June 2025: EDDF 40 °C against 33, LFPG 39/34,
   EDDM 35/31, LSZH 35/32, EGLL 34/31, EHAM 34/30); January 2026 is colder than 2025 at EDDM, LSZH, EDDF (12–20 % of rows below P1).
   The any-of share over 18 features is 10–73 % against a 2025 self-rate of 13–15 %; the excess is almost entirely
   these two features. METAR flags are all supported in 2025 at the airport where they occur in 2026 (snow never occurs
   at LEBL/LIRF in either year); exact flag vectors unseen in 2025 cover at most 1.1 % of an airport-month.
3. **NM coverage.** Departures without an NM record are 1.6 / 0.9 / 1.4 / 1.5 % in the four months against
   0.9 / 0.8 / 1.4 / 2.1 % in the same months of 2025 (all 2025: 1.1 %). Only two airport-months exceed every 2025
   month of their airport: EHAM January (5.0 %, 805 rows; 2025 maximum 2.5 %) and LSZH January (3.0 %, 300 rows;
   2025 maximum 2.7 %).
4. **Episodes.** (a) The "LTFM 19–23 February" snow episode of the journal is **19–23 February 2025**
   (`analysis/review3_diagnostics_2026-09-13.md`), i.e. in training: 2,708 rows, 31–68 % of METARs with snow per day,
   daily movements below the 2025 P1. The same window of 2026 (3,426 rows) has 3.5 % snow METARs and is flagged by
   nothing. February 2026 snow at LTFM falls on 2, 3, 18, 22 and 26 February (3,480 rows, 36 % with `wx_snow`); 2025
   holds 3,510 LTFM snow rows on 20 days, so the flag is supported. (b) EHAM taxiway A/B works: **18,194 of 20,881**
   June departure rows fall on 1–26 June (87.1 %). The counts 23,812 / 20,753 in `docs/final-submission.md` §5 and
   `docs/model-card.md` are rows with `ADEP_mvt = EHAM` over both phases (2,931 arrival-phase records included) and
   should be corrected. None of (1)–(3) separates the works window from the rest of the month (configuration < 100:
   9.9 % inside, 13 % outside; context any-of 17.1 % against 13.9 %; no-NM 2.1 % against 3.1 %); runway `09` is used by
   252 June 2026 departures against 0 in June 2025 (7,870 in all of 2025). (c) EDDF February 2026: 14,654 rows; (1)
   3.0 %, (2) 16.1 % against a 14.6 % self-rate, (3) 1.1 % against 0.5 % in February 2025. The ADS-B collapse
   (`seg` 32.2 % against 96.3 %) is invisible in the tabular inputs, as the design intends.
5. **Reading.** The tabular final input is inside the 2025 training support for runways, delay context and weather
   flags. What lies outside is (i) a traffic level shift of ±5–12 % at five airports, carried by one daily-count
   feature, (ii) a June 2026 heat episode, (iii) Schiphol runway configurations, and (iv) two January airport-months
   with an unusually high no-NM share. None of this changes the model; it bounds what a 2025-trained chain was asked
   to extrapolate.

## Proposed paragraph for `docs/final-submission.md` §5 (not applied)

> **Tabular inputs against 2025 coverage.** A label-free audit of the sealed four-month feature table against the
> 2025 training rows (`analysis/final_months_input_audit_20261009.md`) finds no new departure runway at any airport and
> at most 0.1 % of an airport-month on runways with fewer than 100 training departures. Runway configurations are novel
> only at Schiphol (5.8–11.9 % of EHAM rows in configurations with fewer than 100 training departures, 3.7–7.9 % never
> seen in 2025) and marginally at Istanbul in January/February (2.4–3.5 %). Destination ATFM delay and hourly traffic
> features stay inside the 2025 1st–99th percentile band of their airport (at most 1.3 % and 3.3 % outside). Two inputs
> leave it: the daily movement count (June/July 2026 above every 2025 day at LEMD, +7 %, 51–69 % of rows, and at LEBL,
> LIRF, LSZH, +5 %; January 2026 below 2025 at EHAM, −12 % departures per day, and EDDM) and temperature (June 2026
> above the 2025 maximum at EDDF, P99 40 °C against 33, and at LFPG, EDDM, LSZH, EGLL, EHAM). The no-NM share is
> 0.9–1.6 % per month against 0.8–2.1 % in 2025; only EHAM January (5.0 %) and LSZH January (3.0 %) exceed every 2025
> month of their airport. The EHAM works window holds 18,194 of 20,881 June departure rows (23,812 / 20,753 count both
> movement phases) and no tabular input separates it from the rest of June; the EDDF collapse is visible only in ADS-B.

<!-- GENERATED TABLES -->
<!-- TABLES BELOW ARE GENERATED by scripts/final_months_input_audit_20261009.py; the summary above them is written by hand. -->
<!-- generated 2026-10-09T09:17:18+00:00 in 1 s; FIN1b rows train 2,085,047, rank 670,790 -->

## 1. Runway and runway-configuration usage against 2025

Share of final departures whose (airport, `RUNWAY_mvt`) or (airport, `rwy_config`) combination has fewer than 100 departures in 2025 (`<100`) or none at all (`never`); `all` = whole of 2025, `same mo` = same calendar month of 2025. `cat` = distinct values used in the final month.

| Airport | Month | Departures | Rwy cat | Rwy never | Rwy <100 all | Rwy <100 same mo | Cfg cat | Cfg never | Cfg <100 all | Cfg <100 same mo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EDDF | 2026-01 | 15,667 | 6 | 0.0 % | 0.0 % | 0.3 % | 35 | 0.5 % | 1.4 % | 4.7 % |
| EDDF | 2026-02 | 14,654 | 5 | 0.0 % | 0.0 % | 0.3 % | 32 | 1.4 % | 3.0 % | 7.4 % |
| EDDF | 2026-06 | 19,700 | 6 | 0.0 % | 0.0 % | 0.2 % | 32 | 1.1 % | 1.5 % | 3.2 % |
| EDDF | 2026-07 | 20,650 | 6 | 0.0 % | 0.0 % | 0.7 % | 22 | 0.4 % | 0.9 % | 3.6 % |
| EDDM | 2026-01 | 10,953 | 4 | 0.0 % | 0.0 % | 0.0 % | 25 | 0.2 % | 0.7 % | 5.7 % |
| EDDM | 2026-02 | 10,534 | 4 | 0.0 % | 0.0 % | 0.0 % | 29 | 0.4 % | 0.9 % | 9.6 % |
| EDDM | 2026-06 | 14,372 | 4 | 0.0 % | 0.0 % | 0.0 % | 23 | 0.3 % | 1.4 % | 4.4 % |
| EDDM | 2026-07 | 15,013 | 4 | 0.0 % | 0.0 % | 0.0 % | 23 | 0.2 % | 0.3 % | 6.5 % |
| EGLL | 2026-01 | 19,283 | 3 | 0.0 % | 0.0 % | 0.0 % | 17 | 0.4 % | 0.5 % | 2.4 % |
| EGLL | 2026-02 | 17,719 | 3 | 0.0 % | 0.0 % | 0.0 % | 16 | 0.2 % | 0.3 % | 3.3 % |
| EGLL | 2026-06 | 19,662 | 5 | 0.0 % | 0.0 % | 0.0 % | 15 | 0.0 % | 0.1 % | 0.7 % |
| EGLL | 2026-07 | 20,557 | 4 | 0.0 % | 0.0 % | 0.6 % | 16 | 0.8 % | 1.2 % | 1.9 % |
| EHAM | 2026-01 | 16,232 | 9 | 0.0 % | 0.1 % | 6.9 % | 127 | 7.9 % | 11.9 % | 45.4 % |
| EHAM | 2026-02 | 16,802 | 9 | 0.0 % | 0.0 % | 0.0 % | 125 | 4.3 % | 8.6 % | 30.7 % |
| EHAM | 2026-06 | 20,881 | 9 | 0.0 % | 0.0 % | 1.2 % | 107 | 5.2 % | 10.3 % | 52.7 % |
| EHAM | 2026-07 | 21,950 | 10 | 0.0 % | 0.0 % | 1.9 % | 112 | 3.7 % | 5.8 % | 28.9 % |
| LEBL | 2026-01 | 12,201 | 4 | 0.0 % | 0.0 % | 2.1 % | 27 | 0.3 % | 0.9 % | 4.6 % |
| LEBL | 2026-02 | 12,167 | 4 | 0.0 % | 0.0 % | 1.7 % | 21 | 0.0 % | 0.1 % | 6.8 % |
| LEBL | 2026-06 | 17,067 | 4 | 0.0 % | 0.0 % | 0.6 % | 27 | 0.2 % | 0.2 % | 5.1 % |
| LEBL | 2026-07 | 17,879 | 4 | 0.0 % | 0.0 % | 0.0 % | 28 | 0.2 % | 0.4 % | 2.9 % |
| LEMD | 2026-01 | 16,802 | 5 | 0.0 % | 0.0 % | 0.0 % | 35 | 0.1 % | 2.2 % | 7.2 % |
| LEMD | 2026-02 | 16,191 | 6 | 0.0 % | 0.0 % | 0.0 % | 32 | 0.1 % | 1.5 % | 9.3 % |
| LEMD | 2026-06 | 19,335 | 4 | 0.0 % | 0.0 % | 0.0 % | 31 | 0.3 % | 1.8 % | 7.1 % |
| LEMD | 2026-07 | 20,152 | 6 | 0.0 % | 0.0 % | 0.0 % | 30 | 0.6 % | 1.8 % | 7.5 % |
| LFPG | 2026-01 | 18,129 | 8 | 0.0 % | 0.0 % | 0.2 % | 54 | 0.5 % | 1.5 % | 7.6 % |
| LFPG | 2026-02 | 16,934 | 8 | 0.0 % | 0.0 % | 0.5 % | 56 | 0.8 % | 2.2 % | 8.2 % |
| LFPG | 2026-06 | 20,714 | 9 | 0.0 % | 0.0 % | 0.9 % | 70 | 1.3 % | 3.5 % | 9.0 % |
| LFPG | 2026-07 | 21,743 | 9 | 0.0 % | 0.0 % | 0.5 % | 49 | 0.3 % | 1.1 % | 11.7 % |
| LIRF | 2026-01 | 11,061 | 5 | 0.0 % | 0.0 % | 0.2 % | 29 | 0.1 % | 1.2 % | 3.9 % |
| LIRF | 2026-02 | 10,264 | 4 | 0.0 % | 0.0 % | 0.2 % | 24 | 0.0 % | 0.7 % | 2.2 % |
| LIRF | 2026-06 | 14,983 | 5 | 0.0 % | 0.0 % | 0.6 % | 22 | 0.1 % | 0.9 % | 4.2 % |
| LIRF | 2026-07 | 15,838 | 5 | 0.0 % | 0.0 % | 0.6 % | 36 | 0.6 % | 1.5 % | 4.0 % |
| LSZH | 2026-01 | 9,995 | 5 | 0.0 % | 0.0 % | 0.4 % | 34 | 0.1 % | 1.4 % | 9.3 % |
| LSZH | 2026-02 | 9,374 | 6 | 0.0 % | 0.0 % | 1.3 % | 28 | 0.1 % | 0.5 % | 9.3 % |
| LSZH | 2026-06 | 12,384 | 6 | 0.0 % | 0.0 % | 0.8 % | 41 | 1.1 % | 1.9 % | 6.6 % |
| LSZH | 2026-07 | 13,155 | 6 | 0.0 % | 0.0 % | 1.2 % | 33 | 0.3 % | 1.0 % | 5.6 % |
| LTFM | 2026-01 | 22,396 | 8 | 0.0 % | 0.0 % | 7.1 % | 70 | 1.6 % | 2.4 % | 27.4 % |
| LTFM | 2026-02 | 19,519 | 8 | 0.0 % | 0.0 % | 3.0 % | 71 | 2.2 % | 3.5 % | 23.9 % |
| LTFM | 2026-06 | 22,693 | 8 | 0.0 % | 0.0 % | 11.7 % | 54 | 0.4 % | 1.2 % | 15.0 % |
| LTFM | 2026-07 | 25,185 | 8 | 0.0 % | 0.0 % | 0.2 % | 39 | 0.0 % | 0.5 % | 3.3 % |

Runway labels never used in 2025 at that airport:

| Airport | Month | Runway | Departures |
|---|---:|---:|---:|
| LFPG | 2026-06 | `NA` | 2 |
| LFPG | 2026-07 | `NA` | 2 |

## 2. Traffic, destination-delay and METAR context against the 2025 P1-P99 band

For each airport and feature the 1st and 99th percentiles are computed on 2025 (all year, and separately on the same calendar month). `outside` = share of final rows strictly below P1 or above P99. The 2025 self-rate (share of 2025 rows outside their own band; at most 2 %, less for discrete counts because of ties) is the reference. `any-of` = outside on at least one of the 18 numeric features. Nulls are not counted as outside.

Features: traffic `dep_before_60`, `dep_after_60`, `arr_before_60`, `offblocks_around_ob_60`, `taxiing_at_ob`, `takeoffs_between_ob_mvt`, `day_movements`; destination ATFM `dst_dly_per_arr`, `dst_share_dly`, `dst_w_per_arr`, `dst_cap_per_arr`; METAR `wx_temp`, `wx_wind`, `wx_gust`, `wx_vis_mi`, `wx_ceiling_ft`, `wxr_headwind`, `wxr_crosswind`.

### 2a. Any-of share per airport x month

| Airport | Month | Rows | any-of vs all 2025 | 2025 self-rate | any-of vs same month 2025 | 2025 same-month self-rate | worst traffic feature (vs all 2025) | worst delay feature | worst METAR feature |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EDDF | 2026-01 | 15,667 | 33.8 % | 14.6 % | 40.0 % | 14.1 % | 14.5 % `day_movements` | 1.1 % `dst_w_per_arr` | 12.3 % `wx_temp` |
| EDDF | 2026-02 | 14,654 | 16.1 % | 14.6 % | 27.6 % | 12.7 % | 4.8 % `day_movements` | 0.7 % `dst_dly_per_arr` | 3.7 % `wx_vis_mi` |
| EDDF | 2026-06 | 19,700 | 25.8 % | 14.6 % | 65.8 % | 13.7 % | 1.5 % `offblocks_around_ob_60` | 0.8 % `dst_cap_per_arr` | 19.6 % `wx_temp` |
| EDDF | 2026-07 | 20,650 | 12.8 % | 14.6 % | 24.4 % | 12.6 % | 1.4 % `offblocks_around_ob_60` | 1.0 % `dst_cap_per_arr` | 4.2 % `wx_temp` |
| EDDM | 2026-01 | 10,953 | 50.6 % | 15.0 % | 16.0 % | 13.3 % | 23.7 % `day_movements` | 1.1 % `dst_w_per_arr` | 20.1 % `wx_temp` |
| EDDM | 2026-02 | 10,534 | 18.0 % | 15.0 % | 29.3 % | 12.0 % | 2.8 % `day_movements` | 0.9 % `dst_w_per_arr` | 5.5 % `wx_wind` |
| EDDM | 2026-06 | 14,372 | 23.7 % | 15.0 % | 62.8 % | 11.8 % | 1.7 % `dep_before_60` | 0.9 % `dst_cap_per_arr` | 15.2 % `wx_temp` |
| EDDM | 2026-07 | 15,013 | 17.0 % | 15.0 % | 35.1 % | 12.5 % | 1.6 % `arr_before_60` | 1.2 % `dst_cap_per_arr` | 5.6 % `wx_temp` |
| EGLL | 2026-01 | 19,283 | 18.2 % | 13.7 % | 11.2 % | 13.8 % | 3.1 % `day_movements` | 1.0 % `dst_w_per_arr` | 3.8 % `wx_vis_mi` |
| EGLL | 2026-02 | 17,719 | 15.0 % | 13.7 % | 40.6 % | 12.5 % | 3.3 % `day_movements` | 0.8 % `dst_w_per_arr` | 2.9 % `wx_vis_mi` |
| EGLL | 2026-06 | 19,662 | 24.4 % | 13.7 % | 33.6 % | 12.3 % | 2.9 % `day_movements` | 0.7 % `dst_cap_per_arr` | 9.0 % `wx_temp` |
| EGLL | 2026-07 | 20,557 | 15.9 % | 13.7 % | 14.8 % | 13.3 % | 1.5 % `takeoffs_between_ob_mvt` | 0.5 % `dst_cap_per_arr` | 8.8 % `wx_temp` |
| EHAM | 2026-01 | 16,232 | 38.9 % | 15.0 % | 35.3 % | 13.8 % | 27.7 % `day_movements` | 0.8 % `dst_w_per_arr` | 5.5 % `wx_temp` |
| EHAM | 2026-02 | 16,802 | 24.5 % | 15.0 % | 27.3 % | 14.1 % | 9.4 % `day_movements` | 1.0 % `dst_share_dly` | 3.4 % `wxr_headwind` |
| EHAM | 2026-06 | 20,881 | 16.7 % | 15.0 % | 16.0 % | 13.1 % | 1.8 % `takeoffs_between_ob_mvt` | 0.9 % `dst_cap_per_arr` | 6.4 % `wx_temp` |
| EHAM | 2026-07 | 21,950 | 13.3 % | 15.0 % | 26.0 % | 14.3 % | 2.1 % `offblocks_around_ob_60` | 1.0 % `dst_cap_per_arr` | 1.7 % `wx_temp` |
| LEBL | 2026-01 | 12,201 | 35.8 % | 14.4 % | 16.0 % | 13.4 % | 20.9 % `day_movements` | 1.1 % `dst_w_per_arr` | 8.3 % `wxr_crosswind` |
| LEBL | 2026-02 | 12,167 | 24.1 % | 14.4 % | 55.0 % | 11.7 % | 5.4 % `day_movements` | 1.0 % `dst_w_per_arr` | 6.8 % `wxr_crosswind` |
| LEBL | 2026-06 | 17,067 | 41.4 % | 14.4 % | 44.8 % | 12.9 % | 31.5 % `day_movements` | 1.0 % `dst_cap_per_arr` | 1.6 % `wxr_headwind` |
| LEBL | 2026-07 | 17,879 | 51.9 % | 14.4 % | 72.4 % | 12.3 % | 40.0 % `day_movements` | 0.9 % `dst_cap_per_arr` | 3.9 % `wx_temp` |
| LEMD | 2026-01 | 16,802 | 28.2 % | 14.3 % | 37.2 % | 12.7 % | 5.7 % `day_movements` | 0.8 % `dst_w_per_arr` | 5.0 % `wx_vis_mi` |
| LEMD | 2026-02 | 16,191 | 25.9 % | 14.3 % | 79.2 % | 11.4 % | 4.0 % `day_movements` | 0.9 % `dst_w_per_arr` | 9.9 % `wx_wind` |
| LEMD | 2026-06 | 19,335 | 58.3 % | 14.3 % | 80.8 % | 10.9 % | 51.2 % `day_movements` | 0.9 % `dst_cap_per_arr` | 2.8 % `wx_temp` |
| LEMD | 2026-07 | 20,152 | 72.8 % | 14.3 % | 88.3 % | 12.2 % | 68.5 % `day_movements` | 0.7 % `dst_cap_per_arr` | 2.7 % `wx_temp` |
| LFPG | 2026-01 | 18,129 | 20.2 % | 14.7 % | 23.1 % | 12.6 % | 2.7 % `day_movements` | 0.8 % `dst_w_per_arr` | 4.8 % `wx_temp` |
| LFPG | 2026-02 | 16,934 | 17.0 % | 14.7 % | 28.1 % | 12.2 % | 3.2 % `day_movements` | 0.7 % `dst_w_per_arr` | 5.4 % `wxr_headwind` |
| LFPG | 2026-06 | 20,714 | 28.0 % | 14.7 % | 34.1 % | 12.2 % | 1.8 % `offblocks_around_ob_60` | 1.3 % `dst_cap_per_arr` | 17.3 % `wx_temp` |
| LFPG | 2026-07 | 21,743 | 25.5 % | 14.7 % | 21.0 % | 13.3 % | 3.2 % `offblocks_around_ob_60` | 0.9 % `dst_cap_per_arr` | 12.7 % `wx_temp` |
| LIRF | 2026-01 | 11,061 | 31.2 % | 13.9 % | 32.1 % | 12.3 % | 16.2 % `day_movements` | 0.9 % `dst_w_per_arr` | 5.7 % `wx_temp` |
| LIRF | 2026-02 | 10,264 | 20.5 % | 13.9 % | 42.0 % | 11.3 % | 9.1 % `day_movements` | 0.7 % `dst_w_per_arr` | 3.7 % `wx_vis_mi` |
| LIRF | 2026-06 | 14,983 | 29.3 % | 13.9 % | 38.3 % | 11.7 % | 21.1 % `day_movements` | 1.0 % `dst_cap_per_arr` | 0.9 % `wx_temp` |
| LIRF | 2026-07 | 15,838 | 50.8 % | 13.9 % | 12.5 % | 12.7 % | 43.5 % `day_movements` | 1.3 % `dst_cap_per_arr` | 2.8 % `wx_temp` |
| LSZH | 2026-01 | 9,995 | 31.6 % | 13.2 % | 29.2 % | 12.4 % | 5.5 % `day_movements` | 1.3 % `dst_w_per_arr` | 13.6 % `wx_temp` |
| LSZH | 2026-02 | 9,374 | 10.3 % | 13.2 % | 39.5 % | 12.0 % | 1.4 % `offblocks_around_ob_60` | 1.1 % `dst_w_per_arr` | 3.0 % `wxr_crosswind` |
| LSZH | 2026-06 | 12,384 | 35.6 % | 13.2 % | 38.5 % | 11.9 % | 14.3 % `day_movements` | 0.9 % `dst_cap_per_arr` | 15.1 % `wx_temp` |
| LSZH | 2026-07 | 13,155 | 45.9 % | 13.2 % | 57.6 % | 12.7 % | 36.9 % `day_movements` | 0.8 % `dst_cap_per_arr` | 5.4 % `wx_temp` |
| LTFM | 2026-01 | 22,396 | 24.1 % | 14.3 % | 64.1 % | 11.9 % | 2.6 % `day_movements` | 0.7 % `dst_w_per_arr` | 8.8 % `wxr_headwind` |
| LTFM | 2026-02 | 19,519 | 17.8 % | 14.3 % | 44.5 % | 12.8 % | 2.9 % `taxiing_at_ob` | 0.5 % `dst_w_per_arr` | 4.4 % `wx_vis_mi` |
| LTFM | 2026-06 | 22,693 | 10.4 % | 14.3 % | 19.4 % | 13.0 % | 2.4 % `dep_before_60` | 0.2 % `dst_cap_per_arr` | 0.5 % `wxr_headwind` |
| LTFM | 2026-07 | 25,185 | 18.1 % | 14.3 % | 24.5 % | 12.3 % | 6.7 % `day_movements` | 0.1 % `dst_w_per_arr` | 0.6 % `wxr_headwind` |

### 2b. Per-feature outside share against all of 2025 (airport-months where share >= 5 % and > 2 x self-rate)

| Airport | Month | Feature | outside | 2025 self-rate | rows below P1 | rows above P99 |
|---|---:|---:|---:|---:|---:|---:|
| LEMD | 2026-07 | `day_movements` | 68.5 % | 1.7 % | 25 | 13771 |
| LEMD | 2026-06 | `day_movements` | 51.2 % | 1.7 % | 0 | 9906 |
| LIRF | 2026-07 | `day_movements` | 43.5 % | 1.6 % | 5 | 6881 |
| LEBL | 2026-07 | `day_movements` | 40.0 % | 1.8 % | 12 | 7144 |
| LSZH | 2026-07 | `day_movements` | 36.9 % | 1.8 % | 0 | 4857 |
| LEBL | 2026-06 | `day_movements` | 31.5 % | 1.8 % | 0 | 5376 |
| EHAM | 2026-01 | `day_movements` | 27.7 % | 1.8 % | 4494 | 0 |
| EDDM | 2026-01 | `day_movements` | 23.7 % | 1.7 % | 2599 | 0 |
| LIRF | 2026-06 | `day_movements` | 21.1 % | 1.6 % | 0 | 3162 |
| LEBL | 2026-01 | `day_movements` | 20.9 % | 1.8 % | 2546 | 0 |
| EDDM | 2026-01 | `wx_temp` | 20.1 % | 1.7 % | 2197 | 0 |
| EDDF | 2026-06 | `wx_temp` | 19.6 % | 1.7 % | 0 | 3865 |
| LFPG | 2026-06 | `wx_temp` | 17.3 % | 1.5 % | 0 | 3584 |
| LIRF | 2026-01 | `day_movements` | 16.2 % | 1.6 % | 1787 | 0 |
| EDDM | 2026-06 | `wx_temp` | 15.2 % | 1.7 % | 0 | 2184 |
| LSZH | 2026-06 | `wx_temp` | 15.1 % | 1.5 % | 0 | 1868 |
| EDDF | 2026-01 | `day_movements` | 14.5 % | 1.9 % | 2265 | 0 |
| LSZH | 2026-06 | `day_movements` | 14.3 % | 1.8 % | 0 | 1776 |
| LSZH | 2026-01 | `wx_temp` | 13.6 % | 1.5 % | 1362 | 0 |
| LFPG | 2026-07 | `wx_temp` | 12.7 % | 1.5 % | 0 | 2767 |
| EDDF | 2026-01 | `wx_temp` | 12.3 % | 1.7 % | 1929 | 0 |
| LEMD | 2026-02 | `wx_wind` | 9.9 % | 0.9 % | 0 | 1598 |
| EHAM | 2026-02 | `day_movements` | 9.4 % | 1.8 % | 1581 | 0 |
| LIRF | 2026-02 | `day_movements` | 9.1 % | 1.6 % | 930 | 0 |
| EGLL | 2026-06 | `wx_temp` | 9.0 % | 1.4 % | 0 | 1765 |
| EGLL | 2026-07 | `wx_temp` | 8.8 % | 1.4 % | 0 | 1809 |
| LTFM | 2026-01 | `wxr_headwind` | 8.8 % | 1.8 % | 18 | 1955 |
| LEMD | 2026-02 | `wxr_crosswind` | 8.5 % | 0.8 % | 0 | 1374 |
| LEBL | 2026-01 | `wxr_crosswind` | 8.3 % | 1.0 % | 0 | 1018 |
| LEBL | 2026-02 | `wxr_crosswind` | 6.8 % | 1.0 % | 0 | 823 |
| LTFM | 2026-07 | `day_movements` | 6.7 % | 1.9 % | 0 | 1683 |
| EHAM | 2026-06 | `wx_temp` | 6.4 % | 1.6 % | 0 | 1338 |
| LEBL | 2026-01 | `wx_temp` | 6.2 % | 1.2 % | 751 | 0 |
| LTFM | 2026-01 | `wx_wind` | 5.9 % | 1.1 % | 27 | 1297 |
| LEMD | 2026-01 | `day_movements` | 5.7 % | 1.7 % | 951 | 0 |
| LIRF | 2026-01 | `wx_temp` | 5.7 % | 1.4 % | 635 | 0 |
| LTFM | 2026-01 | `wx_temp` | 5.7 % | 1.3 % | 1286 | 0 |
| EDDM | 2026-07 | `wx_temp` | 5.6 % | 1.7 % | 0 | 845 |
| EDDM | 2026-02 | `wx_wind` | 5.5 % | 1.4 % | 120 | 459 |
| EHAM | 2026-01 | `wx_temp` | 5.5 % | 1.6 % | 891 | 0 |
| LSZH | 2026-01 | `day_movements` | 5.5 % | 1.8 % | 549 | 0 |
| EDDM | 2026-02 | `wxr_headwind` | 5.4 % | 2.0 % | 91 | 476 |
| LEBL | 2026-02 | `day_movements` | 5.4 % | 1.8 % | 662 | 0 |
| LFPG | 2026-02 | `wxr_headwind` | 5.4 % | 1.9 % | 120 | 794 |
| LSZH | 2026-07 | `wx_temp` | 5.4 % | 1.5 % | 0 | 711 |
| EGLL | 2026-06 | `wxr_headwind` | 5.3 % | 1.8 % | 49 | 991 |
| LIRF | 2026-01 | `wxr_headwind` | 5.1 % | 1.9 % | 115 | 445 |

### 2b'. What drives the out-of-range rows: daily movements and temperature, 2026 against the same month of 2025

`day_movements` = departures + arrivals of the airport on the local day (`src/features.py`); the 2025 P99 of the airport is a few per cent above the 2025 median, so a traffic change of that size moves most of a month outside the band. `dep/day` = final departure rows per calendar day. Temperature in degrees C.

| Airport | Month | day_movements median 2026 / 2025 | max 2026 / P99 all 2025 / max 2025 | dep/day 2026 / 2025 | temp median 2026 / 2025 | temp P99 2026 / same month 2025 / all 2025 |
|---|---:|---:|---:|---:|---:|---:|
| EDDF | 2026-01 | 1021 / 1006 (+1.5 %) | 1101 / 1449 / 1468 | 505 / 500 (+1.1 %) | 1 / 3 | 12 / 13 / 32 |
| EDDF | 2026-02 | 1080 / 1076 (+0.4 %) | 1168 / 1449 / 1468 | 523 / 539 (-2.9 %) | 5 / 4 | 17 / 14 / 32 |
| EDDF | 2026-06 | 1328 / 1379 (-3.7 %) | 1396 / 1449 / 1468 | 657 / 690 (-4.9 %) | 23 / 22 | 40 / 33 / 32 |
| EDDF | 2026-07 | 1333 / 1382 (-3.5 %) | 1371 / 1449 / 1468 | 666 / 688 (-3.2 %) | 25 / 21 | 36 / 37 / 32 |
| EDDM | 2026-01 | 713 / 732 (-2.6 %) | 774 / 1127 / 1208 | 353 / 363 (-2.8 %) | -1 / 1 | 9 / 11 / 30 |
| EDDM | 2026-02 | 751 / 790 (-4.9 %) | 863 / 1127 / 1208 | 376 / 371 (+1.5 %) | 4 / 1 | 16 / 13 / 30 |
| EDDM | 2026-06 | 965 / 1044 (-7.6 %) | 1020 / 1127 / 1208 | 479 / 522 (-8.2 %) | 22 / 22 | 35 / 31 / 30 |
| EDDM | 2026-07 | 973 / 1027 (-5.3 %) | 1016 / 1127 / 1208 | 484 / 510 (-5.1 %) | 23 / 19 | 36 / 32 / 30 |
| EGLL | 2026-01 | 1234 / 1257 (-1.8 %) | 1318 / 1371 / 1375 | 622 / 626 (-0.7 %) | 7 / 5 | 11 / 12 / 29 |
| EGLL | 2026-02 | 1273 / 1287 (-1.1 %) | 1341 / 1371 / 1375 | 633 / 641 (-1.3 %) | 9 / 6 | 16 / 14 / 29 |
| EGLL | 2026-06 | 1319 / 1344 (-1.9 %) | 1355 / 1371 / 1375 | 655 / 669 (-2.0 %) | 20 / 21 | 34 / 31 / 29 |
| EGLL | 2026-07 | 1332 / 1354 (-1.6 %) | 1350 / 1371 / 1375 | 663 / 671 (-1.1 %) | 24 / 21 | 33 / 31 / 29 |
| EHAM | 2026-01 | 1147 / 1205 (-4.8 %) | 1244 / 1504 / 1517 | 524 / 595 (-12.1 %) | 3 / 4 | 10 / 11 / 28 |
| EHAM | 2026-02 | 1217 / 1282 (-5.1 %) | 1342 / 1504 / 1517 | 600 / 638 (-5.9 %) | 6 / 4 | 16 / 16 / 28 |
| EHAM | 2026-06 | 1398 / 1444 (-3.2 %) | 1460 / 1504 / 1517 | 696 / 709 (-1.8 %) | 19 / 19 | 34 / 30 / 28 |
| EHAM | 2026-07 | 1420 / 1455 (-2.4 %) | 1457 / 1504 / 1517 | 708 / 725 (-2.4 %) | 20 / 20 | 31 / 34 / 28 |
| LEBL | 2026-01 | 824 / 816 (+1.0 %) | 974 / 1155 / 1180 | 394 / 395 (-0.3 %) | 11 / 11 | 17 / 19 / 31 |
| LEBL | 2026-02 | 907 / 889 (+2.0 %) | 1018 / 1155 / 1180 | 435 / 435 (-0.0 %) | 13 / 12 | 21 / 15 / 31 |
| LEBL | 2026-06 | 1129 / 1094 (+3.2 %) | 1242 / 1155 / 1180 | 569 / 544 (+4.6 %) | 25 / 26 | 31 / 33 / 31 |
| LEBL | 2026-07 | 1147 / 1091 (+5.1 %) | 1207 / 1155 / 1180 | 577 / 540 (+6.7 %) | 28 / 27 | 34 / 32 / 31 |
| LEMD | 2026-01 | 1107 / 1077 (+2.8 %) | 1223 / 1283 / 1300 | 542 / 533 (+1.8 %) | 7 / 9 | 12 / 16 / 38 |
| LEMD | 2026-02 | 1167 / 1096 (+6.5 %) | 1291 / 1283 / 1300 | 578 / 550 (+5.2 %) | 10 / 10 | 21 / 18 / 38 |
| LEMD | 2026-06 | 1284 / 1193 (+7.6 %) | 1364 / 1283 / 1300 | 644 / 600 (+7.5 %) | 29 / 29 | 39 / 39 / 38 |
| LEMD | 2026-07 | 1300 / 1213 (+7.2 %) | 1345 / 1283 / 1300 | 650 / 607 (+7.1 %) | 31 / 29 | 39 / 38 / 38 |
| LFPG | 2026-01 | 1167 / 1130 (+3.3 %) | 1300 / 1494 / 1498 | 585 / 577 (+1.3 %) | 6 / 4 | 12 / 12 / 31 |
| LFPG | 2026-02 | 1210 / 1222 (-1.0 %) | 1315 / 1494 / 1498 | 605 / 600 (+0.8 %) | 9 / 5 | 18 / 16 / 31 |
| LFPG | 2026-06 | 1386 / 1387 (-0.1 %) | 1445 / 1494 / 1498 | 690 / 696 (-0.8 %) | 23 / 22 | 39 / 34 / 31 |
| LFPG | 2026-07 | 1404 / 1424 (-1.4 %) | 1468 / 1494 / 1498 | 701 / 700 (+0.1 %) | 25 / 21 | 36 / 37 / 31 |
| LIRF | 2026-01 | 741 / 758 (-2.2 %) | 881 / 1031 / 1098 | 357 / 365 (-2.3 %) | 11 / 12 | 16 / 17 / 33 |
| LIRF | 2026-02 | 738 / 777 (-5.0 %) | 865 / 1031 / 1098 | 367 / 382 (-4.1 %) | 13 / 12 | 17 / 16 / 33 |
| LIRF | 2026-06 | 1001 / 983 (+1.8 %) | 1054 / 1031 / 1098 | 499 / 490 (+2.0 %) | 26 / 27 | 33 / 34 / 33 |
| LIRF | 2026-07 | 1026 / 981 (+4.6 %) | 1084 / 1031 / 1098 | 511 / 491 (+4.1 %) | 30 / 27 | 34 / 34 / 33 |
| LSZH | 2026-01 | 623 / 622 (+0.2 %) | 793 / 865 / 877 | 322 / 314 (+2.7 %) | 0 / 2 | 10 / 13 / 31 |
| LSZH | 2026-02 | 666 / 650 (+2.5 %) | 760 / 865 / 877 | 335 / 323 (+3.6 %) | 6 / 2 | 18 / 13 / 31 |
| LSZH | 2026-06 | 832 / 814 (+2.2 %) | 889 / 865 / 877 | 413 / 403 (+2.5 %) | 23 / 22 | 35 / 32 / 31 |
| LSZH | 2026-07 | 856 / 815 (+5.0 %) | 906 / 865 / 877 | 424 / 407 (+4.3 %) | 25 / 20 | 36 / 32 / 31 |
| LTFM | 2026-01 | 1444 / 1381 (+4.6 %) | 1547 / 1683 / 1695 | 722 / 689 (+4.8 %) | 8 / 8 | 17 / 17 / 31 |
| LTFM | 2026-02 | 1411 / 1357 (+4.0 %) | 1512 / 1683 / 1695 | 697 / 655 (+6.5 %) | 8 / 5 | 21 / 12 / 31 |
| LTFM | 2026-06 | 1514 / 1572 (-3.7 %) | 1672 / 1683 / 1695 | 756 / 783 (-3.4 %) | 23 / 23 | 30 / 31 / 31 |
| LTFM | 2026-07 | 1630 / 1631 (-0.1 %) | 1696 / 1683 / 1695 | 812 / 812 (+0.1 %) | 25 / 26 | 31 / 37 / 31 |

### 2c. METAR flags: share of rows with the flag set (final month / same month 2025 / all 2025) and 2025 support

| Airport | Month | `wx_snow` | `wx_freezing` | `wx_rain` | `wx_thunder` | `wx_fog` | `wx_icing` | `wx_snow_recent` | `wx_icing_recent` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EDDF | 2026-01 | 7.1 / 0.7 / 0.2 (n25=512) | 0.6 / 0.1 / 0.1 (n25=164) | 15.3 / 11.8 / 11.7 (n25=27,022) | 0.0 / 0.0 / 0.3 (n25=665) | 10.6 / 6.5 / 2.8 (n25=6,457) | 52.3 / 38.3 / 7.4 (n25=17,123) | 13.4 / 3.5 / 0.9 (n25=2,143) | 66.6 / 50.7 / 11.5 (n25=26,484) |
| EDDF | 2026-02 | 2.3 / 0.8 / 0.2 (n25=512) | 0.0 / 0.1 / 0.1 (n25=164) | 21.9 / 9.5 / 11.7 (n25=27,022) | 0.0 / 0.0 / 0.3 (n25=665) | 9.9 / 5.3 / 2.8 (n25=6,457) | 20.4 / 18.4 / 7.4 (n25=17,123) | 4.9 / 2.6 / 0.9 (n25=2,143) | 33.4 / 30.6 / 11.5 (n25=26,484) |
| EDDF | 2026-06 | 0.0 / 0.2 / 0.2 (n25=512) | 0.0 / 0.0 / 0.1 (n25=164) | 12.4 / 11.6 / 11.7 (n25=27,022) | 0.5 / 0.6 / 0.3 (n25=665) | 0.0 / 0.3 / 2.8 (n25=6,457) | 0.0 / 0.0 / 7.4 (n25=17,123) | 0.0 / 1.2 / 0.9 (n25=2,143) | 0.0 / 0.0 / 11.5 (n25=26,484) |
| EDDF | 2026-07 | 0.0 / 0.2 / 0.2 (n25=512) | 0.0 / 0.0 / 0.1 (n25=164) | 4.5 / 19.4 / 11.7 (n25=27,022) | 0.0 / 1.6 / 0.3 (n25=665) | 0.0 / 0.4 / 2.8 (n25=6,457) | 0.0 / 0.0 / 7.4 (n25=17,123) | 0.0 / 0.9 / 0.9 (n25=2,143) | 0.0 / 0.0 / 11.5 (n25=26,484) |
| EDDM | 2026-01 | 6.8 / 3.0 / 0.8 (n25=1,328) | 4.3 / 2.4 / 0.7 (n25=1,155) | 6.2 / 11.2 / 14.2 (n25=23,829) | 0.0 / 0.0 / 0.4 (n25=705) | 26.8 / 15.1 / 7.1 (n25=11,888) | 62.7 / 58.1 / 16.6 (n25=27,831) | 11.5 / 6.8 / 1.9 (n25=3,134) | 75.7 / 72.5 / 22.4 (n25=37,430) |
| EDDM | 2026-02 | 3.9 / 4.3 / 0.8 (n25=1,328) | 0.8 / 0.1 / 0.7 (n25=1,155) | 17.8 / 6.2 / 14.2 (n25=23,829) | 0.0 / 0.0 / 0.4 (n25=705) | 12.9 / 8.4 / 7.1 (n25=11,888) | 41.6 / 45.1 / 16.6 (n25=27,831) | 8.2 / 7.9 / 1.9 (n25=3,134) | 56.2 / 61.0 / 22.4 (n25=37,430) |
| EDDM | 2026-06 | 0.0 / 0.1 / 0.8 (n25=1,328) | 0.0 / 0.0 / 0.7 (n25=1,155) | 16.4 / 15.2 / 14.2 (n25=23,829) | 1.0 / 1.4 / 0.4 (n25=705) | 0.9 / 0.3 / 7.1 (n25=11,888) | 0.0 / 0.0 / 16.6 (n25=27,831) | 0.0 / 0.4 / 1.9 (n25=3,134) | 0.0 / 0.0 / 22.4 (n25=37,430) |
| EDDM | 2026-07 | 0.1 / 0.1 / 0.8 (n25=1,328) | 0.0 / 0.0 / 0.7 (n25=1,155) | 8.1 / 23.7 / 14.2 (n25=23,829) | 0.9 / 1.1 / 0.4 (n25=705) | 0.3 / 1.4 / 7.1 (n25=11,888) | 0.0 / 0.0 / 16.6 (n25=27,831) | 0.5 / 0.5 / 1.9 (n25=3,134) | 0.0 / 0.0 / 22.4 (n25=37,430) |
| EGLL | 2026-01 | 0.1 / 0.6 / 0.1 (n25=138) | 0.0 / 0.1 / 0.0 (n25=66) | 20.4 / 13.9 / 8.4 (n25=20,028) | 0.0 / 0.1 / 0.1 (n25=226) | 2.8 / 2.5 / 0.8 (n25=1,830) | 12.0 / 33.1 / 5.1 (n25=12,283) | 1.0 / 1.2 / 0.2 (n25=370) | 19.3 / 42.9 / 7.7 (n25=18,552) |
| EGLL | 2026-02 | 0.0 / 0.0 / 0.1 (n25=138) | 0.0 / 0.0 / 0.0 (n25=66) | 29.3 / 15.1 / 8.4 (n25=20,028) | 0.0 / 0.0 / 0.1 (n25=226) | 0.5 / 1.3 / 0.8 (n25=1,830) | 1.0 / 12.4 / 5.1 (n25=12,283) | 0.0 / 0.0 / 0.2 (n25=370) | 1.5 / 21.0 / 7.7 (n25=18,552) |
| EGLL | 2026-06 | 0.0 / 0.0 / 0.1 (n25=138) | 0.0 / 0.0 / 0.0 (n25=66) | 9.1 / 4.7 / 8.4 (n25=20,028) | 0.4 / 0.0 / 0.1 (n25=226) | 0.0 / 0.0 / 0.8 (n25=1,830) | 0.0 / 0.0 / 5.1 (n25=12,283) | 0.0 / 0.0 / 0.2 (n25=370) | 0.0 / 0.0 / 7.7 (n25=18,552) |
| EGLL | 2026-07 | 0.1 / 0.0 / 0.1 (n25=138) | 0.0 / 0.0 / 0.0 (n25=66) | 0.3 / 7.1 / 8.4 (n25=20,028) | 0.0 / 0.1 / 0.1 (n25=226) | 0.0 / 0.0 / 0.8 (n25=1,830) | 0.0 / 0.0 / 5.1 (n25=12,283) | 0.6 / 0.0 / 0.2 (n25=370) | 0.0 / 0.0 / 7.7 (n25=18,552) |
| EHAM | 2026-01 | 7.0 / 2.0 / 0.5 (n25=1,136) | 0.0 / 2.2 / 0.3 (n25=795) | 19.1 / 27.9 / 15.5 (n25=38,466) | 0.0 / 0.1 / 0.1 (n25=175) | 19.2 / 24.3 / 5.1 (n25=12,701) | 48.7 / 42.3 / 7.0 (n25=17,476) | 13.7 / 9.8 / 1.6 (n25=4,050) | 58.7 / 54.6 / 10.7 (n25=26,450) |
| EHAM | 2026-02 | 1.0 / 1.0 / 0.5 (n25=1,136) | 0.0 / 0.7 / 0.3 (n25=795) | 23.4 / 12.1 / 15.5 (n25=38,466) | 0.0 / 0.0 / 0.1 (n25=175) | 14.5 / 12.5 / 5.1 (n25=12,701) | 18.1 / 24.5 / 7.0 (n25=17,476) | 1.6 / 1.8 / 1.6 (n25=4,050) | 25.7 / 39.2 / 10.7 (n25=26,450) |
| EHAM | 2026-06 | 0.0 / 0.0 / 0.5 (n25=1,136) | 0.0 / 0.0 / 0.3 (n25=795) | 12.4 / 9.5 / 15.5 (n25=38,466) | 1.1 / 0.0 / 0.1 (n25=175) | 0.0 / 0.2 / 5.1 (n25=12,701) | 0.0 / 0.0 / 7.0 (n25=17,476) | 0.1 / 0.0 / 1.6 (n25=4,050) | 0.0 / 0.0 / 10.7 (n25=26,450) |
| EHAM | 2026-07 | 0.0 / 0.0 / 0.5 (n25=1,136) | 0.0 / 0.0 / 0.3 (n25=795) | 5.1 / 13.2 / 15.5 (n25=38,466) | 0.0 / 0.1 / 0.1 (n25=175) | 0.3 / 0.2 / 5.1 (n25=12,701) | 0.0 / 0.0 / 7.0 (n25=17,476) | 0.0 / 0.0 / 1.6 (n25=4,050) | 0.0 / 0.0 / 10.7 (n25=26,450) |
| LEBL | 2026-01 | 0.0 / 0.0 / 0.0 (n25=0) | 0.0 / 0.0 / 0.0 (n25=0) | 10.7 / 1.7 / 6.6 (n25=11,878) | 0.0 / 0.0 / 0.7 (n25=1,282) | 0.6 / 0.0 / 1.7 (n25=2,974) | 0.7 / 0.0 / 0.0 (n25=26) | 0.0 / 0.0 / 0.0 (n25=0) | 1.5 / 0.0 / 0.1 (n25=91) |
| LEBL | 2026-02 | 0.0 / 0.0 / 0.0 (n25=0) | 0.0 / 0.0 / 0.0 (n25=0) | 6.9 / 6.7 / 6.6 (n25=11,878) | 0.0 / 0.0 / 0.7 (n25=1,282) | 7.1 / 0.0 / 1.7 (n25=2,974) | 0.0 / 0.2 / 0.0 (n25=26) | 0.0 / 0.0 / 0.0 (n25=0) | 0.0 / 0.7 / 0.1 (n25=91) |
| LEBL | 2026-06 | 0.0 / 0.0 / 0.0 (n25=0) | 0.0 / 0.0 / 0.0 (n25=0) | 0.3 / 0.4 / 6.6 (n25=11,878) | 0.1 / 0.0 / 0.7 (n25=1,282) | 0.2 / 0.0 / 1.7 (n25=2,974) | 0.0 / 0.0 / 0.0 (n25=26) | 0.0 / 0.0 / 0.0 (n25=0) | 0.0 / 0.0 / 0.1 (n25=91) |
| LEBL | 2026-07 | 0.0 / 0.0 / 0.0 (n25=0) | 0.0 / 0.0 / 0.0 (n25=0) | 0.5 / 5.4 / 6.6 (n25=11,878) | 0.2 / 1.0 / 0.7 (n25=1,282) | 0.1 / 0.0 / 1.7 (n25=2,974) | 0.0 / 0.0 / 0.0 (n25=26) | 0.0 / 0.0 / 0.0 (n25=0) | 0.0 / 0.0 / 0.1 (n25=91) |
| LEMD | 2026-01 | 0.1 / 0.0 / 0.0 (n25=2) | 0.0 / 0.0 / 0.0 (n25=6) | 13.6 / 9.8 / 6.3 (n25=13,379) | 0.0 / 0.0 / 0.7 (n25=1,381) | 8.2 / 9.6 / 1.3 (n25=2,707) | 16.1 / 9.3 / 3.7 (n25=7,946) | 0.9 / 0.0 / 0.1 (n25=114) | 26.7 / 16.1 / 6.7 (n25=14,203) |
| LEMD | 2026-02 | 0.0 / 0.0 / 0.0 (n25=2) | 0.0 / 0.0 / 0.0 (n25=6) | 13.3 / 1.7 / 6.3 (n25=13,379) | 0.4 / 0.0 / 0.7 (n25=1,381) | 3.5 / 1.2 / 1.3 (n25=2,707) | 6.6 / 13.4 / 3.7 (n25=7,946) | 0.0 / 0.0 / 0.1 (n25=114) | 12.0 / 25.2 / 6.7 (n25=14,203) |
| LEMD | 2026-06 | 0.0 / 0.0 / 0.0 (n25=2) | 0.0 / 0.0 / 0.0 (n25=6) | 1.7 / 2.7 / 6.3 (n25=13,379) | 0.6 / 2.3 / 0.7 (n25=1,381) | 0.0 / 0.3 / 1.3 (n25=2,707) | 0.0 / 0.0 / 3.7 (n25=7,946) | 0.0 / 0.0 / 0.1 (n25=114) | 0.0 / 0.0 / 6.7 (n25=14,203) |
| LEMD | 2026-07 | 0.0 / 0.0 / 0.0 (n25=2) | 0.0 / 0.0 / 0.0 (n25=6) | 0.9 / 0.7 / 6.3 (n25=13,379) | 0.3 / 0.5 / 0.7 (n25=1,381) | 0.0 / 0.0 / 1.3 (n25=2,707) | 0.0 / 0.0 / 3.7 (n25=7,946) | 0.0 / 0.0 / 0.1 (n25=114) | 0.0 / 0.0 / 6.7 (n25=14,203) |
| LFPG | 2026-01 | 1.4 / 0.7 / 0.1 (n25=198) | 0.1 / 2.1 / 0.3 (n25=741) | 12.5 / 15.7 / 7.8 (n25=18,746) | 0.0 / 0.1 / 0.2 (n25=413) | 11.9 / 19.8 / 4.9 (n25=11,629) | 25.4 / 40.2 / 7.4 (n25=17,732) | 2.8 / 2.5 / 0.3 (n25=835) | 32.8 / 48.5 / 10.1 (n25=24,144) |
| LFPG | 2026-02 | 0.5 / 0.1 / 0.1 (n25=198) | 0.0 / 1.4 / 0.3 (n25=741) | 21.4 / 8.1 / 7.8 (n25=18,746) | 0.0 / 0.0 / 0.2 (n25=413) | 5.4 / 13.3 / 4.9 (n25=11,629) | 5.8 / 21.7 / 7.4 (n25=17,732) | 1.4 / 0.7 / 0.3 (n25=835) | 8.6 / 33.6 / 10.1 (n25=24,144) |
| LFPG | 2026-06 | 0.0 / 0.0 / 0.1 (n25=198) | 0.0 / 0.0 / 0.3 (n25=741) | 4.4 / 4.5 / 7.8 (n25=18,746) | 0.9 / 0.3 / 0.2 (n25=413) | 0.2 / 0.0 / 4.9 (n25=11,629) | 0.0 / 0.0 / 7.4 (n25=17,732) | 0.0 / 0.0 / 0.3 (n25=835) | 0.0 / 0.0 / 10.1 (n25=24,144) |
| LFPG | 2026-07 | 0.0 / 0.0 / 0.1 (n25=198) | 0.0 / 0.0 / 0.3 (n25=741) | 0.7 / 9.3 / 7.8 (n25=18,746) | 0.0 / 0.6 / 0.2 (n25=413) | 0.0 / 1.4 / 4.9 (n25=11,629) | 0.0 / 0.0 / 7.4 (n25=17,732) | 0.0 / 0.0 / 0.3 (n25=835) | 0.0 / 0.0 / 10.1 (n25=24,144) |
| LIRF | 2026-01 | 0.0 / 0.0 / 0.0 (n25=0) | 0.0 / 0.0 / 0.0 (n25=0) | 18.9 / 10.0 / 5.2 (n25=8,389) | 1.0 / 0.2 / 0.3 (n25=418) | 0.6 / 2.1 / 1.2 (n25=1,944) | 3.4 / 0.0 / 0.4 (n25=568) | 0.0 / 0.0 / 0.0 (n25=0) | 5.8 / 0.8 / 1.0 (n25=1,670) |
| LIRF | 2026-02 | 0.0 / 0.0 / 0.0 (n25=0) | 0.0 / 0.0 / 0.0 (n25=0) | 14.9 / 9.1 / 5.2 (n25=8,389) | 0.6 / 0.8 / 0.3 (n25=418) | 3.6 / 3.2 / 1.2 (n25=1,944) | 0.7 / 1.4 / 0.4 (n25=568) | 0.0 / 0.0 / 0.0 (n25=0) | 1.8 / 4.1 / 1.0 (n25=1,670) |
| LIRF | 2026-06 | 0.0 / 0.0 / 0.0 (n25=0) | 0.0 / 0.0 / 0.0 (n25=0) | 1.1 / 0.2 / 5.2 (n25=8,389) | 0.4 / 0.3 / 0.3 (n25=418) | 0.3 / 1.4 / 1.2 (n25=1,944) | 0.0 / 0.0 / 0.4 (n25=568) | 0.0 / 0.0 / 0.0 (n25=0) | 0.0 / 0.0 / 1.0 (n25=1,670) |
| LIRF | 2026-07 | 0.0 / 0.0 / 0.0 (n25=0) | 0.0 / 0.0 / 0.0 (n25=0) | 0.2 / 1.9 / 5.2 (n25=8,389) | 0.1 / 0.5 / 0.3 (n25=418) | 0.8 / 1.1 / 1.2 (n25=1,944) | 0.0 / 0.0 / 0.4 (n25=568) | 0.0 / 0.0 / 0.0 (n25=0) | 0.0 / 0.0 / 1.0 (n25=1,670) |
| LSZH | 2026-01 | 9.3 / 1.2 / 0.5 (n25=646) | 4.5 / 3.5 / 0.4 (n25=488) | 11.6 / 15.3 / 15.9 (n25=21,456) | 0.0 / 0.0 / 0.6 (n25=782) | 32.1 / 16.0 / 7.8 (n25=10,520) | 73.1 / 41.5 / 13.8 (n25=18,646) | 15.8 / 4.8 / 1.3 (n25=1,709) | 83.2 / 54.8 / 18.8 (n25=25,342) |
| LSZH | 2026-02 | 3.3 / 0.9 / 0.5 (n25=646) | 1.0 / 1.0 / 0.4 (n25=488) | 27.6 / 17.5 / 15.9 (n25=21,456) | 0.0 / 0.0 / 0.6 (n25=782) | 12.8 / 21.1 / 7.8 (n25=10,520) | 26.7 / 41.0 / 13.8 (n25=18,646) | 6.0 / 4.3 / 1.3 (n25=1,709) | 38.9 / 55.5 / 18.8 (n25=25,342) |
| LSZH | 2026-06 | 0.0 / 0.0 / 0.5 (n25=646) | 0.0 / 0.0 / 0.4 (n25=488) | 12.3 / 10.8 / 15.9 (n25=21,456) | 1.5 / 1.7 / 0.6 (n25=782) | 0.6 / 0.2 / 7.8 (n25=10,520) | 0.0 / 0.0 / 13.8 (n25=18,646) | 0.3 / 0.0 / 1.3 (n25=1,709) | 0.0 / 0.0 / 18.8 (n25=25,342) |
| LSZH | 2026-07 | 0.0 / 0.0 / 0.5 (n25=646) | 0.0 / 0.0 / 0.4 (n25=488) | 5.0 / 23.0 / 15.9 (n25=21,456) | 1.6 / 1.8 / 0.6 (n25=782) | 0.4 / 0.9 / 7.8 (n25=10,520) | 0.0 / 0.0 / 13.8 (n25=18,646) | 0.0 / 0.0 / 1.3 (n25=1,709) | 0.0 / 0.0 / 18.8 (n25=25,342) |
| LTFM | 2026-01 | 8.6 / 0.0 / 1.3 (n25=3,510) | 0.0 / 0.0 / 0.0 (n25=0) | 31.6 / 18.2 / 10.3 (n25=28,091) | 0.9 / 0.1 / 0.2 (n25=625) | 5.4 / 8.6 / 3.8 (n25=10,257) | 14.5 / 3.5 / 2.1 (n25=5,704) | 11.5 / 0.0 / 2.2 (n25=6,000) | 20.4 / 6.3 / 3.5 (n25=9,577) |
| LTFM | 2026-02 | 6.4 / 16.2 / 1.3 (n25=3,510) | 0.0 / 0.0 / 0.0 (n25=0) | 24.6 / 37.6 / 10.3 (n25=28,091) | 0.0 / 0.6 / 0.2 (n25=625) | 12.3 / 7.0 / 3.8 (n25=10,257) | 8.0 / 23.6 / 2.1 (n25=5,704) | 7.6 / 26.6 / 2.2 (n25=6,000) | 10.7 / 36.4 / 3.5 (n25=9,577) |
| LTFM | 2026-06 | 0.0 / 0.0 / 1.3 (n25=3,510) | 0.0 / 0.0 / 0.0 (n25=0) | 0.7 / 0.1 / 10.3 (n25=28,091) | 0.0 / 0.0 / 0.2 (n25=625) | 0.1 / 0.3 / 3.8 (n25=10,257) | 0.0 / 0.0 / 2.1 (n25=5,704) | 0.0 / 0.0 / 2.2 (n25=6,000) | 0.0 / 0.0 / 3.5 (n25=9,577) |
| LTFM | 2026-07 | 0.0 / 0.0 / 1.3 (n25=3,510) | 0.0 / 0.0 / 0.0 (n25=0) | 4.3 / 0.5 / 10.3 (n25=28,091) | 1.5 / 0.0 / 0.2 (n25=625) | 0.2 / 0.5 / 3.8 (n25=10,257) | 0.0 / 0.0 / 2.1 (n25=5,704) | 0.0 / 0.0 / 2.2 (n25=6,000) | 0.0 / 0.0 / 3.5 (n25=9,577) |

Share of final rows whose exact METAR flag vector (8 flags) has < 100 / 0 rows in 2025 at the airport:

| Airport | Month | flag vector <100 in 2025 | never in 2025 |
|---|---:|---:|---:|
| EDDF | 2026-01 | 5.0 % | 0.5 % |
| EDDF | 2026-02 | 1.0 % | 0.0 % |
| EDDF | 2026-06 | 0.0 % | 0.0 % |
| EDDF | 2026-07 | 0.0 % | 0.0 % |
| EDDM | 2026-01 | 1.4 % | 0.1 % |
| EDDM | 2026-02 | 0.5 % | 0.2 % |
| EDDM | 2026-06 | 0.1 % | 0.0 % |
| EDDM | 2026-07 | 0.7 % | 0.0 % |
| EGLL | 2026-01 | 1.3 % | 0.1 % |
| EGLL | 2026-02 | 0.2 % | 0.0 % |
| EGLL | 2026-06 | 0.2 % | 0.0 % |
| EGLL | 2026-07 | 0.6 % | 0.6 % |
| EHAM | 2026-01 | 1.3 % | 0.0 % |
| EHAM | 2026-02 | 0.8 % | 0.1 % |
| EHAM | 2026-06 | 0.3 % | 0.1 % |
| EHAM | 2026-07 | 0.0 % | 0.0 % |
| LEBL | 2026-01 | 1.5 % | 0.0 % |
| LEBL | 2026-02 | 0.0 % | 0.0 % |
| LEBL | 2026-06 | 0.0 % | 0.0 % |
| LEBL | 2026-07 | 0.0 % | 0.0 % |
| LEMD | 2026-01 | 1.1 % | 0.8 % |
| LEMD | 2026-02 | 0.4 % | 0.0 % |
| LEMD | 2026-06 | 0.0 % | 0.0 % |
| LEMD | 2026-07 | 0.0 % | 0.0 % |
| LFPG | 2026-01 | 0.7 % | 0.2 % |
| LFPG | 2026-02 | 0.1 % | 0.0 % |
| LFPG | 2026-06 | 0.1 % | 0.1 % |
| LFPG | 2026-07 | 0.0 % | 0.0 % |
| LIRF | 2026-01 | 0.5 % | 0.5 % |
| LIRF | 2026-02 | 0.7 % | 0.7 % |
| LIRF | 2026-06 | 0.1 % | 0.1 % |
| LIRF | 2026-07 | 0.0 % | 0.0 % |
| LSZH | 2026-01 | 4.2 % | 0.5 % |
| LSZH | 2026-02 | 2.7 % | 0.0 % |
| LSZH | 2026-06 | 0.3 % | 0.3 % |
| LSZH | 2026-07 | 0.0 % | 0.0 % |
| LTFM | 2026-01 | 0.1 % | 0.1 % |
| LTFM | 2026-02 | 1.7 % | 1.1 % |
| LTFM | 2026-06 | 0.0 % | 0.0 % |
| LTFM | 2026-07 | 0.4 % | 0.0 % |

## 3. Departures without an NM record (`LOBT_flt` null)

| Airport | Month | Departures | no NM | share 2026 | same month 2025 | all 2025 | worst 2025 month |  |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EDDF | 2026-01 | 15,667 | 212 | 1.4 % | 1.1 % | 0.8 % | 2.0 % |  |
| EDDF | 2026-02 | 14,654 | 167 | 1.1 % | 0.5 % | 0.8 % | 2.0 % |  |
| EDDF | 2026-06 | 19,700 | 178 | 0.9 % | 1.4 % | 0.8 % | 2.0 % |  |
| EDDF | 2026-07 | 20,650 | 209 | 1.0 % | 2.0 % | 0.8 % | 2.0 % |  |
| EDDM | 2026-01 | 10,953 | 119 | 1.1 % | 0.6 % | 0.6 % | 1.1 % |  |
| EDDM | 2026-02 | 10,534 | 83 | 0.8 % | 0.3 % | 0.6 % | 1.1 % |  |
| EDDM | 2026-06 | 14,372 | 128 | 0.9 % | 1.0 % | 0.6 % | 1.1 % |  |
| EDDM | 2026-07 | 15,013 | 119 | 0.8 % | 1.1 % | 0.6 % | 1.1 % |  |
| EGLL | 2026-01 | 19,283 | 112 | 0.6 % | 0.7 % | 0.6 % | 1.6 % |  |
| EGLL | 2026-02 | 17,719 | 40 | 0.2 % | 0.2 % | 0.6 % | 1.6 % |  |
| EGLL | 2026-06 | 19,662 | 183 | 0.9 % | 0.6 % | 0.6 % | 1.6 % |  |
| EGLL | 2026-07 | 20,557 | 136 | 0.7 % | 1.6 % | 0.6 % | 1.6 % |  |
| EHAM | 2026-01 | 16,232 | 805 | 5.0 % | 1.6 % | 1.6 % | 2.5 % | above every 2025 month |
| EHAM | 2026-02 | 16,802 | 233 | 1.4 % | 1.4 % | 1.6 % | 2.5 % |  |
| EHAM | 2026-06 | 20,881 | 455 | 2.2 % | 2.0 % | 1.6 % | 2.5 % |  |
| EHAM | 2026-07 | 21,950 | 401 | 1.8 % | 2.3 % | 1.6 % | 2.5 % |  |
| LEBL | 2026-01 | 12,201 | 77 | 0.6 % | 0.5 % | 1.0 % | 2.6 % |  |
| LEBL | 2026-02 | 12,167 | 83 | 0.7 % | 0.4 % | 1.0 % | 2.6 % |  |
| LEBL | 2026-06 | 17,067 | 234 | 1.4 % | 1.3 % | 1.0 % | 2.6 % |  |
| LEBL | 2026-07 | 17,879 | 365 | 2.0 % | 2.6 % | 1.0 % | 2.6 % |  |
| LEMD | 2026-01 | 16,802 | 63 | 0.4 % | 0.2 % | 0.4 % | 1.5 % |  |
| LEMD | 2026-02 | 16,191 | 82 | 0.5 % | 0.1 % | 0.4 % | 1.5 % |  |
| LEMD | 2026-06 | 19,335 | 118 | 0.6 % | 0.6 % | 0.4 % | 1.5 % |  |
| LEMD | 2026-07 | 20,152 | 175 | 0.9 % | 1.5 % | 0.4 % | 1.5 % |  |
| LFPG | 2026-01 | 18,129 | 325 | 1.8 % | 1.2 % | 1.6 % | 3.1 % |  |
| LFPG | 2026-02 | 16,934 | 152 | 0.9 % | 0.7 % | 1.6 % | 3.1 % |  |
| LFPG | 2026-06 | 20,714 | 419 | 2.0 % | 2.2 % | 1.6 % | 3.1 % |  |
| LFPG | 2026-07 | 21,743 | 409 | 1.9 % | 3.1 % | 1.6 % | 3.1 % |  |
| LIRF | 2026-01 | 11,061 | 107 | 1.0 % | 0.5 % | 0.9 % | 2.2 % |  |
| LIRF | 2026-02 | 10,264 | 75 | 0.7 % | 0.5 % | 0.9 % | 2.2 % |  |
| LIRF | 2026-06 | 14,983 | 161 | 1.1 % | 1.3 % | 0.9 % | 2.2 % |  |
| LIRF | 2026-07 | 15,838 | 276 | 1.7 % | 2.2 % | 0.9 % | 2.2 % |  |
| LSZH | 2026-01 | 9,995 | 300 | 3.0 % | 1.8 % | 1.7 % | 2.7 % | above every 2025 month |
| LSZH | 2026-02 | 9,374 | 126 | 1.3 % | 1.2 % | 1.7 % | 2.7 % |  |
| LSZH | 2026-06 | 12,384 | 262 | 2.1 % | 2.2 % | 1.7 % | 2.7 % |  |
| LSZH | 2026-07 | 13,155 | 359 | 2.7 % | 2.7 % | 1.7 % | 2.7 % |  |
| LTFM | 2026-01 | 22,396 | 333 | 1.5 % | 1.1 % | 1.3 % | 2.0 % |  |
| LTFM | 2026-02 | 19,519 | 279 | 1.4 % | 2.0 % | 1.3 % | 2.0 % |  |
| LTFM | 2026-06 | 22,693 | 329 | 1.4 % | 1.2 % | 1.3 % | 2.0 % |  |
| LTFM | 2026-07 | 25,185 | 388 | 1.5 % | 1.7 % | 1.3 % | 2.0 % |  |

Per month, all airports: 2026-01 1.6 % (2025 same month 0.9 %); 2026-02 0.9 % (2025 same month 0.8 %); 2026-06 1.4 % (2025 same month 1.4 %); 2026-07 1.5 % (2025 same month 2.1 %); all 2025 1.1 %.

## 4. Named episodes

### LTFM 19-23 Feb 2026 (same calendar window as the 2025 snow episode)

- Rows: **3,426** of 19,519 in the airport-month (rest of month 16,093).
- (1) Runway with < 100 2025 departures: 0.0 %; configuration < 100: 1.8 % (never seen: 1.4 %).
- (2) Context any-of outside the 2025 band: 18.5 % vs all 2025 (31.6 % vs same month 2025; airport self-rate 14.3 %; rest of the month 17.6 %).
  Largest per-feature shares: `wxr_headwind` 6.3 %, `wx_ceiling_ft` 4.0 %, `takeoffs_between_ob_mvt` 3.1 %, `taxiing_at_ob` 2.8 %.
- METAR flags set in the episode: `wx_snow` 3.5 %, `wx_rain` 7.5 %, `wx_fog` 13.8 %, `wx_icing` 9.9 %, `wx_snow_recent` 5.0 %, `wx_icing_recent` 15.5 %.
  Control LTFM 19-23 Feb 2025 (the snow episode of the journal, in training) (2,708 rows): `wx_snow` 53.1 %, `wx_rain` 56.8 %, `wx_thunder` 3.0 %, `wx_icing` 58.8 %, `wx_snow_recent` 82.1 %, `wx_icing_recent` 83.2 %; any-of 100.0 %.
- (3) No NM record: 1.4 % (same month 2025 2.0 %; rest of the month 1.4 %).

### LTFM Feb 2026, days with any snow METAR (02, 03, 18, 22, 26 Feb)

- Rows: **3,480** of 19,519 in the airport-month (rest of month 16,039).
- (1) Runway with < 100 2025 departures: 0.0 %; configuration < 100: 2.6 % (never seen: 2.4 %).
- (2) Context any-of outside the 2025 band: 18.4 % vs all 2025 (36.2 % vs same month 2025; airport self-rate 14.3 %; rest of the month 17.6 %).
  Largest per-feature shares: `wxr_headwind` 10.7 %, `wx_wind` 3.5 %, `takeoffs_between_ob_mvt` 2.5 %, `taxiing_at_ob` 2.4 %.
- METAR flags set in the episode: `wx_snow` 36.1 %, `wx_rain` 50.2 %, `wx_fog` 11.6 %, `wx_icing` 41.6 %, `wx_snow_recent` 42.8 %, `wx_icing_recent` 53.2 %.
  Control LTFM 2025, rows with wx_snow=1 (3,510 rows): `wx_snow` 100.0 %, `wx_rain` 99.3 %, `wx_thunder` 2.0 %, `wx_fog` 1.2 %, `wx_icing` 81.3 %, `wx_snow_recent` 100.0 %, `wx_icing_recent` 87.3 %; any-of 61.2 %.
- (3) No NM record: 1.5 % (same month 2025 2.0 %; rest of the month 1.4 %).

### EHAM 1-26 Jun 2026 (taxiway A/B works)

- Rows: **18,194** of 20,881 in the airport-month (rest of month 2,687).
- (1) Runway with < 100 2025 departures: 0.0 %; configuration < 100: 9.9 % (never seen: 4.5 %).
- (2) Context any-of outside the 2025 band: 17.1 % vs all 2025 (16.0 % vs same month 2025; airport self-rate 15.0 %; rest of the month 13.9 %).
  Largest per-feature shares: `wx_temp` 7.4 %, `offblocks_around_ob_60` 1.8 %, `takeoffs_between_ob_mvt` 1.8 %, `taxiing_at_ob` 1.6 %.
- METAR flags set in the episode: `wx_snow` 0.0 %, `wx_rain` 13.2 %, `wx_thunder` 1.0 %, `wx_snow_recent` 0.1 %.
  Control EHAM June 2025 (21,257 rows): `wx_rain` 9.5 %, `wx_fog` 0.2 %; any-of 18.7 %.
- (3) No NM record: 2.1 % (same month 2025 2.0 %; rest of the month 3.1 %).

### EDDF Feb 2026 (ADS-B ground-reception collapse)

- Rows: **14,654** of 14,654 in the airport-month (rest of month 0).
- (1) Runway with < 100 2025 departures: 0.0 %; configuration < 100: 3.0 % (never seen: 1.4 %).
- (2) Context any-of outside the 2025 band: 16.1 % vs all 2025 (27.6 % vs same month 2025; airport self-rate 14.6 %).
  Largest per-feature shares: `day_movements` 4.8 %, `wx_vis_mi` 3.7 %, `arr_before_60` 2.1 %, `takeoffs_between_ob_mvt` 1.7 %.
- METAR flags set in the episode: `wx_snow` 2.3 %, `wx_rain` 21.9 %, `wx_fog` 9.9 %, `wx_icing` 20.4 %, `wx_snow_recent` 4.9 %, `wx_icing_recent` 33.4 %.
  Control EDDF Feb 2025 (15,099 rows): `wx_snow` 0.8 %, `wx_freezing` 0.1 %, `wx_rain` 9.5 %, `wx_fog` 5.3 %, `wx_icing` 18.4 %, `wx_snow_recent` 2.6 %, `wx_icing_recent` 30.6 %; any-of 8.4 %.
- (3) No NM record: 1.1 % (same month 2025 0.5 %).

LTFM February 2026, per day around the episode (`snow` = share of rows with `wx_snow`, medians of visibility (mi), temperature (C), wind (kt), departures in the previous hour):

| Day | Rows | snow | snow recent | freezing | vis med | temp med | wind med | dep 60 med | ctx any-of | no NM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-02-16 | 714 | 0.0 % | 0.0 % | 0.0 % | 6.2 | 12 | 11 | 41 | 20.6 % | 1.3 % |
| 2026-02-17 | 673 | 0.0 % | 0.0 % | 0.0 % | 6.2 | 12 | 7 | 34 | 32.8 % | 1.2 % |
| 2026-02-18 | 681 | 63.7 % | 66.2 % | 0.0 % | 6.2 | 2 | 12 | 37 | 13.1 % | 1.6 % |
| 2026-02-19 | 680 | 0.0 % | 0.0 % | 0.0 % | 6.2 | 8 | 11 | 39 | 9.6 % | 1.3 % |
| 2026-02-20 | 708 | 0.0 % | 0.0 % | 0.0 % | 6.2 | 13 | 11 | 39 | 16.0 % | 1.6 % |
| 2026-02-21 | 670 | 0.0 % | 0.0 % | 0.0 % | 3.7 | 8 | 16 | 39 | 25.2 % | 1.3 % |
| 2026-02-22 | 710 | 16.8 % | 23.9 % | 0.0 % | 6.2 | 4 | 16 | 39 | 29.4 % | 1.5 % |
| 2026-02-23 | 658 | 0.0 % | 0.0 % | 0.0 % | 6.2 | 6 | 9 | 37 | 11.7 % | 1.2 % |
| 2026-02-24 | 637 | 0.0 % | 0.0 % | 0.0 % | 6.2 | 12 | 16 | 34 | 17.9 % | 1.1 % |
| 2026-02-25 | 649 | 0.0 % | 0.0 % | 0.0 % | 6.2 | 6 | 12 | 38 | 12.6 % | 1.2 % |
| 2026-02-26 | 671 | 15.2 % | 25.9 % | 0.0 % | 6.2 | 5 | 15 | 36 | 6.4 % | 1.3 % |

LTFM 2025 has 3,510 departures with `wx_snow` = 1 on 20 days (2025-02-05 (327), 2025-02-06 (262), 2025-02-07 (248), 2025-02-09 (53), 2025-02-10 (30), 2025-02-11 (517), 2025-02-12 (77), 2025-02-19 (383), 2025-02-20 (192), 2025-02-21 (219), 2025-02-22 (209), 2025-02-23 (436), ...).

The 2025 episode itself (training rows), per day:

| Day | Rows | snow | vis med | temp med | day_movements med | ctx any-of | no NM |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025-02-19 | 563 | 68.0 % | 6.2 | 2 | 1138 | 100.0 % | 15.1 % |
| 2025-02-20 | 614 | 31.3 % | 6.2 | 2 | 1202 | 100.0 % | 1.3 % |
| 2025-02-21 | 431 | 50.8 % | 6.2 | 1 | 918 | 100.0 % | 0.9 % |
| 2025-02-22 | 458 | 45.6 % | 6.2 | 0 | 935 | 100.0 % | 2.2 % |
| 2025-02-23 | 642 | 67.9 % | 6.2 | 0 | 1209 | 100.0 % | 13.1 % |

EHAM June 2026 count reconciliation (final ranking file): departure rows with `ADEP_mvt = EHAM` in June = **20,881**, of which 18,194 on 1-26 June (87.1 %). Rows with `ADEP_mvt = EHAM` over both phases = 23,812 (2,931 of them are arrival-phase records of flights that left Schiphol, i.e. not EHAM departure rows), 20,753 of them on 1-26 June. The figures 23,812 / 20,753 quoted in `docs/final-submission.md` section 5 and `docs/model-card.md` are these two-phase counts; the departure-row counts are 20,881 / 18,194.

EHAM June 2026 runway configurations, works window against the rest of the month (`n25` = 2025 departures with the same configuration):

| Part | Configuration | Departures | 2025 all |
|---|---:|---:|---:|
| 1-26 Jun | `EHAM:18L,22,24|18R,22,27` | 1470 | 571 |
| 1-26 Jun | `EHAM:18L,22,24|18R,27` | 1160 | 1104 |
| 1-26 Jun | `EHAM:18L,24|18R,22` | 1130 | 14282 |
| 1-26 Jun | `EHAM:18L,24|18R` | 1056 | 18429 |
| 1-26 Jun | `EHAM:18L,24|18R,27` | 1014 | 1987 |
| 1-26 Jun | `EHAM:36C,36L|06,36R` | 720 | 15630 |
| 1-26 Jun | `EHAM:18L,22,24|18R` | 675 | 6792 |
| 1-26 Jun | `EHAM:04,36C,36L|06,36R` | 666 | 12190 |
| 1-26 Jun | `EHAM:18L,22,24|18R,22` | 641 | 9961 |
| 1-26 Jun | `EHAM:18L,24|18R,22,27` | 585 | 639 |
| 1-26 Jun | `EHAM:24|18R` | 570 | 4688 |
| 1-26 Jun | `EHAM:24|18R,24,27` | 470 | 189 |
| 1-26 Jun | `EHAM:36C,36L|06` | 466 | 13743 |
| 1-26 Jun | `EHAM:24|18R,27` | 366 | 1087 |
| 1-26 Jun | `EHAM:18L,24|18R,24,27` | 365 | 150 |
| 1-26 Jun | `EHAM:18L,22,24|18R,24,27` | 326 | 34 |
| 1-26 Jun | `EHAM:18L,22,24|18C,18R,22` | 275 | 9836 |
| 1-26 Jun | `EHAM:18L,24|18C,18R` | 269 | 17131 |
| 1-26 Jun | `EHAM:24,36L|27` | 230 | 1451 |
| 1-26 Jun | `EHAM:22,24|18R,24,27` | 224 | 31 |
| 1-26 Jun | `EHAM:24,36L|27,36C` | 221 | 871 |
| 1-26 Jun | `EHAM:24|18R,22,27` | 220 | 152 |
| 1-26 Jun | `EHAM:04,36C,36L|04,06` | 208 | 1156 |
| 27-30 Jun | `EHAM:36C,36L|06,36R` | 279 | 15630 |
| 27-30 Jun | `EHAM:18L,24|18R` | 267 | 18429 |

EHAM June 2026 departure runways (`n25` = 2025 all, `n25_jun` = June 2025):

| Runway | Jun 2026 | 2025 all | Jun 2025 |
|---|---:|---:|---:|
| `24` | 9750 | 85236 | 10468 |
| `18L` | 4207 | 55589 | 4790 |
| `36L` | 4124 | 58590 | 3241 |
| `36C` | 1797 | 30900 | 1959 |
| `22` | 568 | 4697 | 487 |
| `09` | 252 | 7870 | 0 |
| `04` | 179 | 2833 | 209 |
| `HEL` | 3 | 29 | 4 |
| `18C` | 1 | 2007 | 98 |

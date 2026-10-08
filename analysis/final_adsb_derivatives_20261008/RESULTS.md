# Final daily ADS-B derivatives — coverage (2026-10-08T13:20:50.448909+00:00)

Output root `data/final_adsb_derivatives_20261008`; status file `complete.json`: {'input_missing': 0, 'processed': 120, 'processed_no_matches': 0}, 120 days, 670790 final departures, 0 without any product (input_missing days: none). Manifest `MANIFEST.sha256` sha256 `a3240876632843704f56ea0355cb524ba47de4e56050a714f5c9c6538630f9a0` (960 lines). Final ranking sha256 `4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f`.

Columns: `seg` = share of final departures with a surface segment in kt1/offblock2 (coverage>=2), `ident` = airframe identified (n_cand==1), `stand` = stationary dwell seen (coverage 3), `visit` = visit record, `head` = HEAD record. `*25` = same calendar month of 2025 from the historical products (the stage models' training coverage); `*old` = the old Jan/Jul 2026 products built on the old ranking MVT. `collapse` = seg < seg25 − 20 pp, or seg < ½·seg25 when seg25 ≥ 30 %. `drift_vs_old` = |seg − seg_old| > 2 pp on Jan/Jul.

| airport | month | n | seg % | seg25 % | seg old % | ident % | ident25 % | stand % | visit % | visit25 % | head % | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EDDF | 2026-01 | 15667 | 74.9 | 91.1 | 74.9 | 74.9 | 91.1 | 5.7 | 95.6 | 98.2 | 72.2 |  |
| EDDF | 2026-02 | 14654 | 32.2 | 96.3 | — | 32.2 | 96.3 | 2.1 | 96.4 | 98.7 | 25.4 | COLLAPSE |
| EDDF | 2026-06 | 19700 | 78.8 | 93.8 | — | 78.8 | 93.8 | 6.6 | 96.2 | 96.6 | 75.1 |  |
| EDDF | 2026-07 | 20650 | 74.7 | 91.2 | 74.7 | 74.7 | 91.2 | 5.4 | 95.8 | 93.3 | 70.5 |  |
| EDDM | 2026-01 | 10953 | 96.5 | 96.4 | 96.5 | 96.5 | 96.4 | 64.4 | 96.2 | 98.1 | 95.5 |  |
| EDDM | 2026-02 | 10534 | 97.0 | 90.2 | — | 97.0 | 90.2 | 59.4 | 96.9 | 98.3 | 95.7 |  |
| EDDM | 2026-06 | 14372 | 94.2 | 70.4 | — | 94.2 | 70.4 | 51.2 | 96.6 | 97.9 | 93.2 |  |
| EDDM | 2026-07 | 15013 | 96.0 | 42.3 | 96.0 | 96.0 | 42.3 | 58.5 | 95.6 | 96.7 | 95.3 |  |
| EGLL | 2026-01 | 19283 | 84.7 | 69.9 | 84.7 | 84.7 | 69.9 | 22.6 | 96.5 | 99.5 | 82.2 |  |
| EGLL | 2026-02 | 17719 | 82.5 | 68.1 | — | 82.5 | 68.1 | 20.5 | 97.2 | 99.5 | 79.6 |  |
| EGLL | 2026-06 | 19662 | 93.5 | 34.2 | — | 93.5 | 34.2 | 56.1 | 96.5 | 99.4 | 92.1 |  |
| EGLL | 2026-07 | 20557 | 96.3 | 35.9 | 96.3 | 96.3 | 35.9 | 59.6 | 95.7 | 98.5 | 94.9 |  |
| EHAM | 2026-01 | 16232 | 92.7 | 97.3 | 92.7 | 92.7 | 97.3 | 49.2 | 92.4 | 97.0 | 90.5 |  |
| EHAM | 2026-02 | 16802 | 96.2 | 97.6 | — | 96.2 | 97.6 | 45.9 | 96.0 | 97.2 | 93.9 |  |
| EHAM | 2026-06 | 20881 | 92.8 | 96.8 | — | 92.8 | 96.8 | 48.7 | 95.6 | 96.8 | 90.3 |  |
| EHAM | 2026-07 | 21950 | 95.7 | 95.3 | 95.7 | 95.7 | 95.3 | 51.5 | 95.5 | 95.3 | 93.5 |  |
| LEBL | 2026-01 | 12201 | 94.3 | 98.3 | 94.3 | 94.3 | 98.3 | 18.7 | 96.9 | 99.3 | 93.5 |  |
| LEBL | 2026-02 | 12167 | 95.9 | 98.8 | — | 95.9 | 98.8 | 19.6 | 97.7 | 99.3 | 94.0 |  |
| LEBL | 2026-06 | 17067 | 93.2 | 98.0 | — | 93.2 | 98.0 | 34.9 | 94.4 | 98.7 | 92.1 |  |
| LEBL | 2026-07 | 17879 | 95.8 | 92.6 | 95.8 | 95.8 | 92.6 | 30.7 | 97.0 | 97.4 | 94.3 |  |
| LEMD | 2026-01 | 16802 | 96.3 | 2.3 | 96.3 | 96.3 | 2.3 | 23.7 | 95.9 | 31.8 | 95.2 |  |
| LEMD | 2026-02 | 16191 | 96.9 | 0.0 | — | 96.9 | 0.0 | 21.9 | 96.4 | 0.7 | 95.5 |  |
| LEMD | 2026-06 | 19335 | 82.0 | 97.2 | — | 82.0 | 97.2 | 17.5 | 93.7 | 97.4 | 79.3 |  |
| LEMD | 2026-07 | 20152 | 70.4 | 33.7 | 70.4 | 70.4 | 33.7 | 17.0 | 96.5 | 84.3 | 66.4 |  |
| LFPG | 2026-01 | 18129 | 21.0 | 4.7 | 21.0 | 21.0 | 4.7 | 0.6 | 94.4 | 74.5 | 15.5 |  |
| LFPG | 2026-02 | 16934 | 14.9 | 8.8 | — | 14.9 | 8.8 | 0.1 | 95.0 | 91.4 | 8.8 |  |
| LFPG | 2026-06 | 20714 | 31.0 | 8.7 | — | 31.0 | 8.7 | 1.0 | 96.6 | 78.5 | 22.5 |  |
| LFPG | 2026-07 | 21743 | 25.1 | 17.5 | 25.1 | 25.1 | 17.5 | 0.5 | 95.5 | 95.2 | 16.7 |  |
| LIRF | 2026-01 | 11061 | 78.5 | 0.6 | 78.5 | 78.5 | 0.6 | 6.1 | 96.1 | 85.3 | 76.2 |  |
| LIRF | 2026-02 | 10264 | 72.7 | 1.2 | — | 72.7 | 1.2 | 4.9 | 97.0 | 83.3 | 69.7 |  |
| LIRF | 2026-06 | 14983 | 0.7 | 80.3 | — | 0.7 | 80.3 | 0.0 | 51.5 | 98.3 | 0.1 | COLLAPSE |
| LIRF | 2026-07 | 15838 | 6.2 | 93.9 | 6.2 | 6.2 | 93.9 | 0.0 | 84.2 | 97.1 | 1.2 | COLLAPSE |
| LSZH | 2026-01 | 9995 | 93.0 | 94.9 | 93.0 | 93.0 | 94.9 | 45.4 | 94.6 | 98.1 | 92.3 |  |
| LSZH | 2026-02 | 9374 | 95.5 | 96.2 | — | 95.5 | 96.2 | 42.4 | 96.4 | 98.7 | 94.8 |  |
| LSZH | 2026-06 | 12384 | 90.2 | 92.7 | — | 90.2 | 92.7 | 18.3 | 93.1 | 96.7 | 89.6 |  |
| LSZH | 2026-07 | 13155 | 93.8 | 90.4 | 93.8 | 93.8 | 90.4 | 19.2 | 95.0 | 94.5 | 93.2 |  |
| LTFM | 2026-01 | 22396 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.2 | 0.0 | 0.0 |  |
| LTFM | 2026-02 | 19519 | 0.0 | 0.0 | — | 0.0 | 0.0 | 0.0 | 0.5 | 0.0 | 0.0 |  |
| LTFM | 2026-06 | 22693 | 0.1 | 0.0 | — | 0.1 | 0.0 | 0.0 | 4.1 | 0.2 | 0.0 |  |
| LTFM | 2026-07 | 25185 | 0.3 | 0.0 | 0.3 | 0.3 | 0.0 | 0.0 | 4.8 | 0.2 | 0.1 |  |

## Flags: 3 airport-months

- EDDF 2026-02: seg 32.2 % vs 2025 96.3 % vs old — % — COLLAPSE
- LIRF 2026-06: seg 0.7 % vs 2025 80.3 % vs old — % — COLLAPSE
- LIRF 2026-07: seg 6.2 % vs 2025 93.9 % vs old 6.2 % — COLLAPSE

## Per month (departure-weighted)

| month | n | seg % | seg25 % | visit % |
|---|---|---|---|---|
| 2026-01 | 152719 | 67.6 | 50.3 | 81.7 |
| 2026-02 | 144158 | 63.8 | 51.8 | 83.5 |
| 2026-06 | 181791 | 64.0 | 64.0 | 80.4 |
| 2026-07 | 192122 | 62.9 | 55.9 | 83.0 |

## Per-day receipts

| day | status | departures | identified | cov>=2 | visit | head rows | seconds |
|---|---|---|---|---|---|---|---|
| 2026-01-01 | processed | 4971 | 3465 | 3465 | 4188 | 3352 | 6 |
| 2026-01-02 | processed | 5384 | 3588 | 3588 | 4407 | 3448 | 7 |
| 2026-01-03 | processed | 4992 | 3288 | 3288 | 4018 | 3192 | 6 |
| 2026-01-04 | processed | 5167 | 3374 | 3374 | 4126 | 3263 | 6 |
| 2026-01-05 | processed | 4895 | 3180 | 3180 | 3912 | 3098 | 6 |
| 2026-01-06 | processed | 4639 | 3061 | 3061 | 3689 | 2966 | 6 |
| 2026-01-07 | processed | 4359 | 2860 | 2860 | 3497 | 2783 | 6 |
| 2026-01-08 | processed | 4962 | 3372 | 3372 | 4096 | 3288 | 7 |
| 2026-01-09 | processed | 5039 | 3440 | 3440 | 4060 | 3343 | 7 |
| 2026-01-10 | processed | 4702 | 3157 | 3157 | 3806 | 3067 | 6 |
| 2026-01-11 | processed | 5202 | 3715 | 3715 | 4324 | 3586 | 7 |
| 2026-01-12 | processed | 4894 | 3328 | 3328 | 4094 | 3265 | 6 |
| 2026-01-13 | processed | 4593 | 3106 | 3106 | 3727 | 3028 | 6 |
| 2026-01-14 | processed | 4758 | 3345 | 3345 | 3936 | 3273 | 6 |
| 2026-01-15 | processed | 5002 | 3475 | 3475 | 4124 | 3366 | 6 |
| 2026-01-16 | processed | 5273 | 3766 | 3766 | 4378 | 3676 | 7 |
| 2026-01-17 | processed | 4639 | 3222 | 3222 | 3786 | 3128 | 6 |
| 2026-01-18 | processed | 5174 | 3714 | 3714 | 4307 | 3618 | 7 |
| 2026-01-19 | processed | 5128 | 3644 | 3644 | 4247 | 3552 | 6 |
| 2026-01-20 | processed | 4637 | 3196 | 3196 | 3794 | 3078 | 6 |
| 2026-01-21 | processed | 4801 | 3295 | 3295 | 3928 | 3209 | 6 |
| 2026-01-22 | processed | 5155 | 3668 | 3668 | 4311 | 3575 | 7 |
| 2026-01-23 | processed | 5371 | 3118 | 3118 | 3672 | 3052 | 6 |
| 2026-01-24 | processed | 4621 | 3109 | 3109 | 3705 | 3031 | 6 |
| 2026-01-25 | processed | 5026 | 3519 | 3519 | 4268 | 3446 | 7 |
| 2026-01-26 | processed | 4941 | 3451 | 3451 | 4186 | 3368 | 7 |
| 2026-01-27 | processed | 4639 | 3146 | 3146 | 3804 | 3037 | 6 |
| 2026-01-28 | processed | 4725 | 3142 | 3142 | 3909 | 3039 | 6 |
| 2026-01-29 | processed | 5045 | 3044 | 3044 | 4154 | 2961 | 6 |
| 2026-01-30 | processed | 5383 | 3479 | 3479 | 4479 | 3325 | 7 |
| 2026-01-31 | processed | 4602 | 2904 | 2904 | 3813 | 2780 | 6 |
| 2026-02-01 | processed | 5246 | 3294 | 3294 | 4371 | 3162 | 6 |
| 2026-02-02 | processed | 5190 | 3392 | 3392 | 4303 | 3249 | 6 |
| 2026-02-03 | processed | 4617 | 2958 | 2958 | 3764 | 2854 | 6 |
| 2026-02-04 | processed | 4849 | 3001 | 3001 | 3974 | 2898 | 6 |
| 2026-02-05 | processed | 5065 | 3175 | 3175 | 4115 | 3079 | 6 |
| 2026-02-06 | processed | 5407 | 3482 | 3482 | 4507 | 3349 | 6 |
| 2026-02-07 | processed | 4894 | 3060 | 3060 | 4057 | 2925 | 6 |
| 2026-02-08 | processed | 5317 | 3256 | 3256 | 4411 | 3129 | 6 |
| 2026-02-09 | processed | 5267 | 3384 | 3384 | 4406 | 3270 | 6 |
| 2026-02-10 | processed | 4769 | 3040 | 3040 | 3940 | 2912 | 6 |
| 2026-02-11 | processed | 4931 | 3048 | 3048 | 4052 | 2939 | 6 |
| 2026-02-12 | processed | 4873 | 3027 | 3027 | 3997 | 2875 | 6 |
| 2026-02-13 | processed | 5674 | 3615 | 3615 | 4794 | 3501 | 7 |
| 2026-02-14 | processed | 5178 | 3180 | 3180 | 4312 | 3038 | 6 |
| 2026-02-15 | processed | 5202 | 3208 | 3208 | 4281 | 3045 | 6 |
| 2026-02-16 | processed | 5360 | 3459 | 3459 | 4524 | 3335 | 6 |
| 2026-02-17 | processed | 4965 | 3264 | 3264 | 4134 | 3119 | 6 |
| 2026-02-18 | processed | 5015 | 3291 | 3291 | 4208 | 3155 | 6 |
| 2026-02-19 | processed | 5068 | 3210 | 3210 | 4213 | 3056 | 6 |
| 2026-02-20 | processed | 5557 | 3587 | 3587 | 4737 | 3428 | 7 |
| 2026-02-21 | processed | 5111 | 3265 | 3265 | 4312 | 3111 | 6 |
| 2026-02-22 | processed | 5438 | 3486 | 3486 | 4571 | 3306 | 6 |
| 2026-02-23 | processed | 5296 | 3479 | 3479 | 4529 | 3309 | 6 |
| 2026-02-24 | processed | 4953 | 3273 | 3273 | 4147 | 3119 | 6 |
| 2026-02-25 | processed | 5112 | 3295 | 3295 | 4294 | 3161 | 6 |
| 2026-02-26 | processed | 5330 | 3441 | 3441 | 4489 | 3277 | 6 |
| 2026-02-27 | processed | 5649 | 3700 | 3700 | 4835 | 3522 | 7 |
| 2026-02-28 | processed | 4825 | 3144 | 3144 | 4104 | 2982 | 6 |
| 2026-06-01 | processed | 6197 | 4026 | 4026 | 5054 | 3883 | 7 |
| 2026-06-02 | processed | 5809 | 3768 | 3768 | 4690 | 3633 | 7 |
| 2026-06-03 | processed | 5994 | 3970 | 3970 | 4840 | 3818 | 7 |
| 2026-06-04 | processed | 6032 | 3999 | 3999 | 4922 | 3852 | 7 |
| 2026-06-05 | processed | 6262 | 4201 | 4201 | 5164 | 4084 | 7 |
| 2026-06-06 | processed | 5862 | 3926 | 3926 | 4765 | 3787 | 7 |
| 2026-06-07 | processed | 6117 | 3995 | 3995 | 4971 | 3845 | 7 |
| 2026-06-08 | processed | 6123 | 4128 | 4128 | 5125 | 3977 | 7 |
| 2026-06-09 | processed | 5910 | 3813 | 3813 | 4793 | 3665 | 7 |
| 2026-06-10 | processed | 5985 | 3820 | 3820 | 4905 | 3662 | 7 |
| 2026-06-11 | processed | 6099 | 316 | 316 | 3070 | 98 | 3 |
| 2026-06-12 | processed | 6276 | 4007 | 4007 | 5136 | 3819 | 7 |
| 2026-06-13 | processed | 5823 | 3964 | 3964 | 4835 | 3773 | 7 |
| 2026-06-14 | processed | 6164 | 4153 | 4153 | 5182 | 3999 | 8 |
| 2026-06-15 | processed | 6200 | 4140 | 4140 | 5412 | 4006 | 8 |
| 2026-06-16 | processed | 5916 | 3984 | 3984 | 4916 | 3858 | 7 |
| 2026-06-17 | processed | 6058 | 4027 | 4027 | 4996 | 3868 | 7 |
| 2026-06-18 | processed | 6128 | 4108 | 4108 | 4995 | 3986 | 8 |
| 2026-06-19 | processed | 6252 | 4035 | 4035 | 5015 | 3911 | 7 |
| 2026-06-20 | processed | 5908 | 3817 | 3817 | 4748 | 3686 | 7 |
| 2026-06-21 | processed | 6101 | 3935 | 3935 | 4892 | 3812 | 7 |
| 2026-06-22 | processed | 6131 | 4061 | 4061 | 4978 | 3945 | 8 |
| 2026-06-23 | processed | 5978 | 4021 | 4021 | 4814 | 3909 | 7 |
| 2026-06-24 | processed | 6070 | 4096 | 4096 | 4863 | 3974 | 7 |
| 2026-06-25 | processed | 6158 | 4060 | 4060 | 4945 | 3902 | 7 |
| 2026-06-26 | processed | 6325 | 4159 | 4159 | 5047 | 4022 | 8 |
| 2026-06-27 | processed | 5788 | 3814 | 3814 | 4561 | 3702 | 7 |
| 2026-06-28 | processed | 6139 | 4068 | 4068 | 4874 | 3915 | 7 |
| 2026-06-29 | processed | 6104 | 4036 | 4036 | 4887 | 3886 | 7 |
| 2026-06-30 | processed | 5882 | 3876 | 3876 | 4789 | 3735 | 7 |
| 2026-07-01 | processed | 6062 | 3816 | 3816 | 4811 | 3654 | 7 |
| 2026-07-02 | processed | 6289 | 4103 | 4103 | 5079 | 3934 | 7 |
| 2026-07-03 | processed | 6356 | 4153 | 4153 | 5144 | 4019 | 7 |
| 2026-07-04 | processed | 6005 | 3849 | 3849 | 4727 | 3686 | 7 |
| 2026-07-05 | processed | 6201 | 3947 | 3947 | 4935 | 3767 | 8 |
| 2026-07-06 | processed | 6218 | 4087 | 4087 | 5196 | 3901 | 8 |
| 2026-07-07 | processed | 6048 | 4068 | 4068 | 5087 | 3878 | 8 |
| 2026-07-08 | processed | 6227 | 4036 | 4036 | 5122 | 3870 | 7 |
| 2026-07-09 | processed | 6300 | 4134 | 4134 | 5243 | 3979 | 8 |
| 2026-07-10 | processed | 6372 | 4106 | 4106 | 5286 | 3940 | 7 |
| 2026-07-11 | processed | 6013 | 3951 | 3951 | 4964 | 3801 | 7 |
| 2026-07-12 | processed | 6250 | 4075 | 4075 | 5143 | 3908 | 7 |
| 2026-07-13 | processed | 6252 | 4009 | 4009 | 5242 | 3848 | 7 |
| 2026-07-14 | processed | 6088 | 3794 | 3794 | 4997 | 3625 | 7 |
| 2026-07-15 | processed | 6180 | 3772 | 3772 | 5084 | 3576 | 7 |
| 2026-07-16 | processed | 6226 | 3791 | 3791 | 5134 | 3606 | 7 |
| 2026-07-17 | processed | 6321 | 3807 | 3807 | 5173 | 3625 | 7 |
| 2026-07-18 | processed | 5998 | 3620 | 3620 | 4928 | 3457 | 7 |
| 2026-07-19 | processed | 6205 | 3791 | 3791 | 5180 | 3573 | 7 |
| 2026-07-20 | processed | 6223 | 3897 | 3897 | 5233 | 3714 | 7 |
| 2026-07-21 | processed | 6102 | 3903 | 3903 | 5067 | 3751 | 7 |
| 2026-07-22 | processed | 6237 | 3932 | 3932 | 5214 | 3739 | 7 |
| 2026-07-23 | processed | 6256 | 3840 | 3840 | 5220 | 3686 | 7 |
| 2026-07-24 | processed | 6407 | 3836 | 3836 | 5295 | 3680 | 7 |
| 2026-07-25 | processed | 6021 | 3679 | 3679 | 5015 | 3479 | 7 |
| 2026-07-26 | processed | 6259 | 3938 | 3938 | 5367 | 3660 | 7 |
| 2026-07-27 | processed | 6243 | 3804 | 3804 | 5545 | 3609 | 7 |
| 2026-07-28 | processed | 6082 | 3687 | 3687 | 5132 | 3498 | 7 |
| 2026-07-29 | processed | 6153 | 3714 | 3714 | 5236 | 3513 | 7 |
| 2026-07-30 | processed | 6248 | 3825 | 3825 | 5254 | 3611 | 7 |
| 2026-07-31 | processed | 6280 | 3836 | 3836 | 5329 | 3631 | 6 |

## Flagged airport-months, per day (final products; `seg` = coverage>=2 share, `visit` = visit record share)

Computed from `data/final_adsb_derivatives_20261008/data/adsblol_kt1/offblock2_*` and `data/adsblol/visit_*` joined to the final departures.

- **EDDF 2026-02** (32.2 % vs 96.3 % in Feb 2025, 74.9/78.8/74.7 % in Jan/Jun/Jul 2026): every day of the month sits at 15–46 % (min 14.7 % on 02-03, max 45.9 % on 02-24), while `visit` stays at 92–98 %. The airframes are seen airborne (approach/climb-out) but the ground segment is not received for two thirds of them: a month-long drop of ground reception at Frankfurt, not an archive hole or a one-day outage. Stage models (E141/E64/E188) were trained on 2025 EDDF coverage of 91–96 %, so in February the stage reaches ~1/3 of the EDDF rows it would normally reach; the rows it does reach carry the same feature semantics.
- **LIRF 2026-06** (0.7 % vs 80.3 % in Jun 2025): 0.0–2.8 % every day; `visit` 42–65 % except **2026-06-11 = 0.0 %** (no LIRF points of any kind in that surface file, see the point-count check below). This is the known Rome reception collapse of summer 2026 (REVIEW_BRIEF 21j: July Rome empty on reception) extending over the whole of June: no stage input at Fiumicino in June.
- **LIRF 2026-07** (6.2 % vs 93.9 % in Jul 2025; identical to the old Jul 2026 products, 6.2 %): 0.2–1.5 % on 07-01..07-05, then 4.7–10.5 % from 07-06 on; `visit` 42–52 % until 07-05 and 84–98 % afterwards. Known state, unchanged by the final MVT.

Not flagged but worth knowing when reading the stage: LTFM has no ground reception in any month (0.0–0.3 %, as in 2025); LFPG 15–31 % (2025: 5–18 %); LEMD Jan/Feb 2026 at 96–97 % against 0–2 % in Jan/Feb 2025 (reception appeared at Madrid in 2026, so the stage now reaches LEMD winter rows it never saw in training); EGLL Jun/Jul 2026 93–96 % against 34–36 % in 2025; EDDM Jul 2026 96 % against 42 % in 2025. `ident` equals `seg` everywhere by construction of `adsb_offblock_v2.py` (a row is matched only when its last surface point lies within [MVT − 900 s, MVT + 300 s], which already makes the segment non-empty), so `n_cand == 1` and `coverage >= 2` are the same predicate.

## Jan/Jul 2026: final products against the old ones (old ranking MVT)

Same 344 841 departure IDs in both (0 only-new, 0 only-old); kt1/offblock2 differs in `coverage` on 35 rows, `t_move` on 36, `t_to` on 56 of 344 841 — the ±12 s dithering of the final MVT moves a handful of departures across the matching-window and same-day boundaries (13 departures changed UTC day). Every per-airport-month `seg` is within 2 pp of the old products (no `drift_vs_old` flag). The products are stored per UTC day of the final MVT, so the 13 day-crossers now sit in the file of their final day; consumers that glob `*_2026-*.parquet` and join on `MVT_ID_mvt` see them regardless.

## Inputs and run

- Surface files: Jan/Jul 2026 — the 62 `data/adsblol/surface_2026-0[17]-*.parquet` extracted on the Mac 16.09 (MVT-independent raw input); Feb/Jun 2026 — 58 files prefetched on laptop 175 (`/home/a1/claude_adsb_2026/`, run 10:42–12:45Z, 58/58 ok, 0 FAIL), copied with scp into `data/adsblol/` and verified 58/58 OK against the 175 manifest (`data/adsblol/MANIFEST_175_febjun2026.sha256`, `files=58/58 total=914M`). Every receipt records the sha256 of the surface file it consumed.
- Run: Mac, 13:17:00–13:20:24Z, 203.9 s, 4 workers × 2 threads, nice 10; 120/120 days `processed`, 0 `processed_no_matches`, 0 `input_missing`; 670 790 final departures, 0 without products; 840 product files (360 kt1 + 120 sw_kt1 + 360 visit/visit_m/head), 114 MB; `MANIFEST.sha256` 960 lines, sha256 `a3240876632843704f56ea0355cb524ba47de4e56050a714f5c9c6538630f9a0`; `complete.json` with producer source SHAs and git HEAD `ec7525e9`.
- Consumer paths (the stage scripts' env): `ADSB_DIR=data/final_adsb_derivatives_20261008/data/adsblol_kt1`, `WALK_DIR=data/final_adsb_derivatives_20261008/data/adsblol_sw_kt1`, `VISIT_DIR=HEAD_DIR=data/final_adsb_derivatives_20261008/data/adsblol` (the latter also holds the surface symlinks; the stage globs `visit_`/`visit_m_`/`head_` only). No plain 3 kt `offblock*` and no sw_kt1 `offblock3/5` were produced: the seven products of the recipe are the ones the stages read.

## Archive check: 2026-06-11 is a defective adsb.lol day

Row counts of the 120 surface files: median 1 236 090 points/day; the only outlier is **2026-06-11 with 91 473 points** (6 airports only, LIRF/LEBL/LEMD/LTFM absent, almost no ground points). The 175 download log shows the preferred `staging-0` archive is a single 1.60 GB tar (a normal day is ~3.3 GB in two parts) with 42 736 traces. The alternative replica `prod-0` (1.52 GB) was extracted separately on 175 (`/home/a1/claude_adsb_2026/alt/surface_2026-06-11.prod0.parquet`): 42 314 traces → 81 883 points — equally sparse. Both replicas of that day are truncated at the source, so the day stays as produced: receipt `processed` with 6 039 final departures, kt1/offblock2 matched 2 (EDDM) and visit 300; no stage input for 2026-06-11 is recoverable from adsb.lol. The 2025 historical inventory has an analogous gap (364 of 365 days, no 2025-12-31).

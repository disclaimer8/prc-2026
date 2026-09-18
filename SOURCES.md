# Sources

Everything this solution reads or was informed by, with its licence and what it was used for.

Two reasons this file exists. The prize conditions of the PRC Data Challenge 2026 (`eligibility.html`) require that
any external dataset be open and documented, and this repository becomes public under GPLv3 on 9 October 2026 —
so the licence column has to survive a reader who checks it. The second reason is the paper: every source that
shaped a decision is listed here with the verdict it earned, including the ones that were measured and dropped,
because a source that cost a day and produced nothing is worth recording exactly once.

Last updated 18 September 2026.

---

## 1. Data used in the submitted solution

| Source | What it gives | Licence | Where it enters |
|---|---|---|---|
| **PRC Data Challenge 2026 datasets** — EUROCONTROL Performance Review Commission, supported by the OpenSky Network | 12 monthly parquet files of 2025 airport movements joined to Network Manager flight records (4.17 M movements, 2.09 M departures) plus `ranking.parquet` for January and July 2026 | Provided to participants under the challenge terms; not an external dataset | The whole pipeline. `src/features.py` |
| **adsb.lol `globe_history`** — community ADS-B aggregator | Aircraft positions within 7 km of the aerodrome and below 2500 ft, ground and airborne, per day per airframe. Held locally as **364 days of 2025** and **62 days of 2026** (January and July in full) | **ODbL 1.0** | The second-stage corrections only, never the base model: `src/adsb_offblock_v2.py`, `adsb_offblock_v5.py`, `adsb_visit.py`, `e69_visit_merge.py`, consumed by stages E25, E30, E36, E64, E65 |
| **OpenStreetMap**, via the Overpass API | Stand, gate, holding position, runway, taxiway, stopway and apron geometry for the 10 airports (11 extracts) | **ODbL 1.0**, © OpenStreetMap contributors | `src/geometry.py` (stand → runway distance, bearing offset, `geo_*` features), `src/taxi_route.py` (taxiway graph), and the stand anchor of every ADS-B stage |
| **Iowa Environmental Mesonet ASOS archive** (Iowa State University) | Hourly-ish METAR observations for the 10 airports: temperature, dew point, wind, gust, visibility, ceiling, present weather | **Public domain** — "The materials found on this website are in the public domain and may be used freely by anyone for any lawful purpose." Attribution requested, not required | `src/features.py:load_metar` → 15 weather features in the base model (`wx_temp`, `wx_vis_mi`, `wx_ceiling_ft`, `wx_fog`, `wx_freezing`, `wx_snow`, `wx_icing`, `metar_age_min`, …) |

Nothing else is read. In particular the solution uses no commercial feed, no subscription archive and no dataset
whose terms restrict use to non-commercial research.

**Attribution block for the submission and the paper:**

> Data: EUROCONTROL Performance Review Commission Data Challenge 2026, supported by the OpenSky Network.
> Ground and airborne ADS-B observations © adsb.lol contributors, ODbL 1.0.
> Airport surface geometry © OpenStreetMap contributors, ODbL 1.0.
> METAR observations from the Iowa Environmental Mesonet, Iowa State University (public domain).

---

## 2. Data evaluated and not used

The criterion is the organiser's own, given in the challenge Discord on 17 September 2026 in answer to a
participant asking about data carrying a non-commercial clause: *"if their license applies as open data, ok"*.
So what decides is the openness of the licence, not whether the source is documented.

The hole being filled in every case below is the same: ground coverage at **LFPG, LIRF and LTFM for January and
July 2026**. Measured coverage of the ranking set, share of departures with a usable ground segment:
LFPG 0.210 / 0.251, LIRF 0.785 / 0.062, LTFM 0.000 / 0.003 (January / July), against 0.75–0.96 elsewhere.

| Source | Why not | Checked |
|---|---|---|
| **OpenSky Network** (state vectors, Trino/MinIO) | Terms restrict use to non-commercial research — not an open licence. Participants also report Trino access is disallowed for the challenge | 16.09 |
| **ADS-B Exchange** archive | Commercial subscription. Its free sample proves the observations exist — in the same fourteen instants of 01.07.2026 it shows 52 aircraft on the ground at LIRF, 36 at LTFM, 14 at LFPG, where we have 0 and 0 for whole days — but the archive itself is closed by the rules | 18.09 |
| **Airplanes.Live** | Publishes no archive (the "Data" menu holds live API endpoints only) and no licence: its Terms of Use page renders empty, the content sitting in a third-party widget | 18.09 |
| **adsb.fi**, **ADSBHub**, **Planespotters Radar** | Live state only, no archive | 16–18.09 |
| **hpradar** | Eight receivers around Haiphong, Vietnam — none of our airports — and no archive: `globe.hpradar.com/globe_history/…` returns the page shell, not a listing | 18.09 |
| **Virtual Radar** | Software, not a network; the "regular data dump" listed on mode-s.org turns out to be SDM reference tables (routes, airlines, types), not trajectories | 18.09 |
| **adsb-archive.com** | Dead domain; no trajectory dumps in the Internet Archive either | 18.09 |
| **FlyItalyADSB** (regional network covering Rome) | Answered directly: *"we switched systems and no longer have July coverage"* — the central archive was lost in an infrastructure migration, which is exactly what the LIRF collapse from 0.785 in January to 0.062 in July measures. Individual feeders keeping local ring buffers have not been asked | 17–18.09 |
| **EUROCONTROL ADRR** (Aviation Data Repository for Research) | Fails twice: coverage is 2015–2024 and only four months per year, so January and July 2026 are not in it at all; and access is via OneSky under Terms of Use signed per user, "open for all R&D use" — research-only, not an open licence | 18.09 |
| **Open portals and public APIs of the three blind airports** (Groupe ADP, Aeroporti di Roma, iGA Istanbul) | Live flight status only; no historical per-flight export. Everything offering history for those airports (FlightLabs, FR24, FlightAware, FlightStats) is a commercial aggregator | 18.09 |
| **Schiphol Flight API** | Does expose `actualOffBlockTime`, but only for EHAM, where ground coverage is already 0.93–0.96. No value against the hole | 18.09 |
| **ERAU and ROSA P repositories** | Hold US documents and US airports; no per-flight data for the ten European airports | 16–18.09 |

Open leads are people, not sites: individual FlyItalyADSB feeders (`info@flyitalyadsb.com`) for Rome in July,
and spotters running mobile receivers near Istanbul.

---

## 3. Literature

The full survey is `analysis/literature_taxi_out_2026-09-14.md`: 30 works on taxi-out prediction with airports,
data sources, methods, features and reported accuracy for each, plus the EUROCONTROL PRU indicator documentation
(additional taxi-out and taxi-in, editions 00-04 and 01.00) and the OPDI paper. Reading it is what fixed the
shape of the feature set, so it is not repeated here.

### Read on 18 September 2026 (US DOT National Transportation Library, ROSA P)

Each of these was read to the point where it could be accepted or closed, and every closure below is a
measurement rather than an argument. The pattern they establish is worth more than any of them individually:
the mechanisms they describe are correct and the model has already learned them.

| Work | What it offered | Verdict |
|---|---|---|
| Yin, Tian, Wang, Quadrifoglio. **Taxiway aircraft traffic analysis at George Bush Intercontinental Airport**, SWUTC/11/476660-00029-1, 2011. [dot/23699](https://rosap.ntl.bts.gov/view/dot/23699) | Applies Simaiakis's saturation method to IAH (capacity 50 departures/h, saturation at N = 43) and adds a mixed-integer programme for optimal taxi routing | Method already held from the primary source. Its one untried piece — counting the queue **per runway** rather than per airport — measures corr −0.008 against the residual, ceiling 0.03 s against a 0.3 s gate. The MIP half is prescriptive and does not transfer |
| Yeh, Chandra. **Determining minimal display element requirements for surface map displays**, Volpe Center, 2003. [dot/9961](https://rosap.ntl.bts.gov/view/dot/9961) | 43 pilots rank 88 cockpit-chart elements by need per phase; hold short lines and ILS hold lines top the taxi-out list, i.e. where an aircraft must stop | Suggested crossing an active runway as a feature. Closed before building the route graph: the interaction it would encode (runway × hourly configuration) gives corr +0.76 and +0.558 s under a **random** split but corr −0.04 and −0.97 s under a **month** split. The lesson outlived the idea |
| Simaiakis, Khadilkar, Balakrishnan, Reynolds, Hansman, Reilly, Urlass. **Demonstration of reduced airport congestion through pushback rate control**, ICAT-2011-2, 2011. [dot/66521](https://rosap.ntl.bts.gov/view/dot/66521) | The BOS field trial and the primary source of the saturation curve; the curve is built in jets only and parameterised by arrivals per 15 min and by the number of props — the **mix** of the queue, not only its size | The arrival side we already hold. The mix measures corr −0.001 and +0.007 on the two temporal halves, sign flipping, because `nb_ap_30_30_mean` already averages the neighbours' observed taxi proxy — the mix in the currency that matters |
| Badrinath, Li, Balakrishnan. **Integrated surface–airspace model of airport departures**, J. Guidance, Control and Dynamics, 2019, CC BY 3.0. [dot/72571](https://rosap.ntl.bts.gov/view/dot/72571) | A queuing network whose stages are ramp, taxiway, runway **and the terminal airspace** — the departure fix and en-route traffic as their own bottleneck. The first proposal outside the surface-congestion class | Surface decomposition is our E66 (−0.15 s, 6 of 6 folds, short of the gate). Airspace competition measured on a temporal split at −0.005…+0.002 for four keys, including the sharp one (same destination airport). A true departure-fix feature would need the SID assignment, which the challenge data does not carry |
| Balakrishnan, Deonandan, Simaiakis. **Opportunities for reducing surface emissions through airport surface movement optimization**, ICAT-2008-07, 2008. [dot/62143](https://rosap.ntl.bts.gov/view/dot/62143) | PARTNER Project 21. Its section 6 lists, after Idris et al., the five variables that determine taxi time: runway configuration, weather, downstream restrictions, gate location, and the length of the take-off queue the flight experiences | Closed by comparison rather than measurement: four of the five are in the feature set, and the fifth was measured at zero the same day. Their headline variable — take-offs between own pushback and own take-off — is literally `takeoffs_between_ob_mvt` (`src/features.py:109`). Note the inverted task: they must *predict* the queue because take-off time is unknown to them, while our ranking rows carry `MVT_TIME` |
| Yeh, Jaworski, Thomas, Kendra, Hiltunen. **Examining the compellingness of electronic low visibility taxi charts**, 2018. [dot/50641](https://rosap.ntl.bts.gov/view/dot/50641) | Simulator study, 20 airline transport pilots, ownship position on iPad charts at 1200/600/300 RVR | Human factors; taxi times are simulator artefacts. Its only tangent — low visibility slowing taxi — is already carried by `wx_vis_mi`, `wx_fog` and `wx_ceiling_ft`, and the weather pocket was measured on 17.09 at 187 534 rows holding 16.21 s of error that is **unbiased**, worth 0.003 s |

**Class closure recorded from this batch:** congestion features are exhausted for this model. Three independent
proposals of that shape measured at approximately zero, on top of E51 (taxiway-graph route length, +0.01 s) and
the rejected stand/runway graph GNN. A new proposal of this shape is worth a run only when it arrives with a
number, never with a mechanism — the mechanisms are all correct and all already learned.

---

## 4. Software

Python 3.14 with polars, numpy, LightGBM, CatBoost, scikit-learn and TabPFN-2; `pyproject.toml` and `uv.lock`
pin the versions. `pyModeS` is not a dependency but is noted here because its `PipeDecoder(surface_ref=…)`
is the correct way to decode surface position messages should raw Beast dumps ever be obtained: surface CPR
is ambiguous without a reference point, so a decoder without one silently discards type-code 5–8 messages.

# Data sources, licences and attribution

This register describes the **v12075 submitted recipe**, with primary public
source pages checked on **7 October 2026**, and the one input added after it: the
EUROCONTROL monthly additional taxi-out time files read by the E218 stage of v12081
and of the final file (11 October 2026; [below](#sources-added-after-v12075)). The
code licence, a data licence and an organiser's permission to use data in this
challenge are different things. We record them separately. Public downloadability
alone is not an open licence.

The [challenge eligibility page](https://prc-data-challenge-2026.netlify.app/eligibility.html)
requires documented, openly accessible/usable external data, openly available
additional datasets, GPLv3 source code and sufficient reproduction documentation.
The dataset-specific evidence below should be read alongside those conditions.

## Sources in v12075

The E203 replacement adds ordered current/previous-visit observations from the
same adsb.lol archives. It introduces no additional external dataset or licence.

| Source and primary page | Contribution to the submitted chain | Licence or terms evidence |
|---|---|---|
| [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/) — EUROCONTROL PRC, supported by the OpenSky Network | 2025 training movements/NM records and January/July 2026 ranking data; ten airports | Participant-provided challenge data. Access is through the organiser's distribution. We do not assign these files the repository's GPL licence or claim an independently verified redistribution grant. |
| adsb.lol [2025](https://github.com/adsblol/globe_history_2025) and [2026](https://github.com/adsblol/globe_history_2026) `globe_history` | Aircraft surface observations used for movement onset, stand/track geometry, visits and correction stages | The 2026 repository explicitly places the database under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/). Credit adsb.lol and contributors; preserve applicable database attribution/share-alike obligations. Its separate CC0 file is not a reason to relabel the database CC0. |
| [OpenStreetMap](https://www.openstreetmap.org/copyright), retrieved through Overpass | Stand, runway and taxiway geometry; distances, bearings, routing and ADS-B stand anchors | [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/), © OpenStreetMap contributors. The primary copyright page requires attribution and identification of the licence. |
| [Iowa Environmental Mesonet ASOS/METAR archive](https://mesonet.agron.iastate.edu/request/download.phtml), Iowa State University | Temperature, wind, visibility, ceiling and weather indicators | The [IEM usage statement](https://mesonet.agron.iastate.edu/disclaimer.php) identifies its website materials as public domain and appreciates attribution. |
| [EUROCONTROL Aviation Intelligence Portal](https://ansperformance.eu/data/) | Destination-airport daily arrival ATFM delay aggregates, joined by destination and UTC take-off date; E163 columns retained in FIN1b | The linked [portal disclaimer](https://ansperformance.eu/about/disclaimer/) allows copying with attribution for non-commercial purposes and requires written permission for modification. This is not evidence of a general open-data licence. Recorded organiser guidance permits use of published dashboard figures/aggregates in the challenge; scope is described below. |
| [Open Performance Data Initiative v0.0.2](https://www.opdi.aero/data.html), EUROCONTROL PRU and OpenSky Network | Matched exit-parking and taxiway events for the **LSZH-only** final correction, blended at 50% | The data page links the EUROCONTROL portal disclaimer and does not identify a separate dataset licence in the material inspected. Challenge permission is recorded below. The MIT licence of the [OPDI processing software](https://github.com/euctrl-pru/OPDI) must not be attributed to its output datasets. |

Both adsb.lol year repositories explicitly identify the database as ODbL 1.0.
The 7 October source check independently confirms the displayed adsb.lol, OSM,
IEM and EUROCONTROL portal statements. It does not certify that all data and
pretrained-model redistribution questions have been resolved.

## Sources added after v12075

| Source and primary page | Contribution | Licence or terms evidence |
|---|---|---|
| EUROCONTROL monthly **taxi-out additional time** per airport: `taxi_out_additional_time_2025.csv` (2025; obtained from ansperformance.eu) and `taxi_out_additional_time_2026.csv` (January–July 2026; `www.eurocontrol.int/performance/data/download/csv/`, server last-modified 19 September 2026, unchanged when re-downloaded on 9 October 2026). Columns used: `APT_ICAO`, `MONTH_NUM`, `TF`, `VALID_FL`, `TOTAL_REF_NB_FL`, `TOTAL_ADD_TIME_MIN` | **E218 only**: the post-mixture PRU level stage of v12081 and of the final file moves each eligible airport-month half-way from the indicator implied by our predictions towards the published value ([final submission §2.5](docs/final-submission.md#25-e218-pru-level-stage)). The 2026 values for January, February, June and July are the scored months of the final, so this is a published aggregate of the same kind of record as the labels, one number per airport and month. SHA-256 `4aedf195…` (2025) and `b72040e0…` (2026), pinned in the final contract. Before 11 October the same files served only as a comparison reference ([PRU view](docs/pru-view.md), [data quality for PRU](docs/data-quality-regimes.md), the 9 October [PRU check](analysis/final_pru_and_regime_check_20261009.md)) and in the rejected E145 calibration | **No open licence is stated** for either file. The related [ansperformance.eu portal disclaimer](https://ansperformance.eu/about/disclaimer/) allows copying with attribution for non-commercial purposes and requires written permission for modification; we found no separate statement for the eurocontrol.int download. Under the organiser's general rule of 1 October 2026 ("If no open license is present it is not open") the files are not open data by licence. Their use rests on the organiser's specific answers: "I already answered above: yes" (27 September 2026) to our question about using the published monthly taxi-out figures for January and July 2026 as a model input, and a confirmation on 8 October 2026 that `taxi_out_additional_time_2026.csv` may still be used (recorded in `HYPOTHESES_LOG.md` entry 452; see [organiser clarifications](docs/organiser-clarifications.md)). We do not redistribute the files |

If this input were set aside, only the E218 stage is affected: v12081 returns to
v12077 (+0.0334 s on January/July) and the final file to its 9 October predecessor
`76c32426…` (+0.0360 s on its January/July rows); no other stage reads these files.

## Challenge-specific clarifications

The organiser answered data questions in the challenge Discord channel. The
statements that matter for this recipe are quoted verbatim, with UTC times, in
[docs/organiser-clarifications.md](docs/organiser-clarifications.md). In short:

| Date (UTC) | Organiser's answer | Applies to |
|---|---|---|
| 2026-09-16 | All columns except the departure block time and target, in the same row and in other rows, are available to the model | `AOBT_3_flt` and other NM timestamps |
| 2026-09-17 | EUROCONTROL's daily airport files (including arrival ATFM delay, under the portal's non-commercial disclaimer) and the adsb.lol archive (ODbL 1.0) "can be considered open data for the challenge" | Destination ATFM delay features (E163); adsb.lol |
| 2026-09-19, 2026-10-01 | Using open trajectory data, including the flight's own post-pushback track, is allowed | Surface stages and trajectory network |
| 2026-09-27 | "I already answered above: yes", to our question whether EUROCONTROL's published monthly taxi-out figures for January and July 2026 may be used as a model input | E218 PRU level stage (v12081, final file) |
| 2026-10-04 | OPDI, published without a separate licence, is "open" for eligibility | LSZH correction |
| 2026-10-08 | `taxi_out_additional_time_2026.csv` may still be used (answer to another team; recorded in our register, entry 452, not quoted) | E218 PRU level stage (v12081, final file) |

These are challenge-specific permissions. They do not change the general terms of
each portal and do not by themselves grant a right to redistribute EUROCONTROL
files; we therefore do not redistribute them and give acquisition instructions
instead.

## Acquisition, lineage and coverage

| Input | Code or artifact record | Reproduction detail |
|---|---|---|
| Challenge files | [`src/download.py`](src/download.py) | Organiser-issued access credentials are required; credentials are not part of the repository. |
| METAR | [`scripts/get_external.sh`](scripts/get_external.sh) (IEM ASOS, 2024-12-25 to `METAR_END`, default tomorrow UTC) → `data/metar/<ICAO>.csv`, `data/metar_wind/<ICAO>.csv`; [`src/features.py`](src/features.py) | Preserve the station, date coverage and downloaded snapshot used in feature construction. |
| OSM geometry | [`scripts/get_external.sh`](scripts/get_external.sh) → [`scripts/osm_overpass_download.py`](scripts/osm_overpass_download.py) (`OSM_DATE` pins the snapshot; `analysis/sources/scripts/` holds an earlier research version), [`src/geometry.py`](src/geometry.py) | Preserve the extract and retrieval date; a new live OSM query can produce different geometry. |
| adsb.lol | [`src/adsb_offblock_v2.py`](src/adsb_offblock_v2.py), [`src/adsb_offblock_v5.py`](src/adsb_offblock_v5.py), [`src/adsb_visit.py`](src/adsb_visit.py), [`src/e69_visit_merge.py`](src/e69_visit_merge.py) | Freeze daily archive/release selection and extraction settings. Coverage is uneven; do not interpret missing reception as no movement. |
| Destination ATFM | [`src/e163_build_features.py`](src/e163_build_features.py); `data/ext_pru/apt_dly_{2024,2025,2026}.csv` | Joins `ADES_mvt` and the UTC take-off date. Five `dst_*` columns are present in the submitted FIN1b schema. These are retrospective daily aggregates. |
| OPDI in the submitted correction | [`src/e159_fixed_lineage.json`](src/e159_fixed_lineage.json), [`src/e159_opdi_stage_cv.py`](src/e159_opdi_stage_cv.py), [`src/e159b_apply.py`](src/e159b_apply.py) | The frozen `data/opdi/ev_*.parquet` inputs define the v12075 correction. Callsign/airport matching uses a nearby observed event anchor; the correction is applied only at LSZH. |
| EUROCONTROL monthly taxi-out additional time | `data/taxi_out_additional_time_{2025,2026}.csv`; read by `pru_read` in [`scripts/final_tail_assembly_20261008.py`](scripts/final_tail_assembly_20261008.py) (stage `pru_level`, flag `E218_PRU_LEVEL=1`) | Download the two CSV files from EUROCONTROL's performance data; the stage checks SHA-256 `4aedf195…` and `b72040e0…` and refuses any other copy. A later revision of the published 2026 file would not reproduce the final. |
| Expanded OPDI research ingestion | [`docs/opdi_full_ingestion.md`](docs/opdi_full_ingestion.md), [`scripts/opdi_full_ingest.py`](scripts/opdi_full_ingest.py) | Separate research data path. It records source URLs, versions, hashes and incomplete windows; its larger inventory is not part of v12075. |

The development directories `data/` and `submissions/` are excluded from Git.
Listing a local path is not a statement that its contents ship with a fresh clone.
The release must provide acquisition instructions and permitted artifacts with
checksums. An unavailable source window must remain an explicit gap, not be
silently represented as complete coverage.

## Software and pretrained models

Our source code is [GNU GPLv3](LICENSE). Third-party libraries retain their own
licences. [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock) describe the
main Python dependency set (Python ≥3.12), including Polars, NumPy, LightGBM,
CatBoost and scikit-learn. The GPU stack of the neural member (PyTorch with
ROCm, `rtdl_num_embeddings`) is pinned separately in
[`requirements-gpu.txt`](requirements-gpu.txt).

The numerical-embedding method is from Yury Gorishniy, Ivan Rubachev and Artem
Babenko, *On Embeddings for Numerical Features in Tabular Deep Learning*, NeurIPS
2022. We use their [MIT-licensed implementation](https://github.com/yandex-research/rtdl-num-embeddings)
(`rtdl_num_embeddings` 0.0.12); the package's version-B preset is documented in its
[usage guide](https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/README.md).
The PLE method is prior work, not a new algorithm claimed by this solution.

The LIRF tail uses the TabPFN-2 classifier (`tabpfn` 8.5.0, `ModelVersion.V2`
weights) by Prior Labs, licensed under the
[Prior Labs License 1.1](https://github.com/PriorLabs/TabPFN/blob/main/LICENSE):
Apache 2.0 with an added attribution clause. This solution is **Built with
PriorLabs-TabPFN**. We do not redistribute the weights. Exact reproduction of
v12075 uses the 42 saved tail predictions rather than fresh inference, because
TabPFN inference is not bit-identical across runs and machines.

## Evaluated sources outside this recipe

Research considered alternative ADS-B archives, additional OPDI features,
reanalysis weather, ACARS, airport publications and other context. Their presence
in scripts or survey notes does not establish use in the submitted model. In
particular, the v12075 source inventory does not include commercial flight feeds,
OpenSky Trino trajectory queries, Airplanes.Live, ADSB Exchange, ACARS archives
or ERA5 variables. This is an inventory statement, not a current legal judgment
on every source considered.

The dated [source survey](analysis/sources/sources.csv),
[4 October review](analysis/data_sources_2026-10-04_fable.md) and
[experiment register](HYPOTHESES_LOG.md) retain research context and rejection
reasons. Historical survey verdicts may be superseded: OPDI, previously listed as
excluded in this document, is now used in the LSZH stage. The earlier detailed
literature log is preserved in this file's Git history; its older claims were
not revalidated as part of this source inventory update.

## Attribution

> Challenge movement and Network Manager data: EUROCONTROL Performance Review
> Commission Data Challenge 2026, supported by the OpenSky Network.
>
> ADS-B observations: adsb.lol and its contributors, including the feeder networks
> identified by its archive, under ODbL 1.0.
>
> Airport geometry: © OpenStreetMap contributors, under ODbL 1.0.
>
> METAR observations: Iowa Environmental Mesonet, Iowa State University.
>
> Airport ATFM aggregates: EUROCONTROL Performance Review Unit / Aviation
> Intelligence Portal. Monthly taxi-out additional time per airport (2025, 2026):
> EUROCONTROL Performance Review Unit. OPDI events: EUROCONTROL PRU and the
> OpenSky Network, Open Performance Data Initiative v0.0.2.

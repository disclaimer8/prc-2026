# Data sources, licences and attribution

This register describes the **v12075 submitted recipe**, with primary public
source pages checked on **7 October 2026**. The code licence, a data licence and
an organiser's permission to use data in this challenge are different things.
We record them separately. Public downloadability alone is not an open licence.

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

## Challenge-specific clarifications

The organiser answered data questions in the challenge Discord channel. The
statements that matter for this recipe are quoted verbatim, with UTC times, in
[docs/organiser-clarifications.md](docs/organiser-clarifications.md). In short:

| Date (UTC) | Organiser's answer | Applies to |
|---|---|---|
| 2026-09-16 | All columns except the departure block time and target, in the same row and in other rows, are available to the model | `AOBT_3_flt` and other NM timestamps |
| 2026-09-17 | EUROCONTROL's daily airport files (including arrival ATFM delay, under the portal's non-commercial disclaimer) and the adsb.lol archive (ODbL 1.0) "can be considered open data for the challenge" | Destination ATFM delay features (E163); adsb.lol |
| 2026-09-19, 2026-10-01 | Using open trajectory data, including the flight's own post-pushback track, is allowed | Surface stages and trajectory network |
| 2026-10-04 | OPDI, published without a separate licence, is "open" for eligibility | LSZH correction |

These are challenge-specific permissions. They do not change the general terms of
each portal and do not by themselves grant a right to redistribute EUROCONTROL
files; we therefore do not redistribute them and give acquisition instructions
instead.

## Acquisition, lineage and coverage

| Input | Code or artifact record | Reproduction detail |
|---|---|---|
| Challenge files | [`src/download.py`](src/download.py) | Organiser-issued access credentials are required; credentials are not part of the repository. |
| METAR | `data/metar/<ICAO>.csv`; [`src/features.py`](src/features.py) | Preserve the station, date coverage and downloaded snapshot used in feature construction. |
| OSM geometry | [`analysis/sources/scripts/osm_overpass_download.py`](analysis/sources/scripts/osm_overpass_download.py), [`src/geometry.py`](src/geometry.py) | Preserve the extract and retrieval date; a new live OSM query can produce different geometry. |
| adsb.lol | [`src/adsb_offblock_v2.py`](src/adsb_offblock_v2.py), [`src/adsb_offblock_v5.py`](src/adsb_offblock_v5.py), [`src/adsb_visit.py`](src/adsb_visit.py), [`src/e69_visit_merge.py`](src/e69_visit_merge.py) | Freeze daily archive/release selection and extraction settings. Coverage is uneven; do not interpret missing reception as no movement. |
| Destination ATFM | [`src/e163_build_features.py`](src/e163_build_features.py); `data/ext_pru/apt_dly_{2024,2025,2026}.csv` | Joins `ADES_mvt` and the UTC take-off date. Five `dst_*` columns are present in the submitted FIN1b schema. These are retrospective daily aggregates. |
| OPDI in the submitted correction | [`src/e159_fixed_lineage.json`](src/e159_fixed_lineage.json), [`src/e159_opdi_stage_cv.py`](src/e159_opdi_stage_cv.py), [`src/e159b_apply.py`](src/e159b_apply.py) | The frozen `data/opdi/ev_*.parquet` inputs define the v12075 correction. Callsign/airport matching uses a nearby observed event anchor; the correction is applied only at LSZH. |
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
> Intelligence Portal. OPDI events: EUROCONTROL PRU and the OpenSky Network,
> Open Performance Data Initiative v0.0.2.

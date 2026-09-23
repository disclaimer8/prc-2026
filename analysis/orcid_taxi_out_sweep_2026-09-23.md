# Свип ORCID / Crossref / OpenAlex по taxi-out (23.09.2026)

Итог в одной строке: **открытых наземных данных по LIRF, LTFM, LFPG за январь и июль 2026 года не найдено. Работ, которые описывали бы, какое время аэропорт записывает как off-block (фактическое или плановое), тоже нет. Новых методов с измеренным выигрышем на европейских аэропортах, которых нет в закрытом списке, практически нет.** Ниже перечислено, что проверено, и две-три слабые идеи, честно помеченные как слабые.

## 1. Как искали

**ORCID, публичный API** (`pub.orcid.org/v3.0/expanded-search`, `rows=200`, без авторизации). Запросы и сколько профилей нашлось:

| Запрос | Найдено профилей |
|---|---|
| `"taxi-out"` | 31 |
| `"taxi out"` | 31 |
| `"taxi time"` | 32 |
| `"taxi-time prediction"` | 10 |
| `"taxi times"` | 11 |
| `"off-block"` | 11 |
| `pushback` | 217 (взято 200) |
| `"surface movement" airport` | 11 784 (взято 200; поиск по словам, почти всё мимо) |
| `"A-CDM"` | 86 |
| `"departure metering"` | 13 |
| `"ground delay" airport` | 11 648 (взято 200, почти всё мимо) |
| `"airport surface"` | 157 |
| `"airport collaborative decision making"` | 10 |
| `"ADS-B" airport` | 12 143 (взято 200) |
| `"air traffic management"` | 1 013 (взято 200) |

Веб-запрос `orcid.org/orcid-search/search?searchQuery="taxi-out"` работает через тот же индекс, что и строка `"taxi-out"` в таблице: 31 профиль.

- Уникальных ORCID iD: **1 053**. Для каждого выгружен список работ (`/v3.0/<id>/works`), ошибок 0.
- Просмотрено работ: **41 429**. Отобраны работы 2018–2026 годов, в названии которых есть taxi, off-block, pushback, A-CDM, departure metering, airport surface, block time, ADS-B вместе с airport или ground и тому подобное: **656 совпадений у 406 человек**. После отсева по смыслу (такси-машины, электротягачи, покрытия ВПП, геополитический «pushback» и прочее) осталось **290 работ у 187 человек**. Из них по теме действительно около 120.
- Поиск ORCID идёт по тексту профиля, поэтому добавлены поиски по названиям работ:
  - **OpenAlex**: 16 запросов, `from_publication_date:2018`, по 100 записей на запрос. Запросы: taxi-out prediction, off-block time, pushback ADS-B, A-CDM milestones, block time data quality, Istanbul/Fiumicino/CDG taxi и другие.
  - **Crossref**: 7 запросов `query.title`, по 100 записей.
  - **OpenAlex, только датасеты** (`type:dataset`): 6 запросов (taxi, airport ADS-B, A-CDM, off-block, airport surface movement, aircraft ground trajectories).
  - Дополнительно 12 узких запросов про качество записанных времён: OOOI accuracy, AOBT data quality, APDF, ADS-B-derived off-block и так далее.
  - Итого после фильтра по названию **547 записей**, из них авиационных около 240.
- Аннотации прочитаны у 35 работ через OpenAlex `abstract_inverted_index`, Crossref и страницы Zenodo/HAL. Полный текст прочитан у 4: De Falco (ATRDS 2025, Zenodo), Falque и соавт. (DKE 2024, HAL), Schultz и соавт. (AIDA-AT 2020, HAL), Çömez и İnan (IMIENS 2024).
- **Недоступны:** Zúñiga и Argilés-Herrero (TRPro 2024) и Stloukal и соавт. (TRPro 2026). ScienceDirect отвечает 403, Elsevier API требует ключ. У Stloukal есть аннотация, у Zúñiga нет даже её. Недоступны также закрытые статьи IEEE/AIAA: Mori 2021, Kato и Itoh 2023, Stloukal 2026 (pushback), Šimić 2024, Dönmez 2024 (CSTP), Coupe 2023, Park 2025. По ним есть только аннотации или только названия.

## 2. Исследователи (отобраны по теме; число работ = совпадения по теме в ORCID-профиле за 2018–2026)

| Имя | ORCID | Аффилиация (по профилю) | Работ по теме |
|---|---|---|---|
| Duc-Thinh Pham | 0000-0001-5156-8171 | VinUniversity / NTU Singapore | 10 |
| Mary E. Johnson | 0000-0002-6572-0979 | Purdue / UT Arlington | 10 |
| Malte von der Burg | 0000-0002-0190-3274 | TU Delft | 7 |
| Jianan Yin | 0000-0002-1469-8759 | China Univ. of Petroleum / NUAA | 6 |
| Jiansen Wang | 0009-0001-2663-834X | Purdue / CAUC | 5 |
| Hasnain Ali | 0000-0002-2347-7739 | NTU Singapore | 4 |
| Sameer Alam | 0000-0002-7379-8223 | NTU Singapore | 3 |
| Michael Schultz | 0000-0003-4056-8461 | DLR / UniBw München | 3 |
| Susan L. Hotle | 0000-0002-8859-1196 | Virginia Tech / GRA Inc | 3 |
| Xiaowei Tang | 0000-0002-0906-6351 | NUAA | 3 |
| Xavier Olive | 0000-0002-2335-5774 | ONERA | 2 |
| Kadir Dönmez | 0000-0002-1236-0498 | Eskişehir Technical / Anadolu Univ. (**LTFM**) | 2 |
| Ryota Mori | 0000-0002-2804-2048 | ENRI / Kobe Univ. | 2 |
| Bo Stloukal | 0009-0005-9449-245X | ČVUT Praha | 2 |
| Sandeep Badrinath | 0000-0002-9094-1079 | MIT | 2 |
| Paolino De Falco | 0000-0001-5849-6430 | Innaxis / EUROCONTROL-проекты | 2 (+ Zenodo) |
| Thibault Falque | 0000-0003-2803-1530 | Univ. d'Artois / Exakis (**LFPG**) | 2 |
| Eri Itoh | 0000-0002-6957-4366 | Univ. of Tokyo / ENRI | 2 (в ORCID); ещё Liu & Itoh 2026 |
| Tony Diana | 0000-0002-6692-1131 | FAA | 2 |
| Kailin Chen, Daniel J. Graham | 0000-0002-9061-4633, 0000-0002-6971-8737 | Imperial College | 1 |

Профили с совпадениями, но не по теме (электротягачи, эмиссии, покрытия, UAM, камеры), в таблицу не вошли. Их около 170 из 187.

## 3. Релевантные работы

Условные обозначения: **НОВОЕ** — нет ни в `literature_taxi_out_2026-09-14.md`, ни в разделе 0 брифа; **ЛИТ** — уже есть в обзоре литературы; **ПРОВЕРЕНО** — механизм уже проверен командой.

| Год | Авторы | Название | DOI / ссылка | Данные, аэропорты | Открытые данные/код + лицензия | Статус |
|---|---|---|---|---|---|---|
| 2025 | De Falco | ML predictions of TOBT and turnaround duration for all European CDM airports (ATRDS 2025) | [10.5281/zenodo.20127242](https://doi.org/10.5281/zenodo.20127242) | Все 33 европейских A-CDM, **сообщения ETFMS за 2023 год** (закрытые), CatBoost с MAE, 66 % прогнозов TOBT в допуске, MAE 4,3 мин. Лучше всего BRU, хуже всего CDG (≈82 % / 61 %) | Только PDF статьи, **CC BY-SA 4.0**. Данных и кода нет | НОВОЕ как ссылка; механизм «EOBT перекрывает TOBT, если позже» = наши режимы EOBT/LOBT → ПРОВЕРЕНО |
| 2025 | De Falco, Delgado | Probabilistic approach for prediction of flight processes… block time (ESWA) | [10.1016/j.eswa.2025.129313](https://doi.org/10.1016/j.eswa.2025.129313) | Block time рейсов; регрессия + классификатор поправки → распределение | CC BY-NC (статья); данных нет | ПРОВЕРЕНО (наша смесь «классификатор режимов + регрессор» — тот же приём) |
| 2024 | Falque, Mazure, Tabia | ML for predicting off-block delays: Paris-CDG (DKE; ICAART 2023) | [10.1016/j.datak.2024.102303](https://doi.org/10.1016/j.datak.2024.102303), [HAL](https://univ-artois.hal.science/hal-05028916) | **LFPG**, данные ADP, 08.2021–09.2022 (плюс 2019). Цель AOBT − SOBT. LightGBM/FastTree, MAE ≈ 6 мин против 11 у базовой модели. Признаки TSAT − SOBT, число обновлений TOBT, доля задержанных вылетов по аэропорту/терминалу/авиакомпании в окне w мин | Статья CC BY; **данные ADP закрыты** | Признаки «задержка соседей» = E77 → ПРОВЕРЕНО |
| 2024 | Zúñiga, Argilés-Herrero | A post-analysis study of the departure phase based on A-CDM: start-up phase (TRPro) | [10.1016/j.trpro.2024.11.031](https://doi.org/10.1016/j.trpro.2024.11.031) | A-CDM, фаза start-up (ASRT/ASAT/TSAT). Аэропорт по метаданным не виден, видимо испанский | CC BY-NC-ND; **полный текст недоступен (403)** | НОВОЕ, не прочитано |
| 2026 | Stloukal, Pleninger, Hospodka, Had | Analysis of ground bit changes in ADS-B data from airport operations (TRPro) | [10.1016/j.trpro.2026.01.014](https://doi.org/10.1016/j.trpro.2026.01.014) | LKPR, 8 дней января 2024, 2 185 смен ground bit: **9,1 % смен недостоверны (13 % бортов)**. Ложные смены при посадке и аномальные на перроне | Статья CC BY-NC-ND; данных нет | ПРОВЕРЕНО (урок E25: земля по флагу `ground` ненадёжна) — независимое подтверждение |
| 2026 | Stloukal, Had, Hospodka | Analyzing the duration of aircraft pushback operations using ADS-B data (NTCA) | [10.23919/ntca68808.2026.11524162](https://doi.org/10.23919/ntca68808.2026.11524162) | Судя по аффилиации, LKPR; ADS-B | Закрыта, аннотации нет | НОВОЕ, не прочитано |
| 2020 / 2021 | Schultz, Olive, Rosenow, Fricke, Alam | Analysis of airport ground operations based on ADS-B (AIDA-AT); Data-driven airport management… milestones from ADS-B (JATM) | [10.1109/AIDA-AT48540.2020.9049212](https://hal.science/hal-03182721), [10.1016/j.jairtraman.2021.102164](https://doi.org/10.1016/j.jairtraman.2021.102164) | **LSZH**, OpenSky. Вехи из ADS-B: AIBT, **ARDT = включение транспондера на стоянке**, AOBT, взлёт; библиотека `traffic` | Код `traffic` MIT; данные OpenSky (не открытая лицензия) | ARDT ≈ наш `vm_first_seen` → ПРОВЕРЕНО |
| 2025 | Olive, Krummen, Figuet, Alligier | Filtering techniques for ADS-B trajectory preprocessing (JOAS) | [10.59490/joas.2024.7882](https://doi.org/10.59490/joas.2024.7882) | Фильтры шума и квантования в `traffic` | CC BY; код MIT | ПРОВЕРЕНО по сути (E44 признаки приёма; своя фильтрация) |
| 2020 | Badrinath, Balakrishnan, Joback, Reynolds | Impact of off-block time uncertainty on the control of airport surface operations (Transp. Sci.) | [10.1287/trsc.2019.0957](https://hdl.handle.net/1721.1/130429) | EWR, DFW, CLT. Ошибка EOBT авиакомпаний, сетевые очередные модели | CC BY-NC-SA (авторская версия) | ЛИТ-класс (очередные модели); к записи block не относится |
| 2020 | Badrinath, Balakrishnan, Ma, Delahaye | Comparative analysis of departure metering at US and European airports (JAT) | [10.2514/1.D0179](https://enac.hal.science/hal-02617029) | **LFPG** и CLT, стохастическая симуляция | HAL; данных нет | НОВОЕ, но для прогноза не нужно (про metering) |
| 2021 / 2024 | Mori | Off-block time prediction using operators' prediction history (DASC); Prediction of off-block time distribution (J. Air Transp.) | [10.1109/DASC52595.2021.9594354](https://doi.org/10.1109/DASC52595.2021.9594354), [10.2514/1.D0359](https://doi.org/10.2514/1.D0359) | История обновлений TOBT → распределение Johnson-SU (нейросеть) | Закрыта; данных нет | ЛИТ (№ 20). У нас истории TOBT нет |
| 2019 | Mirmohammadsadeghi, Hotle, Trani, Gulding | Taxi event extraction from surveillance for surface performance evaluation (JAT) | [10.2514/1.D0160](https://hdl.handle.net/10919/121088) | 6 аэропортов США, 07.2015: сравнение OOOI (округление до минуты, **у неотчитывающихся авиакомпаний времена оценены по похожим рейсам**) с ASDE-X | Public domain (авторская версия); данных нет | НОВОЕ как ссылка по пункту B (единственное найденное упоминание подстановки времён), но это США и ASPM |
| 2020 | Li, Hotle | Evaluation of taxiing behavior by airport and flight characteristics (AIAA) | [10.2514/6.2020-2883](https://hdl.handle.net/10919/121094) | ASDE-X, 6 аэропортов США; скорость руления по TDG, movement / non-movement area | Авторская версия; данных нет | ПРОВЕРЕНО (тип ВС, стоянка × ВПП, признаки трека E30/E36) |
| 2024 | Çömez, İnan | Performance evaluation of ML algorithms in estimating taxi times at Istanbul Airport (IMIENS) | [10.58190/imiens.2024.102](https://doi.org/10.58190/imiens.2024.102) | **LTFM**, источник данных не назван. 432 стоянки, 11 ВПП-ID, расстояние стоянка–ВПП 490–11 514 м, METAR. PCA + ANN, **случайный split 80/20, метрики на нормированной цели** (R² 0,96) | Статья CC BY-SA; **данных нет** | Методически слабая; нового нет |
| 2024 | Dönmez | The operational dynamics of end-around taxiways: insights from Istanbul Airport (CSTP) | [10.1016/j.cstp.2024.101303](https://doi.org/10.1016/j.cstp.2024.101303) | **LTFM**, end-around taxiways | Закрыта | НОВОЕ, не прочитано; скорее всего геометрия, которая у нас уже есть в `ap_stand` × `ap_runway` |
| 2025 | Dönmez | Algorithm-based runway configuration management: a comprehensive ADS-B data analysis (JTTE) | [10.1016/j.jtte.2024.09.011](https://doi.org/10.1016/j.jtte.2024.09.011) | Турецкий аэропорт с параллельными ВПП (по описанию похоже на LTFM или SAW), ADS-B: время на земле, «zero-touch» остановки, hot spots по конфигурациям | CC BY-NC-ND; данных нет; источник ADS-B не уточнён в аннотации | ПРОВЕРЕНО (остановки и конфигурация ВПП в E30 и регрессоре) |
| 2019 | Aydoğan, Çetek | Runway capacity enhancement analysis of end-around taxiway at Istanbul New Airport | [10.29130/dubited.457436](https://doi.org/10.29130/dubited.457436) | LTFM, симуляция | — | Не нужно |
| 2026 | Momtaz | Rolling-calibrated conformal decision support for seasonal airport taxi-time planning under distribution shift (SOIC) | [10.19139/soic-2310-5070-3782](https://doi.org/10.19139/soic-2310-5070-3782) | **EUROCONTROL CODA**, 17 сезонов IATA, 722 аэропорта, по wake-категориям | CC BY; CODA «information only» (отвергнута в `review5`) | ЛИТ (CODA-приор, № 11) и лицензионно закрыто |
| 2024 | Šimić, Begović, Šabić, Muharemović | Using ML to predict additional taxi-out time as KPI in the EUROCONTROL zone (LNNS) | [10.1007/978-3-031-66271-3_26](https://doi.org/10.1007/978-3-031-66271-3_26) | Судя по названию, агрегаты PRU | Закрыта, аннотации нет | НОВОЕ, но агрегатный KPI: к прогнозу по рейсу не переносится |
| 2025 | Dalmau, Attia | A collection of ML models for improved airport operations amidst adverse weather (EJTIR) | [10.59490/ejtir.2025.25.1.7487](https://doi.org/10.59490/ejtir.2025.25.1.7487) | Европа (NM/EUROCONTROL), пиковый темп обслуживания (P99), вероятность регуляции ATFM по погоде и календарю; **confident learning** против шума меток, монотонные ограничения | CC BY; данных нет | Регуляции: Г6 и DDR2 → ПРОВЕРЕНО. Confident learning → НОВОЕ (слабое, см. п. 5) |
| 2026 | Chen, Anupriya, Bansal, Anderson, Findlay, Graham | Understanding congestion in airport surface operations using 3D fundamental diagrams (TR-B) | [10.1016/j.trb.2026.103429](https://doi.org/10.1016/j.trb.2026.103429) | 5 аэропортов (в аннотации не названы). Задержка против загрузки ВПП и загрузки земли, причинная модель | CC BY; данных нет | Класс «выразимо имеющимися счётчиками» → ПРОВЕРЕНО (разд. 21f) |
| 2025 | Tang, Zhang, Zhang, Wu | A road-level transport network model… taxi-out (Aerospace) | [10.3390/aerospace12080721](https://doi.org/10.3390/aerospace12080721) | ZSPD, A-CDM | CC BY | ЛИТ (№ 23) |
| 2025 | Lim, Alam, Lilith, Tan | GNN with spatiotemporal flow features for taxi-out (J. Air Transp.) | [10.2514/1.D0369](https://doi.org/10.2514/1.D0369) | ATL, A-SMGCS | — | ЛИТ (№ 18) |
| 2026 | Liu, Itoh; Wu и соавт.; Lu и соавт. | Stochastic departure metering…; stacking + SHAP; DDA-SIM-ATT | см. обзор | RJTT, ZGSZ, хаб КНР | CC BY | ЛИТ (№ 25–27) |
| 2025 | Wang, Gupta, Johnson | Reassessing ICAO standard taxi/ground idle time: 71 US hubs (Aerospace) | [10.3390/aerospace12030220](https://doi.org/10.3390/aerospace12030220) | FAA ASPM, агрегаты | CC BY | Не нужно (агрегаты США) |
| 2025 | Park, Hwang, Ryu, Lee | Predicting taxi times using airport surface movement data (DASC) | [10.1109/DASC66011.2025.11257221](https://doi.org/10.1109/DASC66011.2025.11257221) | RKSI, наземное наблюдение, XGBoost, 13 признаков (самолётов на поверхности, самолётов на маршруте, длина маршрута), R² 0,74 | Закрыта | ПРОВЕРЕНО (счётчики соседей E28, граф OSM E51) |
| 2026 | Tagusari, Kayanuma, Umetani | From raw ADS-B signals to reproducible flight trajectories: open workflow for Narita (ISPRS) | [10.5194/isprs-archives-L-4-W1-2026-303-2026](https://doi.org/10.5194/isprs-archives-L-4-W1-2026-303-2026) | RJAA, свой RTL-SDR, 04.2024–03.2025 | CC BY (статья) | Не наш аэропорт |
| 2026 | von der Burg | Data and code underlying the PhD thesis… airport surface movement (4TU) | [10.4121/00e879bd-…](https://doi.org/10.4121/00e879bd-adf5-4d07-b123-cbd0bcf44695) | Граф рулёжек EHAM (SVG) + синтетика | **Apache-2.0**; наблюдений нет | Граф рулёжек у нас уже из OSM (E51 отклонён) |
| 2024 | Navarro и соавт. | AmeliaTF: large model and dataset for airport surface movement forecasting | [10.2514/6.2024-4251](https://doi.org/10.2514/6.2024-4251) | 42 аэропорта **США**, SWIM | CC BY 4.0 | ЛИТ (Amelia) |
| 2023 | Waltert, Figuet; 2026 Hänggi и соавт. | Rapid exit taxiways by ADS-B (JOAS); AROT and exit taxiway prediction (JOAS) | [10.59490/joas.2023.7207](https://doi.org/10.59490/joas.2023.7207) | LSZH, OpenSky | CC BY | ЛИТ (№ 28), к тому же прилёты |
| 2022 | Abdelghany, Guzhva, Abdelghany | The limitation of ML-based models in predicting airline flight block time (JATM) | [10.1016/j.jairtraman.2022.102339](https://doi.org/10.1016/j.jairtraman.2022.102339) | Block time авиакомпаний США | Закрыта, аннотации нет | Не прочитано; к записи AOBT аэропортом отношения, судя по названию, не имеет |

**Датасеты** (OpenAlex `type:dataset`, 6 запросов, около 4 600 записей; после фильтра авиационных несколько штук):

| Датасет | Лицензия | Охват | Годится? |
|---|---|---|---|
| KSFO ADS-B trajectories, four months arriving (figshare 10.6084/m9.figshare.16712956) | figshare по умолчанию (не проверял) | KSFO, прилёты, 2021 | Нет: США, прилёты |
| Jakarta FIR ADS-B 1090ES (Zenodo 10.5281/zenodo.20085852) | — | 11–12.2020, Индонезия | Нет |
| 4TU: surface movement thesis data (von der Burg) | Apache-2.0 | EHAM, только граф | Нет наблюдений |
| Stanford: Filtering framework for aircraft localization during taxiing (10.25740/zr063zq9238) | — | камеры/сегментация | Нет |
| De Falco, Zenodo 20127242 | CC BY-SA 4.0 | **это статья, а не данные** | Нет |

В ORCID-профилях найденных исследователей ни одного датасета с наземными наблюдениями европейских аэропортов за 2025–2026 годы нет.

## 4. Пункт A: открытые наземные данные LIRF / LTFM / LFPG за январь и июль 2026

**Не найдено.** Прямо:

- Ни в одном из 1 053 ORCID-профилей, ни в 547 отобранных записях OpenAlex/Crossref, ни среди датасетов OpenAlex нет открытого набора с ADS-B-точками на земле, A-SMGCS-трассами, вехами A-CDM (AOBT/TOBT/TSAT) или OOOI для Фьюмичино, Стамбула или Шарля-де-Голля за январь или июль 2026 года. Годы и лицензия здесь даже не понадобились: подходящего набора нет в принципе.
- Работы по этим трём аэропортам опираются на закрытые данные. LFPG: Falque (данные ADP), Badrinath (симуляция), Herrema 2018 (ЛИТ). LTFM: Çömez (источник не назван), Dönmez (ADS-B, источник не уточнён, 2019–2023). LIRF: De Falco 2023 и 2025 (A-CDM и ETFMS 2023, закрыто).
- Все европейские ADS-B-работы по земле (Schultz/Olive — LSZH, Stloukal — LKPR, Waltert и Hänggi — LSZH) используют OpenSky или собственные приёмники. OpenSky и OPDI уже отвергнуты по лицензии (SOURCES.md, разд. 2).
- Цена ошибки поиска в этом пункте велика, поэтому оговорка: ORCID ищет по профилю, а не по полным текстам, и Zenodo/figshare отдельно, вне OpenAlex, я не прочёсывал. Но запрос OpenAlex `type:dataset` + «airport ADS-B» (74 записи) и «aircraft ground trajectories» (48 записей) не дал ни одного европейского набора. С закрытием направления в SOURCES.md расхождений нет.

## 5. Пункт B: какое время аэропорт пишет как off-block

**Исследований качества записанных block-времён в Европе (APDF/AODB, подстановка SOBT/EOBT вместо AOBT) не найдено.** Ближайшее, что есть:

- **Mirmohammadsadeghi и соавт. 2019 (JAT, США).** В ASPM времена OOOI у неотчитывающихся авиакомпаний *оцениваются по похожим рейсам*, а отчётные округлены до минуты. Это описание механизма, но для США. Предсказывать подмену эта работа не учит.
- **De Falco 2025.** В A-CDM EOBT перекрывает TOBT, если EOBT позже, а TOBT обновляют только при сдвиге больше чем на ±5 мин. Это объясняет, почему поданные времена (LOBT/EOBT/IOBT) так часто совпадают с «block». У команды это уже выражено режимами IOBT, LOBT, EOBT, SCHED.
- **Stloukal и соавт. 2026.** Недостоверные смены ground bit (9 % смен, аномалии на перроне) — это качество ADS-B, а не AODB. Урок E25 уже выучен.
- **Dalmau и Attia 2025.** Confident learning как защита от шума меток. Общий инструмент, подмену он не объясняет.

Вывод по B: литература механизм подмены **не описывает**. Эмпирика команды (раздел 17 брифа, «block = SCHED ±6 с», AODB только Рима, секундное разрешение меток у EDDF/EHAM/LSZH) глубже всего, что нашлось.

## 6. Топ-5 идей (честно: по-настоящему новых и дешёвых две, и обе слабые)

1. **Confident learning для меток режимов** (Dalmau и Attia 2025; `cleanlab`). *Механизм:* по OOF-вероятностям классификатора найти строки 2025 года, чья разметка «обычный / SCHED / IOBT / LOBT / EOBT» вероятно неверна, и выбросить их из обучения классификатора или снизить им вес. Пример такой строки — обычное руление, случайно попавшее в ±6 с от SCHED вне трёх аэропортов с секундным разрешением. *Популяция:* обучение классификатора режимов; в прогнозе больше всего затронуты строки с NM при D около 0 и узкие совпадения. *Чем отличается от закрытого:* К2 менял допуск разметки (гигиена, +0,02 с), E6/E58 рекалибровали выход, Isolation Forest искал выбросы по признакам. Отбора самих меток по согласию с OOF-моделью не было. *Ожидание:* низкое. Шум ±6 с уже узкий, а выигрыш режимов сидит в LIRF без NM, где правит таблица и TabPFN. Не больше 0,1 с; брать только если дешевле прочего.
2. **Прочитать Zúñiga и Argilés-Herrero 2024 (A-CDM start-up phase) и Stloukal 2026 (длительность pushback по ADS-B)**, когда будет доступ к полным текстам (403 здесь). Если в первой есть распределение ASAT − AOBT или ASRT − AOBT на испанском A-CDM-аэропорту, это даст априорный разрыв «разрешение запуска → фактический off-block». Он может объяснить, какое событие AODB пишет как AOBT: запрос запуска или начало буксировки. *Популяция:* строки LEMD/LEBL с NM. *Статус:* не идея, а непрочитанный источник. Поднимать не раньше, чем будет текст.
3. **Длительность pushback как отдельная компонента в ADS-B-этапе** (Stloukal 2026; Schultz 2020/2021 — вехи ARDT → AOBT → начало руления). *Механизм:* разделить трек на «стоит на стоянке с включённым транспондером», «толкание назад» (движение с разворотом курса около 180° внутри зоны 150 м) и «руление вперёд». Длительность толкания подать признаком. *Популяция:* строки с покрытием ≥ 2. *Чем отличается:* E36 брал уход из зоны 150 м и последнюю остановку, E30 — путь и остановки. Отдельной фазы толкания, похоже, не было. *Но* она почти наверняка выразима через уже имеющиеся «время до ухода из 150 м» и «число остановок» — класс «поглощаемого входа» (разд. 21f). Ожидание около 0; проверить поглощение до прогона.
4. **Турецкий материал по конфигурациям ВПП LTFM** (Dönmez 2024 и 2025: end-around taxiways, режимы параллельных ВПП, zero-touch остановки). *Популяция:* LTFM, 47 581 строка ranking. *Почему почти бесполезно:* наземных точек LTFM ноль в 2025 и 2026 годах, а конфигурация восстанавливается из собственных `ap_runway` и соседей. Может пригодиться только как справка о том, какие пары ВПП × стоянка идут через end-around (длинный путь), если в OSM это не отражено. E51 (граф OSM) уже показал, что категории стоянки × ВПП это несут.
5. **Ничего по методам сверх закрытого списка.** Европейские работы 2023–2026 (Falque — LFPG, De Falco — все CDM, Šimić — KPI, Chen — TR-B, Momtaz — CODA) либо прогнозируют другую цель (AOBT − SOBT, TOBT, агрегатный KPI), либо используют признаки, которые у команды уже есть (задержка соседей — E77; загрузка ВПП и земли — счётчики E28; CODA — отвергнута по лицензии). Пятой содержательной идеи нет, и раздувать список я не стал.

## 7. Файлы свипа

Сырые выгрузки лежат в scratchpad сессии (`…/scratchpad/orcid/`): `orcid_search.json`, `works/*.json` (1 053 профиля), `hits.txt`, `oa_cr.json`, `oacr.txt`, PDF De Falco, Falque, Schultz, Çömez.

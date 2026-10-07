# Разбор предложенных механизмов: ночь 6–7 октября 2026

Исследовательский разбор, а не результат обучения. Проверены текущие локальные таблицы, история закрытых механизмов и первичные публикации/документация. Идеи 4 и 5 про causal inference совпадают: ниже шесть уникальных гипотез и отдельная интеграционная идея digital twin. Обещанные проценты и места не используются для выбора. Протоколы ниже — предложения до регистрации, а не уже запущенные эксперименты.

## 0. Решение и общая постановка

### Takeaway

Первым поставить **событийный контекст аэропорта с проверкой графовой структуры**. Вторым — **последовательность своего/предыдущего визита и наблюдаемой погоды**. Это два способа предъявить модели информацию, потерянную при агрегировании. Физический decoder, условная диффузия и причинная устойчивость заслуживают отдельных небольших сравнений, но их корректная постановка важнее названия архитектуры. RL полезен для управления аэропортом; для этого конкурса его наиболее разумная часть — оценка скрытого состояния очереди по реальным событиям.

### Cited Findings

- На ranking имеется 689 534 движения, из них 344 841 DEP. У 344 835 DEP есть хотя бы одно другое движение того же аэропорта в ±1 час; у 327 402 — не менее 64. У всех 47 581 LTFM DEP есть контекст, у 46 985 — не менее 64 событий. Это новый замер доступности входа, **не измерение его предсказательной силы**. [Замер](event_support.json).
- Для ranking NM-LTFM существующий сенсорный отчёт показывает 46 320 строк без своего и предыдущего ground-трека из 46 862; NM-LIRF июля имеет только 966 строк со своим ground-треком, но 11 359 с предыдущим. Отчёт относится к существующим extractor, а не к ещё не построенной последовательности. [Источник](/Users/denyskolomiiets/prc-2026/data/current_errors_20261005/report.json).
- Цель модели — записанное TAXITIME. Уже реализованы regular, четыре плановых режима и dayshift; метка может отражать переписанные часы, а не реальное время движения. [Определение](/Users/denyskolomiiets/prc-2026/src/train.py:1).

### Inferences

Общий опасный разрыв всех физических предложений: `y_recorded` и фактическое `q_physical` — разные случайные величины. Разумная большая архитектура должна допускать наблюдаемый процесс записи:

`наблюдаемые события → скрытое состояние движения/очереди → q_physical;`

`аэропорт + перевозчик + часы + практика записи → режим Z;`

`y_recorded = q_physical` при обычном режиме, иначе значение соответствующих часов/даты.

Модель движения не должна жёстко запрещать корректную конкурсную метку только потому, что она физически невозможна. У всех новых ветвей нужен одинаковый existing production wrapper, хвост и ограничения; иначе меняется одновременно несколько механизмов.

| Идея | Сильная проверяемая версия | Что доступно сейчас | Приоритет |
|---|---|---|---|
| TGN/TGAT/DySAT | Условные взаимодействия конкретных событий, а не средний остаток аэропорта | Все PRC движения, часы, ВПП, тип, перевозчик; координаты только у части | Первый ночной paired screen |
| PINN | Физически согласованный latent trajectory decoder с мягкой кинематикой | Наблюдаемые ADS-B точки, геометрия; нет фактических массы/тяги/торможения | После sequence encoder, отдельная абляция loss |
| Multimodal | Порядок трека + история погоды + событийный контекст с явными масками | METAR уже почти полностью присоединён; raw tracks доступны частично | Второй ночной механизм |
| Causal, идеи 4/5 | Проверка устойчивого механизма по временным средам и процесса записи | Наблюдаемые ковариаты есть; причинная идентификация не доказана | Диагностика, затем одна заранее выбранная абляция |
| Diffusion | Conditional response diffusion как альтернатива явной смеси | Все табличные входы и target train | Отдельный bounded screen после likelihood baseline |
| RL | Фильтр скрытой очереди по observed policy; управлять виртуальным аэропортом отдельно | Последовательность движения есть, complete action logs и environment нет | State estimator можно исследовать; PPO сейчас не первый fit |
| Digital twin | Модульная оценка состояния + отдельный measurement model | Части доступны, end-to-end twin не откалиброван | Интеграционный план после независимых абляций |

### Gaps

Нет результата нового графа, последовательности, physics loss, diffusion или RL на наших данных. Прежний pre-ADS-B OOF и будущий полный production-chain OOF нельзя объявлять одной и той же метрикой. Ждать Cat90 для первого screen пользователь не требует; ограниченный screen честно подписывается.

## 1. Динамический граф аэропорта: TGN, TGAT, DySAT

### Takeaway

Идея содержательная. При одинаковом количестве вылетов задержка может различаться из-за порядка прибытий, сочетания ВПП, классов wake, стоянок, волны подачи и длинных пауз. Это зависимости между конкретными соседями, которые среднее/счётчик может потерять. **Начинать следует с графа наблюдаемых событий; физические конфликты — отдельная ветвь только на поддержанном координатами подмножестве.**

### Cited Findings

- TGN представляет динамический граф последовательностью событий и сочетает memory с graph operators. Оригинальная работа проверяет динамические графовые задачи, а не наш TAXITIME. [Статья](https://arxiv.org/abs/2006.10637), [код авторов](https://github.com/twitter-research/tgn).
- TGAT учит взаимодействия времени и признаков в temporal neighborhood, поддерживает новые узлы. Для рейсов, появляющихся по одному разу, это ближе к нужной постановке, чем запоминание отдельного ID. [Статья](https://arxiv.org/abs/2002.07962), [код авторов](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs).
- DySAT объединяет structural attention и temporal attention по снимкам графа. Это другой способ дискретизации, а не три взаимозаменяемых названия одной готовой модели. [Статья](https://arxiv.org/abs/1812.09430), [оригинальный TensorFlow-код](https://github.com/aravindsankar28/DySAT).
- В PRC ranking имеются MVT/LOBT/IOBT/EOBT/AOBT, aircraft type, wake, operator, runway, stand, phase. Полей фактического приоритета диспетчера, ACARS door/engine state, taxi clearance или planned taxi route в прочитанной схеме нет. [Схема исходной таблицы](/Users/denyskolomiiets/prc-2026/data/prc-2026-datasets/ranking.parquet), [контекстный замер](event_support.json).

Точные имена raw parquet имеют суффиксы: `MVT_ID_mvt` (Float64), `PHASE_mvt`, `ADEP_mvt`, `ADES_mvt`, `MVT_TIME_UTC_mvt`, `SCHED_TIME_UTC_mvt`, `RUNWAY_mvt`, `STAND_mvt`, `AIRCRAFT_TYPE_mvt`, `AOBT_3_flt`, `LOBT_flt`, `IOBT_flt`, `EOBT_1_flt`, `WK_TBL_CAT_flt`, `AIRCRAFT_OPERATOR_flt`. Времена — Datetime(us, UTC). Выше используются смысловые сокращения, не буквальные имена колонок. Запрещённые input поля: `BLOCK_TIME_UTC_mvt`, `TAXITIME_SEC_mvt`. [Сохранённая проверка схемы](verified_inputs.json).
- E151 проверял агрегаты arrivals/departures вокруг собственного интервала; E35/E60 и E183 — различные соседские taxi/residual proxies. E157 wake ladder дал полезный вход; E184 overdue gate queue не прошёл trim. Эти результаты ограничивают конкретные признаки, но не являются сравнением conditional graph с теми же полными event tokens. [Журнал](/Users/denyskolomiiets/prc-2026/HYPOTHESES_LOG.md), [E183](/Users/denyskolomiiets/prc-2026/analysis/e183_probe_2026-10-04.md).

### Inferences

**Почему может быть новое.** Средний остаток других вылетов отвечает на вопрос «есть ли общий сдвиг». Граф может отвечать «какая группа влияет именно на этот рейс, в какой последовательности». Нулевой средний эффект не исключает разнонаправленные условные эффекты. Но это не лазейка для бесконечного повторения E183: новая модель должна победить matched input control, а не только старую модель с беднее представленными данными.

**Три различных графа.**

1. Event-time graph: один узел — одно движение, связи по времени, аэропорту и наблюдаемой ВПП/stand. Почти полное покрытие, включая LTFM. Узел нельзя назвать «самолётом с известной историей», если ICAO identity не подтверждена.
2. Physical interaction graph: два борта действительно одновременно расположены рядом на земле. Нужны общая временная сетка, допустимый age последней позиции, ground/stale flags и качество identity. Близость сама по себе ещё не конфликт; нужны направления и топология путей. Отсутствие точки — отсутствие измерения, не отсутствие самолёта.
3. Persistent resource graph: аэропорт/ВПП/stand-group — постоянные узлы, события рейсов меняют их latent state. Это разумная область для TGN memory. На первой пробе проще stateless local graph: он исключает утечку memory между train/holdout и уменьшает инженерную нагрузку.

**Точный первый протокол, готовый для регистрации.**

- Query: тот же FIN1b PLE encoder, что в E190. Контекст: до 64 других движений, по 32 перед/после MVT, в ±3600 с, тот же локальный аэропорт (DEP=ADEP, ARR=ADES), deterministic tie-break по ID. Target self всегда исключён. Сохранить missing masks и число доступных событий.
- Tokens: ΔMVT; DEP/ARR; runway; stand-prefix; aircraft type/wake; operator; MVT−AOBT/LOBT/SCHED при наличии. Содержимое raw BLOCK_TIME и TAXITIME **всех** соседей исключить из первой пробы; никаких target-derived planned labels, residuals или скрытых OOF teacher predictions.
- A: query-only PLE baseline. B: те же query+tokens, shared token MLP и mean/max pooling (flat/deep-set control). C: query-conditioned cross-attention, два слоя 64d, 4 heads. D: те же tokens, два graph message-passing слоя + query pooling. Размер hidden/head примерно одинаков, одинаковые epochs/seed/optimizer. Это различает пользу нового входа, условного выбора и собственно topology.
- Для D graph edges фиксируются до fit: каждый event соединяется с ближайшими по времени предшественником/последователем на той же известной ВПП; дополнительно ближайшие по времени события того же известного stand-prefix; padding не образует edges. Edge features — signed Δtime, relation type, same runway/stand/wake. Между соседями происходит реальный message passing, иначе star attention нельзя выдавать за доказательство преимущества GNN.
- Здесь runway/stand edge — наблюдаемая операционная связь, **не утверждение физического конфликта**. Wake encoded as type, а не вручную выдуманный диспетчерский приоритет.
- Regression objective, mask и proxy идентичны E190 regular member: `(y−observable_proxy)/scale` для existing regular label. Не учить поправку к OOF учителям, которые могли видеть внешний holdout. Входы с абсолютными уникальными ID/date не поступают в сеть.
- Holdouts 1+7 и 4+10, один seed, фиксированные 8 epochs. Это exploratory paired screen, не новая оценка всей production chain. Norm/TE/embeddings fitted только на train. Доступность unlabeled соседних событий точно та же, что в ranking; holdout labels никогда не входят в graph/memory.
- Контроль механизма: (i) wrong-day context внутри того же аэропорта/близкого часа; (ii) у обученного D переставить связи между допустимыми context nodes, сохраняя по возможности degree/type/time distribution; (iii) сравнить C и D на неизменных tokens. Permuting порядок списка без изменения Δtime не является полноценным отрицательным контролем времени.
- Gate первого продвижения: body и trim улучшаются в обеих парах; отдельно Jan/Jul, LTFM, LIRF July, missing-context; знак не держится лишь на двадцати наибольших изменениях SSE. Airport-day block bootstrap отражает общую зависимость строк. Если B выигрывает, а D не лучше C/B — сохраняем пользу входа, прекращаем усложнение GNN. Нулевая разница D−C не закрывает физический graph на наблюдаемых trajectories.

**Инженерная форма.** Новый `scripts/event_context_pack_20261007.py` владеет только новым output dir: raw token table, N×64 int32 indices, ID manifest, column allowlist, source hashes, split/missing/support receipts. Новый `scripts/event_context_screen_20261007.py` владеет четырьмя arms. Контекст нельзя дублировать N×64×F float32 целиком: хранить events один раз и собирать batch из indices; индексная матрица ~615 MB для 2,4 млн строк. Parent отдельно владеет регистрацией/journal/queue. Другие workers не меняют эти файлы; текущие E190/E188 sources не трогать.

**Ресурсы, оценка до benchmark.** Packing 0,3–1,5 ч на 6 CPU. Query-only можно переиспользовать лишь при побитовом совпадении preprocessing/fold/seed; иначе пересчитать. B/C/D × 2 пары: ориентир 1–3 GPU-часа, batch 512–2048; сначала 200 batches с time/RSS/VRAM receipt и пересчитать ETA. Если budget ограничен, сначала B против C (четыре fit), D только при наличии сигнала C; порядок фиксируется до просмотра результатов. Веса новой архитектуры в смеси не подбирать на screen.

### Gaps

Условная информативность event tokens ещё не измерена. Физическая simultaneous-track coverage по каждому pair не подсчитана; own/previous coverage не заменяет этот подсчёт. TGN state с долгой памятью и actual ground-conflict graph остаются отдельными неначатыми исследованиями.

## 2. PINN и физика движения

### Takeaway

Сильная версия идеи — учить скрытую траекторию с мягкими кинематическими ограничениями, затем использовать её representation для записанного времени. Слабая постановка «добавить acceleration loss к одному скаляру taxi-out» не определяет, чьё ускорение вычисляется. Это исправимый недостаток постановки, а не доказательство бесполезности физики.

### Cited Findings

- В исходной PINN network представляет функцию состояния `u(t,x)`; differential residual вычисляется из её производных и уравнения, а data loss задаёт наблюдения/границы. [Авторы PINN](https://maziarraissi.github.io/PINNs/).
- У нас surface points содержат t, lat/lon, altitude, ground, speed, track, callsign; в части дней stale отсутствует. Фактические масса, тяга, угол торможения, engine state и разрешённая taxi route не наблюдаются. [Extractor](/Users/denyskolomiiets/prc-2026/src/adsb_visit.py), [E191 аудит](/Users/denyskolomiiets/prc-2026/REVIEW_BRIEF.md).
- BADA доступна по отдельной лицензии с одобрением EUROCONTROL и ограничениями поставщиков данных. Наличие бесплатного доступа для некоторых целей не делает набор открытым. [Официальный BADA](https://www.eurocontrol.int/model/bada), [условия лицензии](https://www.eurocontrol.int/sites/default/files/2020-06/bada-standard-licence-information.pdf). Конкурс требует открыто доступной лицензии для дополнительных данных. [Eligibility](https://ansperformance.eu/study/data-challenge/dc2026/eligibility.html).
- E36/E42/геометрия уже используют физическую информацию; E9 free-flow stand×runway reference и некоторые kin features закрыты в своих реализациях. Жёсткая aircraft-rotation feasibility ранее противоречила записанным clock labels. [Журнал](/Users/denyskolomiiets/prc-2026/HYPOTHESES_LOG.md), [базовая геометрия](/Users/denyskolomiiets/prc-2026/src/geometry.py).

### Inferences

**Физика не идентифицирует ожидание.** Уравнение `m dv/dt = thrust − drag − rolling − braking` описывает движение при известных/оцениваемых управляющих величинах. Один и тот же путь и тип борта допускают тысячи длительностей из-за стоянок и очереди. Свободно обучаемые thrust/brake без наблюдений могут объяснить любую скорость; чрезмерно жёсткие nominal values навязывают ложное знание. Внешний METAR даёт ветер, но не controller action. Размерности также нужно согласовать: MSE секунд² плюс штраф ускорения м/с² без масштаба λ не имеет стабильной интерпретации.

**Честная проба без закрытого BADA.** На естественных треках с известным identity взять sequence encoder из `new_mechanisms.md`, добавить decoder latent `(x(t), y(t), v(t))` и masked reconstruction по имеющимся точкам. Мягкая кинематика `dx/dt≈v cos θ`, `dy/dt≈v sin θ`; неотрицательная скорость; осторожный robust penalty на большие скачки ускорения. Не заполнять gaps неподвижностью и не принуждать положительный minimum physical taxi к recorded target. Отдельные stop/move states допускают произвольное ожидание. Predicted target остаётся supervised recorded y с existing regime wrapper.

Контроль A: тот же encoder и supervised loss. B: + auxiliary reconstruction без physics. C: + тот же reconstruction и dimensionless kinematic loss. Все arms видят одинаковые точки; λ выбрать один до fit по нормировке/градиентной шкале train pilot, не по holdout RMSE. Если C не лучше B, это отсутствие пользы physics regularizer в этой постановке, а не отсутствие пользы траекторий. Вспомогательное reconstruction error — диагностика, gate остаётся recorded-y RMSE после общей цепочки. Разбить complete/natural partial/previous-only и LIRF July; проверять чувствительность к densities/stale.

Ночная выполнимость: bounded pilot двух пар × A/B/C после готового pack, ориентир дополнительно 2–4 GPU-часа, сначала benchmark. До готовой последовательности это не первый fit. Получать BADA ради этой пробы не требуется. OSM path length / empirical feasible speed envelope можно использовать мягким признаком только после точного route/matching, без замены текущих geometry features их переименованием.

### Gaps

Истинная flight mass не выводится однозначно из aircraft type. Физические параметры ground thrust/braking для наших конкретных рейсов не установлены. Strict physical pseudo-label E191 имел очень ограниченное покрытие; это не обеспечивает обучающую правду всего latent decoder. Внешние BADA files не включены в разрешённый открытый набор.

## 3. Мультимодальность: ADS-B, METAR/TAF, расписание и ACARS

### Takeaway

Идея полезна в варианте **учёта временной истории и взаимодействий**: например, последовательность ухудшения погоды и изменение пропускной способности влияют иначе, чем один METAR. Но базовые departure METAR и часы уже используются широко. Следует измерять добавку последовательности/новых наблюдений, а не объявлять всё перечисленное новым источником.

### Cited Findings

- Проверены `features_fin1b.parquet` и фактическая `feature_columns` с `DROP_PREFIX=day_,tq_,wxr_,month`: проходят 15 признаков `wx_*`/`metar_age_min`. Среди них температура/влажность, ветер/порывы, видимость/ceiling, снег/freezing/rain/thunder/fog, icing и недавняя история снега/icing. `wxr_` и `wx_` — разные префиксы. [Извлечение](/Users/denyskolomiiets/prc-2026/src/features.py:145), [selector](/Users/denyskolomiiets/prc-2026/src/train.py:86).
- Новый замер 6 октября: METAR join имеется у всех 2 085 047 train и 344 841 ranking строк; temperature отсутствует лишь у 68 train и 5 rank. Это наличие присоединённой записи, а не гарантия её точности/частоты. Join backward до трёх часов, around AOBT proxy. [Таблица](/Users/denyskolomiiets/prc-2026/data/features_fin1b.parquet), [join code](/Users/denyskolomiiets/prc-2026/src/features.py:181).
- Текущий AWC API предоставляет до **30 дней**; старые testbed страницы ещё говорят 15. METAR OpenAPI использует `date` и `hours`, не `start/end`. Прямой запрос LIRF за 15 июля 2026 вернул HTTP 400 с ограничением 30 дней. [Текущая документация](https://aviationweather.gov/data/api/), [OpenAPI](https://aviationweather.gov/data/schema/openapi.yaml), [сохранённый ответ](awc_historical_probe.json). Предложенный URL не скачает исторический диапазон 2025–2026.
- IEM описывает исторический **NWS TAF** archive, включая issue-time/amendments. Это подтверждает метод получения истории, но не европейское покрытие всех десяти аэропортов. [IEM](https://mesonet.agron.iastate.edu/onsite/news.phtml?id=1477).
- Ограниченная живая проверка этого IEM endpoint: запросы `station=LIRF` за январь 2025 и июль 2026 оба вернули HTTP 200 с `data: []`. Это отсутствие записей в двух конкретных ответах данного NWS-сервиса, не доказательство отсутствия TAF Рима во всех архивах. [Январский ответ](iem_taf_lirf_jan2025.json), [июльский ответ](iem_taf_lirf_jul2026.json).
- 25 сентября уже измерены полные сутки Airframes 02.07.2026 и 15.01.2026: OUT у EGLL 25/669 летом и 0 зимой; LSZH 18/439 и 5/328; LFPG/LIRF/LTFM — 0 OUT/OFF в обоих checked days. Это прежний двухдневный sweep, **не новый полный сезонный аудит**. [Скан, код и raw receipts](/Users/denyskolomiiets/prc-2026/analysis/acars_registry_sweep_2026-09-25.md).
- Новая официальная Airframes licensing page требует attribution, выделяет отдельные arrangements для bulk/commercial use и прямо сообщает, что licensing terms ещё дорабатываются. Установленной стандартной open-data licence эта страница не предъявляет. [Airframes](https://docs.airframes.io/api/licensing/).
- H26 касался погоды назначения; этот отказ нельзя переносить на departure-weather temporal representation. [Журнал](/Users/denyskolomiiets/prc-2026/HYPOTHESES_LOG.md). Проверка координатора Navigraph обнаружила simulation-purpose terms и developer research restriction, а не разрешённый исторический public dataset. [Terms](https://navigraph.com/legal/terms-of-service), [developer restrictions](https://developers.navigraph.com/docs/general/restrictions).

### Inferences

**Что именно могло потеряться.** Один METAR не кодирует полностью длительность низкой видимости, внезапную смену ветра, чередование runway configurations, момент начала snow/icing wave и послесобытийное восстановление потока. TAF может отражать заранее известное ожидание ухудшения, влияющее на планирование; поэтому «METAR уже есть» не математическое доказательство ненужности TAF. Но реальный issue-time и coverage нужны до любого fit. Weather-dependent queuing обычно уже частично наблюдается через часы и потоки; дополнительный эффект должен быть измерен условно на них.

**Первая исполнимая проба.** Из существующих IEM CSV собрать 12 временных bins последних шести часов у origin, включая возраст/coverage, continuous wind/temp/visibility и categorical weather. По каждому bin — только наблюдения, доступные к зафиксированному reference time; current snapshot и существующие snow_recent остаются в обоих arms. Control A: FIN1b/sequence model с текущими scalar weather. B: те же входы + компактный temporal weather encoder (TCN 2×32 или fixed summary history, выбрать до fit). C: gated fusion этого encoder с event/trajectory representation при одинаковом token set. Не одновременно менять MLP и target и источник.

Weather missingness и sensor missingness должны быть отдельными масками. Cross-modal interaction можно проверить перестановкой погодных последовательностей между другими днями того же аэропорта/сезона при сохранении current-weather bin. Если B≈A, а C>A только при genuine history, есть основание исследовать interaction; если выигрывает простой fixed summary, Transformer не обязателен. Evaluate airport×month и условия precip/visibility по предзаданным weather criteria, не cherry-picked target residuals.

ACARS сначала требует новой source/coverage audit: actual message sample, event semantics (OUT/OFF не то же, что door/engine), точный identity join и пригодная лицензия. Наличие raw text/dashboard не доказывает, что engine/door states предоставлены по каждому рейсу. Любой положительный micro-pocket в EGLL/LSZH оценивать поверх уже сильного ADS-B. Нет основания обещать закрытие LTFM/Rome gap нынешним Airframes.

Planned priority и assigned taxi route нельзя синтезировать из Navigraph charts как будто это наблюдаемые разрешения диспетчера. Расписания/stand geometry уже есть; новая operational restriction может быть отдельным признаком лишь после dated public source и точного соответствия 2025/Jan+Jul2026.

**Выполнимость.** Existing-weather pack <1 CPU-часа как предварительная оценка; два matched arms × две пары 0,5–2 GPU-часа после общего query pack. Historical TAF discovery — bounded availability probe без обучения, отдельный от готовой weather-history пробы. ACARS full sweep сегодня не первый ресурсный приоритет: старый отрицательный support надо менять новым конкретным источником.

### Gaps

Утверждение об объяснённой доле дисперсии погодой не проверено на current residual и не используется. Historical European TAF, engine/door/priority и assigned route coverage не установлены. Нет новой таблицы ACARS с широким matching 2025/2026. Полное removal-ablation активных METAR сейчас не запускалось; наличие признака не гарантирует его текущую marginal importance.

## 4. Причинная модель и DoWhy — идеи 4 и 5

### Takeaway

Сильная часть — явно разделить физические причины, решения диспетчера, сенсорную наблюдаемость и правила записи, а затем проверить устойчивость отношений во времени. **Это может улучшить дизайн признаков и перенос.** Одна PSM оценка «эффекта traffic» сама по себе не выдаёт лучший прогноз каждого рейса и не доказывает, какие корреляционные признаки удалять.

### Cited Findings

- DoWhy разделяет modeling, identification, estimation и refutation. Идентификация зависит от заданных причинных предположений; библиотека не узнаёт все скрытые confounders из таблицы автоматически. [Статья авторов](https://arxiv.org/abs/2011.04216), [документация](https://www.pywhy.org/dowhy/v0.13/user_guide/causal_tasks/estimating_causal_effects/index.html).
- Реализация propensity-score estimators v0.13 прямо проверяет один бинарный treatment и отклоняет непрерывный/многозначный. Traffic count не соответствует этому интерфейсу без преобразования или другого estimator. [Исходный код](https://www.pywhy.org/dowhy/v0.13/_modules/dowhy/causal_estimators/propensity_score_estimator.html).
- E103 adversarial importance weighting ранее ухудшил board; E147 удаление конкретных density features принималось после измеренного дрейфа. Это два разных результата, не общий запрет на causal/stability анализ. [Журнал](/Users/denyskolomiiets/prc-2026/HYPOTHESES_LOG.md).

### Inferences

**SCM необходимо расширить.** `Traffic→Queue→Taxi` разумная часть схемы. Но нужны runway configuration/capacity, ATFM/готовность борта, weather, оператор/тип, фиксированный reference time, сенсорная сеть и `recording regime→recorded BLOCK`. Если строить traffic по интервалу, длина которого использует target taxi time, возникает прямая утечка; если по observed takeoff time, это допустимый retrospective predictor, но exposure эндогенен и не «рандомизированная нагрузка». Одно и то же runway assignment может быть mediator политики и proxy скрытых ограничений. Для разных estimands adjustment set различается.

**Для конкурса необязательно удалять следствия.** Мы предсказываем записанные значения при наблюдавшейся политике, а не эффект запрета одного вылета. Законный post-event sensor может быть отличным предиктором. Anti-causal feature не равен leakage, если он выдан в ranking. Отказ от него требует измеренного transfer benefit, иначе причинная чистота улучшает другой вопрос за счёт RMSE.

**Реальный маленький тест.** До fit зафиксировать одну конкретную неустойчивую family, например receiver-density interaction, и causal rationale (изменение measurement mechanism между 2025/2026 при сохранении физического движения). Выполнить availability/overlap и условный residual audit по airport/month/coverage, учитывая источник current OOF. Затем сравнить одинаковый predictor: A со всеми входами; B без этой family; C с отдельным physics representation и recording/missingness branch плюс заранее выбранный stability penalty между train-month environments. Penalty не назначать invariant всем аэропортам: их процессы записи реально различаются. Outer holdouts остаются непросмотренными при выборе family/λ.

Для исследования traffic ATE можно задать binary «высокая нагрузка» по заранее определённому train-only threshold, показать overlap propensity, standardized balance и чувствительность к скрытому confounding. Но это отчёт о выбранном contrast, не новые pseudo-labels для всех строк. Continuous dose-response требует соответствующего метода. Инструмента с одновременно relevance, independence и exclusion restriction у нас пока нет: runway, weather и schedule сами прямо связаны с taxi и режимом записи, поэтому объявлять их IV нельзя.

**Выполнимость.** Подготовка DAG/overlap/shift diagnostics до1 CPU-часа. Одна registered same-capacity ablation — как обычный two-fold screen. Это разумная параллельная CPU работа, но не повод ставить полный DoWhy stack или выдумывать instrumental variable. Результат «нет идентификации ATE» оставляет predictive stability эксперимент открытым.

### Gaps

Не доказаны conditional exchangeability, positivity и выбранная intervention semantics. Наблюдаемого valid IV нет. Invariance между всеми train environments не гарантирует перенос на ranking 2026; measurement process может меняться именно там. Причинные выводы из traffic proxies ограничены даже при хорошем balance.

## 5. Условная диффузия — идея 6

### Takeaway

Содержательный мотив — моделировать мультимодальное `p(y|x)` и неопределённость, особенно когда одинаковые наблюдения допускают несколько clock regimes. Для нашего вопроса лучше поставить **conditional response diffusion**, а не генерацию всей tabular dataset. Сначала нужно сравнение с небольшой явной смесью и детерминированным conditional mean; uncertainty сама по себе не уменьшает RMSE.

### Cited Findings

- TabDDPM — модель генерации неоднородных табличных данных; основные сравнения в работе направлены на качество синтетических таблиц против GAN/VAE. [Оригинальная работа](https://proceedings.mlr.press/v202/kotelnikov23a.html).
- DiffWave разработан для audio synthesis. Его temporal denoiser можно адаптировать, но непосредственная задача scalar taxi-time не является его исходным benchmark. [Статья](https://arxiv.org/abs/2009.09761).
- CARD специально моделирует conditional response distribution, сочетая diffusion и предварительный conditional mean estimator; это ближе к предложенной задаче. [Статья](https://arxiv.org/abs/2206.07275).
- Уже имеющаяся модель использует вероятности шести режимов; E104 показывал, что лучшее classifier logloss не обязательно даёт лучший mixed RMSE, E107/111 тоже проверяли смежные изменения. [Модель](/Users/denyskolomiiets/prc-2026/src/train.py), [журнал](/Users/denyskolomiiets/prc-2026/HYPOTHESES_LOG.md). Новая явная joint likelihood описана отдельно. [Предложение](new_mechanisms.md).

### Inferences

**Что diffusion может добавить.** Неограниченную форму regular conditional distribution и мягкое представление переходов между объяснениями, без жёсткого first-match label. Это реальное отличие от нынешнего independently trained gate/regressor. Однако четыре известные узкие clock modes уже имеют точные аналитические centers; generic continuous denoising может размыть именно ту структуру, которую модель хорошо использует. Более эффективная постановка — известные clock atoms/узкие компоненты плюс flexible regular conditional distribution, с явной measurement branch.

**RMSE и выбор вывода.** При squared error оптимален `E[y|x]`. Один сгенерированный сценарий добавляет случайную ошибку; усреднение S независимых samples имеет Monte Carlo variance `Var(y|x)/S`. Квантили нужны для других задач, но median не заменяет mean. Формула — математический вывод, а не оценка нашей дисперсии. При больших хвостах даже несколько десятков samples могут дать нестабильный mean; нужна convergence check и, возможно, distillation mean head.

**Корректная проба.** A: deterministic PLE mean + существующая mixture. B: joint likelihood MDN из `new_mechanisms.md` с известными clock centers. C: условный diffusion для regular residual при тех же query features, сохранённых planned/day branches; небольшой denoiser 2×256 с timestep embedding, 100 train noise steps и fixed 20-step sampler как одна заранее выбранная конфигурация. Этот первый C проверяет regular distribution, **не всю гипотезу soft assignment**. Следующий совместный variant имеет смысл лишь после него или прямого B signal.

Все normalization/mean guidance внутри outer-train; guidance teacher, обученный на всём 2025, недопустим в holdout. На pilot из фиксированных 10 000 holdout rows сравнить mean при 16/64/256 samples с общими seed и измерить сам шум интегрирования, затем выбрать budget заранее для полного screen, а не лучший seed по RMSE. Сохранять samples/mean variance metadata; compare on same rows/objective/production wrapper. Если MDN равен или лучше при меньшей стоимости, это практический выбор для завтра, а не теорема о диффузии.

Не генерировать synthetic rare-tail flights и не смешивать их с настоящими как независимую истину без отдельного ablation: генератор может воспроизвести неправильные зависимости/target leakage, а перепредставление хвоста меняет squared-error задачу. Сам tail в боевой цепочке уже имеет отдельное решение.

**Выполнимость.** Training компактного denoiser может вместиться в ночь; inference S×steps на сотнях тысяч строк дороже training. Начать с двух outer pairs и bounded validation sample, benchmark до полного ranking. Ориентир 1–4 GPU-часа для screen — непроверенная оценка; дедлайн останавливает после заранее обозначенного inference budget. Отдельный conditional likelihood baseline дешевле и должен считаться рядом.

### Gaps

Для нашего conditional distribution не измерены multimodality beyond known clocks, качество uncertainty и sample convergence. Нельзя приписывать TabDDPM результаты, полученные CARD, или улучшение generative score — улучшению production RMSE. Полноценный all-regime diffusion с точными плановыми компонентами ещё не спроектирован/обучен.

## 6. RL для порядка вылетов — идея 7

### Takeaway

Оптимизация порядка вылетов — содержательная задача управления, но изменяет политику аэропорта. Конкурс просит восстановить время при фактически состоявшихся решениях. Наиболее полезный мост к нашей задаче — **скрытое состояние очереди и service capacity, оценённые по фактической последовательности**, а не «лучшие» времена из оптимизированной политики.

### Cited Findings

- PPO чередует сбор опыта взаимодействия с environment и policy optimization. Нужны заданные observation/action/reward/transition semantics. [Оригинальная работа](https://arxiv.org/abs/1707.06347).
- PRC raw movements дают взлёты/прилёты и часть оперативных/плановых часов, но в прочитанной схеме нет полного action log выдачи pushback/taxi clearances и альтернативных разрешённых решений. [Ranking](/Users/denyskolomiiets/prc-2026/data/prc-2026-datasets/ranking.parquet).
- Простейшие queue proxies и overdue counts уже испытывались: E127/E137/E184 и arrival-flow E151. E157 показывает, что отдельные последовательностные взаимодействия wake всё же могут иметь marginal signal. [Журнал](/Users/denyskolomiiets/prc-2026/HYPOTHESES_LOG.md).

### Inferences

**Почему нельзя непосредственно обучить PPO/DQN по TAXITIME.** Если simulator выдаёт reward=меньше taxi, агент найдёт более эффективную очередь; его y относятся к изменённой политике, а не observed labels. При ошибочных transition rules он также может оптимизировать ошибки симулятора. Offline RL не устраняет проблему: нужны достаточное покрытие actions, корректно восстановленные действия/состояния и ограничение выхода за logged support. Присвоить action=«следующий фактически взлетевший рейс» не восстанавливает доступный диспетчеру выбор и скрытые ограничения.

**Сильная версия для прогнозирования.** Построить observed-policy state-space queue filter: latent readiness/queue membership/service rate; observed departure/arrival events; wake/runway configuration и weather как inputs. Он оценивает распределение состояния при текущих наблюдениях и выдаёт occupancy/backlog/uncertainty features. Ни изменение порядка, ни reward optimization для этого не обязательны. Можно использовать neural transition model или differentiable queue, но нельзя просто переименовать существующий count E184.

**Малый различающий тест.** Сначала проверить, может ли filter лучше простого счётчика предсказывать следующие наблюдаемые interdeparture gaps/порядок на train-only withheld event spans, не используя TAXITIME. Это проверка state information, не финальный metric. Затем добавить cross-fitted latent features к тому же regular model, сравнив A existing queue counts, B counts+filter, C event attention. Для всех fit supervised part только outer-train; не оптимизировать фильтр на holdout target. Оценивать incremental body/trim current wrapper, LTFM и динамичные runway periods; permutation control на последовательности service events.

Readiness из AOBT — imperfect proxy, а не фактическая доступность у threshold; queue membership нужно вероятностное. Последовательность observations после target event может использоваться только в той же retrospective постановке, что baseline; если filter объявлен causal online, он должен иметь отдельный past-only cutoff. Не смешивать эти две задачи в сравнении.

**Выполнимость.** Ночью возможен прототип latent queue filter на одном/двух аэропортах и перенос на all-airport support при успехе. Полный RL environment с route conflicts, wake/safety/ATFM, staffing и validated policy — отдельная многодневная работа. В очередь сегодня first event attention даёт тот же тест идеи взаимодействий быстрее и с меньшим количеством неподтверждённых предпосылок.

### Gaps

Action set, hidden controller constraints, transition dynamics и offline-policy support не восстановлены. Без них score simulator не является CV на competition labels. Наличие calendar/schedule не даёт полного readiness или departure priority. Latent filter ещё не реализован и может повторить уже закрытые агрегаты без дополнительного signal.

## 7. Digital twin как соединение всех ветвей

### Takeaway

В сильном варианте twin — согласованный скрытый state estimator с отдельным процессом измерения/записи. Просто объединить названия TGN+PINN+Transformer+RL не создаёт недостающую информацию и не позволяет понять, какой компонент помог. Собрать целое разумно после независимых проверок двух-трёх модулей.

### Cited Findings

- Наблюдаемые текущие материалы различают покрытие raw events, current/previous trajectories, NM clocks и record regimes; у blind LTFM отсутствует именно ground observation, а не вся информация. [Current errors](/Users/denyskolomiiets/prc-2026/data/current_errors_20261005/report.json), [контекст](event_support.json), [regime implementation](/Users/denyskolomiiets/prc-2026/src/train.py).
- Последовательностный airport model Amelia-TF предсказывает физические траектории с взаимодействиями, что поддерживает общий representation approach, но не доказывает улучшение наших clock labels. [Страница авторов](https://ameliacmu.github.io/amelia-model/).

### Inferences

Минимальное содержательное соединение:

1. Event/trajectory/weather encoders дают согласованное latent state плюс uncertainty/missing masks.
2. State evolution допускает stop/wait и мягкую кинематику там, где её можно проверить.
3. Отдельный measurement head предсказывает recorded-y mixture; он может расходиться с physical state из-за clock regimes.
4. Control/RL head для оптимизации реальных операций остаётся отдельной задачей, не генерирует «улучшенные» конкурсные labels.

Приёмка модульная: fixed baseline→event input→structure→trajectory→weather history→physics auxiliary→measurement likelihood. Каждая ступень получает matched ablation. Фиксировать ID/hashes/availability/feature cutoffs; не обучать все компоненты и менять blend в одном эксперименте. Приоритет задаётся закрытием измеренного информационного пробела и доступностью данных, а не масштабом названия модели.

**Что можно закончить к утру.** Полный обоснованный review всех идей; event context pack и первый paired screen при готовом worker; weather-history/trace feasibility inventory; prereg план physics/diffusion/control. Обещать готовый валидированный twin за одну ночь оснований нет, но отдельные новые механизмы вполне поддаются ночной проверке.

### Gaps

Нет откалиброванного цифрового двойника десяти аэропортов, текущих dispatch clearances или контрфактуальных labels. Предлагаемая интеграция является архитектурным планом. Правильная физическая simulation metric и конкурсный RMSE должны оставаться раздельными.

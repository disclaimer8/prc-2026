# Кто в топ-10 PRC 2026: публичные профессиональные следы (29.09.2026)

Собирались только публичные рабочие материалы: страницы команд у организатора, GitHub, статьи, страницы институтов и компаний. Личные данные не собирались. Ни с кем не связывались, ничего не публиковали.

**Главный новый источник.** Страницы команд на сайте организатора (`https://prc-data-challenge-2026.netlify.app/teams/<team>.html`) показывают **имена участников**. В `teams_raw.json` их нет (`participants: null`). Отсюда же имена в заметке от 16.09 (`analysis/past_winners/prc2026_public_competitors_2026-09-16.md`, раздел 5). Эта работа добавляет к ним аккаунты, прошлые PRC-решения, стиль работы и динамику очков.

Динамика очков посчитана по всем 3394 сабмитам из API доски (`https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard`). Сейчас у всех сабмитов `usedPairs = 344841`, то есть ранние сабмиты пересчитаны на новом ранжирующем наборе. Поэтому числа на 16.09 здесь не совпадают с заметкой от 16.09.

## Сводка

| # | команда | лучший RMSE | сабмитов | человек (стр. организатора) | публичный аккаунт | уверенность связи | код PRC-2026 публично |
|---|---|---|---|---|---|---|---|
| 1 | strong-crane | 220,65 | 65 | Somnath Panigrahi, TU Dresden | GitHub `Somnathab3` | высокая | нет |
| 2 | vigorous-whistle | 222,59 | 42 | madhav pv, ArcNautical | GitHub `SaltyTaro` (Madhav PV), орг. `arcnautical` | высокая | нет |
| 3 | jolly-lobster | 224,50 | 67 | Lee Marshall, Elysium Labs | GitHub `LeeMarshall1113` | высокая | был, теперь 404 |
| 4 | quick-boat | 226,81 | 73 | Arquimedes Canedo | GitHub `arquicanedo` | высокая | только вьювер доски |
| 5 | zesty-puzzle | 228,21 | 83 | Luis Melzer | не найден | — | нет |
| 6 | upstanding-firefly | 234,99 | 17 | Jiawei Jing, Makuhari Development Corp. | сайт mkhr.co.jp, GitHub-орг. `mkhrdev` | высокая (компания) | нет |
| 7 | gentle-igloo (мы) | 235,17 | 72 | — | — | — | — |
| 8 | strong-turtle | 235,45 | 54 | Nathaniel Ehimen | GitHub `EhimenNathan`, ResearchGate | средне-высокая | нет |
| 9 | gentle-tractor | 235,92 | 30 | HOANG ANH THONG LE, ACV (в 2026 заявлено 3 участника) | GitHub `sCoOL` (PRC 2024) | высокая для 2025, средне-высокая для `sCoOL` | нет |
| 10 | jovial-uniform | 236,98 | 22 | Piyush Patil | GitHub `piyush7911` = Discord `piyush7911` | высокая (команда↔Discord), средне-высокая (GitHub) | нет |

## Лучший счёт по датам (на текущем ранжирующем наборе)

| команда | 16.09 | 22.09 | 25.09 | 29.09 |
|---|---|---|---|---|
| strong-crane | 264,2 | 263,4 | 259,4 | **220,7** |
| vigorous-whistle | 246,5 | 239,1 | 234,1 | 222,6 |
| jolly-lobster | 272,0 | 239,5 | 234,5 | 224,5 |
| quick-boat | 260,8 | 256,9 | 233,2 | 226,8 |
| zesty-puzzle | 263,6 | 243,4 | 237,9 | 228,2 |
| upstanding-firefly | 266,4 | 266,4 | 243,9 | 235,0 |
| gentle-igloo (мы) | 258,2 | 243,8 | 240,1 | 235,2 |
| strong-turtle | 265,1 | 260,6 | 257,4 | 235,5 |
| gentle-tractor | 303,0 | 239,5 | 236,3 | 235,9 |
| jovial-uniform | 247,5 | 240,7 | 240,7 | 237,0 |

Одношаговые скачки, одинаковые по форме:
- upstanding-firefly: 23.09, v13, 266 → 245;
- quick-boat: 24.09 21:56 UTC, v102, 254,5 → 238,5;
- jolly-lobster: 25.09 00:28, v106, 233,4 → 228,3;
- strong-crane: 26.09 00:01, v77, 253,2 → 235,9, затем до 220,7 за три дня;
- strong-turtle: 28.09 02:18, v77, 250,6 → 240,4.

Между своими лучшими сабмитами strong-crane продолжает слать варианты на ~260 (v78, v80, v81, v90). Похоже, это контрольные прогоны старой базы.

---

## 1. strong-crane: Somnath Panigrahi (TU Dresden)

**Связь, высокая.** Имя и «TU Dresden» есть на странице команды. README репозитория PRC 2024 `team_honest_turtle` начинается с «I'm Somnath Panigrahi… Master's… Air Transport and Logistics at TU Dresden», коммиты там от `Somnathab3`. В PRC 2025 он выступал как команда `likable-ant`, страница организатора называет его по имени, TU Dresden. Репозиторий `Somnathab3/PRC_2025_Likable-ant` лежит в том же аккаунте.

**Публичные работы:**
- Три PRC подряд. 2024 (TOW): https://github.com/PRC-Data-Challenge-2024/team_honest_turtle и https://github.com/Somnathab3/PRC_TOW_Predictor. 2025 (топливо): https://github.com/Somnathab3/PRC_2025_Likable-ant и https://prc-data-challenge-2025.netlify.app/teams/likable-ant.html.
- Статья по PRC 2024: «Inference of the Actual Take-Off Mass from ADS-B data using Machine Learning», Student Research Task 2024/25, TU Dresden, февраль 2025 (PDF: `analysis/literature/Panigrahi2025_TOM_ADSB_ML.pdf`). AutoGluon WeightedEnsemble_L2 над CatBoost / XGBoost / LightGBM (2 560,85 кг против 2 636,95 у одиночной CatBoost), групповые модели + глобальная на запас + веса по обратному RMSE; 8-е место PRC 2024. (Исправлено 29.09: прежняя строка «многолинейные модели на 2–3 признаках, MAPE 5,95 % и 4,89 %» в статье не подтверждается.) Разбор: журнал №157. https://www.researchgate.net/publication/388835553
- Магистерская диссертация (TU Dresden, поддержка LFV Sweden) о галлюцинациях нейросетей в управлении воздушным движением на эшелоне: https://tu-dresden.de/bu/verkehr/ila/ifl/die-professur/news/1st-graduation-of-a-master-of-air-transport-and-logistics-atl-at-ifl-msc-somnath-panigrahi
- Сейчас аспирант DFG RTG AirMetro, тема «T03/2: Cooperative Conflict Resolution for IAM Urban Environment»: https://tu-dresden.de/bu/verkehr/airmetro/team/phd-candidates
- Сниппет поисковика по его LinkedIn (https://www.linkedin.com/in/somnathab3/) говорит о докладе на **DASC 2026 (Орландо)**: ML-предсказание **распределений задержки вылета**, «event-driven approach». LinkedIn закрыт для чтения, **напрямую не проверено**.
- Прочие репозитории: `ATC_Hallucination` (MAPPO, BlueSky), `EYDS2025` (EY Open Science Challenge), `Accoustic_model_EDTH_Munich` (хакатон).

**Что говорит о методах.** Код 2025 года читался напрямую, `scripts/Stage_9_Fuel_predict_CV.py` и `Stage_6.2_Train_Multiplier.py.py`. Что там есть:
- тяжёлая обработка ADS-B через библиотеку `traffic`: очистка, ресэмплинг, восстановление дыр, сглаживание, ERA5;
- **физическая база плюс ML по остатку** (`--target-type residual`: fuel − acropole);
- **отдельные модели по типу ВС**;
- CatBoost с LightGBM и XGBoost в ансамбле;
- **TimeBasedGroupKFold**, группы по flight_id и разбиение по времени;
- **tail blending**: смешивание с физической базой выше порога.

Перенос на такси-аут: скорее всего, у него «базовое такси (unimpeded или ADS-B) плюс остаток», сегментация по аэропорту и режиму и временная валидация. Если верен сниппет про DASC, у него есть наработка по **моделированию распределения задержки вылета от событий**. Это та же ось, что наш режим «эхо / блок = расписание».

**Динамика.** С 25.09 по 29.09 улучшился на 39 с: 259 → 221. Главный шаг был ночью на 26.09. Сабмитит сразу после 00:00 UTC, когда обнуляется суточный лимит.

## 2. vigorous-whistle: Madhav PV (ArcNautical)

**Связь, высокая.** На странице команды: «madhav pv, ArcNautical». GitHub-организация `arcnautical` создана 17.09.2026, у неё 4 публичных репозитория, и единственный контрибьютор всех четырёх — `SaltyTaro`, профиль «Madhav PV». Сайт https://arcnautical.com: скрининг судов на санкции, скоринг рисков рейса, морская маршрутизация. Это данные AIS, по сути аналог ADS-B.

**Публичные работы (https://github.com/SaltyTaro):**
- `gauntlet-ptcg`: Kaggle Pokémon TCG AI Battle Challenge, команда «notoverkil», 1342/6807 в Simulation. Главная мысль его writeup: «набор проверок прошёл 14 сабмитов подряд, в которых агент ни разу не принял решения». Отсюда правило: каждая проверка должна сначала показать true negative на заведомо плохом входе. Плюс поправка Elo на силу соперников. Правило «лидерборд не используется как сигнал для отбора» зашито в тесты.
- `neuroforecast`: 5th Pazhou Algorithm Competition (AI4S). Extra Trees, конформные интервалы, **отказ от прогноза для непредсказуемых условий**. Два «запечатанных» теста, критерии захешированы заранее. Прогноз всегда показывается рядом с наивной базой «перенести последнее значение».
- `agent-skill-lift`: хакатон BenchFlow по навыкам для агентов, журнал всех прогонов и предсказания, записанные до запуска.

**Что говорит о методах.** Прямых следов PRC нет. Стиль: жёсткая дисциплина валидации против переобучения на доске и сравнение с наивной базой. По этим признакам он вряд ли «упадёт» на финальном наборе. Авиационного домена в его работах нет. Зарегистрировался 13.09, а ставку на ноябрь не делал: 42 сабмита, ровный рост на 1–2 с в сутки с 23.09.

## 3. jolly-lobster: Lee Marshall (Elysium Labs, UCF METIL)

**Связь, высокая.** На странице команды: «Lee Marshall, Elysium Labs». Поисковики проиндексировали `github.com/LeeMarshall1113/prc-taxiout-2026` с описанием «Entry for the PRC Data Challenge 2026…» и упоминанием команды jolly-lobster. Сейчас репозиторий отдаёт **404** (удалён или закрыт). В публичных событиях аккаунта его нет, значит, скорее всего, он стал приватным. Wayback 29.09 был недоступен (429 / «temporarily offline»).

**Что успели проиндексировать из README (сниппет поисковика):**
- на момент индексации «no model yet»;
- «Initial findings (2026-09-04)»: смена ранжирующего набора посреди соревнования; в train цель ровно `MVT_TIME − BLOCK_TIME`; **квадратичная ошибка определяется несколькими десятками хвостовых строк**.
- Во втором поиске был пересказ: «January and July 2025 held out for validation, train on the other ten months». Сборщик сниппетов мог смешать это с соседним репозиторием, **надёжность низкая**.

**Прочие работы:**
- https://github.com/LeeMarshall1113/jetspace-imitation-learning: V-JEPA 2, behaviour cloning. В README пред-регистрация предсказаний, «фальсификаторы названы заранее», журнал из 12 провалов и отзыв цифры, которая не воспроизвелась (85,7 % → 90,7 % с leak check).
- `jetspace-scaffold-rsi`: исследование самоулучшающихся агентов.
- Страница METIL при UCF IST: CS, специализация AI & Robotics, проекты Winterim 2024 и лето 2025 (https://www.metil.org/quickstart-students).
- «Elysium Labs»: публичной технической работы не найдено.

**Что говорит о методах.** С 4 сентября он понимает, что RMSE определяют хвостовые строки. Стиль исследовательский, с пред-регистрацией и проверкой утечек. Большой шаг сделал 25.09.

## 4. quick-boat: Arquimedes Canedo

**Связь, высокая.** Имя на странице команды. GitHub `arquicanedo`: Arquimedes Canedo, Siemens, Princeton NJ. Там лежит репозиторий `prc-challenge-2026`, это вьювер доски (уже в `competitor_repos/`, решения в нём нет).

**Публичные работы:**
- Distinguished Engineer в Siemens: графы знаний, цифровые двойники, автономные системы (dblp: https://dblp.org/pid/03/4986.html).
- Статьи 2026 года про **LLM-агентов для кода**: «Architecture as Capability Equalizer for Coding Agents», «Self-Reflective APIs», paper.json, arXiv 2609.10237.
- https://github.com/arquicanedo/awesome-air-traffic-control, август–сентябрь 2026: симуляторы, игры, библиотеки данных ATM. Про такси-аут там ничего нет.

**Что говорит о методах.** Метод не раскрыт. Профиль указывает на итерацию с помощью агентов: 73 сабмита. Самый крупный одношаговый прыжок среди лидеров: 24.09, −16 с.

## 5. zesty-puzzle: Luis Melzer

**Ничего не найдено.** На GitHub нет аккаунта с таким именем (проверены `lmelzer`, `melzerr` и другие, совпадений нет). Kaggle, LinkedIn и публикаций тоже не найдено. Есть только самоописание: студент CS в Leibniz Universität Hannover, недавнее AutoML-соревнование. У него больше всех сабмитов среди лидеров, 83, и ровный рост. Догадка без доказательств: он может использовать AutoML-ансамбли вроде AutoGluon.

## 6. upstanding-firefly: Jiawei Jing (Makuhari Development Corporation)

**Связь, высокая.** Имя на странице команды. Сайт компании https://mkhr.co.jp/ называет 荆嘉瑋 (JING JIAWEI) представительным директором. У компании есть GitHub-организация https://github.com/mkhrdev (Makuhari Development Corporation, Japan). Там лежат плагин для Claude Code и расширение Chrome, авиации нет.

**Публичные работы.** На сайте есть демо «тепловая карта реальных траекторий ВС и анализ наземного шума», то есть опыт с треками ADS-B и геопространственным стеком. Отчёты компании касаются энергетики, BESS и инфраструктуры. В блоге про авиацию и OpenSky ничего нет. В самоописании команда обещает код на GitHub и статью в JOAS.

**Что говорит о методах.** Прямо сказано: «feature-engineering and gradient-boosting toolkit» из шумового проекта на OpenSky ADS-B. Значит, почти наверняка используют наземные треки ADS-B и бустинг. Самый экономный из лидеров: 17 сабмитов, скачок 23.09 на −21 с.

## 8. strong-turtle: Nathaniel Ehimen

**Связь, средне-высокая.** Имя на странице команды. ResearchGate: «Nathaniel EHIMEN, Skoltech, Department of Petroleum Engineering» (https://www.researchgate.net/profile/Nathaniel-Ehimen). Совпадает с самоописанием «reservoir engineer». GitHub https://github.com/EhimenNathan («Nat»): репозиторий `Projects` с CRM (capacitance-resistance model), FNO для хранения CO₂, infill drilling и геологией скважин. Прямой перекрёстной ссылки «Ehimen Nathaniel = EhimenNathan» не найдено, поэтому не «высокая». Есть Kaggle `ehimennathan`, содержимое не прочиталось (JS).

**Что говорит о методах.** Самое показательное — https://github.com/EhimenNathan/tramcast-moscow, выложен 26.09: соревнование по почасовому прогнозу пассажиропотока трамваев Москвы, метрика WAPE. Что в нём есть:
- валидация строго out-of-time;
- структурная модель уровень × форма × календарь × погода;
- **байесовский фильтр частиц для адаптивного уровня с точками смены режима**;
- рекурсивный LightGBM как хедж с весом 0,2;
- оценка «теоретического потолка» неустранимого шума;
- бюджет ошибки;
- стекинг из 18 кандидатов не превзошёл лучшую одиночную модель, поэтому финальная модель — интерпретируемая структурная.

На такси-аут это ложится как «такси как поток в трубах»: признаки очереди и потока, адаптивный уровень по дню и аэропорту. Скачок 28.09 на −10 с.

## 9. gentle-tractor: HOANG ANH THONG LE (Airports Corporation of Vietnam)

**Связь, высокая.** Имя на странице 2026 года, в деталях заявлено 3 участника. В PRC 2025 он же выступал командой `tiny-marshmallow` с двумя коллегами из ACV (Canh Trieu Tran, Duc Phu Nguyen): https://prc-data-challenge-2025.netlify.app/teams/tiny-marshmallow.html. Публичного репозитория 2025 года в орг. `PRC-Data-Challenge-2025` нет. В PRC 2024 была команда `team_modest_scooter` («Software Engineer at ACV»). Её обоснование почти дословно совпадает с текстом `tiny-marshmallow`, а метаданные коммитов аккаунта `sCoOL` совпадают с именем Thong Le. Уверенность в `sCoOL` = он: средне-высокая.

**Что говорит о методах.** Решение 2024 года: https://github.com/PRC-Data-Challenge-2024/team_modest_scooter.
- Агрегаты признаков по ADS-B на полёт: TAS, ветер, вертикальная скорость на взлёте.
- **Отдельная модель для строк без ADS-B**, результаты объединяются.
- Он замечал особенности данных: диверсии; внутри дня одной авиакомпании TOW двух ВС меняется местами, если их рейсы поменялись местами по времени.
- Хотел данные о расписании и ротациях ВС.

Команда из отрасли с интересом к A-CDM, вероятно, хорошо понимает вехи AOBT/TOBT/TSAT. Прогресс остановился 25.09 на 235,9.

## 10. jovial-uniform: Piyush Patil

**Связь.** Имя на странице команды. В Discord Piyush Patil упоминал `jovial-uniform_v1` (заметка от 25.09), значит, Discord ↔ команда — высокая. GitHub https://github.com/piyush7911: имя Piyush Patil, логин совпадает с Discord-хэндлом `piyush7911`, связь средне-высокая. Других Piyush Patil среди 475 команд нет.

**Что говорит о методах.** Портфолио https://piyush7911.github.io/: Junior ML Engineer в Mobius by Gaian, GenAI, RAG, агенты. Из временных рядов там «Storm Oracle»: ансамбль и конформные интервалы для прогноза геомагнитных бурь. Кода PRC нет. 22 сабмита, медленный рост.

## Discord-хэндлы без привязки

| хэндл | вывод |
|---|---|
| `piyush7911` / Piyush Patil | jovial-uniform (см. выше) |
| `kind-mango (skylinkapi)` | по странице организатора kind-mango = Sergio Nicolini и David Schnaider (SkyLink API), сейчас №31 |
| `arnavhm13` | низко-средняя: возможно, Arnav Hemanth (versatile-violin, №76). Основание только в форме хэндла |
| `GREKI [VTSM]`, `Vudueprajacu [TECH]` | связей не найдено. На GitHub одноимённые пустые аккаунты без репозиториев |

## Прошлые победители PRC в 2026 году

Сопоставлено по страницам команд и всем сабмитам.

| кто | прошлое | команда 2026 | место |
|---|---|---|---|
| Richard Alligier (ENAC) | 1-е в 2024 | positive-kiwi | 114 |
| Fazlur Rahman (TU Delft) | 1-е в 2025 (resourceful-quiver), 3-е в 2024 | silly-net | 51 |
| Euranova (7 человек) | 3-е в 2025 | gentle-violin | 165 |
| Joao Dantas (ITA/IEAv) | 2-е в 2024 | inspiring-kitten | 86 |
| Eftychios Eftychiou | статья JOAS 2025 | dynamic-emu | 141 |
| Rade Kačar | статья JOAS 2025 | joyous-rainbow | 100 |

**Ни один прошлый победитель не входит в топ-50.** Из топ-10 в прошлых PRC участвовали двое, оба без призов: Panigrahi (2024, 2025) и Thong Le (2024, 2025).

Не исследовались сознательно:
- humble-giraffe (№11): школьники;
- unbelievable-snowflake (№12, Daniel Lindner): распространённое имя, без опоры для однозначной связи.

---

## Выводы для нас (по убыванию пользы)

1. **Разрыв до третьего места — 10,7 с (235,17 против 224,50). Лидеры сейчас улучшаются на 2–5 с в сутки.** strong-crane: −39 с за 4 дня; vigorous-whistle: −11,5 с с 25.09; zesty-puzzle: −9,7 с. Если темп сохранится, к 11.10 третье место будет около 215 или ниже. Мелкими добавками его не догнать, нужен ступенчатый выигрыш.
2. **Ступенчатые скачки у пяти независимых команд пришлись на 23–28.09**: upstanding-firefly 23.09 (−21), quick-boat 24.09 (−16), jolly-lobster 25.09 (−5), strong-crane 26.09 (−17), strong-turtle 28.09 (−10). Похоже на общий, «находимый» сигнал или публичную подсказку. 23.09 в Discord был совет по хвостам («make the submission outliers look like training outliers»), см. `competitor_repos/_IDEAS_kindmango_zestful_alligator.md`, п. 48. Там он замерен как вредный, но возможно, его реализовали иначе. **Стоит перечитать Discord и коммиты публичных репозиториев за 22–26.09, это самая дешёвая проверка.** Это гипотеза, а не факт.
3. **Лидер (strong-crane) строит модели так:** физическая или наивная база плюс ML по остатку, отдельные модели по сегментам, временной GroupKFold, «хвостовое смешивание» с базой выше порога, тяжёлый ADS-B через `traffic`. По непроверенному сниппету у него ещё есть работа про **распределение задержки вылета, event-driven**. Проверить у себя, предсказываем ли мы для режимных строк **смесь E[y] = Σ p_k·μ_k с калиброванными p**, а не жёсткий выбор режима. Для RMSE оптимальна именно смесь.
4. **Как минимум три команды из топ-9 умеют работать с наземными треками ADS-B**: upstanding-firefly (шумовой проект), strong-crane (`traffic`, 2025), gentle-tractor (признаки ADS-B в 2024 и отдельная модель для строк без ADS-B). Схема «отдельная модель для строк без трека» стоит проверки у нас, если покрытие треками в январе и июле 2026 неполное.
5. **Физика не решает.** Сильные в физике победители TOW/топлива 2024–2025 (Alligier, Rahman, Euranova) стоят на местах 51–165. Выигрывает обработка режимов данных (эхо, расписание, хвосты), а не физическая модель такси. Вкладываться в физику руления не стоит.
6. **Хвосты: jolly-lobster знал с 04.09**, что RMSE определяют несколько десятков строк. Наша картина та же. У лидеров вопрос не «видят ли они хвост», а чем его кормят.
7. **Защита от переобучения на доске у №2 (vigorous-whistle) и №8 (strong-turtle) строгая:** запечатанные out-of-time тесты, сравнение с наивной базой, отказ от прогноза. Значит, их доска, скорее всего, перенесётся на финал без просадки. На их откат рассчитывать не стоит.
8. **GPLv3 и публикации.** Репозиторий jolly-lobster скрыт. Makuhari обещает GitHub и JOAS. Раньше 11.10 кода лидеров ждать не стоит. После 11.10 проверить `LeeMarshall1113`, `Somnathab3`, `SaltyTaro`, `mkhrdev`, `EhimenNathan`, `piyush7911` и орг. `PRC-Data-Challenge-2026`.
9. **Темп итераций.** 65–83 сабмита у лидеров против наших 72, темп сопоставимый. У нескольких лидеров виден профиль тяжёлого использования LLM-агентов: исследования Canedo, хакатон агентских навыков у SaltyTaro, Claude Partner у Makuhari, агентский скаффолд у Lee Marshall. Разница, скорее всего, не в скорости, а в найденном сигнале (см. п. 2).

## Источники
- Страницы команд 2026: https://prc-data-challenge-2026.netlify.app/teams/strong-crane.html (и аналогично для остальных команд)
- Доска (все сабмиты): https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard
- PRC 2025: https://prc-data-challenge-2025.netlify.app/outcome.html, выпуск JOAS https://journals.open.tudelft.nl/joas/issue/view/1058
- PRC 2024: https://prc-data-challenge.netlify.app/ и https://prc-data-challenge.netlify.app/teams.html
- GitHub: https://github.com/Somnathab3, https://github.com/SaltyTaro, https://github.com/arcnautical, https://github.com/LeeMarshall1113, https://github.com/arquicanedo, https://github.com/mkhrdev, https://github.com/EhimenNathan, https://github.com/sCoOL, https://github.com/piyush7911
- Прочее: https://arcnautical.com, https://mkhr.co.jp/, https://www.metil.org/quickstart-students, https://www.researchgate.net/profile/Nathaniel-Ehimen, https://www.researchgate.net/publication/388835553

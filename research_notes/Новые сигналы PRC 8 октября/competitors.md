# Свежие публичные решения участников PRC: дельта на ночь 7–8 октября

## Что действительно обновилось после предыдущего обзора?

### Takeaway

Новые коммиты есть; подтверждённого уникального крупного рычага для нашей модели в проверенных дельтах не найдено. Самая свежая работа — правило gentle-octopus для рейсов без NM, использующее уже знакомый нам сигнал соседних рейсов. Это ограниченный вывод по перечисленным репозиториям, а не утверждение об отсутствии новых решений вообще.

### Cited Findings

Проверка выполнена 7 октября 2026, завершение источников около 23:15 UTC, то есть уже 8 октября по Lisbon/WEST. Даты ниже — **committer.date GitHub**, отдельно от времени retrieval. Все загруженные исходники сохранены с префиксом `competitor_`; [manifest](competitor_receipt_manifest.json) содержит SHA-256 локальных квитанций. Исследовательские файлы не исполнялись.

| Источник | Фактическая свежесть | Содержание и доказательство |
|---|---|---|
| gentle-octopus, `joris-decombe/taxiout` | HEAD `12f3b471cf67085e5861cb0fa42a578e1a3ac1d0`, **07.10 22:36:50Z** | Реальная правка inference: нижняя граница для NM-missing вне LIRF по живому уровню NM соседей. Предыдущие вечерние изменения — resumable build (21:02), обновления отчёта/README (21:13–21:15). [Коммит](https://github.com/joris-decombe/taxiout/commit/12f3b471cf67085e5861cb0fa42a578e1a3ac1d0), [API](https://api.github.com/repos/joris-decombe/taxiout/commits?per_page=15). |
| genuine-cabbage, `RavindraTarunokusumo/prc-data-challenge-2026` | HEAD `3e85b4cb9125cb1ec1f81e5315226f228e8f7834`, **07.10 22:17:32Z**; push 22:22 | Дневная ветка добавила реальный METAR-блок, parser fix и pilot; финал — отрицательный weather gate и план дальнейшей работы. Это не новый принятый компонент. [Коммиты](https://api.github.com/repos/RavindraTarunokusumo/prc-data-challenge-2026/commits?per_page=12), [внедрение weather](https://github.com/RavindraTarunokusumo/prc-data-challenge-2026/commit/d5065169740ea475e97abe67f5088f07b31a9ab1). |
| smart-jigsaw, `yookeew/dumb-puzzle` | HEAD `cb191e22fba5b4954c3d2fafc34544c89c9b5d6d`, **06.10 17:44:59Z** | Arm C действительно добавляет NM-unmatched в обучение корректора; это старше дневного обзора 7 октября, но позже SHA `567586f` из нашего отчёта 6 октября. Тезис Fable «коммитов с 01.10 нет» текущим API опровергнут. [API](https://api.github.com/repos/yookeew/dumb-puzzle/commits?per_page=8), [кодовый коммит](https://github.com/yookeew/dumb-puzzle/commit/878515348ed5bff1ef3ebe724de80ad1a52af08e). |
| outgoing-boat, `EnioAguiar/prc-taxiout-2026` | HEAD `30e23deb77d0c2301ca4223b9c5f070b58a68fe2`, **03.10 13:02:24Z**; автор 13:02:14Z | Новый обход не выявил изменения: единственная публичная ветка `main`, README/checklist с завершением работы 3 октября. Не новая находка. [HEAD](https://github.com/EnioAguiar/prc-taxiout-2026/commit/30e23deb77d0c2301ca4223b9c5f070b58a68fe2), [ветки](https://api.github.com/repos/EnioAguiar/prc-taxiout-2026/branches?per_page=100). |
| Официальный сайт, связанный с espinielli/PRU | HEAD `37d2f8874d30896ba659f8af128d1b27b315252a`, **06.10 10:43:19Z** | Последние пять commit subjects — regeneration team pages. Проверен список коммитов, не установлено новое техническое решение или новое разъяснение правил. [API](https://api.github.com/repos/euctrl-pru/prc_data_challenge_website_2026/commits?per_page=5). |

Повторён GitHub repository search по трём строкам: `prc-data-challenge-2026` — 23 результата (включая наш репозиторий в авторизованном поиске), `taxiout-2026` — 5, `PRC2026` — 6 (последний содержит нерелевантные проекты). У двух сохранённых поисков `incomplete_results=false`; это свойство запросов, не гарантия полноты публичного кода. Дополнительного полезного нового репозитория сверх дневного списка 7 октября не выделено. [Поиск PRC](https://api.github.com/search/repositories?q=prc-data-challenge-2026&sort=updated&per_page=100), [taxiout](https://api.github.com/search/repositories?q=taxiout-2026&sort=updated&per_page=100), [PRC2026](https://api.github.com/search/repositories?q=PRC2026&sort=updated&per_page=100).

### Inferences

- Кодовые дельты gentle-octopus и genuine-cabbage реальны, а не только смена score в README. Из них ни одна не предоставляет новый источник наблюдений. Доказательств связи этих решений с методами лидеров нет.
- У smart-jigsaw нужно исправить полноту прежнего обзора: отрицательный ответ конкретного старого запроса не означал отсутствия октябрьских коммитов. Причина того старого отрицательного ответа сейчас не установлена; force-push/даты автора не доказаны.

### Gaps

- Нового полного обхода всех веток всех 36 прежних репозиториев не делалось. Для Enio проверены ветки, для трёх кодовых дельт — default-head и relevant files.
- Доска и Discord заново не проверялись. Здесь нет новых результатов нашей команды или новых слов espinielli; прежняя доска 22:00Z остаётся историческим снимком.

## Какие механизмы заслуживают внимания и чем отличаются от наших?

### Takeaway

Есть две полезные наблюдаемые реализации: явная нижняя граница NM-missing и исправление обучения на отсутствующих NM-входах. Ни одна пока не заслуживает нового большого обучения у нас. Первую можно опровергнуть дешёвым replay без fit; вторая требует сначала доказать аналогичный дефект нашего конкретного этапа.

### Cited Findings

**1. Gentle-octopus: жёсткий floor по ±30-минутной медиане соседей.** Функция `neighbour_taxi_level` берёт все departures данного аэропорта в окне ±1800 с от известного MVT, требует минимум три конечных `MVT−AOBT`, возвращает медиану. Для строк без NM вне LIRF, когда median >1500 с, `orphan_congestion_floor` заменяет прогноз на `max(pred, 0.95×median)`. В `correct.rules` это применяется после 3600-секундного cap, затем общий plausibility clip. Это не только README: `correct.apply → rules → orphan_congestion_floor`, а `build_final.main` вызывает `correct.apply` для ranking. Прочитаны целиком новый `experiments_round8.py` (218 строк), `correct.py` (98), `build_final.py` (93), полные новые функции и константы `train.py`. [train.py](https://github.com/joris-decombe/taxiout/blob/12f3b471cf67085e5861cb0fa42a578e1a3ac1d0/pipeline/taxiout/train.py#L408), [inference](https://github.com/joris-decombe/taxiout/blob/12f3b471cf67085e5861cb0fa42a578e1a3ac1d0/pipeline/taxiout/correct.py#L73), [build](https://github.com/joris-decombe/taxiout/blob/12f3b471cf67085e5861cb0fa42a578e1a3ac1d0/pipeline/build_final.py).

- **Фактическая связь со score:** автор прямо пишет о 431 изменённой строке и 275.03 → 274.07 с (v9 → v10), validation −0.16 с. Это самоотчёт с кодом построения v10, а не независимо воспроизведённая метрика. Его own model предсказывала около 1240 с на 223 EHAM orphan rows с уровнем соседей 2800–3700 с. [round8](https://github.com/joris-decombe/taxiout/blob/12f3b471cf67085e5861cb0fa42a578e1a3ac1d0/pipeline/experiments_round8.py#L1).
- **Наш аналог:** E3/E4 (NM-missing в дни сбоя), E72 (`x_hprox_med`, медиана NM по аэропорт-часу), текущий E185 с обучением без NM-полей. E72 дал +0.06 с, E4 −0.07 на старой доске и не прошёл первоначальные ожидания. Наши результаты не доказывают отказ любой возможной нижней границы, но закрывают заявление о новом сигнале. [Наши E3/E4](/Users/denyskolomiiets/prc-2026/REVIEW_BRIEF.md:448), [E72](/Users/denyskolomiiets/prc-2026/REVIEW_BRIEF.md:2156), [hour proxy](/Users/denyskolomiiets/prc-2026/src/features.py:403).
- **Точная новизна:** скользящее симметричное окно и принудительная нижняя граница *после* ADS-B, вместо признака или множителя в выбранные дни. Новый алгоритм ограничения, но прежние исходные наблюдения.
- **Вход/ranking:** MVT, ADEP, AOBT соседей и null-NM доступны в ranking. Число затронутых строк нашего текущего best не измерено; 431 относится исключительно к файлу конкурента.
- **Контрпример:** NM может отсутствовать по механизму записи, не связанному с очередью; самолёт мог рулить быстрее соседей. Конкурент сам сообщает иной коэффициент в LTFM. Наша E4 уже показала слабый перенос зимнего поднятия EHAM; наш текущий прогноз может и без floor быть выше него.
- **Самый малый решающий тест (предложение, не запуск):** сначала label-free число/RMS изменённых строк текущего ranking, затем строго фиксированный replay этого правила на шести имеющихся текущих OOF, без fit и без подбора 0.95/1500/окна. Решающий отказ: pooled decided/body/trim хуже, меньше 4/6 пар лучше либо выигрыш удерживается одним зимним событием. Учитывать, что имеющиеся current-chain OOF исторические и не полностью nested.
- **Риск переноса:** высокий — у чужого orphan model не было live inputs, у нас есть контекст и E185; 2025/2026 различаются зимними нарушениями. Это низкоприоритетный фиксированный sanity replay, не существенная новая линия исследований.

**2. Smart-jigsaw: NM-unmatched действительно включены в fit корректора.** Arm C1 объединяет прежнюю обучающую маску с `nm & ~line_rows & stack<=7200 & finite(y)` и добавляет `nm_unmatched`; C2 добавляет operator echo rate, который на NM-unmatched всегда null. Прочитаны полные `ctx_armc.py` (258 строк) и `armc_build.py` (65). В `prod()` строятся и train/ranking inputs, вызывается `fit_apply_ens`, затем сохраняется ranking; финальная сборка копирует LIRF NM-unmatched из исходного файла. Это реальная реализация и готовый файл, но **PROGRESS §88 на текущем HEAD прямо говорит “Not uploaded”**. Файл сначала назывался v45 против локальной rebuild, затем переименован; актуальный кандидат — v42 против uploaded v41. Ему нельзя приписывать результат доски. [Код](https://github.com/yookeew/dumb-puzzle/blob/cb191e22fba5b4954c3d2fafc34544c89c9b5d6d/src/post/ctx_armc.py), [финальная сборка](https://github.com/yookeew/dumb-puzzle/blob/cb191e22fba5b4954c3d2fafc34544c89c9b5d6d/src/post/armc_build.py), [PROGRESS §87–88](https://github.com/yookeew/dumb-puzzle/blob/cb191e22fba5b4954c3d2fafc34544c89c9b5d6d/PROGRESS.md#88-context-corrector-arm-c-nm-unmatched-rows-in-training-on-v41s-chain----c1-gate-pass-v42-built-2026-10-06).

- **Авторские измерения:** C1 trimmed −0.91/full −0.80/body −0.25 относительно их rebuilt v41; never-seen null patterns на NM-unmatched ranking 13.6% →0.1%. Это их gate и baseline; показатели не переносим на нашу модель.
- **Наш аналог/различие:** E185 целенаправленно учится на обычных строках с 22 замаскированными NM-колонками и 36 пересчитанными зависимостями, затем применяется к 4907 NM-missing вне LIRF; это непосредственно работает с поддержкой missing inputs. E13 был более ранним отрицательным вариантом отдельной модели. Smart C1 добавляет реальные missing rows в residual-corrector, E185 использует синтетически одинаковый missing-input вид в ordinary member. Они не тождественны, но вопрос поддержки уже отдельно решался у нас. [E185 регистрация](/Users/denyskolomiiets/prc-2026/HYPOTHESES_LOG.md:270), [E185 CV](/Users/denyskolomiiets/prc-2026/HYPOTHESES_LOG.md:276).
- **Вход/ranking:** null-NM и прогнозы базы доступны; у нас не доказано, что какой-либо текущий этап обучается только на complete cases, а применяется на unseen missing patterns. Нельзя объявлять наш баг по чужому.
- **Контрпример/transfer:** добавление редких тяжёлых NM-missing labels может ухудшить весь residual-corrector. У конкурента для EHAM Jan2026 обнаружены подъёмы +974 с против +74 с в 2025 и собственный жёсткий запрет менять LIRF — их gate не проверяет перенос этих изменений.
- **Самый малый решающий тест:** без fit проверить реальную train/apply mask и поддержку missing-pattern у одного конкретного подозреваемого нашего этапа, включая E185. Если поддержки достаточно или этап не применяется там, закрыть этот lead. Новый fit оправдан только при измеренном провале поддержки и независимой регистрации.

**3. Genuine-cabbage: свежая погода уже отклонена самим автором.** 16 признаков IEM/METAR: ветер/порыв/компоненты ВПП, видимость/потолок, температура/разность точки росы, коды осадков/тумана, число precipitation reports за 3ч и snow reports за 6ч. Привязка latest report ≤ offblock proxy−10мин, max age3ч. Прочитан целиком актуальный `weather.py` (118 строк), весь новый pilot и build script из commit diff. Две design-month пробы дали +1.43 и −0.48 с; требовалось ≤−1с в обеих. Это не принятый inference компонент. У нас погода и её история уже проверялись, включая E193/E205; нового источника здесь нет. Минимальный следующий тест не предлагается: материал не даёт нового механизма. [Актуальная реализация](https://github.com/RavindraTarunokusumo/prc-data-challenge-2026/blob/3e85b4cb9125cb1ec1f81e5315226f228e8f7834/src/prc/weather.py), [pilot](https://github.com/RavindraTarunokusumo/prc-data-challenge-2026/blob/d5065169740ea475e97abe67f5088f07b31a9ab1/scripts/pilot_weather_D08.py), [negative result commit](https://github.com/RavindraTarunokusumo/prc-data-challenge-2026/commit/2d53d34e15572e09d0505728716a29e27edf3dea), [наша погода](/Users/denyskolomiiets/prc-2026/src/features.py:162), [свежие наши итоги](/Users/denyskolomiiets/prc-2026/REVIEW_BRIEF.md:19).

### Inferences

- Конкурентные дельты сейчас полезнее как проверка конкретного переноса/поддержки, чем как источник нового класса модели. Ни floor, ни Arm C не обосновывают обещание секунд или запуск большого обучения.
- «Деайсинг-площадки приняты за стоянки» у gentle-octopus — объяснение автора для ошибки dwell-detector, не доказанная локализацией точек причина. Эту фразу нельзя превращать в подтверждённый новый датчик или в объяснение разрыва с лидерами.

### Gaps

- Чужие target fits и наш OOF/replay в этом задании не запускались; собственная достижимая популяция floor и остаточный эффект неизвестны.
- Импортируемые общие модули smart-jigsaw не проходили полного source-to-sink аудита. Подтверждена конкретная Arm C маска/путь применения, но не корректность всех их CV, категориальных кодировок и импортированных моделей.

## Насколько полон вывод и какие действия выполнены?

### Takeaway

Обзор завершает последние две вечерние дельты и исправляет конкретный пропуск smart-jigsaw. Он не заменяет поиск новых источников или новых способов использования сырых траекторий, который ведётся другими исследователями.

### Cited Findings

- Базовые предыдущие состояния сверены с [обзором 6 октября](/Users/denyskolomiiets/prc-2026/reports/Новые%20репозитории%20PRC%206%20октября.md), [обходом 4 октября](/Users/denyskolomiiets/prc-2026/analysis/competitor_repos/_UPDATE_2026-10-04.md), [Fable 7 октября](/Users/denyskolomiiets/prc-2026/analysis/review_fable_2026-10-07/fable_A_gap_forensics.md), актуальными AGENTS/HANDOFF и закрытыми идеями REVIEW_BRIEF/HYPOTHESES_LOG. Старые лидерборд-числа не объявлены повторно проверенными.
- Сохранены JSON commit/search/branch receipts и только небольшие кодовые/текстовые файлы; список и контрольные суммы — [manifest](competitor_receipt_manifest.json). Никаких target fits, CV, изменений production, коммитов, submissions или сообщений участникам не выполнялось.

### Inferences

Приоритет этому направлению: сохранить две конкретные проверки в запасе, а новый существенный бюджет отдавать только независимым новым данным/механизмам. Это решение о приоритете, а не доказательство математического нулевого потенциала любого расширения floor или residual-corrector.

### Gaps

Публичность всех команд, GitLab/Kaggle/Hugging Face, закрытые репозитории и новые материалы после 23:15Z 7 октября этим обзором не установлены. Абсолютное «у всех команд выше нас нет публичного кода» не подтверждается этим ограниченным исследованием.

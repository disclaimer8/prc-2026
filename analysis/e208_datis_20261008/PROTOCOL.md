# E208: открытые D-ATIS, проверка покрытия без меток

Статус: подготовка к независимому review и пререгистрации. **До root prereg commit разрешён только inventory метаданных релизов и synthetic/parser tests.** Реальное сканирование архивов, metadata pack и coverage run не запущены. Никакого model fit, OOF, target evaluation или подачи.

Точная строка: `E208-CC0-DATIS-fixed16days-label-free-60min-v1`.

## Вопрос и фиксированные данные

Доступны ли для наших10 аэропортов timestamped ARR/DEP ATIS с операционными состояниями и какая доля настоящих вылетов имеет свежий, уже полученный бюллетень? Дни заранее фиксированы:1/8/15/22 января и июля2025/2026,16 суток. Порядок при ограничении бюджета — сначала число месяца, затем2025Jan/2025Jul/2026Jan/2026Jul; внутри дня asset-name ascending. Цифры покрытия не являются RMSE и не разрешают модель.

Только официальный [adsb.lol aircraft-data-links](https://www.adsb.lol/docs/open-data/aircraft-data-links/), **CC0**, репозитории `adsblol/aircraft-data-links-2025` и`2026`. Это не лицензия ODbL ADS-B globe. Airframes и другие ACARS-провайдеры исключены. Для Jan2026 используется исторический release в repo2025. Inventory фиксирует API/release URL, asset IDs, sizes, digest, updated_at и время получения. Новые релизы проверяются по всем страницам assets API; отсутствующий релиз остаётся unavailable. Лицензия указана для этого источника, не как blanket-разрешение для внешних наборов.

Полные Jan15/Jul15 2026 уже исследованы:253assets/143913547байт. Повторная загрузка исключается только после точной проверки полного множества asset URLs, release SHA, каждого compressed SHA против GitHub digest, размеров, status=ok и отсутствия malformed JSON. В reuse сохраняются только результаты нынешнего parser и counters: station/tail/flight identity и посторонние ACARS не копируются. Старый candidate selector является тем же superset: uppercase target ICAO и`ATIS`; произвольная смена case-selection сделала бы reuse неэквивалентным. Локальная2025 проба не имеет доказанного full-release inventory и не считается полным днём.

## Parser и временная семантика

Фиксированные regex, без LLM. Допускаются только A9, явный заголовок`ICAO ARR|DEP[/ARR|DEP] ATIS <letter> HHMMZ` либо документированный испанский порядок`ICAO ATIS ARR|DEP <letter> HHMMZ`, routing-prefix и assembly_status=complete. Запросы, clearance prose, неизвестная роль, частичные сборки и неоднозначные timestamps считаются отдельно и не присоединяются как valid state. Сохраняется только текст бюллетеня начиная с ICAO; receiver/aircraft identity исключены. Неожиданное personal-content сообщение отклоняется без сохранения текста.

Время received — timezone-aware `_adsblol.received_at`. HHMM выпуска привязывается к ближайшим не будущим UTC-суткам относительно received; если age>6ч или часы неверны, дата считается неоднозначной. Это консервативное правило восстановления даты, не утверждение об официальном сроке ATIS. Полночь тестируется явно. RSCD observation clock обрабатывается отдельно от времени выпуска; ambiguous condition-clock не становится уверенным свежим condition-state.

Привязка требует **issued≤MVT и received≤MVT**. Свежесть фиксирована60мин **от issued**, верхняя граница исключительная; задержанный приём не продлевает state. ARR и DEP присоединяются отдельно, ARR является только контекстом прилётов. Новый полученный полный выпуск supersedes прежний. Разные операционные состояния одного issue-time дают conflicting_same_issue до следующего выпуска. Дубликат текста/issue/role хранится с самым ранним received полного сообщения.

Closure parser извлекает только явно названные runway/taxiway/exit и CLSD/CLOSED/UNAVAILABLE; runway-condition code поддерживает явные5/5/5 и RSCD666+observation-clock. Ограниченная консервативная грамматика времени распознаёт FROM/UNTIL/TILL/EXPECT*/CHANGE TO/AT/BTN/BETWEEN и следующие не дальше65 символов часы HHMM либо HH:MM (с необязательным Z или HOURS). При таком совпадении **все closures и runway-condition codes этого бюллетеня исключаются из текущего состояния**, scheduled_or_future_text=true; обычное покрытие ATIS сохраняется, operational coverage не увеличивается. Интервалы не разрешаются; смешанный текущий и запланированный текст также консервативно исключается. Неизвестные обороты, даты без часов и временные оговорки за пределом65 символов эта грамматика не гарантирует: это ограниченный parser pilot, а не полный авиационный парсер. Пропуск, частичный текст, expiry или отсутствие слова CLSD **не означают открытие объекта**. Rejected/ambiguous counts обязательны.

Сырые candidate count, accepted complete count и unique bulletin count публикуются раздельно. Coverage считает уникальные flight IDs, ARR/DEP coverage, union(any role), operational union, expired/conflicting/no-prior. Один flight с обоими ролями в union считается один раз. Для первых60мин каждого фиксированного дня отдельно отмечается left-censoring: предыдущий день не запрашивался. Все coverage-строки читаются вместе с source-day complete/partial/unavailable; неполный архив не доказывает отсутствия ATIS.

## Метаданные вылетов и воспроизводимость

Canonical sources —12 training parquet и ranking в`data/prc-2026-datasets`. Arrow/Polars projection читает только`MVT_ID_mvt,PHASE_mvt,ADEP_mvt,MVT_TIME_UTC_mvt`; PHASE фильтрует DEP и не сохраняется. Cache содержит **только id,airport,mvt** выбранных16 дней. BLOCK/TAXITIME, OOF и prediction files не открываются для значений. SHA исходных Parquet считается по непрозрачным байтам для provenance, без декодирования label columns.

После prereg `pack` сверяет source/test/protocol/inventory с полным Git commit, создаёт свежие`analysis/e208_datis_20261008/inputs/`: projected departures, sanitized reuse, inventory и hash manifest. `run` проверяет все source/input SHA и registration перед любым archive request. Выход только в новом`execution/`; повтор/resume в существующий каталог запрещён. Старые исследования и production не меняются.

## Бюджеты и порядок исполнения

Hetz existing `.venv`,≤4CPU/nice10/MemoryMax4GiB/swap0. Python standard library+имеющиеся pyarrow/polars; штатный`/usr/bin/zstd`, никаких env/install changes. Network concurrency4, без retries. Не более2GiB **новых compressed bytes**: целые assets резервируются по manifest до запроса; oversize/непоместившиеся явно skipped. Повторно используемые253assets не расходуют новый network-budget. Максимум32MiB compressed/asset,1MiB JSON-line,512MiB decompressed/asset; zstd memory128MiB, вывод читается построчно. Сырые архивы только в bounded RAM, на диск не пишутся. На диск попадают sanitized ATIS и metadata receipts.

Wall cap45мин; worker deadline44мин оставляет минуту для summary. Каждая network read имеет timeout и проверку deadline; decoder имеет отдельный timer≤75с и kill на ошибке. Внешний root-owned systemd unit должен иметь RuntimeMaxSec=2700/KillMode=control-group; никаких nested SSH из script. При cap/errors остаётся честный partial manifest. `complete.json` всегда содержит accepted_for_model=false. Гейта по проценту покрытия и адаптивного выбора дат нет.

## Команды после регистрации

```sh
.venv/bin/python scripts/e208_datis_coverage.py inventory --root /Users/denyskolomiiets/prc-2026
# Первая команда только metadata, разрешена до prereg.
.venv/bin/python scripts/e208_datis_coverage.py pack --root /Users/denyskolomiiets/prc-2026 --registration-commit FULL_ROOT_COMMIT
# Root переносит frozen script/protocol/inventory/tests+inputs на Hetz штатным direct SSH/scp.
.venv/bin/python scripts/e208_datis_coverage.py run --root /root/prc-2026 --registration-commit FULL_ROOT_COMMIT
```

До применения модели необходимы отдельные source-coverage review и научная регистрация. Здесь нет обещания выигрыша и нет автоматического перехода к fit.

## Выполненный source inventory (без архивного extraction)

Metadata16 дат завершена:12 релизов перечислены полностью,1800assets/766417808 объявленных байт. Четыре января2025 далиHTTP404 и остаются source-unavailable; это не отсутствие самолётных сообщений вообще. Проверенный reuse253assets/143913547байт означает **622504261байт новых запросов** при полном проходе доступных релизов (около594MiB), ниже2GiB cap. Архивы пока не запрашивались. Неудачная sandbox DNS попытка сохранена отдельно; её пустые результаты не принимаются за source availability.

# HANDOFF — передача работы (08.10.2026; оперативные обновления выше исторических снимков)

Правила, машины, рецепты — в `AGENTS.md`. Здесь — позиция, что считается и что делать дальше.

## ТЕКУЩЕЕ СОСТОЯНИЕ — 11.10.2026 09:30Z (владелец: Claude; агенты на Opus)

**Финал:** `data/final_tail_20261011/work/gentle-igloo_final.parquet` **SHA bbae2cc5…34a1** = файл 09.10 (76c32426) + стадия E218 (уровень PRU, `E218_PRU_LEVEL=1`; №451/452/457). Валидатор PASS, пересборка на маке побайтно; контроль **v12082 = 231,1445** (−0,036 к v12080, коридор ✓; №458/459). Копии: Hetzner `/root/prc-final-backup-20261011/`, Home `/home/a1/prc-final-backup-20261011/`, PRCEXT.

**✅ ФИНАЛ ЗАГРУЖЕН 12.10 08:44:59Z** (разрешение Denys; №462): объект в бакете побайтно = bbae2cc5, обычные подачи закрыты ledger'ом. Дальше — только репо/доки до 14.10 10:00Z, затем открытие (OPENING-CHECKLIST). Выполненная команда (повторять НЕЛЬЗЯ, скрипт откажет):
`cd /root/prc-2026 && /root/prc-2026/.venv-upload/bin/python scripts/final_transport_20261009.py upload --manifest analysis/final_transport_20261009/manifest_final_v2.json --registration-proof analysis/final_transport_20261009/proof_final_v2.json --registration-commit 6b130265f467b050182600cfe0d7feb6bb72100e --source-sha256 831712bfd165c3e94a20f85a801047514ab6f462231e1c8cdfedc6264650ab4f`
(preflight пройден 11.10 на маке и Hetzner; старый `manifest_final.json` для 76c32426 больше не проходит — сборщик теперь 21f46756.)

| Блок | Статус |
|---|---|
| Best обычной доски | **v12081 = 230,4478** (v12077 + E218), 18/242 на 11.10 08:51Z |
| Цена честного рецепта | E81 +0,568 (v12078), свежий хвост +0,696 (v12079); контроль финала без E218 v12080 = 231,1805, с E218 v12082 = 231,1445 |
| Закрыто 09–11.10 | E215, E216 (скаляры на месте), E217 (зимний вес NN), I1–I8, внешние разборы; пересборка на prc-cloud не побайтная (TabPFN: Accelerate vs OpenBLAS; E141 — подпись Python), №454 |
| Документы | агент обновляет docs/paper под bbae2cc5 и E218 (раскрытие использования PRU) |
| Статья | `paper/` готова кроме чисел после 14.10; ORCID 0000-0003-3212-3251 и e-mail denys@himaxym.com вписаны 11.10 (от Denys); «no external funding» подтверждено Denys 11.10 — от автора больше ничего не нужно |
| Открытие | 14.10 после 10:00Z: репо public + письмо со ссылкой и SHA последнего коммита на challenge@opensky-network.org (по «да» Denys) |

## Передача управления Claude — 08.10.2026 12:19UTC

Пользователь передал все задачи Claude. Начать с **[CLAUDE_TAKEOVER_20261008.md](CLAUDE_TAKEOVER_20261008.md)**: реальные результаты, пути, SHA, drafts и очередность продолжения. Codexagents остановлены.

Mac featurebuild №385 завершён PASS:11стадий/64.89с/peak12.53GiB, готовый WORK `data/final_feature_work_mac_20261008`,2755837строк, latestfinalrankSHA4b0692a7. НаMac сейчас нет долгой Codexочереди;0finalmodel-fits. E20718/18 завершён12:00:56UTC, **gateFAIL** (pooledbody+0.007327с/decided+0.015615с); child№383не запускать. E208итог№384. Журналдо№386. Tabular/NN independentREADY drafts, actualfit ещё не зарегистрирован. Tail/7dailyderivative adapters ещё не реализованы.

Дедлайн всего конкурса продлён до **14.10 10:00UTC**, подтверждено главной страницей и письмом в записях Claude ниже. Финальныеfeatures уже построены по окончательным файлам11:04Z. Best230.9387/15из237 относится к08.10 10:31UTC; liveквота08.10не проверена. [Квитанции](analysis/claude_transfer_20261008/state.json).

## Историческое обновление Codex —08.10.2026 11:10UTC

E207 v3 зарегистрирован №379, commit `f43405e7c6836230862837b66fad081ae4d69777`.
Фактический старт controller55670 —08.10 10:06:07.756092UTC. Все365 дней и6 fold
таблиц готовы. Materialized spec `56e1417bb71aa917847a7536ffbf8f0a9110c768a66bbaa6bf32e851fbffc4ca`.
Synthetic full-shape GPU smoke PASS (flat/ordered, peak13.02/12.75GB);
[квитанция](analysis/e207_full_token_v3_20261008/smoke_snapshot.json).
Fold1 завершён к11:01:50UTC: **3/18 fit завершены**. На11:10UTC идёт второй фолд
(seed1 epoch4/8); это обучение, не оценка качества. Folds1/2/3 допущены,
fold4/5 ещё передаются. GPU-unit PID870055, Invocation`4ccf79f86a144e4d9aaedd1ef0f169bd`,
active/running. Не менять frozen source/protocol и не дублировать очередь.
После всех6пар/18fit нужен отдельно зарегистрированный current-chain CV; нового RMSE нет.
Best230.9387;15/237 — полная доска08.10 10:31:23UTC, authenticated квота08.10 не проверена.
E208 №380 реально обрабатывает фиксированный16-дневный label-free пилот наHetzner.
Пакет saved-components v12075 №381 зарегистрирован, **не запущен и отложен** ради
единого финального конвейера. E201 отменён; ARR-пилот остановлен до extraction; LLMoff.

**Ближайшая работа:** единый finalWORK и сохранение всех моделей/преобразований;
fresh2025+final feature view (E165/E157 зависят от union train+ranking),55productionfits
по уже выбранным параметрам плюс10TabPFN calls;16cachedstage-моделей повторно
используются только при совпадении training signatures. Это план, ни один final-fit
ещё не запущен. [Точный состав/ресурсы](analysis/final_pipeline_readiness_20261008.md).

175 параллельно скачивает58rawADS-B дней февраля/июня в `/home/a1/claude_adsb_2026/`;
это задача Claude, не дублировать. На11:02UTC журнал подтверждал9дней; ETA не является
завершением. Авто-review ранее отказал Claude в transfer final_ranking на175; не обходить
отказ альтернативным транспортом. Производные строить в отдельном разрешённом WORK.

Текущие владельцы: root — train/save adapter, внешние входы и регистрация;
fresh_mechanisms — final feature/workspace adapter; fresh_competitors — E207 CV-only
child (№382 зарезервирован после review), затем final neural adapter;
fresh_external — E208 результаты/review child. Финальный default пока E203ordered,
E207 входит только после исходного и current-chain гейтов. Репо/design/docs11.10,
слепой файл14.10 10:00UTC; опубликованный ниже блок Claude с новыми правилами сохранён.

## ДЕДЛАЙН ПРОДЛЁН — 08.10 12:10Z (письмо Enrico/espinielli командам + сайт index.html)
- **Весь конкурс, включая репо/доки, — до 2026-10-14 10:00Z** («The competition runs … up until 14 October 2026 12:00:00 CET»; «Deadline is extended till 2026-10-14T12:00 CET (2026-10-14T10:00Z)»). Финальная фаза: 4 месяца, ранжирование янв/июл, фев/июн и все 4 + ревью кода и документации.
- Следствия: заморозка дизайна фактически сдвигается к 14.10; обычные подачи янв/июл по сайту тоже до 14.10 10:00Z (5/сутки — больше слотов под S1–S3); финальный файл грузить после последней обычной подачи; репо открывать 14.10 ≈ 07:00–08:00Z после финальной загрузки (OPENING-CHECKLIST).

## ОКОНЧАТЕЛЬНЫЕ ФИНАЛЬНЫЕ ФАЙЛЫ — 08.10 12:00Z (Claude)
- espinielli 11:55Z: «yeah, sorry, that is the "randomness" I had to introduce...just use the final_ranking.parquet / final_submission.parquet for the final phase». ±12 с — намеренная случайность; финал — по текущим файлам.
- **По каноническим путям теперь окончательные (загрузка в бакет 11:04Z):** `data/prc-2026-datasets/final_ranking.parquet` SHA-256 `4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f` (85 853 399 Б), `final_submitting.parquet` `b3445d80e29bf599a5583ca9675f63ddf57b51fdf1ca3f498facb38ea84f3924` (3 003 722 Б). Утренние (08:47Z) перенесены в `data/prc-2026-datasets/final_v1_0847Z/` — не использовать. Новый `scripts/final_feature_workspace_20261008.py` (untracked) на момент замены не запускался и workspace не строил.

## ОТВЕТ ОРГАНИЗАТОРА 08.10 11:16Z (правка 11:30Z) на вопрос gentle-igloo — дословно
- (3) «Thank you for spotting this error: there shouldn't be anymore +/-12s discrepancy. Final prediction on final_ranking.parquet. Please re-download the data.» → `final_ranking`/`final_submitting` перезалиты; старые копии — `data/prc-2026-datasets/final_*.parquet` (08:47Z), новые (загружены 11:04Z) — `data/prc-2026-datasets/final_v2/` (ID те же). **Проверено 11:38Z: ±12 с не исчезли** — янв/июл всё ещё отличаются от ranking.parquet (MVT сдвинут −12…+12 с у 91,9 %, RMS 4,92 с, taxi-in == BLOCK − MVT), и почти все строки отличаются от версии 08:47Z: свежий розыгрыш дизеринга при выгрузке. Уточнение организатору отправлено 11:40Z (какой файл окончательный). **Финальный прогон делать на последней версии из бакета и записать её SHA-256 в манифест** (v2: final_ranking 85 853 399 Б, etag `8970d7057c0c72b5138020578f6bc861-6`).
- (1)/(2) «maybe it is just better to extend the deadline to Oct 14th 12:00 CET. I will update the webpage and anyway send an email to all team correspondents» → **вероятно, дедлайн репо/доков сдвинут на 14.10 10:00Z**; ждём обновления сайта и письма. До подтверждения — прежний план (11.10).
- (4) «deadline is noon 14Th Oct CET, or 2026-10-14T10:00Z».

## ФИНАЛЬНЫЙ ЭТАП — проверенные факты, 08.10.2026 10:30Z (сессия Claude; решения по конвейеру — за Codex/Denys)

- **Объявление espinielli в Discord 08.10 07:34Z и 07:46Z** (дословно — `docs/organiser-clarifications.md` после обновления): финальный датасет для одного слепого ранжирования; ранжирование по янв/июл, по фев/июн и по всем 4 месяцам; на ревью — топ-30 по янв/июл **или** топ-30 по финалу; причина — признаки «optimizing for the ranking». **«Repos/docs deadline will remain the same (the design of the model shouldn't change)»**; финальная подача до **2026-10-14T12:00+02:00 = 10:00Z**.
- Сайт (eligibility/ranking/data, 08.10): одна подача `gentle-igloo_final.parquet`; условие приза «Final single submission is uploaded by 2026-10-14T12:00 CET».
- Бакет `prc-2026-datasets` (08:47–08:48Z): `final_ranking.parquet` (1 341 197 строк, 30 колонок, схема = ranking) и `final_submitting.parquet` (670 790 строк, MVT_ID_mvt Float64 + TAXITIME_SEC_mvt Int32). Скачаны в `data/prc-2026-datasets/`. **Месяцы: 2026-01, 02, 06, 07**; DEP 152 719 / 144 158 / 181 791 / 192 122. Все 344 841 DEP-ID янв/июл совпадают со старым ranking.
- **Дизеринг, не новый шум (исправлено 11:00Z по отчёту `analysis/final_jitter_sensitivity_20261008.md`):** в строках янв/июл `final_ranking` время взлёта сдвинуто относительно старого ranking (~92 % движений, RMS 4,9 с). Это свежий розыгрыш дизеринга поставщика, который есть во **всех** файлах, включая 2025: MVT = T − η/2, BLOCK = T + η/2, η ~ U(−12, 12); taxi-in == BLOCK − MVT в 100 % строк (сдвиги антикоррелированы, −0,99; прежняя запись «независимы» была ошибкой — я сравнил разность с неверным знаком). Плато ±6 с (TOL=6) и окно 3606 — это и есть дизеринг. Ожидаемое ΔRMSE = 0, менять дизайн/делать аугментацию **не нужно**; чувствительны только ~20 строк хвоста LIRF без NM фев/июн (усреднять TabPFN по сидам).
- adsb.lol `globe_history_2026`: релизы за февраль и июнь 2026 есть (prod ≈ 3 ГБ/сутки, 58 суток).
- На маке свободно ~13 ГБ диска (10:30Z).
- **Загрузка финала (Claude, 10:45Z):** `scripts/validate_final_submission.py` (13 тестов OK) и `docs/final-submission-runbook.md`. **Ловушка:** замороженные транспорты (`bucket_inventory` в `src/e185_submission.py`, `read_ledger`, `submission_guard.ROWS=344841`) принимают только `gentle-igloo_v\d+.parquet`; после появления `gentle-igloo_final.parquet` в бакете старые загрузчики падают с `ambiguous team bucket object`. Значит: финал грузить **после** последней обычной подачи (11.10 ≤ 21:59Z) и новой датированной обёрткой (runbook §3).
- **Вопрос организатору опубликован в Discord 08.10 10:38Z** (от Denys): можно ли держать репо приватным с доступом организаторам до 14.10; коммиты «только запуск» после 11.10; использовать ли только `final_ranking` (шум ±12 с); дедлайн = 10:00Z. Ответа пока нет — до ответа план: открыть репо 11.10 вечером.
- **Стратегия финала (Fable, 10:50Z; рекомендация, решают Codex/Denys) — `analysis/final_strategy_20261008.md`:** рецепт финала = v12075, сделанный периодонезависимым: **убрать E81** (доноры v44/v60 есть только на янв/июл, без OOF) и **заменить 42 замороженные строки TabPFN детерминированным ансамблем E156** на всём сегменте (в `final_ranking` LIRF без NM, D ≥ 4 ч: янв 15, фев 8, июн 12, июл 28 — для фев/июн доноров нет в любом случае). Остальное оставить; E188 — по желанию. Копировать v12075 в янв/июл финала **не** рекомендуется (ровно «optimizing for the ranking»). Слоты 09.10: S1 = v12075 без E81, S2 = свежий TabPFN-ансамбль вместо 42 строк, S3 = оба (рецепт финала на янв/июл). OPDI в `data/opdi` покрывает 31.12–09.02 и 29.06–08.08 — фев/июн для LSZH докачать или fallback «нет события → без поправки». Бустеры Cat15/LGB5/якорей не сохранены (только `.npz` прогнозов) — для новых месяцев их надо переобучить/пересохранить.
- **Документация (Claude):** раздел «Leaderboard use disclosure» в `docs/development-history.md` §6 и объявление 08.10 в `docs/organiser-clarifications.md` — в master (e67ec58).
- **ADS-B фев/июн 2026 предзагружен (Claude, ноутбук 175):** `/home/a1/claude_adsb_2026/` — те же `scripts/get_adsb_day.sh` + `src/adsb_extract.py` (md5 = мак), выход `data/adsblol/surface_<day>.parquet`, 58 дней, старт 10:42Z, P=4, ETA ≈ 12:35Z; прогресс `logs/progress.tsv`, проверка: 58 файлов и `grep -c FAIL` = 0, манифест `bash manifest.sh` → `MANIFEST.sha256`. Производные (offblock/visit/head) не строились — им нужен `ranking.parquet`. Для финала: прилинковать surface-файлы в `$WORK/data/adsblol/`, `adsb_day` пропустит скачивание. Отчёт `analysis/final_period_transfer_20261008.md`: февраль — самый «лёгкий» месяц (no-NM 0,91 %, хвост LIRF 64 строки), июнь ближе к июлю; главный риск — покрытие ADS-B по новым месяцам (EDDF 01.02 просел), нужна таблица покрытия до E70/E64; в Feb-2025 режим «расписание» в хвосте LIRF был 78 % против 33–49 % в другие месяцы.
- **Новый узел `prc-cloud` (Hetzner Cloud, HEL1, заказан Denys 08.10; квалифицирован Claude 11:35Z):** `ssh prc-cloud` (root, ключ `~/.ssh/id_ed25519` мака; алиас в `~/.ssh/config`). ARM Neoverse-N1 **16 vCPU, 30 GiB RAM**, 287 ГБ свободно, Ubuntu 26.04. `/root/prc-2026` = `git archive` master, `.venv` через `uv sync --frozen` (Python 3.12.14; polars 1.44.2, lightgbm 4.7.0, catboost 1.2.10, numpy 2.5.3, pandas 3.0.5, sklearn 1.9.1, pyarrow 25.0.1 — **как на маке**). Скопирован `data/features_fin1b.parquet` (SHA = мак). **Пробник на продакшн-пути данных** (FIN1b, DROP_PREFIX, PROXY_SCALE=1, `train.regime_label`/`add_encodings`/`proxy_base`, CatBoost depth 8 lr 0.08 thread 8 seed 0 × 60 итераций, LightGBM deterministic × 100 раундов): прогнозы на 344 841 строке ranking **побитно равны маку** (proxy base, CatBoost, LightGBM). Пик RSS 22,3 ГБ → **один CatBoost за раз**; 0,46 с/итерацию на 8 потоках (мак 0,27) → ~43 мин на сид 5600 итераций. Пробник: `/root/prc-2026/node_probe.py`, лог `/root/probe_cloud.log`. Узел свободен, задач на нём нет.
- Claude параллельно (без правки конвейера, отчёты в `analysis/final_*_20261008.md`): чувствительность к шуму ±12 с, перенос на фев/июн (покрытие ADS-B), валидатор финального файла, риск-аудит ревью.

## Исторический снимок 08.10.2026 10:28 UTC — E207 v3: подготовка завершена, передача на Home

E207 v3 зарегистрирован №379, commit `f43405e7c6836230862837b66fad081ae4d69777`.
Фактический старт controller55670 —08.10 10:06:07.756092UTC; ожидание разрешения
до этого не является вычислением. Все365 дней извлечены, materialized spec
`56e1417bb71aa917847a7536ffbf8f0a9110c768a66bbaa6bf32e851fbffc4ca`.
На10:28UTC все6 fold tables подготовлены на171/175; оба prep-unit retained
active/exited/MainPID0. Идёт передача fold1/3 наHome; GPU-unit PID870055
active/running, лог `WAIT PREPARED 1`: **0 подтверждённых target-fit**, smoke ещё
не подтверждён. [Точный статус E207 v3](analysis/e207_full_token_v3_20261008/STATUS.md).
Не менять frozen source/protocol и не дублировать очередь. Далее admission,
synthetic GPU smoke и6пар/18fit; ranking требует отдельного current-chain гейта.
Best230.9387 прежний;15/237 — полная доска08.10 10:31:23UTC (100страниц/4981подача).
На08.10 обработанных подач команды нет; authenticated live-квота не проверена.
E208 D-ATIS coverage и переносимый saved-component пакет v12075 готовятся;
их bulk-запуски ещё не зарегистрированы. E201 отменён, LLMoff.

## Исторический снимок 08.10.2026 02:30:30 UTC — E207 v2 технически остановлен

- **V2 controller38371 завершён**, оба raw-unit inactive/dead/MainPID0. 171 сохранил183day receipts и shard`complete.json`;175 —145day receipts без shard complete. **Всего328 сохранённых дней,0 prep/target-fit/ranking.** [Полная квитанцияv2](analysis/e207_full_token_v2_20261008/STATUS.md).
- После завершения171 systemd GC очистил InvocationID; `wait_unit` ошибочно завершил очередь с `owned unit invocation changed`. Owned175 cleanup02:28:24.955675UTC, terminal`failed_no_retry`02:28:53.186665UTC. Это технический lifecycle-сбой, не scientific FAIL. Operational snapshot дополнен terminal-событием.
- Данные/partials и frozen v2 source/protocol сохранены. Author готовит отдельнуюv3 с bounded real-systemd smoke; **№379 пока не зарегистрирован**, новый запуск не выполнен. Не возобновлятьv2 и не переиспользовать его каталоги без отдельного протокола.
- Best230.9387 прежний;14/237 — снимок07.10 22:00UTC,4/5слотов были7октябряUTC; live-квота8октября не проверена. E201 отменён, LLMoff. Ниже текущая история запускаv2 и неизменённый блок документации Claude.

## Исторический снимок 08.10.2026 02:26:18 UTC — E207 v2 реально обрабатывает дни

- **E207 v2 запущен02:24:15.531271UTC**, controller PID38371, №378/commit`ea16e05330d3fe4fb5573dc17938002f14645876`. [Статус E207 v2](analysis/e207_full_token_v2_20261008/STATUS.md). Source/protocol после регистрации не менялись.
- На02:26:18UTC сохранены **71 дневной receipt на171 и54 на175 —125/365 дней**. Оба raw-unit active;171 PID365405/Invocation`a5e6e4a3afa347dbb3440143ba5e54d5`,175 PID168255/`e92ac9eae23f4f979182c7ce98ea2601`. Это подтверждённый extraction-progress, не только CPU/liveness. MemoryPeak3.017/2.467GB.
- **0 target-fit, нового RMSE нет.** Далее автоматически обмен shard→prep/admission→synthetic GPU smoke→6пар/18fit. Около2ч — оценка workflow, hard6ч; ranking требует отдельного current-chain протокола, автоматической подачи нет. Не дублировать запущенную очередь.
- V1 остановлен технически с0day/0fit и сохранён. Best230.9387 прежний;14/237 — снимок07.10 22:00UTC,4/5слотов были7октябряUTC. Live-квота8октября не проверена. E201 отменён, LLMoff. Ниже сохранены история запуска/остановкиv1 и независимые изменения документации к призу.

## Документация к ревью приза — 08.10.2026 00:55Z (сессия Claude, коммиты a95ed2b, 92c7278, merge 84ddbaa)

- README и `docs/` переписаны под ревьюеров приза (английский): README с таблицей открытости данных, `docs/development-history.md`, `docs/data-insights.md` + `docs/figures/` (`scripts/make_doc_figures.py`), `docs/negative-results.md`, `docs/glossary.md`, `docs/organiser-clarifications.md` (дословные цитаты организатора с UTC), переписанный `docs/reproduction.md` (история v12074 → `analysis/reproduce_v12074_20261007/HISTORY.md`), `requirements-gpu.txt`, `CITATION.cff`, метаданные `pyproject.toml`. Прошло ревью Fable в роли проверяющего PRU: `analysis/review_fable_2026-10-07/`.
- При новом best обновлять числа в README (результат, таблица фаз), `docs/development-history.md` (шаг), `docs/tables/leaderboard_gentle-igloo.csv`, график 1 (`scripts/make_doc_figures.py`). Структуру не откатывать; правки — по месту.
- Открыто для Denys до 09.10: раздел Team в README (имя, фраза про AI-ассистентов, нужны ли affiliation/контакт), что делать с внутренними файлами (AGENTS/HANDOFF/hz_queue/outreach/review/research_notes/reports, LAN-IP в tracked-файлах) при открытии.

## Исторический снимок 08.10.2026 00:08:36 UTC — первый E207 технически остановлен

- **E207 №377 остановлен до завершения первого дня и до fit.** Root обнаружил повторную распаковку NPZ в per-flight `validate_arrays(NpzFile)` и отправил SIGTERM controller87152. Это техническая проблема выполнения, не научный FAIL F-only. [Полная квитанция старого запуска](analysis/e207_full_token_20261008/STATUS.md).
- Cleanup остановил owned175 Invocation`070a9f744fd24d6db4e05d5b7ce13a32` в00:07:42.600974UTC и171 Invocation`69d7d4c5f736471ba82c4c5fca6324bb` в00:07:43.351569UTC; `failed_no_retry` записан00:07:51.862671UTC. Проверка00:08:36UTC: оба unit inactive/dead/MainPID0, controller отсутствует. **0 completed day receipts на обоих shard,0 prep/target-fit/ranking.**
- Старые source/protocol/metadata frozen, partials и логи сохранены. Author готовит отдельные v2 sources/namespaces; **№378 пока зарезервирован после review, не зарегистрирован и не запущен**. Старую очередь не возобновлять и её каталоги не переиспользовать. Текущего E207 fit нет; прежние заявления о работающих unit ниже — история.
- Bestv12075=230.9387с прежний;14/237 — снимок07.10 22:00UTC,4/5слотов были7октябряUTC. Live-квота8октября не проверена. E204/E205 завершены FAIL, E201 отменён, LLMoff; новых подач нет.

## Исторический снимок 08.10.2026 00:05:31 UTC — E207 реально запущен

- **Controller E207 работает:** старт00:02:05.287063UTC, PID87152 под caffeinate, пререгистрация №377/`741045e9fe10c17f68a7ce6bed4216e6160ee7b1`; source не менялся после commit. [Текущий статус E207](analysis/e207_full_token_20261008/STATUS.md).
- **Extraction-unit активны:**171 — PID352872 / InvocationID`69d7d4c5f736471ba82c4c5fca6324bb`,175 — PID157961 /`070a9f744fd24d6db4e05d5b7ce13a32`; старты00:02:46/47UTC подтверждены events и systemctl. На00:05:10UTC оба worker-log пусты, обработка первого дня ещё не подтверждена. На00:05:31UTC подтверждено по3 дочерних worker на узел (~100%CPU каждый); cgroup CPU за21с вырос на63.49/63.17CPU-с. Первый завершённый день пока не заявляется. **0 target-fit; prep и GPU smoke ещё не подтверждены стартовавшими.**
- PACKED spec`fa19112c8cdeb217bd40269f04a9cd5866eeebd61cc3ec9bcdb5379940e87e68`; входы переданы до старта extraction. Idle/disk admission: Home711576444928/171162340503552/175415114645504байт. Очередь автоматически продолжит extraction→prep/admission→synthetic smoke→6пар/18fit. Около2ч — оценка, hard6ч; после сбоя нет retry/resume/fallback. Ranking потребует отдельного current-chain протокола, автоматической подачи нет.
- **Best v12075=230.9387с прежний.** Доска14/237 — старый снимок07.10 22:00UTC;4/5слотов относятся к7октябряUTC. Сейчас уже8октябряUTC, **live-квота8октября не проверена**. E204/E205 завершены FAIL, E201 отменён, LLMoff. E207 — текущая очередь; прежнее «все расчёты завершены» ниже относится к истории.

## Исторический снимок 07.10.2026 23:23:12 UTC — исследование завершено, E207 готовится

- **Best v12075 = 230.9387 с не изменился; новых подач нет.** Доска **14/237** — прежний снимок **7 октября 22:00 UTC**, не новая проверка. **4/5 слотов относятся к 7 октября UTC**: местная полночь в Лиссабоне не начинает новые UTC-сутки. E201 отменён, LLM выключены.
- **Исследование и metadata завершены, журнал №376:** [Новые сигналы PRC 8 октября](<reports/Новые сигналы PRC 8 октября.md>). Это фиксация прочитанных источников и наблюдений, **не пререгистрация обучения**. E194 проверен по427 дневным файлам: собственный ranking-поток raw>64 у119699, недозаполнен у118881; предыдущий —172406/171008. Потоки пересекаются; числа не являются выигрышем RMSE.
- **E207:** реализация F-only и независимое review в работе; **0 фактических подготовок таблиц и0 target-fit**. Предложены6пар/18fit с сохранением старых индексов и заполнением свободных слотов, затем отдельный контроль текущей E203-цепочки. №377 зарезервирован root для пререгистрации после review и пока не добавлен. Наличие кода не означает запуска.
- **D-ATIS:** полный CC0-скан15 января/15 июля2026 —253assets,143913547байт,77.6с,0ошибок. В июле466сообщений/268текстовых вариантов/8хабов; январские сутки дали0 распознанных этим parser ATIS наших10хабов, не отсутствие за месяц. Закрытия LFPG08R/08L и LEMDEA5 наблюдались в открытом сырье. Присоединение к ranking с проверкой свежести ещё не измерено. ARR344693 с предоставленным BLOCK — резерв, трековое покрытие не проверено.
- **Конкуренты:** новые коммиты найдены; smart-jigsaw Arm C/v42 в проверенном PROGRESS не загружен, новый LB-выигрыш ему не приписывать. Отчёт отделяет новые сигналы от повторов журнала; нового RMSE нет.
- **Назначенные ночные расчёты завершены:** E204/E205 — FAIL, независимый replay v12075 и ASUS metadata-audit завершены. Подготовка E207 не является обучением; новый вычислительный запуск требует отдельной квитанции. Переносимый пакет к открытию остаётся отдельной задачей; обещания непрерывной ночной нагрузки нет.

Ниже сохранены исторические снимки; текущее состояние задаёт верхний блок.

## Исторический снимок 07.10 22:41Z — ночные опыты и очистка диска

- Best остаётся **v12075 = 230.9387 с**. Доска **14/237** проверена именно в22:00UTC,
  после неё новых подач нет;7октября4/5 слотов. E201 отменён, LLM выключены.
- **Mac:** независимый replay v12075 v2 №372/70aff7199731c381945c45754ee70c7e3a964aa5
  завершён22:33UTC за101.19с. Все344841 прогнозов, схема/ID/42донора и Parquetbytes
  exact, SHA90d3cd0;0 новыхfit. Оба complete.json проверены, root отдельно пересчитал SHA.
  Peak combined RSS8.49GB/7.90GiB, ниже16GiB. Прежний memory-failed partial сохранён.
  Состояние: `data/reproduce_v12075_v2_launch_20261007/`; выход:
  `data/reproduced_v12075_v2/`. Процесс закончен; повторять replay для занятости не нужно.
- **Home GPU:** E204 №369/05a1b2e завершён22:39:58UTC,6fit/48epochs,24мин02с.
  Unit `prc-e204-night-gpu-20261007.service` inactive/MainPID0/exit0, audit PASS,
  scientific screen FAIL: pair1+7 body+0.092642/decided+0.096240с, pair4+10
  body−0.203661/decided−0.203193с. Требовалось улучшение обеих пар, поэтому
  средний выигрыш−0.043281/−0.024573 не разрешает продолжение.12confirmationfit,
  ranking и E206 не запускались. [Итог](analysis/night_gpu_20261007/RESULTS.md).
  **E206 только подготовлен**, не зарегистрирован/не запущен; исходники и STATUS
  в `analysis/night_trajectory_chain_20261007/` остаются черновиком.
- **Hetz/171/175:** E205 все12fit завершены и независимо проверены, №371.
  CB сверх прежней погоды помогает−0.009903с decided, но полная поправка против
  текущего E203 ухудшает decided+0.087097с;3/6 вместо4/6, trim хуже. Гейт FAIL,
  ranking/подачи нет. Эти CPU сейчас свободны, не считать их обучающимися.
- **ASUS:** независимая metadata-проверка №373 завершена22:23:34UTC под
  PowerShell7.6.6: все2429888ID/4CB-колонки exact,5.79с/680MiB. Сейчас свободен.
- По запросу пользователя удалена21 старая таблица; реально освобождено11.3GB,
  после удаления свободно20.2GB. Проверенная SHA-копия на
  `home-pc-lan:/home/a1/prc-2026/data/mac_cold_archive_20261007/`.
  Текущие зависимости сохранены. [Квитанция](analysis/disk_cleanup_20261007/RESULTS.md).
- **Все назначенные расчёты завершены; новых очередей нет.** Далее подготовить переносимый пакет
  артефактов к открытию9октября. [Проверка зависимостей](analysis/reproduction_bundle_20261007/RESULTS.md)
  завершена: минимум5.232GB до Git/runtime; Home нужны ещё~2.729GB, источник/spec/Git
  objects и отдельная квалификация runtime/launcher. Архив не построен/не опубликован.
  Журнал№375. Новых принятых кандидатов нет;
  не обещать непрерывные8часов обучения на всех машинах.

## Оперативное обновление 07.10 22:00Z — E203 подан, новый best v12075

- **v12075 = 230.9387 с**, улучшение **−0.5051 с** против v12074. Обработан
  21:53:59.722232 UTC, все 344841 пары. Полная доска проверена 22:00 UTC:
  **14/237**, лидер 213.5212. Сегодня **4/5 слотов**, один остаётся.
- Это исходный E203 candidate SHA90d3cd0, без нового подбора весов/строк.
  Транспорт №366/3bcb89d, один PUT; серверная среда `.venv-upload`.
  Повторно upload не запускать. Журнал intent/upload и ответ доски сохранены в
  `analysis/e203_upload_20261007/`; [результат](analysis/e203_upload_20261007/RESULTS.md).
- **Независимый replay нового v12075 ещё предстоит.** Существующий
  `scripts/reproduce_best.py` закреплён на v12074: его 344841 прогнозов и байты
  уже воспроизведены точно, но это теперь исторический контроль. Следующая
  приоритетная работа — новый replay и доступный пакет артефактов к открытию9октября.
- Все назначенные PRC-очереди завершены; нового обучения/следующей подачи в очереди
  нет. ASUS доступен после короткого завершённого аудита. E201 остаётся отменённым,
  LLM не включать. Не описывать свободные машины как работающие.
- README/модель/валидация/источники/чеклист/бриф приведены к новому best; журнал №367.
  Ниже сохранены исторические снимки; их «E203 не отправлен» больше не актуально.

## Оперативное обновление 07.10 17:55Z — E203 готов, best воспроизведён

- E20224remainingfitиобаgatePASS,16:19UTC; E203currentpreE81body−0.614049/decided−0.545761с,6/6/trimPASS. Этоselectioninformed/preE81, неfullnested/неboard.
- E203fullcandidateготов16:45:17UTC, `data/e203_candidate_20261007/candidate.parquet`,SHA90d3cd0...,344841IDs/42protected проверены. **Пока неотправлен**; нуженотдельныйsinglePUTtransport/preflight+регистрацияпередподачей, прежниеOct6uploadwindowsнеиспользовать.
- Bestv12074точно воспроизведён16:47:44UTC, все344841+bytes/SHAe5c78abсовпали;11моделейE203reuse,0новыхfit. QueuePID83363,E20351200,transport33495завершились. Теперьдокументыreproductionможнообновлять,историческоесодержимое№361сохраненовgit.
- Сейчас активныхназначенныхPRCрасчётов наMac/Home/Hetz/171/175/ASUSнет. HomeGPU54%вразовомдатчике неотноситькPRC: PRCunitsуже завершены. Новогоопытавочерединет. E201неподниматьсамовольно.
- Журнал№364–365; [результат](analysis/e203_reproduction_status_20261007.md). Историческиеожиданиянижеустарели.

## Оперативное обновление 07.10 15:47Z — ASUS отработал, очередь воспроизведения запущена

- **ASUS снова в пуле и реально выполнил ADS-Baudit №360/f2fc896**: PythonPID15204,15:30:38–15:30:47UTC,9.364с,peak1.255GiB,2CPU/hard6GiB. PowerShell7; старыйrebootwatcher безпроцессов/триггеров. Terminal15:34:51: LastTaskResult0,0futuretriggers,0processes. Сейчас ASUS свободен, не утверждать, что он продолжает fit. СтаруюCatqualification не повторять.
- **Результат №362**: fixedfirstmove+crosshalflag body+20.520237с/decided+18.266188; standwalk+23.597194/+21.040133. Обаhalf/trimхуже. Идеальныйy-awareselector лишь−0.689/−0.862body дляэтихготовыхproxy. НикакогообщегоотрицанияADS-B/TGN. Полныйаудит/1932comparisons PASS, [отчёт](analysis/adsb_stage_audit_20261007/RESULTS.md).
- **E202 реальнообучается** наHomeGPU: fold2+8 и3+9 полностьюготовы(12fit), fold5+11 flatзакончен, orderedначался. Originalunit626873,24fitrecipeнеизменен. E203PID51200 ждётremaining24fit/independentproof; selected2pairCV иfullrankprep закончены. Полныйgate ещёнеизвестен. Mac E202transport33495 жив.
- **Bestreproduction №361/c62911c96d8631b99e1a8443f7dd002d153578f0**: durablecaffeinatePID83363 реально стартовал, `data/reproduce_best_queue_20261007/events.jsonl` = waiting_for_e203. ТочныйE203terminal+exit→reuseverified11modelcache/replay; приscientificFAILбезcache→одинcapture11+replay; technicalfailure→blocked/noautoretry. Independentreview+actualorphan/SIGTERMsmokePASS. Собственные4registeredfiles не менять покаqueueактивна. Реальных11fit/replay ещёнет, exactv12074 пока незаявлять. [Receipt](analysis/reproduction_execution_20261007.json).
- **Документызакоммичены9bddce9**: README/currentmodel/validation/SOURCES/openingchecklist, пятьавиационныхидей,CBmetadata иGramutility. CBпропусков2880train/452JanJul2026METAR, не число рейсов/не новыйqualityresult. ПоздниеOSMнаходкиисправляютслишкомширокоеноль. №363.
- Бриф/HTML обновлены подE201canceled/E202→E203/ASUS/reproduction; текущаяшапка разрешаетраскрытие9октябряпо плану, но **публикации/отправкисообщенийнебыло**. Сменаvisibilityраскрываетвсюtrackedhistory. Cleanclone/rawretrain+distributionartifactarchive ещё не готовы; не называтьготовыми.
- **E201 остаётсяотменённым** наMac/HomeCPU/Hetz/171;175recoveryотменён. Сохранённыемодели не удалять. LLMвыключены. СвободныеHetz/ноуты не называтьнагруженными, пустыхfitдлязанятостинезапускать.
- Bestv12074=231.4438,3/5UTCслотов прежние. Новыхподачвэтойработенет. Следующийшаг — E202terminalproof→E203gate; приPASSполныйcandidate, затемрешениеподачи. Reproductionqueue самостоятельна от этойсессии.

Ниже исторические снимки; верхнее обновление имеет приоритет.

## Оперативное обновление 07.10 15:10Z — E201 отменён, E203 запущен, ASUS возвращён

- Пользователь передал `analysis/review_fable_2026-10-07/SYNTHESIS.md`: дизайн/документация важны для приза, скрытого финала по последним сообщениям нет. Root принимает приоритет воспроизводимости/документации; статистические «полы62%» и догадки о соперниках не принимаются как доказанные факты.
- №356/5cb4d38 выполнен: Home E201 memoryfix, Hetzner E201 и171 E201 остановлены по точным InvocationID; Mac child27607 завершён SIGTERM после проверки PID/PGID/команды. Все partials/готовые модели сохранены.175 уже failed missing sklearn, восстановление отменено ДО файлов/fit. Не возобновлять E201. Receipt `data/prc_prize_priority_20261007/complete.json`.
- Старый day-controller передал владение запуском через durable `e203_not_started.json`; E202transport33495 продолжает. **E203 PID51200 старт14:59:25.379082UTC**, registration6e95e3a: localpack завершён14:59:31, selectedCV15:02:27. Оба selectedfold прошли вычисление, это не полный шестипарный gate. Homeprep unit `prc-e203-rank-prep-20261007.service`,PID663478/invocation09b3b8281c9345f689dcbb459c10de61 завершил427дней. E203 ждётremainingE202. Логи `data/e203_execution_20261007.json`, `data/e203_controller_20261007.log`.
- **E202**: все4таблицы допущены14:53:46; originalGPUunit626873 продолжает24fit, настоящий прогрессepoch проверен. Новых quality/leaderboard результатов пока нет.
- **ASUS**: пользователь15:00UTC снова разрешил расчёты на нормальной зарядке. SSHadmin@192.168.1.176, PowerShell7.6.6 по прежнему пути. Последнийboot14:55UTC; видимаяRAM15419252KiB (~14.7GiB), свободно8796176KiB, AC/chargingtrue,56%; GPU1GiB/postbootPASS. Нет работающегоPython. Catqualification_v2 оборвалась2000/5600, report/model отсутствуют; НЕ повторять. Агентtoday_e190_rank переносит зарегистрированный ADS-Baudit№360/f2fc896 portablebundle,2CPU/6GiBkernelcap/<2GiBRSS, безMLfit.
- **Код/документация**: README/SOURCES/docs/model-design.md/docs/validation.md переписаны и проходятrootreview. Новый scripts/reproduce_best.py проверил3464inputs+207sources; реальный replay ещёне выполнен. Queue послеE203 готова к регистрации (следующий свободныйномер): reuseexactE203cache→replay; отрицательныйE203gate безcache→capture11→replay; техническийсбой→blocked безautoretry. Нельзя выдавать архивную сборку за обучениевсегорецептаизсырыхданных.
- **Новые пять авиационных идей:** `analysis/aviation_feature_review_20261007.md`. TSAT нет; takeoffразрешён/полон; icing/wake/thunder ужеесть. В rawMETAR обнаружены4608CB-сводок без текущегоwx_thunder изwxcodes; грубыйrunwayflip параллельно проверен в№357 и ухудшилMSE+36с²/строку. ВыигрышCB не измерен, production не менялся.
- **Грам:** новый scripts/blend_score_identity.py проверен синтетически; для заранеефиксированнойполовины12061/12070 неокруглённыйRMSE231.608594, доска12071=231.6095. Послеокругления точное значениеиздвухscoreне определяется, публикуетсяграница. Не выбиратьвесаподоске/непрощупыватьстроки. Римские ставки не приняты.
- Best231.4438v12074 и3/5подачи прежние. Publicrelease запланирован09.10; сейчас внешнихпубликаций/сообщений нет. Прямыеопросыорганизатора и открытие оформлять отдельным действием после готовностиrelease.

Ниже исторические снимки; требование снова считать E201 было заменено пользовательским приоритетом призового ревью.

## Оперативное обновление 07.10 14:43Z — восстановление после перезагрузки Mac

Пользователь поручил снова запустить Mac и продолжать очереди без ручных подтверждений. Перезагрузился Mac около14:08UTC; причина зависания не установлена. SSH восстановлен через существующий ключ macOS Keychain. Завершённые расчёты повторять не нужно.

- **Mac снова обучает E201 v60 fold2:** guard PID27601, worker PID27607, фактический старт14:28:39UTC. Шесть потоков, nice10, `caffeinate`, контроль памяти/диска каждые10с; в14:42 около79% свободной памяти по `memory_pressure`,13.3GB свободного диска, ни одного срабатывания порога. Первый LGB seed1000 завершён. Fold1 завершился до reboot14:05UTC, повторный аудит22файлов PASS, small packet сохранён. Новый launcher `scripts/e201_mac_fold2_launch_20261007.py`; повторно не запускать. Состояние `data/e201_mac_fold2_guard_20261007`, fit log `data/e201_mac_20261007/v60_fold2_run-fold.log`.
- **Home CPU продолжает E201:** `prc-e201-donors-home-memoryfix-20261007.service`,PID627644,6CPU/16GiB/swap0/nice10. Fold3 пять LGB seeds закончены, CatBoost достиг500/5600. Исходный12thread запуск завершился OOM, его partial сохранён; рабочий output `data/e201_home_20261007_memoryfix`, регистрация№351/2c857544.
- **Hetzner продолжает E201:** SYSTEM unit `prc-e201-donors-hetzner-20261007.service`,PID3897923,6 физических ядер/16GiB/nice10. Fold5 CatBoost4000/5600. Для этого unit использовать `systemctl` без `--user`.
- **171 продолжает E201:** первый v44 fold1 завершился и прошёл original audit; очередь перешла дальше. User unit `prc-e201-donors-171-20261007.service`,6CPU/10GiB.
- **175 требует технического восстановления:** все10 базовых аэропортовых моделей v44 fold4 сохранены, но E14 завершился `ModuleNotFoundError: sklearn` в `.venv-cat-parity`. Нового fit пока нет. Владелец `prc_leader_repo_map` готовит продолжение сохранённых bases в новом output, без изменения frozen worker и действующих сред; root резервирует№356. Ошибка также остановила E201 collector33496, новый collector должен использовать отдельный путь/state. Не перезапускать старый collector вслепую.
- **E202 передача восстановлена:** postreboot controller PID33495, регистрация№354/5ef410c. Две таблицы проверены и допущены: fold3 в14:41:31UTC, fold2 в14:42:33UTC. Передаютсяfold5/6. Исходный Home GPU unit `prc-e202-trajectory-20261007.service`,PID626873/invocation30d9322da32d4cfdb55ef5c7c3d5ecbf, сохранён; в14:43 загрузка входа началась, epoch ещё не подтверждён. Не называть ожидание GPU обучением. Старые неполные файлы перенесены в `data/e202_postreboot_archive_20261007`, готовые файлы повторно не копируются.
- **Автоматическое продолжение:** day controller PID33481,№355/f9aa25c, `data/prc_day_recovery_20261007`. E203 зарегистрирован№352/6e95e3a; ВСЕ его локальные операции ждут успешного завершения Mac guard, затем fixed CV/подготовка/условная GPU и полная сборка. Два тяжёлых Mac fit не перекрываются. E203 ещё не начался. Источники действующих контроллеров не редактировать.
- **Подачи:** новых после reboot нет. v12070=231.4449,v12071=231.6095,v12074=231.4438(best);3/5UTC слотов. E19712fit PASS, E199/E200 FAIL остаются в силе. Подробно `analysis/prc_mac_reboot_20261007.md`, `analysis/e201_execution_20261007.md`, `analysis/e202_trajectory_execution_20261007.md`.

Ниже сохранены исторические оперативные снимки; их PID, статусы и ресурсные лимиты заменены обновлением выше.

## Оперативное обновление 07.10 13:10Z — длинные очереди и подтверждённый E197

- Пользователь требует автоматического продолжения полезных задач на всех доступных узлах. Не называть ожидание, завершённый процесс или заготовленный скрипт активным обучением. ASUS исключён, LLM выключены. Системное ожидание разрешений около09:43–12:42UTC действительно оставило свободные узлы; CPU-only задачи запускаются в обычной песочнице, прямые SSH/scp уже разрешены. Локальное shell-redirection вокруг SSH теряет распознавание разрешённого префикса: получать файлы через scp, не через `ssh ... > localfile`.
- **E197 завершён09:58:59Z и независимо проверен:** все12fit/32artifactSHA/ID/y/month/exactmeans/mixture/pairedmetrics/bootstrap PASS, оба исходных гейта истинны. Ordered−PLE allRMSE−4.465932/−7.113443с; ordered−равноёмкийflat−0.249545/−0.126464с. Это2selectedscreenпары LGB5/preADSB, не fullchain/доска. №349/27e6f89; `analysis/e197_verified_results_20261007.md/json`, auditSHAcae24cc29438a6d3023329a07664f653a94c40b2802737fbc98c93b7362fac62. Большие19GiBtables осталисьHome.
- **E202:** неизменныеflat/ordered на4оставшихсяпарах,24fit; worker/controller/protocol готовы, независимое ревью перед регистрацией№350. Владелецtoday_e190_rank уполномочен послеPASS зарегистрировать и запустить без нового вопроса.171готовитfold2/5,175fold3/6; после обеихCPUprep marker освобождает E201, передачи immutableтаблиц напрямуюHome продолжаются параллельно. GPU начинаетfold2послеadmission, не ждётвсе37GiB. ДоactualPID/прогресса не считатьE202обучением. `analysis/e202_trajectory_protocol_20261007.md`.
- **E201 реально обучается наMac/Home/Hetz:** №346/d664e153b3f0732b4047011cc062f58f42581a6c. НаMacqueue77126/worker77133,8threads;13:09Z уже завершеныouterLGBseed1000/1001/1002fold1. Homeunit`prc-e201-donors-home-20261007.service`,PID618590/invocationebf174d4f933451987e314d91be999bd,12CPU/16GiB. Hetzunit`prc-e201-donors-hetzner-20261007.service`,PID3897923/invocation473da44c223e4d85afb5cecdb9b4aff3,6physicalcores/16GiB. Два ноутбука ждутE202prep→своиE201. Точные фактические статусы/collector обновляетprc_leader_repo_map в`analysis/e201_execution_20261007.md`.
- **E201 назначение:** восстановитьmissingdonorOOFv44/v60 дляE81:288LGB+6Cat5600basefits+до66ADS-Bstages, recipe-faithfulnewcrossfit, не byte-identicalстарыеOOF/неfullynestedhistoricalstages. Macv60fold1→2/Home3→4/Hetz5→6;171v44fold1→2→3/1754→5→6. Nodequeue сам audit→smalltar→nextfold,48hbound/noautoretry. Homeиспользуетсуществующую`.venv-cat-oof`,175`.venv-cat-parity`,остальные`.venv`. Среды не менять, модели оставлятьнаузлах; наMac≈7GiBfree, толькомалыеverifiedpackets.
- **E199 иE200 закрыты:** E19912fitspatialscreenFAIL,24extensionнеидёт; E2006pairpreE81body−0.200152/decided+0.054472с,3/6improvedпри4required, no ranking/capture/GPU. №347/348, отчёты`analysis/e199_results_20261007.md`, `analysis/e200_mdn_results_20261007.md`. Не запускать отрицательные кандидаты, не чинить ненужнуюlateE197queueE200.
- **E203 готовится:** фиксированноеorderedreplacement вcurrentpreE81joint. Владелецtoday_e188_rank, после ревьюE202. Нужна собственнаярегистрация перед новымcomparison/fit. ТолькоE202+E203PASS разрешают3seedfulltrainsequence/exactv12074control/capture/replay. Не выдавать эту подготовку за активныйсчёт.
- **Подачи:** v12070=231.4449,v12071=231.6095,v12074=231.4438best;3/5UTC7октября,2слотаостаются. Ранг14/233 — только07:22Zsnapshot. Повторныхподачнебыло.

## Оперативное обновление 07.10 09:18Z — результаты и реальные очереди

- **Прямое указание пользователя:** все доступные машины сегодня должны переходить к следующей полезной задаче автоматически. Не считать завершённые jobs активными; не запускать дубли ради загрузки. ASUS исключён. CPU-линия E199 и MDN-линия E200 готовятся параллельно с уже работающей GPU.
- **Три подачи обработаны:** v12070=231.4449, v12071=231.6095, v12074=231.4438 (best). Использовано3/5 слотов7октябряUTC; остаются2. Место14/233 — снимок07:22Z. Квитанции: analysis/submission_results_20261007.json/.md. Повторно эти файлы не загружать.
- **Cat90 завершён и проверен:** Home44+Hetz20+Mac16+1717+1753; data/cat_extra175_20261006/merged,6×15/2085047ID/exact seedmeans/SHA/runtime. analysis/cat_completion_20261007.json. Не повторять.
- **E193 история погоды не прошла:**12/12fit, history−current body+0.015381/+0.006401с, trim хуже. analysis/e193_weather_results_20261007.md/json; №333/338. Продолжения нет.
- **E196 MDN4 прошла screen, Gaussian не прошла:**18/18fit; body−.30528/−.47118с противmean. №335/340, analysis/e196_density_results_20261007.md/json. Это компонентная проверка, не fullchain.
- **E198 завершён и независимо проверен:**24/24fit,08:59:11–09:11:42Z. Четыре новые пары MDN−mean all−0.098291/body−0.410251/decided−0.123667с; все6пар body−0.399541/decided−0.142199с. Гейт PASS противmean иPLE,5/6 decided; одна пара ухудшилась. Все2085047ID, exact3seedmeans/source/runtime/mixture/SHA; локальный evaluator byte-exact. analysis/e198_density_results_20261007.md/json и execution.json. Регистрация25ef25db84657bccefa48be7ac42c76c873520e3; inputspec37a13dbb83e852c5ef4a6bde9b84dfd9a2882affc5fc44d8c2654658602c691f. Это LGB5/preADSB, accepted_for_submission=false. Сервис завершён, не активная нагрузка.
- **E194 raw extraction проверена:**427дней/2429888DEP/104524037actualpoints; data/e194_sequence_20261007_s0,s1, auditdata/e194_verification_20261007/verification.json. analysis/e194_trajectory_execution_20261007.md. Входы, не качество модели.
- **E197 реально обучается на HomeGPU:** persistentunit prc-e197-trajectory-20261007.service, PID591693, invocation0b48f00de65a4f1daada3b2fdcd094cc. Очередь поставлена09:06:13Z, проверила окончаниеE198 и автоматически начала fit09:12:01Z. В09:18Z journal показывает flat seed0 epoch6/8. Всего12fit:2пары×2равноёмкихarms×3seeds,8epochs. Home .venv-numemb,2CPU/12GiB/swap0/nice10. Registrationbfe793c62f7cecf633c7cc31c747f48eb0676f2d, inputspec93d90131cdfa32c04b8f8c3513f3e7f7cc19c9695d8f47d3fe98b4cc8c0a407d. Prepared tables data/e197_prepared_20261007 проверены09:04:47Z; два9.2GiB файла переданы notebook→Home без копии наMac. GPU synthetic peak≈13GB. Очередь scripts/e197_wait_queue_20261007.py — точная копия Home data/e197_queue_assets_20261007/wait_then_run.py; timeline data/e197_queue_20261007.jsonl. Агент prc_leader_repo_map сопровождает. analysis/e197_trajectory_execution_20261007.md содержит реальные команды/receipts. Качество пока неизвестно.
- **Mac/171/175/Hetzner/HomeCPU в данный момент не имеют длительного fit:** старые короткие очереди завершены. E197 laptop prep занял66/57с, завершён08:51Z. Это не непрерывная загрузка. E199 пространственное состояние аэропорта проходит ограниченную проверку новизны и готовится к распределению между CPU-узлами; новых E199fits ещё нет. Владелец prc_leader_repo_map, независимый reviewer today_e188_rank, зарезервированы №344/345. После review/регистрации владелец уполномочен запускать и продолжать без нового подтверждения пользователя. Не выдавать обычные counts/E183residuals за новую spatial state модель.
- **E195 завершён:**96stagefits наMac/Home/Hetz, всеSHA проверены; apply4arms×6pairs занял115.81с наMac. Оценено2084921/2085047ID, исключены126TabPFN, changedfeasibility0. E190 decided−0.339646/body−0.366200с,6/6; E188−0.006994/−0.004913,3/6; joint−0.347029/−0.371560,6/6. analysis/e195_pre_e81_results_20261007.md и stage_execution.json; output data/e195_pre_e81_20261007_v2. №341/342/343, source repairbc19847d23d4485717bafa84744a14b49af0da19. Exactseedmean проверяется ДО перестановки IDs (C/F rounding); failedpartial сохранён. MacPID69703 уже завершён.
- **Граница E195:** послеE152/доE81, historical upstreamOOF не полностьюnested. Exactv44/v60 donorOOF отсутствуют; e38/e46scalar не заменяют полныецепочки, e16_lomo globalregularmedian не productionperrow. Не выдавать результат за fullCV; classifier rawP не изобретать.
- **E200 готовится:** today_e190_rank владеет новым фиксированным MDN replacement противE195joint доE81; затем при registered gatePASS — ranking3seeds и полная production-сборка. GPU только ПОСЛЕE197, CPU preparation/apply отдельно. №346/347 зарезервированы; fits/apply по новой регистрации ещё не запускались. Не менять веса по доске. Два свободных слота остаются для готовых полных кандидатов.
- **Лимиты:** HomeCPU12CPU/16GiB при GPU2CPU/12GiB; Hetzner6physicalcores0,2,4,6,8,10/nice10/PASSIVE;171/175 не более6CPU и существующих memorycaps; Mac8threads. Общие среды/CI/LLM не менять. Mac≈8.5GiB свободно — не копировать19GiB sequence tables.
- **Операционные ошибки:**07:29–08:43Z root read-onlyps и HomeSSH ждали системногоapproval; независимые GPU services завершились. Проверять конкретные units/logs, не общий список процессов. Systemd раскрывает$variables в ExecStart: циклы помещать в отдельный payload, не embedded loop. HomeE195 первая попытка упала доfit, literalargs v2 завершил32fit. Очереди не связывать с ожиданием разрешения посторонней команды.

## Оперативное обновление 07.10 после 07:02Z — три доставки завершены, best v12074

- Зарегистрированный транспорт №324/`5ea69b7b4ace45abd3d62f2fa15e82e3db737f82` успешно выполнил по одному PUT в порядке v12070 → v12071 → v12074. Канонический Hetzner ledger независимо прочитан: для каждого ровно `intent → uploaded`, SHA совпадает; повторов/ambiguous нет. Использовано **3/5 слотов 07.10 UTC**, осталось 2 при отсутствии других доставок. v12072/v12073 не повторялись.
- Все оценки обработаны, `usedPairs=344841`: **v12070=231.4449**, **v12071=231.6095**, **v12074=231.4438**. Новый лучший файл — `submissions/gentle-igloo_v12074.parquet`, SHA `e5c78ab061492b69fe72d4a46d2e286f7a2f6b3129ebf2ae4a34731bb662374b`; улучшение относительно v12073 на 0.5372с, относительно v12061 на 0.5444с. Это неизменённый совместный рецепт E190+E188 с полной цепочкой и половинным LSZH, зарегистрированный 6 октября до оценок.
- Доказательства: `analysis/submission_results_20261007.json` (полные ответы доски, журнал, SHA), краткий отчёт `analysis/submission_results_20261007.md`. Все три файла прошли локальный и серверный preflight, ID/finite/42donors. Код/рецепты/среды не менялись. Этот результат заменяет статус «три доставки в работе» в следующем историческом блоке.

## Оперативное обновление 07.10 06:44Z — отчёт готов, E192 завершён, три доставки в работе

- Полный подробный русский отчёт `reports/Гипотезы PRC подробный разбор.md` (~4000слов) уже передан пользователю; обновляется измеренным E192. Шесть уникальных идей +digitaltwin, causal4/5одинповтор. Главныйследующийисследовательскийприоритет — sequence/погоднаяистория; новыхfitпоэтимидеямещёнет. Не выдавать планы за запущенныерасчёты.
- E192 зарегистрирован№323/bb80c3e345727175abf24d0270b5281577aa74d4,workerSHA427b3ab0e063f421f8cc3b150c06a4030d5870d0417e63d0b868a36ba4d0dff5. Сервис `prc-e192-event-20261007.service` завершился02:10:29Z/exit0,42m16s,12fit. Macinput `data/e192_event_inputs_20261006`,output `data/e192_event_20261006`;registered evaluation+independentreconstructionPASS, обаqualityscreenFAIL. `analysis/e192_event_results_20261007.json`,№325. Не масштабировать/подавать этуконфигурацию. Реальныепараметры1856101/1856957поouterpair, flat=graph;2408917синтетические, ранееошибочнообобщенывописании.
- v12072/v12073 поданы06.10 21:18Z, обработаны~21:19Z:231.9884/231.9810. Новыйbest12073. Остальные12070/71/74полностьюготовы: SHAbe753aeb9f752359faad3e403e63a3b7e57dbad1defd5a076cb8e7571b21575b /976b0b8617db6fd5dfeb63d42b87915356eec3f72e99972f81a6c4c9172d8a11 /e5c78ab061492b69fe72d4a46d2e286f7a2f6b3129ebf2ae4a34731bb662374b. Frozenbuildercd3598eи5рецептовнеизменны, моделиcompleteE1903/E1885.
- Доставка7октября отдельно№324/5ea69b7b4ace45abd3d62f2fa15e82e3db737f82, `scripts/submission_transport_20261007.py`, SHA1f029f911bb17c9530a929da1aad10607ee323eeb1aee16e75704a1cc8a5a1ab. Агентtoday_e188_rank получилкомандуseal/transfer/preflight/PUT12070,12071,12074; конкретныеуспехиещёпроверить. ОбщийHetznerledger `data/submission_upload_ledger.jsonl`, PythonТОЛЬКО`.venv-upload/bin/python`, frozenhelpers, никакогоminioinstallвактивнуюсреду. ПриambiguousintentнеповторятьPUT.
- SSHlive06:40ZвсеCatочередизавершеныHome44,Hetzner20,Mac16,1717,1753=90. today_e190_rankсобираетновые15 и делаетregisteredoverlay87+3merge черезcat_extra175.py/05f7d2804a2e399d2d20d45423aaa0a2a17e285c сbase41129d5003aa54619259e67dd65af974860e2a1e. Ожидаемыеanalysis/cat_completion_20261007.json/.md; mergedпутьdata/cat_extra175_20261006/merged. Поканетreceiptнеутверждатьmergeготовым. ВсеCPUочередиостановилисьпослеуспеха, GPUсвободна; среды/CI/LLMнеизменны.
- ДлительныйsystemSSHapprovalпоставилrootнаожидание01:27–06:40Z; вычислениясервисовшлисамостоятельно. E192evaluationзадержалсяпослеcopyожиданиемservicejournalдо06:40Z. Отчётнеследовалоудерживатьиз-заэтого; пользователюссылкаотданапослеегонапоминания. На6октябряреально2/5, недоставленные3слотасгорели; не переписыватьэто5/5.


## Приоритет пользователя 06.10 21:01Z — подать пять сегодня, новые механизмы завтра

Denys явно отменил ожидание полного CV для сегодняшних подач: «подавай что готово», использовать пять слотов до00:00UTC. План фиксирован до новых оценок: E190, E188, по одному полному50:50бленду каждого сv12061, совместныйE190+E188. Нужны полные ranking и корректная полная цепочка; базовые проверки ID/finite/42donors/лимита/однократной доставки остаются. Старые блокировки accepted_for_submission=false не переписывать в исторических отчётах; новая регистрация фиксирует разрешение отдельно. Работают два агента: today_e190_rank (новыйrankingrunner+GPU) и today_e188_rank (новыйE64runner); root собирает/регистрирует/загружает. Фитов/подач по новой регистрации на21:01Z ещё нет. На завтра пользователь потребовал новые гипотезы с существенным потенциалом; prc_leader_repo_map исследует новые механизмы в research_notes/tomorrow_hypotheses_20261007. Состояние REVIEW_BRIEF и HTML обновляются вместе. Не останавливать текущие Cat-очереди.

## Оперативное обновление 06.10 20:56Z — остальные машины и сегодняшние слоты

- Live SSH20:51–20:52Z: Home37/44, worker≈2016%CPU; Hetzner14/20, worker≈583%CPU в исходном6CPUлимите;171 6/7 retained, последнийfit≈572%CPU;175 сноваонлайн,2/3, последнийfit≈578%CPU. Оба ноутбука продолжают без перезапуска. Прежние failedunits относятся к сохранённым историческим попыткам, текущие Catunits active/running.
- Mac16/16 complete; PRCобучения уже нет. Домашняя GPU0% на обоихDRMустройствах; новой GPUочереди нет. Ничего нового наASUS не назначать: пользователь переключил внимание на остальные машины; ранее поставленная защита/отложенный restart не менялись.
- Все опубликованные результаты скопированы Mac без attempts/models: **75/90unique independently verified**, receipt/runtime/SHA/ID/finite/overlayassignments.175qualification344419exactHome/max_abs0 независимо проверена, activation совпадает с armed171. Отчёт `analysis/cat_progress_verified_2026-10-06_2059.json`, фактическая проверка20:56:18Z. Merge ещё не выполнялся. Дальнейший merge только overlay87+3 черезcat_extra175.py, как вышеописанная регистрация№313.
- Общий serveruploadledger не содержит6октябряintent/uploaded; **0/5использованы, 5свободны, 0новыхготовыхфайлов**. Последние пять загрузок были5октября. ПриоритетE190, затемE188, условная фиксированная комбинация только после отдельнойрегистрации/полнойпроверки; места4–5 пока без кандидатов. План `analysis/submission_plan_2026-10-06.md`; сам план не являетсяпредрегистрацией/разрешениемподачи.
- Текущий темп: Home последниеfit51–53мин, Hetzner≈78мин. Без перераспределения Home ожидаемооколо02:45Z7октября, Hetznerоколо03:40Z; затем полнаяцепочка/гейты/сборки. Это оценки, не обещание; в сегодняшние5слотовE190/E188 пока не успевают. Никакого новогоfit/CV/ranking/PUT в этом обновлении не запускалось.

## Оперативное обновление 06.10 19:09Z — ASUS онлайн, 1GiB GPU сохранён, перезагрузка отложена

- SSH176 восстановлен, прежняя загрузкаWindows15:49WEST и CatPID13812 сохранены. На19:08Z записанаитерация1669/5600, процессCPU23191с; квалификацияещёнеполная. Приподключениизаряд7%, WMI PowerOnline/Charging=true, ChargeRate32064mW; затемаккумулятордостиг20%, PowerOnline=true (последнийснимокCharging=false). Питаниенеотключать. Причинапредыдущейнедоступноститочнонеустановлена.
- **GPU reserve4→1GiB уже записан через установленный OEM ASUS WMI** 19:02:34Z. Сверены модельRC71L, AC/charging, реальныеметодыAsusAtkWmi_WMNB. До DSTS/APU_MEM0x600C1 raw0x10105; одинDEVS Control_status0x102 вернулresult1; обратноеDSTSraw0x10102 подтверждено. До/послефизическивидимаяRAMнеизменилась:12273524KiB; CPUfitпродолжился. Это **pending-next-boot**, фактические14.7GiBWindowsещёнеподтверждены. Firmwareпароль/BIOSвход не использовались, драйверы/GHelper неустанавливались.
- НаASUS `data\ally_vram_change_20261006\before.json,pending.json` сдоказательствами; копияводноимённомdataкаталогеMac. Источниккодов: G-HelperAsusACPI.csSHA480b07e; егоExtra.cs явно сохраняетвыбордо отдельногопредложенияперезагрузки. ТекущийCPUfitнепрерывался.
- **Отложенный перезапуск действительно поставлен и работает:** Task `PRC-Ally-VRAM-After-Training-20261006`, ASUS\admin/S4U/Highest, pwsh7; watcherPID19480/start19:08:18Z. `scripts/ally_vram_finish.ps1`, SHAb1e5346fbd293647417aa5d716912c4ce88a0e94f576acd1c7fee2be0c72cf09. Ждёт отсутствиялюбогоpython.exe, полногоreport344419/registration9c0d087/protocolv2, SHAprediction/model, ACиbattery≥20%; копируетmanifest/invocation/report/prediction/model/fitlogвtraining_backup иперепроверяетSHA, повторноreadback1GiB, затемshutdown/r/t0 **без /f**. Прираннемсбое/нетreport/дрейфе —watch_failed, безперезагрузки/автоповтора. Deadline8h/Task9h, sleeprequestтолькожизньwatcher, releasefinally. **Не назначать ASUS новыеPythonзадачи до перезагрузки и postboot-проверки.**
- Task `PRC-Ally-VRAM-Verify-20261006` SYSTEM/Startup+30s/pwsh7 проверитGPU1GiB, visibleRAM≥14GiB, новыйBoot; пишетpostboot.json/Pass, затемотключаетсебя. TaskWatchRunning, VerifyReady, reboot_requested.jsonещёнет на19:08Z. ПослеbootскопироватьпроверенныеартефактыMac, независимооценитьexactqualification; не путатьRAMPassиCatPass. Автозапускновыхfitнепредусмотрен. 7nativeWindowsguardchecksPASS, отдельныйliveCheck=training_or_other_python_running; никакойперезагрузкивтестах.

## Оперативное обновление 06.10 18:56Z — запрос уменьшить UMA, ASUS недоступен

- Пользователь разрешил уменьшить резерв встроенной GPU для CPU-расчётов после завершения текущего задания. План: **4GiB → 1GiB**, ожидаемо Windows получит около14.7GiB. На последней успешной аппаратной проверке установлены4×4294967296bytes=16GiB, Windows12568088576bytes, registry `HardwareInformation.qwMemorySize`=4294967296bytes. Свободная RAM и установленная RAM — разные величины.
- Изменение **ещё не выполнено и не поставлено в автоматическую очередь**. На18:55Z SSH176 несколько раз timeout; ping0/2, порт22 недоступен также с171. По известномуARP MAC00:41:0e:23:fe:b1 отправлен один WoLпакет с171, повторныйSSH всёещёtimeout. Причина не установлена, состояние квалификации неизвестно; нельзя считать её завершённой или продолжающейся. Пользователю отправлена просьба разбудить ASUS/проверитьпитание/Wi-Fi/актуальныйIP.
- После восстановления: PowerShell7, проверитьtask/process/failed/guard_stop/report, сохранить результаты; не прерывать activefit. Официальная настройка ArmouryCrate → Settings → OperatingMode → GPUSettings → MemoryAssignedToGPU, требуетсяreboot (https://rog.asus.com/articles/guides/how-to-increase-the-rog-allys-vram-allocation/). ВходвBIOS для штатнойнастройки не требуется. АльтернативныйOEMAPI изучен толькопоисходнику G-Helper SHA480b07ece661b6e682939940778e6838a1f6baf5/app/AsusACPI.cs: APU_MEM0x000600C1, SetAPUMem(1) код0x102; локальныеCIMметоды/текущееустройство **непроверены**, командызаписи иперезагрузки невыполнялись. Перед любойAPIзаписью проверить реальный интерфейс/readback наASUS.

## Оперативное обновление 06.10 17:56Z — освобождена RAM ASUS

- По просьбе пользователя штатно закрыты Chrome и Steam через пользовательскую сессию; принудительное завершение браузеров не применялось. Для оставшихся фоновых Edge/PhoneExperienceHost выполнен одноразовый EmptyWorkingSet,13 процессов/0ошибок. Состояние приложений сохранено; автозапуск и системные службы не менялись. Все команды через PowerShell7; одноразовая задача закрытия приложений удалена после выполнения.
- На17:56Z доступно3919312KiB (~3.74GiB), в момент перед trim3292272KiB; это изменяющийся снимок, не обещанный постоянный объём. CatPID13812 продолжает расчёт: CPU6261с, итерация464/5600 записана, RSS~1.16GiB. Python/SSH/защита от сна не тронуты. Отчёт `analysis/ally_ram_cleanup_20261006.json`, оригинал наASUS `data\ram_cleanup_20261006_result.json`.

## Оперативное обновление 06.10 17:38Z — ASUS176 подключён и обучает квалификацию

- **ASUS ROG Ally `admin@192.168.1.176` доступен по SSH-ключу Mac.** Ryzen Z1 Extreme 8c/16t, физически16GiB RAM, Windows видит11.7GiB; около146GiB свободного диска при подключении. Windows11Home, AMD Radeon Graphics. BIOS и пароль учётной записи не менялись.
- **Пользователь потребовал PowerShell7 для дальнейшей работы.** Установлен официальный7.6.6 ZIP, SHA проверен против GitHub release; `C:\Users\admin\prc-2026\runtime\PowerShell-7.6.6\pwsh.exe`. Создан ярлык «PowerShell7» в меню Пуск. Вызывать именно pwsh с `-NoProfile -EncodedCommand` (UTF16LE); remote executable с обратными слешами в едином quoted SSH argument. Отдельно установлен подписанный Python3.14.4, SHA+AuthenticodePSF PASS; `.venv-prc\Scripts\python.exe`. Старый системный Python3.11 сохранён. Cat1.2.10/LGB4.7.0/NumPy2.5.3/Polars1.44.2/Pandas3.0.5/PyArrow25.0.1, imports/pipcheck PASS.
- **Сейчас обучается только техническая квалификация №317**, registration `9c0d087a0dfc96bbb57e7a343ad984418b3c08a6`: Task `PRC-Ally-Cat-Qualification-20261006-v2`, ASUS\admin/S4U/Limited, PowerShell7, без пароля и автоповтора. На17:37Z Running, PythonPID13812, 1622159trainrows, seed0/fold1+7/5600iter, первая итерация выполнена.6threads/affinity0,2,4,6,8,10/BelowNormal; Windows Job committed-memory cap6GiB; RSS3.17GiB/private4.20GiB/свободно~0.91GiB. На первой итерации ETA~3ч, устойчивый темп ещё не измерен. AC-only, timeout6h; временная защита сна привязана к процессу, после выхода освобождается. Guard прекращает процесс при отключенииAC/availableRAM<256MiB; failed/guard_stop сохранять, не перезапускать автоматически.
- Пути наASUS: root `C:\Users\admin\prc-2026`; inputs `data\cat_laptop_inputs_20261005`; output `data\cat_ally_20261006_v2`; логи `data\cat_ally_launcher_20261006_v2.log` и `...\fit.log`; definition `data\cat_ally_task_20261006_v2.xml`. Worker `scripts\cat_ally_windows.py`, SHA `d1639e072b395ba1d41fd49fc54103c466519a2417f727ebcd76dd181221bfa4`; launcher.ps1 SHA `825b371c88bd455e36e5d3e695fc9d44d269ebf5778f06e9f68160f26916eb0b`.
- **PASS ещё нет.** Требуются344419ID/finite/полное np.array_equal сHome/max_abs0. Ни одним из90fit ASUS не владеет, merge/submission запрещены этим протоколом; послеPASS отдельная полезная очередь с регистрацией. Исходные helpers/bundles/назначения остальных узлов не менялись.
- Первая попытка№315 (`531905f`, output `data/cat_ally_20261006`) завершилась доfit ArrowMemoryError: копияfloat-матрицы1557272640bytes не поместилась в6GiB. Failed task отключён; failed.json/invocation/fit.log и исходники сохранены, диагностические файлы скопированыMac. В№317 единственное изменение — `to_pandas(split_blocks=True)`. Red/green memory-sharing regression +7testsPASS; nativeWindows samplePASS; наMac полные1622159train+344419rank×170columns точное pd.assert_frame_equalPASS, `analysis/ally_pandas_layout_20261006.json`. Порог квалификации не ослаблялся.
- Mac завершил свои16/16fit по логу очереди; последние2receipt ещё требуют независимой проверки. НаMac выполнена полная проверка pandas-layout; нового долгого обучения после Cat не назначено. Hetzner на17:37Z active/MainPID3372713,12/20готовы, считает13-й fit.175 по-прежнему не отвечал при последней проверке этого хода; Home/171 заново не проверялись. Новых сабмитов нет.

## Оперативное обновление 06.10 16:08Z — сбор результатов и доступ к Ally

- На Mac скопированы опубликованные результаты Home, Hetzner и 171. Независимая проверка `Q.verify`, SHA metadata/prediction, ID/finite, назначений, runtime и уникальности `(seed, fold)` прошла для **60 fit**: Home31/44, Hetzner11/20, Mac14/16, 1714/10 исходных назначений. Это промежуточный сбор, не полный merge и не новый score. Mac начал свой15-й fit. При активном overlay175 ноутбук171 сохраняет7, а не10 заданий.
- **175 недоступен по SSH** с Mac и171 на проверках15:39–16:07Z; `ip neigh` на171 показывал FAILED. Причина (сон/питание/сеть) не установлена. Нельзя утверждать, что его обучение продолжает работать. Watchdog `prc-cat-extra175-drain-20261006` на171 уже active, `armed.json` создан14:34Z; передача последних3 заданий активирована. Сам qualification report/результаты175 ещё не получены, поэтому exactPASS независимо наMac не перепроверен. Не откатывать назначения и не запускать дубликаты до восстановления связи и проверки артефактов175.
- **Шестое устройство — ASUS ROG Ally, Windows, `admin@192.168.1.176`.** `ASUS\admin` — вывод whoami, а не логин/пароль. По словам пользователя, уWindows-аккаунта нет пароля; забытый BIOS-пароль ограничивает только вход вBIOS, ОС загружается. OpenSSH_for_Windows9.5 отвечает, авторизация ключомMac пока отклоняется. Пользователю передан блокPowerShell для добавления существующего публичного `id_ed25519.pub` в `C:\ProgramData\ssh\administrators_authorized_keys` с ACL Administrators/SYSTEM. Ждём выполнения/диагностики; инвентаризация, установка среды и обучение наAlly **не выполнены**. Не менять BIOS/Windows-пароль и не ослаблять политику пустых паролей.
- Home/Hetzner/171 подтверждены active в15:39Z, Mac продолжает писать COMPLETE/START. Новых обучений, сборок и подач в этом обновлении нет. Нужен повторный live-check перед заявлениями о текущей загрузке.

## Оперативное обновление 06.10 12:03Z — пятый узел175 подключён

- **a1@192.168.1.175**, hostname `a1-G1619-01`, Ubuntu26.04.1, i5-1035G7/4ядра8потоков, около14GiB RAM,433GiB диска приподключении. `/home/a1/prc-2026`, отдельная `.venv-cat-parity` Python3.14.4. Cat1.2.10/LGB4.7.0/NumPy2.5.3/Polars1.44.2/Pandas3.0.5/PyArrow25.0.1; imports/pipcheckPASS. Linger=yes, входMac поSSHключу, пароль не сохранён.
- **Уже обучает полный технический повтор:** userunit `prc-cat-extra175-20261006.service`, PID14885,6threads/CPU600%/Memory12GiB/swap0/nice10. На12:03Z1500/5600iter, текущаяRAM4,69GiB. Log `data/cat_extra175_20261006.log`; детальный `data/cat_extra175_20261006/qualification.log`. Рабочийoutput `data/cat_extra175_20261006`. PASS ещёнеизвестен, требуются все344419предсказания/max_abs0, не промежуточнаяtrain-loss.
- **Регистрация №313:** `05f7d2804a2e399d2d20d45423aaa0a2a17e285c`, новыйкод `scripts/cat_extra175.py`, overlay `data/cat_extra175_inputs_20261006`, lineage `78ae852b4b0fccec580fda8c724ff476e8f7405add1af0cf73f6936c299a2b78`. 31CatcontracttestsPASS, read-onlyreview; исправлены атомарнаяпубликацияarmed иранняяпроверкаchildlog. Базоваярегистрация41129d5, Q/C/L/helpers/4узловыевходы не изменены.
- После **exactPASS**175 автоматически отдаёт171 ограниченнуюSSHкомандуактивации. Проверяетвсеretainedreceipts/отсутствиеattempts+логов+готовыхtailjobs/active171. Watchdoguserunit `prc-cat-extra175-drain-20261006.service` на171 ждёт конкретныепервые7из10, затемостанавливаетвесьстарый171unit/MainPID0, сохраняетpartial иSHA7receipts. Егоoutput `data/cat_extra175_drain_20261006` на171. Только послеarmedподтверждения175учит фиксированные `(seed14,fold4),(seed14,fold5),(seed14,fold6)`. ДоPASS/активации171сохраняетвсе10 задач; неостанавливатьего заранее. Read-onlySSHpreflight успешно выполнен, активации ещёнет.
- Доступ175→171 ограничен ключом `/home/a1/.ssh/prc_transfer175_20261006`, `restrict/from=192.168.1.175/forcedcommand`; pin171 в`/home/a1/.ssh/prc_transfer175_known_hosts`. Ключнедаётshell/portforwarding. Диагностика `ssh ... preflight` делает только проверку ивернётошибкупослеактивациипоfresh-workправилу. Не запускатькомандубезpreflightвручнуюдоквалификации.
- Послеактивации **финальноеобъединение через cat_extra175.py merge**, строго87оригинальных+3новых/90unique/6×15; basework —скопированное`data/cat_night_20261006`, work —175output, drainwork —скопированное171`data/cat_extra175_drain_20261006`, snapshot/bundleкакраньше, overlayновый. ОбычныйQ.mergeпослеtransferнеподходит: онждётвсе10от171. Если175FAILдоактивации, исходныйQ.merge остаётсядействующим. Ниfallbackпоscore, ниauto-resume; приSSHответеlostпослеактивациисначалапроверитьwatchdog/receipts.
- Сон175 блокируетrootunit `prc-laptop175-awake-20261006` сRuntimeMaxSec2d. Попытка заменитьегонаinhibitorпоPIDнепрошлаsudo-timeout; job-boundunitнебылсоздан. **Послеокончания175очередиостановитьэтотrootunit**, показащитаотснаактивна.
- MacквалификациязавершиласьточнымPASS, max_abs0/outputSHA008f3a29f387bc02a5eafe6e0eb6138521dd9754a0b91b9f757fbba92bac779e,16-fitочередьпродолжается. Снимок11:24Z:Home26/44,Hetzner7/20,Mac6/16,1711/10; всевработе. Новыхподачнет. Второйдополнительныйноутбукещёнедан.

## Оперативное обновление 06.10 07:13Z — Cat90 распределён на четыре машины

Регистрация №310: `41129d5003aa54619259e67dd65af974860e2a1e`, код `scripts/cat_night_queue.py`, execution lineage `6cd125245ccb67db4d4add728aa4c7ada68c7bb1e3a8a1b1d17e4281082acfa7`. Общий bundle `data/cat_night_inputs_20261006`, отдельные выходы `data/cat_night_20261006/{home,hetzner,mac,laptop}`. 90 уникальных fit, без сокращения 15 сидов. 26 contract tests PASS, read-only review: непосредственных блокеров запуска нет.

| Узел | Проверенное состояние | Очередь и ограничения |
|---|---|---|
| Home | **21/44 опубликованы**, идёт seed3/fold4+10. Старый service остановлен после 15-го полного fit, MainPID0; 15 результатов проверены и импортированы, partial сохранён. | user `prc-cat-night-home-20261006`, `.venv-cat-oof`, 24threads/24GiB/swap0/nice10. Snapshot остаётся `data/cat_fin1b_oof_20261005`. Log `data/cat_night_home_20261006.log`. Старый awake service активен. |
| Hetzner | **4/20 опубликованы**, идёт seed8/fold1+7. Точная квалификация №309 пройдена, все344419 прогнозов и NPZ SHA совпали с Home. | system `prc-cat-night-hetzner-20261006`, `.venv`, affinity0,2,4,6,8,10,6threads/26GiB/swap0/nice10. Log `data/cat_night_hetzner_20261006.log`. Это шесть физических ядер сервера; production сохранён. |
| Mac | Полная квалификация seed0/fold1+7 **обучается, пройдена итерация1000/5600**; после absolute max≤1e−6 автоматически16 назначенных fit. Ещё не PASS. | `.venv/bin/python`, 14threads/nice10/caffeinate, PID caffeinate97790. Log `data/cat_night_mac_20261006_authorized.log`, fit log `data/cat_night_20261006/mac/qualification/fit.log`. Подготовка всех6Pools завершена до этого запуска. |
| Ноутбук171 | Квалификация №308 **точный PASS**, max_abs0/344419;7812,67с. Уже обучает **первый из10**, seed13/fold3+9. | user `prc-cat-night-laptop-20261006`, `.venv-cat-parity`,6threads/12GiB/swap0/nice10. Log `data/cat_night_laptop_20261006.log`; per-fit `.../laptop/fold_3_9-s13.log`. Root awake unit `prc-cat-night-laptop-awake-20261006`, release unit автоматически снимет запрет сна после завершения очереди; cap2d. CI сохранён. |

- Home и Hetzner стартовали около01:35–01:39Z и продолжали без сессии. Запуск Mac застрял на системном разрешении nice с01:39 до около07:04Z; первый sandbox-run завершился до создания рабочего каталога. Failed log/receipt сохранены. Ноутбучная передача входов завершилась, но зависимый запуск был отложен этим ожиданием и выполнен только07:06Z; не утверждать, что все четыре считали всю ночь.
- **25 опубликованных результатов** Home+Hetzner уже скопированы на Mac без attempts и независимо проверены по назначению/receipt/SHA/ID/finite. Это промежуточные копии, `complete.json` всех узлов ещё нет; merge не выполнен. Laptop qualification тоже скопирована и независимо сверена полностью с Home: NPZ SHA `008f3a29f387bc02a5eafe6e0eb6138521dd9754a0b91b9f757fbba92bac779e`.
- У Mac квалификация с абсолютным допуском, не bitwise-обещание для всех фолдов. Квалификационные повторы не входят в90 и не заменяют Home seed0. При failure остановиться и сохранить артефакты; **автоматического resume нет**. Не перезапускать runner в existing work. Новую восстановительную процедуру зарегистрировать отдельно.
- Следующий шаг: проверить Mac qualification, затем собирать receipts. После complete всех4 узлов: `scripts/cat_night_queue.py merge` с тем же registration, snapshot и bundle; независимо проверить все6×15 средних/ID/y/month/SHA. Только затем полный текущий v12061-chain контроль/кандидаты E188/E190, без подбора weights. ПодготовленныеPool-файлы и оригинальные6fold inputs уже есть на ноутбуке.
- **GPU нового задания не имеет**, E19018fit завершены. LLM остаётся отменён. Два дополнительных ноутбука пользователь отложил до дня; сейчас их не подключали. Новых подач06.10 в этом ходе нет.

## Оперативное обновление 05.10 23:48Z — E190 screen PASS, квалификация CPU-работы на других машинах

- **E190 №299/306 завершён:** все18 GPUfit,6/6пар,2085047ID проверены исходным evaluator. Решающий показатель гейта −0,4851316с, body−0,4528099,trim−0,4561267; ниже во всех6парах. Это фиксированная нормировка strict_oof_gate на matched LGB5-only/pre-ADS-B контроле; **не полныйv12061OOF и не измеренный выигрыш доски**. Отчёт `analysis/e190_screen_2026-10-05.json`, полные входы/выходы `data/e190_inputs_20261005` и `data/e190_numemb_20261005`. Следующий шаг — полная текущая цепочка с Cat15 и неизменным хвостом/LSZH. Не подбирать веса и не выдавать screen за разрешение на подачу. GPU после завершения свободна.
- **Cat Home продолжает исходные90fit:** последнее наблюдение12/90, около68ч по последнему fit. На Hetzner запущен **один** полный ресурсный повтор fold1+7/seed0/5600iter: `prc-cat-runtime-parity-20261005`, CPU0,2,4,6,8,10/6threads/nice10/RAM26GiB/swap0. Уже идёт обучение. Код `scripts/cat_runtime_parity.py`, регистрация `c982056fbe7ac50cd347cd79c7310ac12e2acd8f`; данные `data/cat_parity_snapshot_20261005`, результат `data/cat_runtime_parity_20261005`. PASS только точное совпадение344419reg сHome, max_abs=0. Даже PASS не разрешает вставить результаты в активныйHome manifest: нужны отдельные эксклюзивные назначения `(fold,seed)` и merge-provenance. Рецепт15сидов не сокращать.
- **Ноутбук:** E191 supportaudit завершён, все2085047ID/SHA verified;187дней имеют stale,177дней безфлага отмечены unknown,31.12 безсенсоров. Строгих наблюдений223. Ограниченное восстановление6дней №303/307 завершено:4дня совпали состарыми точками и дали13наблюдений,01.09/01.12 отличаются; production surfaces не заменены. Расширять на177дней или обучать auxiliary model по этому результату не надо. Отчёты `analysis/e191_support_verification_2026-10-05.json`, `analysis/e191_recovery_verification_2026-10-05.json`; файлы наMac. Failure первогоrun и exit1 восстановления сохранены, причины документированы №301/307.
- **Для ноутбука зарегистрирован №308:** `scripts/cat_laptop_pools.py`, commit `55304792ee0bc0e39868c9138904723522f5fee9`. Mac готовит прежние target/proxy/170features через исходный frozen trainer preprocessing; ноутбук обучает один seed0/5600iter с поочерёдным созданием Pool и освобождением DataFrame. CPU600%/RAM12GiB/swap0/nice10/runtime6h; отдельная `.venv-cat-parity` сreadonlybase,Cat1.2.10/sixpackages/pipcheckPASS. Вход `data/cat_laptop_inputs_20261005`, output `data/cat_laptop_parity_20261005`. Подготовлены1622159fitстрок/344419rank/170колонок, точныйroundtripPASS. **Запуск проверен:** `prc-cat-laptop-parity-20261005` active/running, лог `data/cat_laptop_parity_20261005.log` содержит TRAINING иитерацию0; CPU около504%, memory4,26GiB/peak5,98GiB на раннем снимке. UnitCPU600%/12GiB/swap0/nice10/runtime6h; отдельный `prc-cat-laptop-inhibit-20261005` active блокируетсон до завершения. Пока это первыйfit, а не завершённаяквалификация памяти/совпадения. После PASS №304/308 разработать и зарегистрировать эксклюзивное распределение оставшихся(fold,seed), сохраняя15сидов; до этого исходнуюочередьHomeне трогать. Точное совпадение344419прогнозов и реальный пикRAM — условия квалификации, автоматического merge нет.
- **Maccollector №305 не сработал:** sandbox-попытка оставила каталогlock, повтор завершился на нём. Активногоcollector нет; E190 скопирован явнымrsync и проверен напрямую. `data/.e190_collector_20261005.lock` и лог сохранены; этотодноразовыйскрипт не перезапускать без отдельного решения. Лучшийфайл/05.10пятьподач не менялись, новыхsubmissionнет.

## Оперативное обновление 05.10 21:32Z — LGB30 завершён и проверен

- Hetzner завершил все 30 fit / 6 фолдов, service `prc-lgb-fin1b-oof-20261005` inactive, Result=success. Готовые компоненты скопированы в `data/lgb_fin1b_oof_20261005` на Mac. Независимо проверены manifest/source/runtime/production-rank SHA, все 2085047 уникальных ID, y, месяцы, конечность и точное среднее пяти сидов. Отчёт `analysis/lgb_component_completion_2026-10-05.json`. Большие fold-inputs и fit-логи остались на Hetzner. Это ретроспективный компонент контроля, не новая модель для подачи.
- Home Cat продолжает исходную очередь: 10/90 fit на 21:29Z, MemoryCurrent около 18 GiB; темп последнего fit означает около 70 часов остатка. Не уменьшать число сидов по промежуточным результатам. Точного полного OOF текущей цепочки пока нет. Ожидание Cat не является причиной блокировать независимые эксперименты или проверку доступных последующих этапов; их результаты нужно явно отличать от полной проверки.
- Масштаб E188 уточнён без нового обучения: natural ΔSSE=−14318041,275685 на 35007 строках; при нормировке на все 2085047 training-строк это ΔMSE=−6,867011. При условном глобальном RMSE 250 и неизменности остальных строк эквивалент составляет −0,013734 с. Это иллюстрация масштаба screen, **не измеренный current-chain выигрыш, не прогноз ranking и не потолок метода**. Synthetic стресс сюда не складывается.
- Нового ranking/подачи/качества на доске нет. После завершения LGB сервер свободен от этого PRC-задания; Mac, GPU и ноутбук нового обучения не ведут. Новая задача автоматически не поставлена. Сегодняшние 5/5 слотов уже использованы.

## Оперативное обновление 05.10 19:28Z — E188 screen пройден, E189 отклонён

- **E188 завершён по №293/296/297:** natural LIRFNM35007строк ΔRMSE−0,606867с,4/6пар,body−0,408053,trim−0,463909; synthetic60327строк −2,772305с,6/6,body−2,296020,trim−2,546534. Исходные обе ногиscreenPASS. Это локальные показатели этапа на прежнемtraining-parent`nomonth`, **не полныйv12061OOF/глобальныйscore/принятыйкподaче файл**. Новогоrankingнет. Следующий шаг — полнаяcurrent-chainпроверка; недостающиеCat15/LGB5 ещёсчитаются. Не повторятьпройденныйscreen и не менятьweightпоегоитогам.
- Все2085047ID сохранены;364наблюдаемыхдня2025 original8vi parityPASS. Upstream31декабря отсутствует на всехмашинах:4715рейсов остаются бездатчиков/безE64поправки. Первыйrun остановилсядоfit наFeb28: два рейса находятсявсоседнихraw-файлах. ИсправленаразбивкапоactualUTCmonth; ID185722540/196701297 сохранены. Failedrunсохранён, строгиеворота неослаблены.
- Регистрацияv2`da89a73846aa675d591ba5121bd2391241652a80`; input`data/visit_mask_inputs_20261005_v2`, output`data/visit_mask_20261005_v2` наMac/ноутбуке. ВсеSHA/ID/y/month/finite/perfoldtrim проверены после копирования. resultSHA`02b53e496d06ad0aff6842408dfa392c9b1c82754a0152e0f9a86bdc484f5c78`. Отчёт`analysis/e188_groundmask_screen_2026-10-05.md`. 60fitза134,78с; ноутбучныйservice`prc-visit-mask-v2-20261005` иsleepinhibitor успешно завершены, ноутбук свободен.
- **E189 TabICLv2 отклонён** по№294/295: две парыLTFMNM +1,4338/+3,4396с;8192строки вместе +2,5382с,trim+2,3450. Это локальные секундывыборки, не глобальнаядоска. 18,39сGPU, maxallocated2,794GiB. ВсеартефактыскопированыMac, SHA/ID/y/controlverified. Отчёт`analysis/tabicl_ltfm_screen_2026-10-05.md`. ПолныйCV/ranking/weightrescueдляэтогопротоколанезапускать. Иные масштабы/fine-tuning непроверялись. GPU свободна; `.venv-tabicl`изолирована, checkpointSHAзакреплён.
- **Ресурсы на19:28Z:** HomeCat90 активен,7/90готовы; HetznerLGB30 активен,20/30готовы, предварительноосталось1,54ч. MLP18 завершёнранее. MacиGPUновыйсчётневедут; ноутбукE188закончил. Текущийлучшийфайл и5использованныхслотов05.10 неизменны, новыхподачнет.
- Вжурнале№293–296 поправлены только редакционныеUTC-времена поgitcommit:19:08/19:12/19:15. Commit-before-fitпорядок и научныепротоколы не менялись. HTML-бриф пересобран черезсуществующийpure-PythonMarkdown renderer безизменениярабочей`.venv`.

## Оперативное обновление 05.10 17:18Z — v12061 лучший, GPU OOF и ноутбучный аудит завершены

- **Новый лучший — v12061 = 231,9882 с**: заранее фиксированный полный бленд **50:50 v12048 + v12060**, где v12060 — родитель v12048 + E159b/c только LSZH. Все 42 значения TabPFN сохранены. Это **−0,2529 с к v12048**, **−0,1118 к v12060** и **−0,6318 к v1139**. Обработка v12061 — 16:42:50.78501Z; снимок `analysis/board_snapshot_20261005_after_lszh.json` на 16:51:22Z: **13/230**, третье место 217,2782 с, разрыв **14,7100 с**. Прогноз v12061 −0,05 [−0,20; +0,10] оказался **лучше коридора**, а не попал в него. v12060 = 232,1000 (−0,1411 к родителю), его прогноз −0,10 [−0,30; +0,15] попал в коридор. Оба файла и веса были зафиксированы до первой новой оценки; отдельного принятого CV-выигрыша для 50:50 нет, перенос теперь подтверждён доской. **Все 5/5 слотов использованы и обработаны:** ledger содержит 5 intent + 5 uploaded, неоднозначных исходов нет.
- **Транспорт пары — только новый `src/e159_fixed_single_put_upload.py`, №286.** Неиспользуемый при загрузке OPDI-исходник на сервере отличается от Mac, поэтому прежний preflight остановился до PUT. Новый транспорт проверяет неизменённый receipt `2958dae1ed7e8cab6ae5aa31463463a129443bf9b808ec4d6b81148a91e326f8`, старый builder и полный inventory внутри receipt, реально используемые runtime-модули и оба файла. Сам рецепт, готовые прогнозы и старые серверные исходники не менялись. Общий серверный ledger/lock, одна попытка PUT без retry, 344841 ID/schema/finite/LSZH/LOBT/42 донора и точный бленд остаются обязательными. Итоги и времена обработанных файлов — `analysis/submission_status_2026-10-05.md`; сегодняшние номера повторно не загружать.
- **Home: Cat15 FIN1b OOF активен, 5/90 фиксированных fits готовы** (№282). Это 6 пар месяцев × 15 сидов; полное среднее только после 15/15 для каждого фолда. Предварительный ETA порядка **73 ч**, не обещанный срок; не уменьшать число сидов по промежуточным результатам. Рецепт и отдельную `.venv-cat-oof` сохранять. Это ретроспективная проверка компонента, не новый кандидат доски и не полный OOF v12048.
- **Hetzner: LGB5 FIN1b OOF активен, готов первый фолд (5/30 fits), считается второй** (№283): 30 фиксированных fits, 2319 раундов, шесть реальных потоков на CPU0,2,4,6,8,10, nice10, OpenMP PASSIVE. Второй LightGBM CV рядом не запускать. Сервис `prc-lgb-fin1b-oof-20261005`, лог `data/lgb_fin1b_oof_20261005.log`. Это недостающий производственный компонент OOF, а не новая модель для подачи. По первому фолду2791,8с, предварительно ещё3,88ч.
- **GPU MLP завершён по №287/292**: все6пар×3сида,2085047уникальныхID/y, единыйROCm backend, SHA/finite проверены наMac. МаксимальныйworkerRSS3,6662GiB, каждыйGPUfold~3мин. Компоненты `data/mlp_split_20261005/outputs`, отчёт `analysis/mlp_component_completion_2026-10-05.json`. Macкоординатор завершён, GPUсвободен; это ретроспективныйкомпонент, не новыйкандидат и не полныйOOFлучшегофайла.
- **Ноутбук подключён и выполнил первый расчёт**: `a1@192.168.1.171`, i3-1215U/14GiB, `.venv` Python3.14.4, Linger=yes, CI сохранён. Аудит №289/290/291:124дня/125049648точек/689260ID,122,516с,peak996,2MiB, все контрольные8vi и SHA совпали. Завершённый результат `data/visit_groundmask_audit_20261005_v2`, старыйfailedrunсохранён. После завершения нового фонового PRC-задания на ноутбуке ещё нет.
- **Диагностика №281 завершена**, отчёт `analysis/current_error_mechanisms_2026-10-05.md`. В pre-ADS-B OOF членов E182+E185 LTFM с NM имеет широкую ошибку обычного регрессора: при y≤7200 RMSE 216,32 с, принудительный regular даёт 216,03 с. Для LIRF с NM в июле отключение режимов вредит: 417,04 → 496,44 с. На ranking из 46862 LTFM-NM строк 46320 не имеют ни своего ground-трека, ни предыдущего визита; у LIRF-NM-July предыдущий визит есть у 11359/15562. Это наличие входов и диагностика, не измеренный выигрыш сверх E64 и не точный OOF текущего полного файла.
- **Ground-mask удаление: механическая проверка пройдена**, качество ещё не измерено. LIRFNMJuly2025 даёт12298 stable-identity синтетических строк с изменёнными действующимиvi; Jan2025 лишь28. July2026 реальнаямаскаE64 сохраняется10528→10521. Число11359выше — все previousvisitsвключая831строкуcoverage≥2, не маскаE64. Следующий шаг требует отдельно подготовить остальные месяцы2025 и зарегистрировать train-only аугментацию/полныйOOF; такойfitещёнезапущен. Отчёт `analysis/visit_groundmask_verdict_2026-10-05.md`. OPDIoccupancy, E184/E186 не возвращать.


## Предыдущий снимок Codex 05.10 10:17Z

- **Новый лучший v12048=232,2411с**, E182+E185, −0,1175к12008 и −0,3789к1139; место12/230, до третьего14,6756с на10:13Z. Отдельный E182v12028=232,5004 (−0,1196к1139). Оба исходных прогноза попали в коридор. За5октября **3/5слотов**, все три доставки подтверждены одним PUT/intent/uploaded; неоднозначных нет. Отчёт `analysis/submission_status_2026-10-05.md`, снимок `analysis/board_snapshot_20261005_after_e182.json`, журнал№277/278.
- **Обе сборки завершены и выгружены**: bare12028 в09:58:32Z, combo12048 в10:09:31Z. Их final+immutable manifest наMac иHetzner; все344841ID/schema/finite/42TabPFN проверены. Комбинация зарегистрирована№273 до первой подачиE182, weights не менялись. Итог12048SHA `0c72511bf5be05ecb9d68a72245e244feeb6088b92372732bb3aae0dc7dddbb1`. Все пять FIN1b-ranking членов5/5сидов наMac/Hetzner.
- **E184 завершён и отклонён** по исходному №254: decided−0,215059,5/6, body−1,558346, **trimmed+0,162186**. Аудит всех6фолдов/2085047ID/y/month и SHA PASS; ranking/подачи нет. Журнал№280, отчёт `analysis/e184_verdict_2026-10-05.md`. Сервис `prc-e184-cv-6t-20261005.service` закончил работу; snapshot и старый отдельныйfold1_7 сохранены. Hetzner свободен для следующей зарегистрированной задачи при прежнем ограничении6потоков/nice10.
- **OPDI: безметочный аудит завершён**, №279.44/45окон/17FLIGHT;64с/RSS2670MiB подCPU2/MemoryMax3G. Ranking266011 уникальных MVT-связей; при известномADSBcoverage<2 безsource-gap —22025runway,4971taxiway,1814parking. Это согласованные полигонные интервалы, не подтверждённое наземное движение; только20runway-MVT также имеют целуюпару до публичногоtake-off. Исходный occupancy-residual proposal не запускать. Отчёт `analysis/opdi_support_verdict_2026-10-05.md`, полныйJSON `analysis/opdi_support_gate_20261005.json`, артефакт `data/opdi_matched_support_20261005.parquet` (SHA/2429888уникальныхID проверены). Наhome output `/tmp/opdi_support_gate_20261005_v2`, service `prc-opdi-support-20261005-v2.service` завершёнуспешно. Первый schema-failure сохранён; исправление№276/`03a146b` меняло только временные projections. Цели/OOF/модели не читались. Домашний ranking и OPDI готовы, новой принятой задачиGPU пока нет.

- E186 отклонён по полномуCV; GPU доступен, LLM остановлены. Mac ranking завершён; свободнооколо3,9GiB, полнуюOPDI сюда не копировать. Оставшиеся два слота — для прошедших ворота кандидатов, следующий приоритет — диагностика текущих ошибок и недостающие OOF-компоненты. Оценки новых данных пока отсутствуют.

## Предыдущий снимок Codex 05.10 09:13Z

- **Лучший v12008 = 232,3586 с**, улучшение на 0,2614 с к v1139. Фактическая загрузка 06:56:28Z, обработка 06:57:03Z; один из пяти слотов 5 октября использован. Последняя проверка позиции в 07:01Z: 12/230, разрыв до третьего места 14,7931 с. Отчёт: `analysis/submission_status_2026-10-05.md`.
- **E182 ranking: готовы 4/5 членов**, каждый с пятью сидами: rawf, lobtf, eobtf, aobtf. Файлы уже на Mac и Hetzner; домашний schedf имеет 2/5 сидов и продолжает работу на 24 потоках. Mac закончил свою очередь в 09:00:20Z. Последний файл забирать только после атомарной публикации final, затем сверить SHA и provenance. Подготовка сборщика E182 v12028 идёт отдельно; сборка и загрузка пока не запускались.
- **E184 успешно перезапущен около 09:00Z** по №268/269. Сервис Hetzner `prc-e184-cv-6t-20261005.service`, новый OOF `data/cv_preds_e184_hz_6t`, snapshot `data/e184_resource_20261005`, лог `data/e184_resource_20261005.log`. Только шесть реальных потоков на CPU0,2,4,6,8,10, OpenMP PASSIVE, nice10. После исправления perf показал 96,6–96,8% циклов в LightGBM и 0,18% в libgomp; виден прогресс обучения. Повторно не перезапускать. Старый fold_1_7 и логи сохранены отдельно, не смешивать с новым запуском. После 6/6 нужен strict audit и исходный гейт №254; автоматической загрузки E184 нет.
- **OPDI: 44/45 архивных окон**, 115 713 336 EVENT и 127 567 332 MEASUREMENT. Точная декабрьская перекодировка проверена и применена: +2 572 369 EVENT и +2 684 670 MEASUREMENT, 51,62 с, peak RSS около 1,125 GiB. Единственный отсутствующий архив EVENT — 20251002_20251012 (HTTP 404); также сохраняются исходные пробелы FLIGHT внутри января 2026. Набор остаётся incomplete. `analysis/opdi_full_inventory_2026-10-05.json` — более ранний снимок 43/45, текущий авторитетный manifest на home-pc: `data/opdi_full_v002/manifest.json`. Идёт профиль новых наблюдений без меток.
- **E185 поверх E182:** подготовлена отдельная фиксированная проверка добавочности (вес .35, прежняя маска и гейт №255); до коммита кода с №270 реальные метрики не считать. E186 отклонён; GPU доступен, LLM службы остановлены. Mac имеет около 4,4 GiB свободного места: полные OPDI сюда не копировать.
- Все будущие загрузки используют **общий серверный** `data/submission_upload_ledger.jsonl` и его lock. Для E185 проверен `src/e185_single_put_upload.py`; стандартный MinIO uploader с автоматическими PUT retries для новых подач не использовать. E182 получит свой проверенный entry point. Исторический импорт E64 запускает CV: сборку E182 на Hetzner согласовать с единственным активным E184 CV, временная заморозка всего systemd service допустима только со страховочным независимым thaw timer.

## Предыдущий снимок Codex 05.10 07:01Z

- **Новый лучший v12008=232,3586 с**, −0,2614 к v1139; доска обработала файл06:57:03Z,344841 строка. Текущее место12/230; третье charming-eel217,5655, разрыв14,7931с. Использован1/5слотов5октября. Журнал№267, report `analysis/submission_status_2026-10-05.md`.
- Сборка E185 завершилась04.10 22:46:26Z; отправка фактически05.10 06:56:28Z после паузы на сетевое разрешение. Старый upload с retry не запускался: новый single-PUT транспорт №264 / `84c29f8` прошёл реальные HTTP-тесты и независимое ревью. Ledger находится **на Hetzner** `data/submission_upload_ledger.jsonl`; все дальнейшие uploader используют тот же lock/ledger. До конца5октябряUTC ещё4слота. SHA final `bb7896d5b2698faf2de95cc0109604b34474bb36342e7f972e982bdd846dfb4a`, все42TabPFN-значения сохранены;4619 изменений строго внутри зарегистрированной маски (LOBT отсутствует, аэропорт не LIRF).
- **E182 принят №265 и ranking запущен №266**, commit `84c29f8`: home24threads raw/lobt/sched, Mac12threads eobt/aobt, по5сидов. Логи `data/e182_home_ranking_20261005.log` / `data/e182_mac_ranking_20261005.log`; файлычленов `submissions/e130_<mode>f_rank.npz` публикуются только5/5. MacPID51368. Членынеперезаписывать. После всех5×5 — отдельная пред-регистрациясборки; веса (.0333333333,.10,.10,.10,.10,.25), MLPполный. StrictCV−0,226814/5из6/trimmed−0,209718.
- **E184 пока0/6 после ночи:** подтверждён OpenMP busy-wait ≥94,91% CPUcycles, памяти/IO-дефицита нет. Старый лимитCPU0–5 оказался тремяфизическимиядрами; горячаясмена на0,2,4,6,8,10 сохранила6logicalCPU, но spinостался≥92,79%. Готовится изолированное ресурсное исправление до перезапуска; исходные hyperparameters/gateне меняются. Не расширять серверный PRCлимит.
- **OPDI43/45 окон**:113140967EVENT,124882662MEASUREMENT, всегообработано25,945GBисходныхархивов; итогincomplete. Октябрь2–12HTTP404. Для декабря1–11 найдена полная точная перекодировка старых ID с проверкойобоиххешей; ремонт пока в review, не считать применённым. Артефакт `analysis/opdi_full_inventory_2026-10-05.json`.
- **E186 закрыт:**+0,123132CV/1из6; GPUработает, этоткандидатнесдавать. LLMслужбыдомаinactive. Полныйнаборкода **119testsPASS**.

## Текущий снимок Codex 04.10 22:41Z

- **E186 завершён и отклонён**, журнал №263: все шесть фолдов и strict audit корректны, decided +0,123132 с, ниже лишь 1/6, body +0,126584, trimmed +0,121079. Гейт №259 не выполнен. Ranking/сабмит не запускать. Полный расчёт на RX7900XTX занял около 22 минут (22:15:24–22:37:16Z); лимит RSS20GiB выдержан, пик19,832GiB. Логи `data/e186_home_rss20_*`, результат `data/cv_preds_e186be_w015_f1b/gate_result.json` на home-pc.
- **E185 ranking готов 5/5 + done**, 344841 ID проверены, SHA256 `ed13473f97202fd21589257f5cb98c81c5ddbdf4c1fc68380eac544ed36d3cd2`. Сборка зарегистрированного v12008 запущена на Hetzner как `prc-e185-build-v12008.service`, лог `data/e185_build_20261004.log`, регистрация `f309ea1` (основная №262 в `3cdc720`). Пока файл не готов, upload не назначен.
- На время E185 сборки процессы E184 PID2894663/2894664 приостановлены SIGSTOP с сохранением памяти: старый импорт E64 запускает исторический CV, а два CV на сервере запрещены. Страховочный timer `prc-e185-resume-e184-safety.timer` через20мин заморозит E185 service и продолжит E184. После сборки обязательно CONT обоим E184 PID и отменить этот timer; проверить состояние/affinity. Сам E185 ограничен CPU0–5/Nice10.
- **Полный OPDI возобновлён** после исправления исторических MEASUREMENT ID (`fed8dc9`): строгий XXH64 seed0 для формата `YYYYMMDD_id`, исходные ключи сохранены. На 22:37Z завершены первые15/45 окон, считается16-е. Лог home-pc `data/opdi_full_ingest_legacyfix_20261004.log`. Октябрьский отсутствующий архив и декабрьская несовместимость FLIGHT/EVENT пока остаются явными пробелами; набор не объявлять полным.
- Mac продолжает E182 EOBT и сам запустит обе сетки. Новые E182 ranking пока не запущены: сначала полный исходный гейт №248. LLM службы дома остаются остановлены. Проверка кода в основном checkout: **118 tests OK**.

## Запуск E186 04.10 22:12Z

- Пред-регистрация E186 №259 и код в `176c394` (22:11:03Z), затем runner `97036b2`; запуск на домашнем GPU 22:12:32Z. Исходники изолированы в `~/prc-2026/gpu_e186/src`, хеши всех десяти файлов и FIN1b сверены с Mac до запуска. `hz_queue/hp_e186.sh` сам выполняет все шесть фолдов, фиксированный gate .15 и strict audit. Логи `data/e186_home_{runner,cv,gate}.log`; `.ready` означает завершение расчёта, решение только по `data/cv_preds_e186be_w015_f1b/gate_result.json`. `.failed` означает операционную ошибку. Нельзя подбирать параметры по первому фолду.
- E185 CV принят по №260: decided −0,047346, body −0,041495, trimmed −0,032365, 5/6. Ranking ещё неполный (3/5 сидов на 22:14Z).
- Полная OPDI загрузка стартовала, но остановлена для исправления совместимости исторических ID: EVENT2025 имеет uint64, MEASUREMENT2025 сохранил строковые `YYYYMMDD_id`. Простое числовое преобразование отвергло измерения, а не связало неверно. Проверяется совместимость xxHash64; исходные EVENT/checkpoints сохранены. До исправления не считать набор полным.

## Обновление Codex 04.10 22:00Z

- Denys разрешил полную загрузку CPU/GPU домашнего ПК и временно отменил LLM. Пользователь выполнил остановку через sudo; SSH-проверка подтвердила `inactive` и `MainPID=0` у `flight-llm-api.service` и `ollama.service`. Автозапуск не отключался, после перезагрузки проверить снова.
- Домашняя карта — **Radeon RX 7900 XTX 24 GiB**. Создана отдельная `.venv-rocm` с официальным AMD torch 2.13.0+rocm10.0.0; рабочая `.venv` E185 и системный amdgpu не менялись. Проба E186 FP32: максимальное расхождение forward 3,73e-8, градиентов 3,73e-9; медиана шага batch512 GPU 0,00360 с, CPU4 0,52056 с при параллельном E185. Это синтетический замер, не время полного CV. Артефакты `analysis/home_gpu_probe_2026-10-04.{py,json}`. E186 ещё требует регистрации кода и журнала перед CV.
- **OPDI разрешён организаторами для этого конкурса**, по уточнению Denys в текущей сессии: положительный ответ был дальше в чате. Это снимает прежний вопрос допуска; не приписывать данным лицензию MIT от репозитория кода. Новая загрузка хранит URL, релиз и SHA источников.
- Полный загрузчик OPDI интегрирован (`scripts/opdi_full_ingest.py`, `docs/opdi_full_ingestion.md`), 18 предметных тестов повторно прошли. Он сохраняет EVENT.id/info, MEASUREMENT и прилёты, исправляет неполный прежний отбор по месяцам. Октябрьский EVENT `20251002_20251012` отсутствует у источника: полный итог будет явно `incomplete`, пока источник не восстановлен. Старые OPDI файлы не перезаписываются.
- Старые сроки ниже не являются текущими: E185 на домашнем ПК завершил CV и начал ranking 5 сидов; финал можно брать только после всех пяти. Mac продолжает E182 EOBT. E184 на Hetzner работает, но прежний прогноз 2,5 ч на весь CV пока не подтверждается.

## Обновление Codex 04.10 20:30Z

- Завершён E187 (№256 → №257): другой критерий выбора весов дал decided **−0,004902 с, 3/6 ниже и 3/6 без изменений**. Требование ≥4/6 не выполнено; **не собирать и не подавать**. Протокол+код зафиксированы до запуска в `47c1005`. Контроль воспроизведён точно, все ID/y/месяцы проверены; подробности `analysis/e187_verdict_2026-10-04.md`, результаты `data/e187_20261004_47c1005/`.
- Добавлен самостоятельный `src/strict_oof_gate.py`: ровно шесть полных фолдов, уникальные ID и train-покрытие, y против features, месяцы, численные проверки. Exit0 = accepted/audit_valid, exit1 = valid reject, exit2 = malformed. Для **уже идущих E182/E184/E185** запускать сначала `--audit-only --features data/features_fin1b.parquet`, затем применять исходный гейт №248/254/255; не менять пред-регистрацию задним числом. Старые `gate.py`, `fold_metrics.py`, E172 и очереди не изменены.
- `src/e130_rank.py` теперь требует **новое имя** `E130_OUT` для FIN1b, например `submissions/e130_rawf_rank.npz`. Любой существующий final и `E130_OVERWRITE=1` отклоняются до обучения. Прогресс пишется отдельно в `.checkpoint.npz`; final публикуется атомарно только после всех сидов, с provenance и проверкой хеша признаков. E185 использует свой прежний ranking-скрипт: перед сборкой отдельно проверить 5/5 сидов и `done`, частичный файл не брать.
- Эти три файла (`e130_rank.py`, `rank_artifact.py`, `strict_oof_gate.py`) также перенесены на Hetzner; SHA256 совпали с Mac, py_compile и strict CLI проверены. Прежний серверный e130_rank.py сверён с базой 08dfb59 и сохранён как `src/e130_rank_before_47c1005.py`.
- Финальная проверка выявила E184 с ≈961% CPU: `cv_fold_preds_nested.py` явно передаёт `POLARS_MAX_THREADS=10`, а `train.py` задаёт `num_threads=10`, игнорируя внешнее OMP=6. После проверки CV_OUT/команды процесса **без рестарта** ограничена affinity всех потоков controller PID2894663 и текущего worker PID2894664 на CPU0–5; повторная проверка всех TID подтвердила этот набор. Следующие fold workers наследуют affinity controller. Исходники/параметры модели не менялись; сами пулы по-прежнему могут иметь 10 потоков, но занимать больше шести CPU не могут. Новые запуски nested CV на Hetzner запускать под `taskset -c 0-5 nice -n 10` до отдельного исправления обработки числа потоков.
- Сегодня использованы все пять слотов (лог загрузок подтверждён). Новый лимит 5 октября в 00:00 UTC = **01:00 WEST**. Очередь: E182, E185, их комбинация после проверки добавочности, E184, резерв E186; каждый только при выполнении своих ворот и полной готовности. Пять готовых новых файлов пока не подтверждены. План: `analysis/submission_plan_2026-10-05.md`.
- E186 остаётся кандидатом после synthetic MPS smoke, **полный CV не запускался и регистрация ещё нужна**. Modal/Kaggle изучены в `analysis/compute_opdi_2026-10-04.md`: для Modal GPU нужен payment method, несмотря на $30 кредит. OPDI отсутствует в рецепте текущего лучшего v1139; более позднее уточнение Denys о разрешении организаторов записано выше.
- 61 новый synthetic test и независимое ревью пройдены; после интеграции в основном checkout **67/67** вместе с прежними тестами E186. Ошибки публикации и malformed exit устранены до запуска E187. В 20:32Z подтверждено: Mac считает AOBT E182; Hetzner закончил SCHED/raw E182 и начал E184 CV (его лог в UTC+2: 22:23). E185 ещё ждёт Mac. Ни одно текущее обучение не остановлено.
- Ниже сохранён прежний снимок. В его строке про топ-3 арифметическая поправка: 232,6200 − 218,9864 = **13,6336 с** (доска ранее проверена 04.10 около 20:00Z), а не 13,37 с.

## Позиция на 04.10 17:00Z

- **Лучший: v1139 = 232,6200, 12-е из 226.** Третье место — jolly-lobster 218,99, разрыв **13,37 с**.
  Лидеры: strong-crane 213,92, vigorous-whistle 216,88, charming-eel 219,94.
- Темп: у нас 235,35 (28.09) → 232,62 (04.10), то есть −2,7 с за 6 дней. У лидеров −4…−13 с за 3 дня.
  Текущая линия (члены смеси, сиды, этапы, таблица FIN1b) даёт десятые доли за день. Разрыв ей не закрыть.
  Публичного кода у топ-11 нет; механизм их скачков неизвестен.
- Доски 2–4.10 и их чтение — журнал №241, №245, №250; бриф, раздел 21cn.
- Закрыто 04.10: обход источников данных (Fable и Sonnet): выигрыш ≥ 0,05 с не даёт ни один источник; лучшее —
  забастовки MIT Italia (LIRF, ≤ 0,05 с). Наземного ADS-B по LFPG/LIRF/LTFM за январь и июль 2026 в открытом доступе нет.

## Что считается

| Задача | Где | Лог / флаг | Ожидаемое окончание |
|---|---|---|---|
| **E182** (журнал №248): члены LOBT → AOBT → EOBT на FIN1b, затем промежуточная сетка 3f | мак, `src/mac_e182.sh` | `data/mac_e182.log`, `data/e182_<m>f.log` (фолд ≈ 22 мин, член ≈ 2,2 ч) | lobtf ~18:40Z, aobtf ~21:00Z, eobtf ~23:15Z, сетка 3f ~23:20Z |
| **E182**: члены SCHED → raw на FIN1b | hetzner, `hz_queue/96_e182_members.sh` | `/root/prc-2026/data/e182_hz.log`, флаг `data/e182_hz.ready` / `.failed` | ночь 4→5.10; после него мак сам заберёт каталоги и посчитает финальную сетку 5f |
| Шаг 0 E182 (готов): E181a под контролем FIN1b −0,419, 6/6 | — | журнал №249 | — |
| **E184** (№254): CV базы FIN1b + «просроченные у гейта» | hetzner, `hz_queue/97_e184_cv.sh`, ждёт E182 | `data/e184_cv.log`, флаг `data/e184_hz.ready` / `.failed` | ≈ 2,5 ч после E182 |
| **E185** (№255): член «только движения» | **home-pc** (перенесён с мака 04.10 20:51Z, чтобы не ждать E182), `hz_queue/hp_e185.sh` | на home-pc: `~/prc-2026/data/hp_e185.log`, флаг `data/hp_e185.ready` / `.failed` | ≈ 23:00Z 04.10 |

## Что делать дальше (по приоритету)

1. **Вердикт E182** по финальной сетке (`data/e182_grid_5f.log`, строка `5f:` в `data/mac_e182.log`): гейт ≥ 4/6 и
   усечённые < 0 (замена членов — не добавка ёмкости). Прогноз был −0,15 (−0,40…+0,10); перенос на доску ждать
   ≈ 30 % (как E181a, журнал №250).
   - При проходе — прогнозы членов на ranking: `FEATURES=data/features_fin1b.parquet E130_MODE=<m> E130_OUT=submissions/e130_<m>f_rank.npz .venv/bin/python
     src/e130_rank.py <seeds> <rounds>`, rounds — среднее шести фолдов из `data/e182_<m>f.log`.
     **Защита добавлена 04.10:** FIN1b требует отдельное новое имя `E130_OUT`; существующий final и флаг overwrite отклоняются.
     Если запускаем не на Mac, предварительно сверить SHA256 `e130_rank.py` и `rank_artifact.py` с текущим кодом.
   - Сборка — рецепт v1139 (`hz_queue/95_builds_4oct.sh`, T4) с новыми файлами членов; веса LightGBM-членов —
     ×2/3 от весов сетки 5f, MLP — полный вес.
2. **E184** (журнал №254) — CV базы FIN1b + «просроченные у гейта» на hetzner, стартует сам после E182
   (`hz_queue/97_e184_cv.sh`, лог `data/e184_cv.log`, флаг `data/e184_hz.ready`/`.failed`, ≈ 2,5 ч). Почти повтор
   закрытого E137; прогноз decided −0,03, P(проход) ≈ 20 %. Добавка ёмкости: решает нога y ≤ 7200. При отказе —
   строка «очередь у гейта (E127, E137, E184)» в раздел 0 брифа (сейчас E127 и E137 там нет).
3. **E185** (журнал №255) — считается на home-pc; результаты забрать на мак:
   `rsync -a home-pc-lan:prc-2026/data/{cv_e130motion,e185_cv_sub,e185_cv.log,e185_grid.log,e185_rank.log,e185_motion_rank.npz,hp_e185.log} ~/prc-2026/data/`.
   Дальше — — член «только движения» для строк без NM вне LIRF; `src/mac_e185.sh` стартует сам после
   членов E182 (лог `data/mac_e185.log`, OOF 45–90 мин, ranking 15–35 мин). Прогноз decided −0,07, P(проход) ≈ 55 %.
   При проходе — доска на hetzner по рецепту v1139: скопировать `src/e185_apply.py` и `data/e185_motion_rank.npz`
  , сверить md5; `E185_W=<пул по телу × 2/3> src/e185_apply.py 1121 V` →
   `e167_nm_window_rule.py V V+1` → `t2 V+1 V+2 361` (с `E36_BASE_DIR=data/cv_preds_e116prod`) → `protect_tabpfn_rows.py V+7 V+2 V+8`.
   При принятии встроить в `run_final_any_period.sh` (построитель вида читает фиксированный `data/prc-2026-datasets`).
4. ~~E183~~ — **закрыт пробой** (журнал №253): ADS-B-прокси остатка соседей стоит +0,002 с; строки без трека и соседи с треком — в разных аэропортах.
5. **Прогноз доски и выбор файлов** — по калибровке 04.10 (журнал №252, `analysis/board_vs_cv_slices_2026-10-04.md`):
   прогноз = decided-срез (тело y ≤ 7200) × коэффициент типа: этап ADS-B ≈ 1,0; член смеси ≈ 0,7 (LightGBM-члены ×2/3,
   MLP — полный вес); новые признаки базы 0,2–0,3; правила/калибровки по 2025 — 0 или минус; хвост/TabPFN — только доска.
   Хвост y > 7200 в CV доску не предсказывает. Финал — по доске; при разнице < 0,1 с — по decided-телу с поправкой на тип.
6. **Слоты:** 5–11.10, по 5 в день — **загружает Codex** (решение Denys 04.10). 4.10 потрачены все 5 (лучший v1139 = 232,6200). Пред-регистрация каждой доски — до сборки (журнал + код одним коммитом).
7. **9.10** — открыть репозиторий и уведомить организаторов (`OPENING-CHECKLIST.md`).
8. **11.10** — финальный файл; последняя загрузка до 21:59Z.

## Инциденты 04.10

- На маке свободно ≈ 12 ГБ диска (занято 98 %). Перед новыми таблицами признаков освободить место
  (например, `data/e183_probe_rows.parquet`, 286 МБ, больше не нужен).

- В 16:43Z агент запустил `uv run` и `.venv` на маке пересоздался: Python 3.14 → 3.12, версии пакетов те же
  (polars 1.44.2, lightgbm 4.7.0, catboost 1.2.10, torch 2.14.0, tabpfn 8.5.0, minio 7.2.20). Шедший член E182
  (lobtf) работает на старом интерпретаторе в памяти. Следующие шаги E182 идут уже на 3.12. Если результаты
  lobtf и следующих членов покажут странности, первым делом проверять это.

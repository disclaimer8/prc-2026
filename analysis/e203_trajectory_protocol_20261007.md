# E203 — траекторный компонент в текущей цепочке и условный production

Protocol ID: `E203-ordered-trajectory-fixed025-joint-preE81-sixpair-gated-fullrank-fulltail-v1`.
Новые файлы: `scripts/e203_trajectory_production.py`, `scripts/e203_execution_queue_20261007.py`, этот протокол и `analysis/e203_input_receipt_20261007.json`. Код, протокол, inventory и журнал регистрируются одним commit до новых сравнений, подготовки или fit. Старые E197/E200/E202, helpers, окружения и submissions не меняются. В этом протоколе **нет upload**.

## Фиксированный вопрос и граница оценки

E197 подтвердил ordered против равноёмкого flat и сохранённого PLE на двух парах. Эти результаты уже известны; пары 1+7 и 4+10 явно selection-informed. E202 независимо от результата E203 завершает фиксированные 24 fit на остальных четырёх парах. E203 проверяет, сохраняется ли польза ordered при единственной замене **PLE-member на ordered-member в прежнем слоте веса 0,25** текущего joint E195. Никакого выбора по fold, аэропорту, seed или leaderboard.

Первый CPU apply использует доступные пары 1/4; второй — все пары 2/3/5/6 после полного E202. Первая половина не разрешает ranking и не изменяет рецепт второй. Контроль должен побитово воспроизвести E195 joint на исходной поддерживаемой популяции, прежде чем сравнивать кандидата. Действует исходный retrospective scope: Cat15 + LGB5, фиксированные anchors и probabilities, E185/E167, E141/D7 и E64/E188/D3 по пяти seeds, E152. Оценка заканчивается **до E81**.

На каждой паре label-free feasibility mask вычисляется до ошибок и объединяет изменения допустимости кандидатов control/ordered. Общие 126 TabPFN-строк и вновь изменившаяся mask исключаются симметрично, полные ID сохраняются, на исключённых строках обе стороны получают одинаковый контрольный fallback. Не восстанавливать потерянную raw probability mass и не выдавать conditional invariant-population comparison за всю популяцию. Дополнительные исключения и покрытие обязательны в отчёте.

Критерий current-chain PASS фиксирован: decided SSE меньше в ≥4 из 6 пар; pooled body SSE (`y≤7200`) меньше; сумма **within-fold DECIDED** symmetric trim3 и trim20 SSE меньше. Pooled decided знак сообщается отдельно, дополнительным gate не становится. Этот trim population соответствует E195/E200; компонентный gate E202 использует ALL-row trims и остаётся своим отдельным условием. Для production нужны одновременно current-chain PASS и оба E202 gates: ordered против flat/PLE и flat против PLE.

Selected-two и remaining-four публикуются отдельными группами вместе с общим результатом. Вся адаптация к current chain selection-informed после компонентного screen. Исторические upstream predictors могут видеть текущие outer labels через другие folds: это **не fully nested CV**. Точные OOF-доноры v44/v60 и post-donor LSZH здесь не оцениваются; E201 — отдельная линия. Нельзя называть этот gate полным production CV.

Экспертный prior до нового сравнения: pooled body ΔRMSE ordered−joint = 0 с, некалиброванный диапазон [−1,+1] с. Это не обещание улучшения и не доверительный интервал. Измеренный результат показывается отдельно от gate.

## Точные зависимости

- E195 parent registration `bc19847d23d4485717bafa84744a14b49af0da19`, результат `data/e195_pre_e81_20261007_v2`, полная исходная SHA-inventory и 96 уже обученных stage models. Cat/LGB заново не обучаются для CV.
- E197 registration `bfe793c62f7cecf633c7cc31c747f48eb0676f2d`, complete `84842a0734975ae76bc27ebdf0fdc5841bc769a7dc8322255672d0c48c4f75db`, независимый audit `cae24cc29438a6d3023329a07664f653a94c40b2802737fbc98c93b7362fac62`.
- E202 registration `55fe9b68f7cc8405681f6c85d576979e067b55c3`, input spec `6d604eaaca830fa83c18ebb78986b028d338869123d5f8c10b2450dde1ec4309`. Операционный перезапуск до fit не меняет эту научную регистрацию. Перед CV проверяются все ordered models/seed outputs, exact mean, mixture, ID/y/month; E202 completion и независимый локальный отчёт обязательны.
- Full E190 ranking queries `data/e190_rank_inputs_20261006`, registration `f40a465633cc32bff959c598d27d59f9bdc11cc3`, spec `9c9e67f37c16db24d1e4a40245c28f56fdb206f8a5a85fca5ff47415cb2c9399`.
- Frozen E197 worker SHA `6be13d401dd2903c50c9584f5ca2d6ca084a1feb0eb1f67fcba0cd94850db54a`; E200 worker SHA `2fe1328214128a0f6ded5a178d954fa3ba0634c4043e42e7c69bba461062ad25`. E200 stage arithmetic скопирована в новый namespace; AST selftest сравнивает paired/gate/signature/capture-replay/branch функции с источником. Старые модули не monkeypatch.

## Full-training/ranking mapping и normalization

Повторно используются **готовые** FIN1b query tensors: 1 944 054 ordinary training rows, 344 841 ranking rows, 272 numeric channels и 14 categorical slots. Не строятся новые query transforms и не меняется target scaling. Новый порядок контекста — full train IDs из исходных FIN1b features, затем exact ranking IDs: всего 2 429 888.

Перед публикацией pack проверяются SHA исходного FIN1b и ranking tensor manifest, полный training-ID digest, отсутствие ranking targets, та же regime_label на полном 2025, обычная популяция, ranking-ID порядок. Восемь ранних числовых query columns восстанавливаются на **всех** ordinary/ranking строках и побитово сверяются. Полностью пересчитываются proxy и scaled ordinary target; они обязаны побитово совпасть с сохранёнными `base_p`/`y_scaled`. Только после этих проверок записываются train/pred indices. Это устраняет предположение о порядке старого join.

Используются все 427 прежних immutable E194 sequence archives, provenance связывается с двумя original complete receipts. Stats channels 0…5 считаются только по known observations ordinary training IDs 2025. Ranking никогда не входит в mean/std, даже при доступном raw. Два визита, 64 реальных tokens, все known/present masks, clip±8, missing→0 остаются как E197. Никакой интерполяции, изменения сегмента, mask augmentation или дополнительных labels. Условия retrospective boundary matching E194 остаются; онлайн-прогноз не заявляется.

Подготовка разрешена до финального gate как фиксированная подготовка входов без model fit. Она не разрешает ranking. Worker последовательно читает один день, пишет mmap и flush/MADV_DONTNEED после каждого дня. Лимит **1 CPU / 2 GiB / swap0 / nice10**, один worker. Старт подготовки ждёт admission всех четырёх E202 таблиц и минимум8GiB фактического MemAvailable на Home; этим исключается одновременный запуск с краткими admission jobs и сохраняется запас для существующих CPU/GPU работ. Ожидание ограничено8часами; чужие jobs не останавливаются. Большой итоговый table создаётся только на Home: values float16, masks uint8, примерно **10,72 GiB**, а не на Mac. На Home требуется ≥14 GiB свободно. С Mac передаются лишь уже существующие 2,6 GiB compressed E194 archives и малый pack; существующие raw files не перезаписываются, несовпадение SHA останавливает процесс.

## Условный GPU fit

Тот же `E197.fit/network`, arm ordered, seeds 0/1/2, восемь epochs, batch4096, AdamW lr0.002/wd1e−5, OneCycle pct_start0.1. Full query bins вычисляются только по full ordinary training tensors. Никакой новый loss, output distribution, early stopping или подбор.

Home `.venv-numemb`, GPU 7900 XTX 24 GiB, CPUQuota200%, MemoryMax12G, swap0, nice10. Full table передаётся кусками8192; CPU mmap pages освобождаются. Новый synthetic smoke резервирует точные 11 507 949 568 bytes таблицы на GPU и делает шесть full4096 backward steps с фактической numeric width и full category cardinalities; synthetic numeric distribution conservatively даёт больше continuous channels. Finite gradients, точные runtime/source и peak VRAM обязательны до target fit. OOM запрещает автоматическое уменьшение batch/precision.

GPU unit стартует только после terminal E202, durable complete/evaluation с pinned source/spec и обоих шести-парных gates. Очередь разрешает **late arming**: не требует увидеть predecessor live, если совпадают завершённые SHA, регистрация, inventory и независимый evaluation. Если live был замечен, смена InvocationID — ошибка. Новые фит-процессы однократные и bounded 2h; провал не вызывает retry.

Результат содержит три model states, seeds, bins, exact mean `reg/reg2`, 344 841 unique ID, hashes и receipts. Неполные/NaN/переставленные строки останавливают продолжение. Авторизация gate переносит полные fold reports; Home заново пересчитывает gate и сверяет E202 proof, а не доверяет одному boolean.

## Полная production-сборка

Контроль: `submissions/gentle-igloo_v12074.parquet`, SHA `e5c78ab061492b69fe72d4a46d2e286f7a2f6b3129ebf2ae4a34731bb662374b`. Новый private workspace `data/e203_chain_20261007`, scratch versions930100…930115 (контроль) и940100…940115 (кандидат), окончательный артефакт `data/e203_candidate_20261007/candidate.parquet`; никаких новых root submission versions.

Рецепт неизменен: Cat0.65/LGB0.35 и прежние anchors, neural slot0.25; E185, E167, 42 TabPFN donors, E141/D7/five seeds, E64/D3/five seeds, E188 models, E1520.95, E81 weights0.1214/0.0270, LOBT lift/clip, повторная защита42donors, half LSZH. Контроль с исходным PLE обязан **точно совпасть во всех целочисленных строках** с v12074. Только после этого новый ordered-member проходит те же стадии.

Как в проверенном E200, первый control pass сохраняет 11 production LGB models. Replay проверяет dataset bytes/dtype/shape, labels, weights, categorical semantics, feature order, params, rounds и runtime; несовпадение запрещает использование кэша. Replay заново рассчитывает входозависимые base/relative features, не подставляет старые поправки. SHA control, model-cache и final validations сохраняются. Final IDs/finite/nonnegative/schema и неизменность42donors обязательны.

На Mac capture ждёт завершения E201 donor job и ≥8 GiB reclaimable memory. До начала требуется ≥4 GiB free disk, перед каждой стадией ≥2,5 GiB; сохраняется минимум2GiB. Существующие данные hardlink/symlink, тяжёлые fullrank tables отсутствуют локально. После сборки только ready artifact, `uploaded=false`, `accepted_for_submission=false`; внешняя подача отдельно.

## Порядок, ошибки и проверка

Durable controller делает pack и лёгкий selected apply (2 CPU, ожидание ≥4GiB доступной памяти), remote streaming preparation, ожидание независимого E202 completion, remaining apply, общий gate. При FAIL фиксирует `complete_gate_failed_no_rank`, не запускает GPU fitting/production. При PASS — small gate transfer, fullshape smoke, три ranking fits, небольшая коллекция, ожидание освобождения Mac, exact control/candidate. Все machine jobs имеют named units, PID/InvocationID и receipts. Запуск через purpose-specific `--launch` проверяет регистрацию, создаёт новый лог и отделённый caffeinate process. Потеря агента не останавливает OS-controller; потеря сети даёт failure receipt без автоматического повтора. Другие задачи не завершаются принудительно.

До регистрации выполняются compile и контрактные selftests, включая AST parity gate/branch, 6fold requirement, DECIDED trim boundary и объём full table. Новые ошибки/OOF RMSE до регистрации не вычисляются. Ориентир после E202: CV минуты, full ordered rank три fits приблизительно15–20мин по E197 timing с большей train population; фактическое время и ожидания ресурсов сообщаются отдельно. Это не обещание срока.

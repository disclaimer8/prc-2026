# E207 v2: техническая остановка после завершения первого shard

Срез проверки **08.10.2026 02:30:30 UTC (03:30:30 WEST)**. **Controller v2 завершён, оба raw-unit остановлены;0 prep/target-fit/ranking.** Это технический сбой обработки lifecycle systemd, не научный результат F-only. Ниже сохранён предыдущий снимок реального extraction.

171 завершил свой shard: найдено **183 дневных `receipt.json` и итоговый `s0/complete.json`** (531250байт). На175 сохранено **145 дневных receipt**, `s1/complete.json` отсутствует. Всего328 дневных результатов сохранены; это не завершённые365 дней и не независимый полный аудит их payload. Данные/partials не удалялись и не изменялись этой проверкой.

По диагностике root, после успешного завершения transient unit171 systemd GC очистил InvocationID; `wait_unit` трактовал это как смену владельца. [controller.log](controller.log) указывает строку114 и `ValueError('owned unit invocation changed')`. Cleanup затем остановил точный owned175 InvocationID`e92ac9eae23f4f979182c7ce98ea2601` в **02:28:24.955675UTC**. Terminal **02:28:53.186665UTC**: `failed_no_retry` с той же ошибкой. [Полный operational snapshot](execution_snapshot.jsonl), [исходные events](execution/events.jsonl).

Read-only проверка02:30:30UTC: оба `prc-e207-v2-raw-{171,175}-20261008.service` — **inactive/dead, MainPID=0, InvocationID пуст**; `ps -p38371` не нашёл controller. Исторический InvocationID171 —`a5e6e4a3afa347dbb3440143ba5e54d5`. `Result=success/ExecMainStatus=0` после выгрузки transient units не заменяет ошибку controller и не означает завершение всего workflow.

Author готовит отдельнуюv3 lifecycle-поправку с ограниченным настоящим systemd smoke; научный рецепт не меняется. **№379 пока не добавлен**: новая регистрация только после review. V2 source/protocol/metadata остаются frozen на №378/`ea16e05330d3fe4fb5573dc17938002f14645876`; его очередь не возобновляется. Новые расчёты, smoke или изменения данных этой документационной проверкой не запускались. Best230.9387с прежний, нового RMSE/подач нет, live-квота8октября не проверена.

## Исторический снимок работающего v2 — 02:26:18UTC

Срез **08.10.2026 02:26:18 UTC (03:26:18 WEST)**. Controller стартовал **02:24:15.531271UTC**, PID38371; detached Python подтверждён `ps`. Пререгистрация №378, commit `ea16e05330d3fe4fb5573dc17938002f14645876`, запуск выполнен после sandbox approval. [launch.json](launch.json), [controller.log](controller.log), [events.jsonl](execution/events.jsonl).

**Текущий этап: extraction реально записывает завершённые дневные результаты.** Read-only count файлов `receipt.json` на02:26:18UTC: **171 —71 день,175 —54 дня**, всего125 из365. Это моментальный снимок продолжающейся работы, не завершённый shard. Логи обоих unit содержат `EXTRACTED` с датой и selected-point count; уже наблюдались март–апрель2025. В отличие от одной CPU-активности это подтверждает сохранение дневного результата после встроенных проверок. Полный независимый аудит всех будущих таблиц ещё впереди.

| Узел | Unit / MainPID | InvocationID | MemoryCurrent / MemoryPeak, байт |
|---|---|---|---:|
|171|`prc-e207-v2-raw-171-20261008.service` /365405|`a5e6e4a3afa347dbb3440143ba5e54d5`|2566791168 /3017306112|
|175|`prc-e207-v2-raw-175-20261008.service` /168255|`e92ac9eae23f4f979182c7ce98ea2601`|2206711808 /2467024896|

Оба `systemctl --user show` дали **ActiveState=active**. Events фиксирует старты171 в02:25:20.184446UTC и175 в02:25:21.316021UTC. Рабочие логи: `/home/a1/prc-2026/data/prc-e207-v2-raw-171-20261008.log` и соответствующий`...175...log`; receipts — `data/e207_v2_raw_20261008/s0/<day>/receipt.json` на171 и`s1` на175.

PACKED spec SHA-256 `b41c15ec34287f7b6c94c0b60066553b5502934bcc7a88ab98db54f0590f0eb8`. Idle/disk admission перед запуском: Home711145914368байт; узел171 —161867833344байт; узел175 —414684033024байт. Входы переданы до старта extraction; worker SHA`52867eb1772ed4bdcfc9088e097e5d2e12d72736796e61ab67dc7ecb7107a452`, queue SHA`5ed43d38c6d669b4766bd25f3a9d6aee196aab9b9133e8cf835e77417eb90a13` закреплены в launch receipt.

**На этом срезе0 target-fit, нового RMSE нет.** Prep/normalization, Home admission и synthetic GPU smoke ещё не подтверждены стартовавшими. Очередь автоматически продолжит extraction→обмен shard→materialize→prep/admission→обязательный full-shape synthetic GPU smoke→6пар×3сида×8эпох,18fit. Около2ч — исходная оценка workflow, не измеренное время завершения; hard6ч, без retry/resume/fallback. Научный рецепт №377 сохранён в №378; [протокол](PROTOCOL.md), [review](REVIEW.md). Ranking потребует отдельного current-E203-chain протокола; автоматической подачи нет.

[V1 технически остановлен](../e207_full_token_20261008/STATUS.md) с0 завершённых дней/fit; его source/protocol/partials и исследовательский отчёт23:20UTC сохранены. V2 использует новые namespaces и однократное чтение NPZ, старый запуск не возобновлялся.

Best **v12075=230.9387с** прежний;14/237 — снимок07.10 22:00UTC,4/5слотов относятся к7октябряUTC. Live-квота8октября не проверена, новых подач нет. E204/E205 завершены FAIL; E201 отменён, LLM выключены. Это не утверждение, что все расчёты сейчас завершены: E207 v2 работает.

# E207: первый запуск технически остановлен

## Техническая остановка 08.10.2026 — подтверждена00:08:36UTC

Root отправил SIGTERM controller PID87152 после обнаружения повторной распаковки NPZ внутри per-flight проверки `validate_arrays`, получавшей `NpzFile`. Высокая загрузка трёх worker на узел подтверждала занятость CPU, но не завершённую полезную обработку дня. Это **техническая остановка реализации**, не отрицательный научный результат F-only.

[events.jsonl](execution/events.jsonl) сохраняет точные terminal-события:

| UTC08.10.2026 | Событие |
|---|---|
|00:07:42.600974|`stopped_owned_invocation`,175, unit`prc-e207-raw-175-20261008.service`, InvocationID`070a9f744fd24d6db4e05d5b7ce13a32`|
|00:07:43.351569|`stopped_owned_invocation`,171, unit`prc-e207-raw-171-20261008.service`, InvocationID`69d7d4c5f736471ba82c4c5fca6324bb`|
|00:07:51.862671|`failed_no_retry`, `InterruptedError('E207 controller received signal 15')`|

Независимая read-only проверка в00:08:36UTC: оба unit **ActiveState=inactive, SubState=dead, MainPID=0**, InvocationID пуст после завершения; совпадение остановленных InvocationID установлено по events. `ps -p87152` не нашёл controller. `Result=success/ExecMainStatus=0` у удалённых transient units не является научным PASS: controller записал `failed_no_retry`.

Проверка `find .../data/e207_raw_20261008/s0 -maxdepth2 -name receipt.json` на171 и соответствующего`s1` на175 дала **0 файлов**. Итого: **0 completed extraction days, 0 prepared tables, 0 synthetic GPU smoke, 0 target-fit, 0 ranking/подач**. Исторические рабочие логи и partial directories сохранены; эта проверка ничего не останавливала, не перезапускала и не удаляла.

Старые source/protocol/registration metadata остаются frozen на №377/`741045e9fe10c17f68a7ce6bed4216e6160ee7b1`. Author готовит отдельные v2 sources/namespaces; новая очередь ещё не зарегистрирована и не запущена. **№378 зарезервирован root для технической v2 после review и здесь не добавлялся.** Прежняя оценка workflow≈2ч не является измеренным временем успешного выполнения. Best230.9387с прежний, live-квота8октября не проверена.

## Исторический снимок запущенного E207 — 00:05:31UTC

Срез **08.10.2026 00:05:31 UTC (01:05:31 WEST)**. Это квитанция исполнения; исследовательский отчёт23:20UTC и frozen-протокол не менялись.

**Текущий этап: два активных extraction-unit; 0 target-fit.** Controller запущен08.10 в00:02:05.287063UTC, PID87152, под caffeinate; `ps` подтвердил живой detached Python-процесс. Пререгистрация №377, commit `741045e9fe10c17f68a7ce6bed4216e6160ee7b1`. Sandbox approval получено перед запуском; зарегистрированный код не менялся.

[launch.json](launch.json), [controller.log](controller.log), [events.jsonl](execution/events.jsonl) подтверждают запуск, проверку idle/disk и последующий старт unit. Упаковка закончилась: spec SHA-256 `fa19112c8cdeb217bd40269f04a9cd5866eeebd61cc3ec9bcdb5379940e87e68`. Передача входов выполнена до старта extraction. Диск на admission: Home —711576444928байт; узел171 —162340503552байт; узел175 —415114645504байт.

| Узел | Подтверждённый unit / PID | InvocationID | Наблюдение |
|---|---|---|---|
|171|`prc-e207-raw-171-20261008.service` /352872|`69d7d4c5f736471ba82c4c5fca6324bb`|Старт00:02:46.649495UTC; `systemctl --user show`:active|
|175|`prc-e207-raw-175-20261008.service` /157961|`070a9f744fd24d6db4e05d5b7ce13a32`|Старт00:02:47.870662UTC; `systemctl --user show`:active|

Read-only `tail` обоих worker-логов в00:05:10UTC вернул пустой вывод: **завершение первого дня extraction пока не подтверждено**. Controller `progress` подтверждает liveness unit, не обработанные дни или fit. Логи на каждом узле: `/home/a1/prc-2026/data/prc-e207-raw-171-20261008.log` и соответствующий `...175...log`. Table preparation, Home admission, synthetic GPU smoke и обучение пока не подтверждены стартовавшими; готовых результатов и нового RMSE нет.

**CPU-активность подтверждена отдельно от liveness.** В00:05:10UTC `ps --ppid` показал на171 child PID352959/352962/352963 и на175 PID158055/158058/158059, каждый около100%CPU. Последовательные `systemctl --user show` в00:05:10 и00:05:31UTC дали:

| Узел | CPUUsageNSec: первая → вторая проверка | Прибавка CPU за≈21с | MemoryCurrent / MemoryPeak, байт (вторая проверка) |
|---|---:|---:|---:|
|171|434403363000 →497896903000|63.493540CPU-с|685871104 /1132810240|
|175|430746259000 →493920057000|63.173798CPU-с|701710336 /1031659520|

Это подтверждает исполнение трёх worker на узел. Завершённых extraction-day receipts пока не наблюдали; первое сообщение `EXTRACTED` печатается только после полного дня и его receipt. Ожидание этого события не подменяется заявлением об обучении. Процессы не перезапускались, исходники не менялись.

Очередь автоматически ведёт extraction→обмен shard→materialize→prep/admission→обязательный full-shape synthetic GPU smoke→6пар×3сида×8эпох, **18 fit**. Оценка всего workflow около2ч — расчёт, не измеренная длительность; единый hard limit6ч, без retry/resume/fallback. [Протокол](PROTOCOL.md), [независимое review](REVIEW.md). Перед ranking потребуется отдельный current-E203-chain протокол; **ranking и upload не автоматизированы и не разрешены этой регистрацией**.

Best **v12075=230.9387с** прежний. Доска14/237 — исторический снимок07.10 22:00UTC;4/5слотов были использованы7октябряUTC. Уже наступило8октябряUTC, его live-квота не проверена; новых подач в E207 нет. E204/E205 завершены FAIL; E201 отменён, LLM выключены. Их завершение не означает отсутствия текущей очереди E207.

# E197 — локальный порядок реальных точек собственного борта

Protocol ID (вставить дословно в строку журнала при регистрации):
`E197-own-previous-local-order-vs-flat-K64-2pair-3seed-8epoch-v1`.
Код: `scripts/e197_trajectory_encoder.py`. До root review и общего коммита
код+протокол+журнал разрешены только synthetic tests; target-fit не запускался.

## Проверяемый механизм

Сеть может использовать последовательность остановок, изменения направления,
наблюдаемого движения и качества приёма сверх одинакового набора отдельных точек.
Проверяется **локальное смешивание соседних выбранных наблюдений**, а не вся
временная информация: возраст точки и gap присутствуют в обоих плечах.

E134a/HEAD показал последовательный малый сигнал и вошёл вE141. E39/E134b
отрицательно проверили prefix augmentation с ручными признаками; ни искусственных
prefix, ни новых псевдометок здесь нет. E191 физический auxiliary имел223 строгих
наблюдения, недостаточный stale recovery pilot не масштабирован. Здесь используется
широкое реальное наблюдение с UNKNOWN masks; stale не превращается вFalse,
физическое движение не объявляется истинной recorded TAXITIME.
E192 проверял связи между соседними рейсами; E197 читает точки самого борта
в двух визитах. Отрицательный E192 не переносится на этот другой вход.

## Входы и уже измеренное покрытие

E194№332, commit `fc7107de145a87a5a1ff93314f318b440b1de825`, завершён на171/175.
Все427дней/2 429 888ID проверены на Mac, включая provenance samples и Mac/Linux
parity; `analysis/e194_trajectory_execution_20261007.md` и
`data/e194_verification_20261007/verification.json`.

|Population|Рейсов|Точки current visit|Точки previous visit|Previous only|
|---|---:|---:|---:|---:|
|Train2025|2 085 047|1 251 417|1 345 381|322 974|
|January2026|152 719|105 052|100 834|16 213|
|July2026|192 122|122 645|132 153|30 607|

E197screen использует только365дней2025 из
`data/e194_sequence_20261007_s{0,1}/YYYY-MM-DD/sequence.npz`.
Все ranking observations остаются доступным будущим источником, но **не входят**
ни в обучение, ни в normalization данного screen.31.12.2025 остаётся полностью
masked для4715рейсов. Никаких новых запросов к API/лицензионным источникам.

E194 содержит два потока по≤64 **реальных** выбранных точек; нет интерполяции.
Сохраняются ages/gaps, east/north, altitude/speed, sin/cosheading, on_ground,
stale, callsign-match, duplicate-timestamp. `known` отмечает измеренность
каждого канала, `present` — наличие точки. Полный raw доступен через E194
raw_ranges/source_rows; этот опыт использует фиксированный64-token view.

Same-day уникальный callsign/airport→ICAO, отсутствие/неоднозначность masked.
Токены строго t≤MVT. Как вE64/E194, выбор последнего сегмента использовал окно
MVT−12h…MVT+600s, затем токены обрезаны. Поэтому это retrospective boundary
matching в разрешённой постановке конкурса, **не онлайн-прогноз**. Предыдущий
визит не продолжается во вчерашний файл. Точки airborne не удалены; это
наблюдение визита, а не гарантированная непрерывная наземная траектория.

## Одинаковые query, target и границы оценки

Существующие immutable E190 tensors: `data/mlp_split_20261005/inputs/fold_1_7`
и`fold_4_10`, original registration`a926f3848baaae7415bd3cb773cbd39728ce1851`.
Сам исходный MLP recipe, query transformation и ordinary residual target сохранены.
Контекстный row order берётся из независимо проверенного E192 mapping:
`data/e192_event_inputs_20261006/query_ids.npy`, train_indices и pred_indices.
E192 проверил восемь ранних numeric columns на всех строках, exact ordinary
train-ID digest и heldout IDs/targets. E197 повторяет SHA/query spec, train-ID
digest, heldout permutation и meta checks; собственного нового regime labeling нет.

Тот же stored control/PLE/evaluation, неизменные production anchor weights и
MLP coefficient; LGB5/pre-ADSB screen. Это **не новый full current-chain CV**.
Target читается только из неизменных query tensors для обучения собственного
регрессионного выхода и из evaluation для оценки. Нет label-bearing token,
соседских targets, auxiliary physical labels или target-conditioned extraction.

## Равноёмкие плечи

Общий E190 PLE query encoder, width512,48bins×12emb; head заменён наIdentity,
query representation256. Общий point encoder24→64→64 сSiLU:12 values +12 known;
values дополнительно умножаются наknown. После каждого шага PAD обнуляется.
Потоки current/previous обрабатываются отдельно одним набором весов.

Два residual message-блока, каждый `Linear(192,64), SiLU, Linear(64,64)`:

- `flat`: concat(сам token, mean всех present tokens этого визита, max этого визита).
  Полностью permutation-invariant по набору точек одного визита.
- `ordered`: concat(предыдущий выбранный token, сам token, следующий выбранный
  token). За краями/вместоPAD — ноль. Междуcurrent/previous/чужими строками
  соседства нет. Направление задаёт исходная сортировка timestamps/source-row.

После двух блоков одинаковые mean/max pools по каждому визиту и два числа
долей present tokens. Empty pools строго0, не−inf. Общий fusion:
concat(query256, means128, maxima128, support2)→256→1.
Нет learned position embeddings и дополнительных параметров только уordered.
Число параметров обязано быть **точно одинаковым** внутри каждой пары/seed;
фактическое число зависит от query fold и фиксируется в receipts.

Оба плеча видят ages и raw gaps. Поэтому результат отвечает на вопрос о
полезности локальной последовательной обработки при одинаковых входах, а не
«есть ли вообще временной сигнал». Final token pools также одинаковые.
Flat против сохранённого PLE смешивает эффект входов и добавленной ёмкости:
положительный результат означает полезность этого контекстного варианта,
для чистой атрибуции новым входам нужен отдельный контроль.

## Подготовка на171/175 и ресурсы

`pack` на Mac после регистрационного коммита создаёт небольшой immutable bundle
с scripts, protocol, IDs/indices, controls/evaluation, manifest. Все E194 source
SHA и complete receipts, parent spec и original query spec закреплены. Source
и protocol обязаны точно совпадать сGit-объектами регистрации; журнал содержит
точный Protocol ID. Реальные источники передаются отдельно без изменения.

Ноутбук171 готовитfold1+7,175 fold4+10. Каждому нужна недостающая половинаE194
(около1,3GB; суммарные compressed outputs около2,6GB). Во времяCPU preparation
target values не нужны: исходный ordinary fit subset задан готовыми IDindices.
Три дневных worker thread, CPUQuota≤600%, MemoryMax≤10G, swap0, nice10,
sleep inhibitor. Всё в существующих средах, безpip/uv/глобальных изменений.

Первый streaming pass считает mean/std для каналов0…5 (age,gap,x,y,alt,speed)
только по known present observations ordinary **fit rows**, с отдельным отказом,
если выбран fit-ID из heldout месяца. Каналы6…11 сохраняют свои естественные
sin/cos/binary значения. Фиксированный z-score clip±8, std0→1; missing снова0
с known0. Все fit observations учитываются, случайного subset/подбора quantile нет.
Суммы/sumsq float64, общий normalization дляcurrent/previous, чтобы шкала
двух потоков оставалась одинаковой. Нет обучения нормализации поranking.

Второй pass пишет полный query-order mmap: `values.npy`float16,
`known.npy`uint8, `present.npy`uint8, `support.npy`uint8 плюс IDs и norm receipt.
Примерно9,2GiB/fold. Группы трёх дней заканчиваются передflush/MADV_DONTNEED;
memory pages освобождаются, весьtable не держится вRAM. Все2 085 047 ID
должны встретиться ровно один раз; day provenance pinned, никаких implicit joins.
Конечныйcomplete публикуется atomic rename только при полном покрытии и SHA.

Home: существующая **`.venv-numemb`**, где уже квалифицированы E190/PLE и
`rtdl-num-embeddings==0.0.12`; runtime сверяется через `E.package_runtime()`.
CPUQuota200%/MemoryMax12G/swap0/nice10, единственная GPU очередь **послеE198**
(E196 уже завершён; его зарегистрированное расширениеE198 выполняется первым).
Таблица переносится изmmap наGPU
кусками8192строки, CPU pages освобождаются после каждого куска. НаGPU таблица
около9,2GiB; к ней добавляются query network/batch activations. CPU держит
исходные E190 query arrays и один ограниченный transfer batch, не весь token table.
Таблица используется всеми6fit пары и удаляется перед следующей парой.
Предварительный synthetic smoke резервирует **точное число байтов таблицы**
и измеряет full batch4096 наGPU; неудачный memory smoke запрещаетtarget-fit.
Никакого автоматического измененияbatch/precision/architecture после ошибки.

Подготовленные9,2GiB таблицы остаются на ноутбуках и передаются **сразу наHome**.
На Mac недостаточно места для двух таких копий: локальной коллекции больших
`values/known/present.npy` не делать. Возможен прямой rsync с уже существующим
доступом notebook→Home; без такого доступа использовать `scp -3 -r` изMac,
который соединяет два удалённых SFTP endpoint с существующими ключами Mac и
передаёт поток без промежуточного файла на диске. Ключи не копировать,
agent forwarding и изменение ssh trust не нужны. После доставки Home сверяет
всеSHA/IDs; на Mac возвращаются только небольшие complete/norm/smoke/evaluation
receipts и будущие predictions/model artifacts по доступному бюджету диска.

## Обучение, решения и ограничения

Фиксированы две пары1+7/4+10, обе рукиflat/ordered, seeds0/1/2,8epochs,
batch4096,AdamW lr0,002,wd1e−5,OneCycle pct_start0,1. Всего12fit.
Same seed даёт одинаковую инициализацию параметров и порядок mini-batches.
LossMSE на том же scaled ordinary residual; одинаковая сборка three-seed mean.
Никаких новых loss, reconstruction, physics, earlystop или hyperparameter search.

`local_order_screen_pass` требует **в обеих парах** отрицательные bodySSE(y≤7200),
decidedSSE,trim3SSE,trim20SSE дляordered противflat **и** противsavedPLE.
`context_capacity_screen_pass` — те же воротаflat противPLE. Нельзя считать
прохождение одного контроля победой над другим. Фиксированные own/previous-only/
empty/LIRFNM/LTFM/month diagnostics и airport-day bootstrap описательные;
по ним нельзя выбирать новые airport weights или thresholds после факта.
Неполныеseeds/строки не оцениваются, final metadata и seed→mixture arithmetic
повторно реконструируются. SHA,finite,ID,source drift — техническийstop.

При отрицательном результате эту конфигурацию не расширять автоматически.
Это не закрывает все trajectory representations/physics/полные raw треки.
При положительном результате root отдельно регистрирует расширение/full-chain
или ranking; этот worker не подаёт и не обещает board-score. Для обязательной
пред-регистрационной записи задан **экспертный prior ΔbodyRMSE=0с**,
**некалиброванный диапазон[−1,+1]с**: это предположение до данных, не измеренный
confidence interval, не прогноз доски и не основание сдвигать критерии принятия.
После опыта фактический результат сопоставляется с этой записью отдельно отgate.
Время fit оценивается из synthetic
секунд/batch, а не из названия архитектуры; ориентир очереди12fixedfit с
общим внешним watchdog4h, без автоматического resume/retry.

## Команды после root регистрации

```bash
.venv/bin/python scripts/e197_trajectory_encoder.py selftest
.venv/bin/python scripts/e197_trajectory_encoder.py pack --root . --output data/e197_inputs_20261007 --registration-commit SHA
```

На171 (для175 заменитьfold1→4, суффиксfold_1_7→fold_4_10 иPythonна`.venv-cat-parity`):

```bash
.venv/bin/python data/e197_inputs_20261007/scripts/e197_trajectory_encoder.py prepare-fold --root . --input data/e197_inputs_20261007 --output data/e197_prepared_20261007/fold_1_7 --fold 1 --registration-commit SHA
```

Payloadыprep запускаются в описанных systemd resource guards иsleep inhibitor.
Home послеE198, подhomeGPUguard:

```bash
.venv-numemb/bin/python data/e197_inputs_20261007/scripts/e197_trajectory_encoder.py smoke --root . --output data/e197_smoke_20261007.json
.venv-numemb/bin/python data/e197_inputs_20261007/scripts/e197_trajectory_encoder.py run --root . --input data/e197_inputs_20261007 --prepared data/e197_prepared_20261007 --output data/e197_trajectory_20261007 --smoke data/e197_smoke_20261007.json --registration-commit SHA
.venv-numemb/bin/python data/e197_inputs_20261007/scripts/e197_trajectory_encoder.py evaluate --root . --input data/e197_inputs_20261007 --prepared data/e197_prepared_20261007 --output data/e197_trajectory_20261007 --registration-commit SHA
```

Перед реальнымfit обязательны passedGPUsmoke/runtime/sourceSHA и обаprepared
receipt; `verify --input... --prepared...` независимо проверяет inputs/обе пары.
Syntheticконтракты: train-only normalization, unknown/PAD invariance,
empty pools, отсутствиеcross-visit mixing, flat permutation invariance,
ordered adjacency sensitivity, equalparameters, finite gradients иreal-sizeVRAM.

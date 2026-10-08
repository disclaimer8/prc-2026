# Передача всех PRC-задач Клоду — 8 октября 2026

Срез **12:14:42 UTC / 13:14:42 WEST**. Пользователь в 13:10 WEST передал управление Клоду. Codex прекращает самостоятельные эксперименты и запуски; три его агента остановлены, своих фоновых работ у них нет. Ниже измеренные результаты и оставшаяся работа. Этот файл и свежие записи Claude в `HANDOFF.md` имеют приоритет над старыми сводками.

## 1. Что реально готово сейчас

- Лучший обычный сабмит **v12075, 230.9387 с**. Последняя полная проверка доски — **08.10 10:31:23 UTC, 15/237**, файл `analysis/leaderboard_snapshot_20261008.json`. Это не обновление места на время передачи. Сегодняшнюю authenticated квоту Codex не проверял, новых подач сегодня от Codex нет.
- **Mac закончил свежие финальные признаки**, все 11 стадий, 64.89 с, peak RSS группы 12.53 GiB. Supervisor exit0, PASS; PID63325 уже завершён. Реального обучения финальных моделей **0**. Новой очереди после завершения признаков нет.
- **E207 v3 закончен: 18/18 fit, научный гейт FAIL.** Terminal12:00:56UTC, GPU-unit завершён успешно. Новый current-chain child №383 не запускать: обязательный parent PASS не выполнен.
- **E208 D-ATIS закончен**; узкое покрытие, модели/выигрыша RMSE нет. Итог №384, `analysis/e208_datis_20261008/RESULTS.md`.
- Tabular и original-E203 NN adapters реализованы, независимо reviewed READY, но **ещё нет регистрации actual final-fit, контрактов и обученных финальных моделей**. Tail assembly и семь ADS-B производных **ещё не реализованы**.

Первичные квитанции, исходные результаты E207, хеши подготовленного кода и снимок рабочего дерева скопированы в [`analysis/claude_transfer_20261008/`](analysis/claude_transfer_20261008/). `state.json` связывает всё по SHA. Четыре готовые feature-таблицы повторно проверены по SHA при передаче; заново их строить не нужно.

## 2. Сроки и правильные финальные входы

**Новый срок всего конкурса — 14.10.2026 10:00 UTC.** Главная страница независимо прочитана Codex при передаче и прямо подтверждает продление; Claude записал письмо организатора в `HANDOFF.md` и обновил план открытия. Старые записи о 11.10 как дедлайне репо устарели. Источники: [официальная главная](https://prc-data-challenge-2026.netlify.app/), `docs/organiser-clarifications.md`, `docs/final-submission-runbook.md`.

Финал: одна `gentle-igloo_final.parquet`, **670790 DEP**, январь/февраль/июнь/июль2026. Все четыре месяца обрабатываются одним рецептом. Использованы **последние файлы, перезалитые организатором в11:04Z**, а не утренние:

| Канонический файл в `data/prc-2026-datasets/` | SHA-256 |
|---|---|
| `final_ranking.parquet` | `4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f` |
| `final_submitting.parquet` | `b3445d80e29bf599a5583ca9675f63ddf57b51fdf1ca3f498facb38ea84f3924` |

Утренние файлы находятся в `final_v1_0847Z/`. Готовый WORK ниже уже использует правильные новые SHA; подмена входов не требуется. Дизеринг сохраняется по разъяснению организатора; старые Jan/Jul прогнозы и MVT-производные нельзя переносить просто по совпавшему ID.

**Финальную загрузку делать после последней обычной:** старый bucket inventory отвергает имя `_final.parquet`. Валидатор/инструкции загрузки — работа Claude, `scripts/validate_final_submission.py`, `docs/final-submission-runbook.md`. Не запускать старый обычный транспорт на final-файле. Вопросы/сообщения организаторам ведёт пользователь/Claude в уже согласованном порядке.

## 3. Готовые признаки Mac: пути и ограничения

Регистрация **№385 / `c0230421b7d60589fd9b358d707ad638f6e5785b`**.

```text
WORK = /Users/denyskolomiiets/prc-2026/data/final_feature_work_mac_20261008
WORK/manifest.json
WORK/feature_complete.json
data/final_feature_mac_execution_20261008/complete.json
data/final_feature_mac_execution_20261008/worker.log
```

Manifest SHA `14e210cadf8491a1bcd596b0ecd0a2f398271290656a07d273ca854d915a8260`.
Completion SHA `8494279d18351ba56706999985d171ba705ca9753b7a86724e8a47c1d5b57ea8`.
Все четыре таблицы содержат **2755837 строк = 2085047 train + 670790 final**:

| Путь относительно WORK | SHA-256 |
|---|---|
| `data/features_fin1b.parquet` | `2bb90b763e8e841f13047c6116f42b7f339f9a7ad60b42e5b02344c297f3ad98` |
| `data/features_e106.parquet` | `f337bdeecc854de42555502048685025136d6d97226782cf29eb4bf31bb9a682` |
| `data/features_e113.parquet` | `86efa94fc9cc66c1cb7e888edf5c3301d15f78c131eb318968408c2f844ed348` |
| `data/e185_motion_features.parquet` | `0c2a3ea1bc9fc1c9009907906b6611d6540714b0b0cab1fd80255effe546e24b` |

Новый union train+ranking важен: E157/E165 зависят от объединённого набора. Старые обучающие feature-values автоматически подставлять нельзя. Raw/static inputs уже staged в WORK, финальные данные представлены стандартными именами `ranking.parquet`/`submitting.parquet` внутри WORK.

Первая sandbox-попытка остановилась до FEATURE_STAGE из-за запрещённых `ps`/проверки группы. Её данные сохранены с суффиксом `.sandbox_attempt1`; повтор вне sandbox завершён. Это техническая попытка, не второй эксперимент. Не удалять/перезапускать успешный WORK.

Mac:48GiB,14 ядер; свободно около10.16GB на12:14UTC. Интерпретатор только существующий `.venv/bin/python`; **никаких `uv run`/`uv sync`**. Linux-only generic build не запускать повторно наMac: он требует25GiB; успешный Mac adapter удалял только свои новые промежуточные таблицы после последнего потребителя.

## 4. E207: закрытый результат, не очередь

Регистрация №379 `f43405e7c6836230862837b66fad081ae4d69777`.
Controller начал10:06:07UTC, terminal12:00:56UTC. Все365 raw-дней,6prep,18GPUfit завершены. На Home unit `prc-e207-v3-trajectory-20261008.service`, Invocation `4ccf79f86a144e4d9aaedd1ef0f169bd`, terminal MainPID0 / ExecMainStatus0. Controller55670 наMac завершён.

Источник: `analysis/e207_full_token_v3_20261008/execution/evaluation.json`, SHA `831f37e5c4677af4fe06e07bb32c09edbdca995965b1a9b5d9ce9a19dfe07549`. Копия сохранена в передаче. Это результат компонента **до ADS-B/current-chain**, а не всей боевой модели.

| Пара | ΔRMSE body, с | ΔRMSE decided, с |
|---|---:|---:|
| 1+7 | -0.106754 | -0.094719 |
| 2+8 | -0.000545 | +0.056357 |
| 3+9 | -0.094286 | -0.101226 |
| 4+10 | +0.489613 | +0.490549 |
| 5+11 | -0.134302 | -0.084363 |
| 6+12 | -0.097188 | -0.081197 |
| Pooled | **+0.007327** | **+0.015615** |

4/6 decided-пар лучше, но pooled ΔSSE body+6141594, decided+15023261, trim3+12930015, trim20+9187708: все положительные, поэтому **исходный гейт FAIL**. Сводные RMSE пересчитаны из сохранённых sums/rows при передаче; удалённые checkpoints целиком заново не аудировались. Не выбирать отдельные пары/веса постфактум.

Child current-chain №383 (`ef46d84e9f969879fe7e1d28e30c7726b87ac5e3`) имеет READY-код и локальную `registration.json`, **но parent-аудит/CV не выполнялись**. Его запуск обусловлен parent PASS; условие не выполнено. Final NN остаётся исходным **E203 ordered/unfilled**.

## 5. Подготовленный код и что ещё не сделано

| Блок | Файлы | Проверенное состояние | Следующий шаг |
|---|---|---|---|
| Feature build | `scripts/final_feature_workspace_20261008.py`, `scripts/final_feature_mac_20261008.py` | Registered;12tests+4lifecycle PASS; actual build PASS | Использовать готовый WORK |
| Tabular models | `scripts/final_tabular_models_20261008.py`, одноимённый test; `analysis/final_tabular_models_20261008/` | Independent READY;9tests+18merge checks; sourceSHA `ba9a5e3aef9ef93b6d692806cba844c8fb0c1d075a6346245ac60c9ddf1bea37` | Создать actual input contract, зарегистрировать fixed queue, attest, fit/merge |
| Original NN | `scripts/final_ordered_trajectory_20261008.py`, test; `analysis/final_ordered_trajectory_20261008/` | Independent READY;13tests; sourceSHA `510f3d7c03455da75681e6b3cb5e1aa81dba787a7a5211ff0a957c7ddc3355f2`; pending=false | Регистрация, raw/queries/context/smoke/run/replay на Linux/HomeGPU |
| Daily tail derivatives | `scripts/final_adsb_derivatives_20261008.py` | **Файла, теста и согласованного receipt interface нет** | Реализовать reuse существующих7producers, uniform finalMVT |
| Tail assembly | `analysis/final_tail_assembly_20261008/PLAN.md`, `source_cache_inventory.json` | План одобрен; hashes16cachedmodels проверены; **adapter/test отсутствуют** | Реализовать tasks1–3 плана |
| Portable v12075 | `scripts/v12075_portable_bundle.py`, №381 | Reviewed7tests; регистрация есть; **сборка/архив/replay не запускались** | Отложено ради финала; не путать со старым exact replay |

В draft tabular PROTOCOL внизу ещё написано7tests/ожиданиеreview; фактический окончательный review —9+18, см. `REVIEW_FINDINGS.md`. Коммит сохранения draft-кода при передаче **не является регистрацией запуска final-fit**.

Tabular контракт обязан связать готовый `feature_complete.json`, FIN1b/motion/template SHA, ordered train/final IDs и helper sources. `attest` создаётся наMac против полного commitSHA; remote fit читает receipt, Git-history наHome не нужен. CLI: `attest`, `fit --kind ... --seeds ... --threads ...`, `merge`; точные аргументы в `main()`/PROTOCOL.

**Tabular production guard пока Linux-only.** Cat≤12GiB, LGB≤32GiB, CPU≤заданных потоков, swap0,nice10,diskfloor4GiB; внешний systemd unit должен задавать RuntimeMaxSec. Mac CatBoost supervisor **не реализован/не квалифицирован**. Не обходить guard, молча меняя платформу. Synthetic Mac native tests требовали OMP_NUM_THREADS=1/OMP_THREAD_LIMIT=1 доimports; без них был segfault. Среду не менять.

NN queries требует Linux с Torch даже дляCPUprep. Home `.venv-numemb` подходит как ранее проверенный GPUruntime; Cloud tabular import success не доказывает Torch. Adapter сохраняет16TE/TER,8групп stats, categories,quantiles,proxy,normalization,checkpoints. Исходные2025 E194raw разрешено переиспользовать только при точном source/day/MVT/ID/receipt совпадении. **Все2026tokens пересчитать по новым finalMVT.** После этого3seeds×8epochs и точный checkpoint replay.

## 6. Рецепт финала и критические зависимости

Точный исходный план: `analysis/final_pipeline_readiness_20261008.md`. План до смены владельца:

1. Base classifier180 + proxy regressor2319; Cat15×5600; LGB5×2319.
2. Пять anchors по5seeds: raw2565,lobt3065,sched2958,eobt2736,aobt2738; E1855×2481. Всего52tabularfits.
3. Original E203 ordered3seeds×8epochs; суммарно55productionfits. Без E207fill.
4. Cat/LGB .65/.35; anchors raw1/30, остальные4по.10; NN.25 ровно один раз; E185 .35×2/3 по прежней маске.
5. E167 → **fresh E156 TabPFN seeds0..9/pfx на всём served tail** → E141/E64 → E188 → E152 → LOBT lift/clip → защита всех E156servedIDs → LSZH half.

**E81 и42historicaldonors исключены из нового uniform final recipe.** Нужны raw classifier probabilities из saved classifier, а не уже masked `base.proba`. Защита E156 применяется ко **всем servedIDs**, даже если округлённый E156предикт случайно не изменился. Эти два места подробно разобраны в tail PLAN, не исследовать повторно.

Семь свежих суточных продуктов на все final-дни: kt1/offblock2,3,5; sw_kt1/offblock2; visit; visit_m; head(default3kt). Producers уже существуют (`adsb_offblock_v2/v3/v5.py`, `adsb_visit.py`, `e69_visit_merge.py`, `e134_head_features.py`). E194tokens их не заменяют. Same-day/threshold/identity semantics сохранять. Нужен receipt, отличающий обработанный день без совпадений от пропущенного дня.

Исторические2025stageinventories содержат364дня (нет31декабря) — это captured fit input. Не дополнять их молча и не инвалидировать кэш. Для11E141/E64/LSZH caches signature включает полный MacPython3.12.14 buildstring,np2.5.3,pl1.44.2,LGB4.7.0, матрицы/порядок/веса/параметры. ПятьE188cache имеют свои368boundinputs. Кэш использовать лишь при точном соответствии; иначе отдельное явное fixed-refit решение, не silent fallback. Проверка сигнатуры может читать trainlabels, поэтому actual replay/refit зарегистрировать.

OPDI:44local event files~1.53GB, Feb/Jun полнота не подтверждена; при отсутствии матча оставлять прогноз по исходному правилу. TabPFN cached V2 weight SHA `cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49`; настоящее offline/runtime qualification и10seedcalls ещё не выполнены.

## 7. Машины и чужие процессы

| Узел | Доступ / каталог | Последнее известное состояние и ограничения |
|---|---|---|
| Mac | этот репо, `.venv/bin/python` | Feature complete12:11UTC; новых долгих Codexjobs нет.48GiB/14cores/~10.16GBdisk. Не менять `.venv` |
| Home | `ssh home-pc-lan`, `/home/a1/prc-2026` | E207GPU закончился12:00UTC по terminalreceipt. Ryzen7900X3D/32GiB/RX7900XTX24GB. `.venv-numemb` GPU; `.venv-cat-oof` CPU. Никакого нового finalfit не запускалось; чужиеjobs заново проверить перед назначением |
| prc-cloud | `ssh prc-cloud`, `/root/prc-2026` | 2.29.61.190, ARM16cores/30GiB/~279GiBdisk. Другая сессия уже создала `.venv`,926MBdata,`node_probe.py`,catboost_info к11:49UTC. **Не переустанавливать и не перетирать.** Root не проверил runtime/parity результаты; инвентарь `analysis/prc_cloud_20261008/` |
| Hetzner | `ssh hetzner`, `/root/prc-2026` | E208 завершён10:54UTC;~8.5GiBdisk на старом снимке. Прод FlightFinder:1LGBjob,≤6physicalcores0,2,4,6,8,10,nice10/passiveOMP |
| Laptop171 | `ssh a1@192.168.1.171`, `/home/a1/prc-2026` | E207raw/prep закончены; новых Codexjobs нет.14GiB RAM/6CPU разумный предел; текущие чужиепроцессы заново не проверены |
| Laptop175 | `ssh a1@192.168.1.175` | **Claude raw-prefetch** `/home/a1/claude_adsb_2026/`,58FebJunдней,logs `progress.tsv`,`run.log`. Старт10:42UTC, root видел9/58на11:02UTC; позже не обновлял. Не считать это текущим числом, не дублировать/останавливать |
| ASUS176 | Windows/PowerShell7 | Квалификация принадлежит Claude, журнал№382; root не имеет свежего статуса. Никакой новой Codexfitочереди |

E201 отменён, LLM/Ollama off — не возобновлять. Не останавливать чужиеCI/FlightFinder/Claudejobs. Ранее automatic review отказал в передаче final_ranking на175; обход другим transport/переименованием не допустим. Производные можно делать на разрешённом узле с готовым WORK.

SSH/ps иногда запрещаются sandbox: повторять **тот же** read/launch с необходимым escalation, а не многократно менять команды. Для общего `ps` не выводить command arguments: в чужих процессах встречаются credentials. Пользователю не обещать загрузку CPU/GPU без проверенных PID/stage/receipt.

## 8. Порядок продолжения для Claude

1. Принять этот handoff и готовые признаки; проверить собственный Cloud probe и175prefetch. **Не повторять featurebuild и E207**.
2. Зафиксировать actual tabular input contract и ресурсы, затем распределить готовые fixedseeds по допустимым узлам. Это ближайшая реально готовая вычислительная работа. ДляMac сначала минимальный корректный supervisor, если выбран этот host.
3. Реализовать7dailyderivatives и привести rawFebJun/JanJul к source-bound inventory. Параллельно зарегистрировать/запустить original NNraw/queries/context наqualifiedLinux, потом3GPUfits.
4. Реализовать tail PLAN, собрать uniform finalcandidate, сохранённые модели/преобразования/receipts и end-to-end replay. Проверить final schema670790IDs перед какой-либо загрузкой.
5. Обычные слоты/новые S1–S3 предложения Claude решает сам по актуальной квоте и журналу; готового нового ordinarycandidate от Codex при передаче нет. Смеси с известными родителями предварительно оценивать по Граму.
6. Довести README/reproduction/data-license/docs и открытие по обновлённому плану Claude. Старый `run_final_any_period.sh` всё ещё v808; exact saved-components replayv12075 уже доказан07.10, но clean-clone bundle№381 не сделан. Не называть эти два результата одним.

## 9. Рабочее дерево и границы передачи

История сохранена, без reset/clean. До передачи чужой tracked diff был в `analysis/e197_trajectory_execution_20261007.md`; его не трогали. Много untracked research/старых executionfiles — список в `analysis/claude_transfer_20261008/git_status_before_transfer.txt`. Коммит передачи включает только явно перечисленные собственные drafts и квитанции, а не всё дерево. Файлы данных/models остаются на своих узлах; скрытой отправки или публикации нет.

Все Codexagents подтвердили stop. После просьбы пользователя новые fits/downloads/submissions не назначались. Последняя Mac задача завершилась сама успешно во время подготовки передачи. E207/E208 результаты сохранены; нового достигнутого RMSE на доске нет. Дальнейший владелец всех задач — Claude.

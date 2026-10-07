# E207 — заполнение свободных позиций исходного K64 sampler

Статус: код и протокол для независимого review и последующей регистрации. До root prereg commit запрещены real extraction, новая подготовка fold tables / normalization и target fits. Разрешены synthetic tests и metadata из существующих E194 массивов. Никакой ranking prediction или подачи этот протокол не разрешает.

Точная строка регистрации: `E207-full-K64-retain-original-raw-index-fill-sixpair-3seed8epoch-v1`.

## Гипотеза и единственное изменение

Исходный E194 выбирает `unique(searchsorted(t, linspace(t[0],t[-1],64)))`. При неравномерном приёме несколько grid points могут дать один raw index; массив остаётся K64, но часть позиций — PAD. E207 проверяет, полезны ли дополнительные **реальные позиции исходного raw ряда**. Это изменение observation count и последовательности соседей при прежней ёмкости модели. E204 менял receptive field при прежних tokens и эту гипотезу не проверял. E25/E190 summaries первого движения/остановок не восстанавливают потерянный trajectory context.

Один новый arm F против сохранённого ordered O. Все старые индексы сохраняются. Для n исходных точек:

```python
old = original_sample_time_indices(t)
remaining = flatnonzero(boolean_mask_with_old_indices_removed)
need = min(64, n) - len(old)
extra = remaining[rint(linspace(0, len(remaining)-1, need)).astype(int)]
new = sort(concatenate([old, extra]))
```

При need=0 возвращается old. Получается ровно min(64,n) уникальных возрастающих raw indices; при n≤64 массив полностью прежний. Вычисление линейно по n с сортировкой не более64 элементов. Дубликаты timestamps сохраняют исходный raw-index порядок; позиции не приравниваются к независимым наблюдениям. Нет интерполяции, новых timestamps, event priority, нового stale/receiver feature или нового окна. Вся выборка внутри исходного visit и t≤MVT. Не изменяются identity resolution, own/previous visit, query rows, 12 source channels или исходные правила masks/PAD.

`encode_day` использует прежний E194 code object в отдельном globals dictionary с единственной заменой `sample_time_indices`. Глобальные функции E194 и numpy не меняются. Для каждого дня до сохранения F проверяются равенство IDs/metadata/raw_ranges/raw_counts, old⊆F, точное равенство channels/known/timestamps сохранённых индексов, полное равенство arrays при n≤64, prefix PAD, finite values и cutoff. RAW duplicate-timestamp semantics сохраняется.

Риск: time-uniform выборка могла защищать от reception-density shift. F добавляет burst points, возможно дубликаты и stale observations; меняет mean/max pooling, соседство и train-only numerical moments. Это не доказанный bugfix или выигрыш. Экспертный незакалиброванный prior body ΔRMSE=0с, диапазон предположения [−1,+1]с; metadata не является оценкой RMSE.

## Данные, модель, контроль

365 дней 2025, 2 085 047 query rows; удержанные пары (1,7), (2,8), (3,9), (4,10), (5,11), (6,12). Источник — зарегистрированные E194 raw surface+label-free departure payloads; прежние NPZ содержат channels только old64, а raw_ranges — ссылки на полный source parquet, поэтому одного NPZ недостаточно. Каждый shard извлекается ровно один раз на ноутбуке, где уже есть его raw inputs. 2026 не извлекается и в fit не участвует.

Контроль: saved ordered E197 для folds1/4, E202 для остальных. Перед extraction проверяются source SHA, исходные регистрации и specification SHA, model/prediction hashes, seed mean и точная final-mixture reconstruction через прежний `Q.checked_arm`. E197 registration `bfe793c62f7cecf633c7cc31c747f48eb0676f2d`; E202 `55fe9b68f7cc8405681f6c85d576979e067b55c3`, E202 spec `6d604eaaca830fa83c18ebb78986b028d338869123d5f8c10b2450dde1ec4309`; E197 spec берётся из frozen `Q.OLD_SPEC`. Исторический отобранный upstream сравнивается попарно и не выдаётся за независимый full-current CV.

Прежние E190 query tensors/targets, numerical bins, embeddings, exact E197 ordered `network` и `fit`; seeds0,1,2, epochs8. Всего **18 fits**, все6пар, без раннего отбора удачных первых пар. Runtime должен совпасть с сохранённым контролем; каждый fit обязан иметь прежнее число parameters,8 losses и peak≤20GiB.

Прежний номинальный .25 neural slot сохраняет **точный inherited per-row coefficient** `ctrl['mlp_coefficient']`: на E185 subset23/120, остальные .25. Нельзя заменять его literal .25. Нет новых весов смеси или калибровки.

Нормализация использует exact code прежнего E197 prepare_fold: moments первых6 continuous channels только по train indices данного fold, std fallback и clipping[−8,+8], float16, unknown/PAD→0. RULE прежнее; moments пересчитываются для F, поэтому сами значения не обязаны совпадать с O. Ни heldout trajectories, ни labels heldout не входят в estimation. По наблюдениям actual channels сохранённых индексов обязаны совпадать **до** новой нормализации. Prepared shape/parameter count прежние, число present tokens увеличивается.

## Заранее фиксированный gate и диагностика

Для каждого fold ΔSSE = SSE(F)−SSE(saved O), отрицательное лучше. E207 проходит только если одновременно:

1. Decided ΔSSE<0 как минимум в4 из6пар.
2. Сумма body ΔSSE<0 и сумма decided ΔSSE<0 по всем6парам.
3. Сумма within-fold symmetric trimmed ΔSSE<0 и при3, и при20 удалённых наблюдениях с каждого края. Как в frozen `G.measured`, сортировка/trim выполняются **в decided population каждого fold**, не среди всех rows и не одной общей сортировкой.

Body=y≤7200; decided mask точный из saved evaluation. Никакого изменения gate после первой метрики. Все2085047 heldout IDs должны быть уникальными и полностью покрыты. Артефакты candidate независимо проверяются через `Q.checked_arm`, затем против evaluation по ID/y/month.

Фиксированные дополнительные diagnostics, без routing и выбора модели: all/body/decided ΔSSE по own-present, previous-only, no-points, аэропортам и удержанным месяцам. Label-free extraction receipts: original/new/added positions, added duplicate timestamps/stale/unknown-stale и selected/added по аэропортам. До extraction отдельная metadata receipt сравнивает train2025 и ranking Jan/Jul2026 по month×airport: raw compressed support, old slots и min(64,raw_count); это prospective cardinality, не новые extracted values, не independent-observation count и не target evidence. Разрешённая проверка427 существующих NPZ дала train88,770,275→120,520,519 позиций (+35.77%), ranking15,753,762→22,656,722 (+43.82%). EDDF+37.59% против+73.51%, EGLL+39.74%/+56.41%, LFPG+37.66%/+54.78% особенно подчёркивают transfer risk. Доля больше означает больше новых позиций, не улучшение модели.

PASS означает только LGB5/preADSB component gate. Перед любым ranking нужен **отдельный preregistered fixed current E203 preE81 chain protocol**; exact full-tail OOF отсутствует. E207 всегда пишет `accepted_for_submission=false`; E201 не возобновляется.

## Исполнение и ограничения

Новые immutable namespaces: `data/e207_inputs_20261008`, `e207_raw_20261008`, `e207_materialized_20261008.json`, `e207_prepared_20261008`, `e207_incoming_20261008`, `e207_trajectory_20261008`, `e207_smoke_20261008.json`. Существующие E194/E197/E202 inputs/outputs остаются источниками только для чтения. Повторный запуск в существующие output paths запрещён; partial artifacts сохраняются. После сбоя нет retry/resume/автоматического fallback.

Одна очередь на Mac упаковывает малый sealed bundle, предварительно сверив registered code/protocol/test с commit и точную protocol string в журнале. Перед запуском проверяются реальные running PRC units/processes на171/175/Home, свободный диск Mac≥2GiB, Home≥70GiB, каждый ноутбук≥40GiB. Ранний detached PID не доказывает запуск fits: подтверждение — `execution/events.jsonl`, unit invocation/MainPID и реальные progress logs.

171 использует existing `.venv`, raw shard0, folds1/2/5;175 — existing `.venv-cat-parity`, shard1, folds3/4/6. CPU extract/prep:6CPU,10GiB RAM,swap0,nice10; каждый raw shard вычисляется один раз. Shards обмениваются через `scp -3`; только complete receipts приходят наHome. После двух extraction manifests создаётся derived manifest с exact raw hashes. Prepared tables передаются непосредственно laptop→Home через `scp -3`, не сохраняются наMac. Home admission1CPU/1GiB проверяет SHA, нормализацию, shape/dtype, masks/cutoff-derived counts, ID order и неизменные support counts. Home GPU2CPU/12GiB/swap0/nice10, existing `.venv-numemb`, VRAMcap20GiB. Среды не меняются.

Перед первым target fit отдельный процесс выполняет frozen full-shape **synthetic** GPU smoke: полный trajectory allocation, batch4096, capacity/RNG/PAD checks и backward pass; проверяются runtime и peak VRAM. Smoke не использует labels. На Mac тестируется exact alias model/fit; full GPU smoke локально не заявляется выполненным.

Шесть таблиц ≈55.2GiB наHome, три≈27.6GiB накаждомноутбуке, новые compressed raw ориентировочно3–5GB (не измерено), модели/receipts дополнительно. Mac хранит только малый input bundle и receipts, не55GiB tables. Historical E204 GPU6fits≈24мин даёт оценку E20718fits≈72–75мин; extract+verify≈5–15мин, prep≈5–10мин, LAN55GiB≈25мин при40MB/s — оценки, не E207 timings. Плановый workflow≈2ч, единый hard deadline6ч.

Все remote stages — existing systemd user units с `KillMode=control-group`, `TimeoutStopSec=30`, `RuntimeMaxSec` равным оставшемуся общему6ч budget. Controller журналирует starts/InvocationID, table admission и completion; workers печатают extraction days/preparation blocks/fit epochs. Controller progress30с — liveness, не доказательство научного прогресса. При ошибке или SIGINT/SIGTERM/SIGHUP прерываются только принадлежащие controller local transport process groups и unit invocations (при потерянном ACK — exact unique Description). Повторные сигналы не прерывают уже начавшуюся cleanup; reentrant lock покрыт real-signal test. Cleanup SSH/stop waits ограничены. Недоступный host может задержать подтверждение остановки: unit RuntimeMaxSec остаётся независимым пределом и cleanup error сохраняется.

## Review и команда

Owned review files: `scripts/e207_full_token_sampler.py`, `scripts/e207_full_token_queue_20261008.py`, `tests/test_e207_full_token_sampler.py`, этот protocol и metadata/test receipts в текущей analysis directory. Frozen upstream scripts не изменяются. Root фиксирует review+code+protocol+журнал единым prereg commit. До этого команда ниже **не запускается**:

```sh
.venv/bin/python -u scripts/e207_full_token_queue_20261008.py --root /Users/denyskolomiiets/prc-2026 --registration-commit <FULL_ROOT_PREREG_COMMIT> --detach
```

Первый launch пишет exclusive `analysis/e207_full_token_20261008/launch.json` (PID/registration/source hashes), `controller.log`, затем `execution/events.jsonl`. Termination относится к controller process group с этим PID; не применять broad pkill. Конечный `evaluation.json` и `complete_no_ranking` обязательны для утверждения о завершении. Targeted tests:

```sh
.venv/bin/python -m unittest discover -s tests -p test_e207_full_token_sampler.py
```

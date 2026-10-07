# E202 — фиксированное расширение E197 на оставшиеся четыре пары

Protocol ID: `E202-trajectory-remaining4pairs-flat-ordered-3seed8epoch-v1`.
Код: `scripts/e202_trajectory_extension.py`; durable controller:
`scripts/e202_execution_queue_20261007.py`. Требуется один коммит code + protocol
+ journal до pack, preparation и target-fit. Старый E197 и среды не меняются.

## Основание и гипотеза

E197, регистрация `bfe793c62f7cecf633c7cc31c747f48eb0676f2d`, завершил12fit
на парах1+7 и4+10. Independent audit подтвердил оба исходных gate, все32
artifact hashes,12fit×8finite losses, exact IDs/y/month, three-seed means и
mixture. Complete SHA `84842a0734975ae76bc27ebdf0fdc5841bc769a7dc8322255672d0c48c4f75db`,
evaluation SHA `e1b6903892865a948d896ecbaf895bca456ad7de2ea23427d13977bd81f3fb3e`.
`analysis/e197_verified_results_20261007.json` содержит независимую проверку.

Теперь проверяется перенос того же локального смешивания соседних наблюдений
своего борта на пары2+8,3+9,5+11,6+12. Это продолжение выбранного screen,
**не четыре новых настройки и не независимая оценка на всех шести парах**.
Первые две пары явно отмечаются selected_screen, оставшиеся четыре — confirmation.
Отрицательный full current-chain E200 MDN не меняет этот trajectory recipe.

## Неизменная наука и входы

Прямые вызовы неизменных `e197_trajectory_encoder.fit`, network, gpu_tables,
numeric helpers; fixed flat и ordered,3seeds0/1/2,8epochs,batch4096,
AdamW lr0.002,wd0.00001,OneCycle pct_start0.1. Всего24 новыхfit.
PLE48/d12/B,width512; two streams×64actual tokens×12channels;
point encoder24→64→64, два одинаково параметризованных message blocks и
общая fusion. Flat использует token/mean/max; ordered — previous/self/next.
Оба плеча видят одинаковые age/gap/known/present. Равноёмкость проверяется
точным числом параметров. Полный механизм описан в исходном протоколеE197.
Не меняются initialization, batch order, loss, target, normalization, precision,
seed count, смесь или heldout query feature transformations.

Сохранённые immutable E190 query tensors всех четырёх пар, registration
`a926f3848baaae7415bd3cb773cbd39728ce1851`, используются по ссылкам и SHA,
без дублирования многогигабайтных query files. Новые indices/control/evaluation/
savedPLE берутся из E193 parent spec
`2b8f65e7212c91523d87a166ceb955376784223ee1dfb9cf1b403fa2f88a0f70`:
только проверенная карта строк и прежние controls, **никакие weather inputs**.
Pack повторяет все SHA, восемь проверенных numeric columns в parent receipt,
disjoint train/holdoutindices, exact ID/y/month alignment и original query spec.
Полный query ID order обязан совпасть с E197 spec
`93d90131cdfa32c04b8f8c3513f3e7f7cc19c9695d8f47d3fe98b4cc8c0a407d`.

Raw365days2025 — прежние immutable E194 shards0/1,2 085 047ID; all427days
и ranking2026 не входят в normalization. E194 provenance/cutoff unchanged:
actual tokens t≤MVT, retrospective same-day segment boundary matching;
не заявляется онлайн-прогноз или непрерывная ground-truth taxi trajectory.
31December4715rows полностью masked, ambiguity/missing остаются UNKNOWN.
Никаких neighbor targets, BLOCK/TAXITIME tokens, новых auxiliarylabels.

Подготовка normalizes только known observations ordinary training rows,
с прямым запретом heldout months. Channels0…5 float64 mean/std, clip±8,
zero variance→1; otherchannels сохраняются; missing again0. `prepare_fold`
и `verify_prepared` скопированы дословно из E197 в новый модуль, чтобы работать
с новой provenance/четырьмя парами без monkeypatch старого модуля.
AST selftest проверяет точное равенство этих функций; numerical helpers —
прямые ссылки на оригинальные функции. Никакой normalization по validation.

## Публикация, ресурсы и непрерывная очередь

Mac `.venv/bin/python`: pack после регистрации, контроллер и небольшие receipts.
Результирующий bundle `data/e202_inputs_20261007` содержит источники/протокол,
metadata/control,selected screen predictions, runtime/smoke и все hashes.
E197 screen перед pack заново реконструируется по всем12fit и gate.
Сохранённый full-shape synthetic GPU smoke E197 допускается только при точном
совпадении source unchangedS и runtime. Он уже проверял4096batch с полным9.2GiB
GPUtable и finitebackward. Новых modeloperations в E202 нет.

171 готовит2+8 затем5+11,175 готовит3+9 затем6+12. Unit names:
`prc-e202-prep-171-20261007.service`, `prc-e202-prep-175-20261007.service`.
Существующие `.venv`/`.venv-cat-parity`,CPUQuota600%,MemoryMax10G,
swap0,nice10,RuntimeMax14400s и sleep/idle inhibitor. Внутри3prepworkerthreads.
Перед стартом требуется≥24GiB free на каждом ноутбуке и≥45GiB наHome.
Никакие исходные файлы, E201 inputs/outputs или окружения не перезаписываются.

Каждая prepared table публикуется atomic rename после полного SHA/IDcoverage.
После двух проверенных prep worker публикует
`data/e202_handoff_171_20261007.json` или175 со schema:
`status=two_prepared_complete`,protocol,registration_commit,input_spec_sha256,
node,folds[2,5]или[3,6],prepared_receipts[{fold,complete_sha256,files}].
E201 CPU release требует этот marker и фактический success/inactive/MainPID0
prepunit. **Home admission и завершение передачи не являются условием release**:
файлы неизменны, paths disjoint, проверен свободный диск. Это заранее разрешает
E201 считать параллельно остаточной передаче по сети.

Два потока `scp -3 -r` соединяют ноутбук→Home через Mac без intermediate disk
files, key copying или agent forwarding. Большие37GiB таблицы на Mac не копируются.
Home получает fresh `data/e202_transfer_20261007/fold_N_N6`; admission по очереди
в одном одновременно unitCPU100%/MemoryMax1G/swap0/nice10/RuntimeMax1800s.
Он проверяет все files SHA, IDs, normtrainingdigest/heldoutmonths, shapes/dtypes,
finitevalues, binarymasks, known≤present, zeroPAD, clip8, support==anypresent.
Coverage фиксировано:88 770 275points;1 251 417own,1 345 381previous,
1 022 407both. Atomic admission.json + rename в `data/e202_prepared_20261007`.

GPUunit `prc-e202-trajectory-20261007.service`, Home `.venv-numemb`,
CPUQuota200%/MemoryMax12G/swap0/nice10/RuntimeMax18000s. Только один GPUfit;
контроллер сверяет опубликованный old E197 completeSHA и отсутствие activeE197.
GPU ждёт admission первой пары2, затем считает по фиксированному порядку2,3,5,6.
Он может считать2, пока3/5/6 ещё передаются; не ждёт все37GiB. Для каждой пары
повторно проверяются hashes, считывается однаGPUtable; шестьfit заканчиваются
до освобождения таблицы. Wait однойпары≤2h, без измененияrecipe при задержке.
Home E201 CPU12/16G разрешён параллельно; admissionодновременно только1G.
E203 successor может ждать успешного complete всех24fit, не gatePASS.

One-shot controller сохраняет PID/unit/invocation/статусы в
`data/e202_execution_20261007.{json,jsonl}`. Сетевой/fit/OOM/source failure —
failed_no_retry; никакого автоматического повторного fit/resume/уменьшенияbatch.
Remote units bounded, и потеряMac не создаёт бесконечный worker. Опубликованные
частичные результаты не считаются complete. Complete создаётся только после24fit.
Mac собирает только models/predictions/logs/малые receipts, повторяет все mean/
mixture/ID/metrics в отдельной hardlinked review copy, не изменяя оригинал.

Оценка времени до данных: E19712fit заняли около47мин от actualfit доcomplete;
E20224fit ожидаются≈95–110мин плюс первыеprep/transfer/admission.
Это operational estimate, не обещание срока. CPUprep двухпар≈2–5мин/ноутбук,
сеть≈20мин для37GiB, перекрыта первымfit и E201CPU после marker.

## Решение и отчёт

Все шесть пар: `local_order_six_pair_pass` требует **для каждого сравнения**
ordered−flat и ordered−savedPLE: ≥4/6пар с decidedSSE<0, pooledbodySSE(y≤7200)<0,
сумма within-fold symmetric trim3SSE<0 и trim20SSE<0. Trimpopulation —
**ALL rows**, как исходный E197/D.metrics; не DECIDEDtrim E195/E200.
Pooled decided знак отдельно сообщается, но не добавляется как новый gate.
`context_capacity_six_pair_pass` применяет те же условия flat−savedPLE.
Нельзя подменять localordergate одним успешным сравниваемым контролем.

Отдельно публикуются remainingfour pool/каждаяпара, чтобы не скрывать selection
двухпервых; all6selection-informed, new4confirmatory. All2 085 047IDs должны
встретиться ровно один раз, finite и exact metadata обязательны. Каждый seed
имеет8finite losses, всеmodel/predictionSHA и exactseedmean/productionmixture
проверяются. Диагностика own/previous-only/no-points/LIRFNM/LTFM/month
описательная, не основание подбирать threshold/вес/аэропорт.

Это LGB5/pre-ADSB component wrapper, **не current production chain CV**.
Даже passed24fit не разрешает здесь ranking/submission: последующий pipeline
регистрируется отдельно. При отрицательном результате этотfixedextension закрыт;
автоматического архитектурного/seed/весового поиска нет.
Экспертный pre-data prior для pooled remainingfour ΔbodyRMSE ordered−flat:
0с, некалиброванный диапазон[−1,+1]с. Это не confidenceinterval и неboardforecast;
измеренное попадание в диапазон сообщается отдельно от gate.

До регистрации: py_compile и selftestPASS (trainonly moments, missing/constant,
float16finite, exactIDs/duplicates, ASTequalprep, ≥4/6иstrictnegativegateboundaries).
Протокол не ослабляется после просмотра новыхtargets/ошибок.
